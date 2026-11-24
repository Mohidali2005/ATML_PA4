import math

import torch
import torch.nn.functional as F


def allowed(lq,lk,device,causal=True,mask=None):
    """Boolean table of the key positions each query may see"""
    ok = torch.ones(lq,lk,dtype=torch.bool,device=device)
    if causal:
        # aligning the last query with the last key so a decode step sees the whole prefix
        ok = torch.tril(ok,diagonal=lk-lq)
    if mask is not None:
        ok = ok & mask
    return ok


def dense_attention(q,k,v,causal=True,mask=None,stable=True):
    """Explicit attention that builds the full score table in the input precision

    mask may hold one table per head and an unstable softmax skips the max subtraction"""
    scale = 1/math.sqrt(q.shape[-1])
    s = torch.matmul(q,k.transpose(-1,-2))*scale
    s = s.masked_fill(~allowed(q.shape[-2],k.shape[-2],q.device,causal,mask),float("-inf"))
    if stable:
        s = s-s.amax(-1,keepdim=True)
    p = torch.exp(s)
    p = p/p.sum(-1,keepdim=True)
    return torch.matmul(p,v)


def tiled_attention(q,k,v,tile=128,causal=True,mask=None):
    """Exact blockwise attention with an online softmax that never holds the full score table

    scores and the running max sum and output stay in fp32 like the fused kernels and a row
    with no visible key returns zeros"""
    b,h,lq,d = q.shape
    lk = k.shape[-2]
    scale = 1/math.sqrt(d)
    shift = lk-lq
    rows = []
    for a in range(0,lq,tile):
        qt = q[...,a:a+tile,:]
        n = qt.shape[-2]
        m = torch.full((b,h,n),float("-inf"),device=q.device)
        l = torch.zeros(b,h,n,device=q.device)
        o = torch.zeros(b,h,n,v.shape[-1],device=q.device)
        # skipping key tiles that lie entirely in the future of this query tile
        stop = min(lk,a+n+shift) if causal else lk
        for c in range(0,stop,tile):
            kt,vt = k[...,c:c+tile,:],v[...,c:c+tile,:]
            s = torch.matmul(qt,kt.transpose(-1,-2)).float()*scale
            ok = allowed(n,kt.shape[-2],q.device,False,None if mask is None else mask[...,a:a+n,c:c+kt.shape[-2]])
            if causal:
                qi = torch.arange(a,a+n,device=q.device)[:,None]+shift
                kj = torch.arange(c,c+kt.shape[-2],device=q.device)[None,:]
                ok = ok & (kj <= qi)
            s = s.masked_fill(~ok,float("-inf"))
            new = torch.maximum(m,s.amax(-1))
            # keeping rows that have seen nothing yet away from minus infinity minus minus infinity
            safe = torch.where(torch.isinf(new),torch.zeros_like(new),new)
            alpha = torch.exp(m-safe)
            p = torch.exp(s-safe[...,None])
            l = alpha*l+p.sum(-1)
            o = alpha[...,None]*o+torch.matmul(p.to(v.dtype),vt).float()
            m = new
        rows.append(torch.where(l[...,None] > 0,o/l.clamp(min=1e-30)[...,None],torch.zeros_like(o)))
    return torch.cat(rows,-2).to(q.dtype)


def sdpa_attention(q,k,v,causal=True,mask=None):
    """PyTorch scaled dot product attention with whatever backend it dispatches to"""
    if mask is None:
        return F.scaled_dot_product_attention(q,k,v,is_causal=causal)
    return F.scaled_dot_product_attention(q,k,v,attn_mask=allowed(q.shape[-2],k.shape[-2],q.device,causal,mask))


def window_mask(length,window,device):
    """Causal sliding window over the current and previous window minus one positions"""
    i = torch.arange(length,device=device)[:,None]
    j = torch.arange(length,device=device)[None,:]
    return (j <= i) & (i-j < window)


def dilated_mask(length,step,count,device):
    """Causal mask over the positions at offsets zero step two step and so on"""
    i = torch.arange(length,device=device)[:,None]
    j = torch.arange(length,device=device)[None,:]
    return (j <= i) & ((i-j) % step == 0) & (i-j < step*count)


def feature(x):
    """Positive feature map elu plus one"""
    return F.elu(x)+1


def linear_attention(q,k,v,eps=1e-6):
    """Causal linear attention that materializes the running state at every position"""
    fq,fk = feature(q.float()),feature(k.float())
    state = torch.cumsum(fk[...,:,None]*v.float()[...,None,:],dim=-3)
    norm = torch.cumsum(fk,dim=-2)
    num = torch.einsum("bhtd,bhtde->bhte",fq,state)
    den = (fq*norm).sum(-1,keepdim=True)+eps
    return (num/den).to(q.dtype)
