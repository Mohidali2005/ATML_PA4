import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from task2_attention.ops import dense_attention,dilated_mask,linear_attention,window_mask


def sinusoid(length,dim,device):
    """Fixed sinusoidal position table"""
    pos = torch.arange(length,device=device,dtype=torch.float32)[:,None]
    freq = torch.exp(torch.arange(0,dim,2,device=device,dtype=torch.float32)*(-math.log(10000.0)/dim))
    table = torch.zeros(length,dim,device=device)
    table[:,0::2] = torch.sin(pos*freq)
    table[:,1::2] = torch.cos(pos*freq)
    return table


class Attention(nn.Module):
    """Multi head attention whose only changing part is the attention operator"""

    def __init__(self,dim,heads,mechanism,mq):
        super().__init__()
        self.heads,self.mechanism,self.mq = heads,mechanism,mq
        self.qkv = nn.Linear(dim,3*dim)
        self.out = nn.Linear(dim,dim)

    def mask(self,length,device,test_window):
        """Boolean table of visible positions for this mechanism or none for full causal attention"""
        if test_window is not None:
            return window_mask(length,test_window,device)
        if self.mechanism == "window":
            return window_mask(length,self.mq["window"],device)
        if self.mechanism == "dilated":
            local = window_mask(length,self.mq["window"],device)
            spaced = dilated_mask(length,self.mq["dilation"],self.mq["dilated_positions"],device)
            # giving the first half of the heads the local window and the second half the spaced positions
            half = self.heads//2
            return torch.stack([local]*half+[spaced]*(self.heads-half))
        return None

    def forward(self,x,test_window=None):
        b,t,d = x.shape
        q,k,v = self.qkv(x).view(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        if self.mechanism == "linear":
            o = linear_attention(q,k,v)
        elif self.mechanism == "local_linear":
            # the first heads attend exactly inside a short window and the rest keep a linear state
            n = self.mq["intervention"]["local_heads"]
            local = window_mask(t,self.mq["intervention"]["window"],x.device)
            o = torch.cat([dense_attention(q[:,:n],k[:,:n],v[:,:n],causal=True,mask=local),
                           linear_attention(q[:,n:],k[:,n:],v[:,n:])],1)
        else:
            o = dense_attention(q,k,v,causal=True,mask=self.mask(t,x.device,test_window))
        return self.out(o.transpose(1,2).reshape(b,t,d))


class Block(nn.Module):
    """Pre normalized transformer block"""

    def __init__(self,dim,heads,ratio,mechanism,mq):
        super().__init__()
        self.ln1,self.ln2 = nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.attn = Attention(dim,heads,mechanism,mq)
        self.mlp = nn.Sequential(nn.Linear(dim,ratio*dim),nn.GELU(),nn.Linear(ratio*dim,dim))

    def forward(self,x,test_window=None):
        x = x+self.attn(self.ln1(x),test_window)
        return x+self.mlp(self.ln2(x))


class RecallModel(nn.Module):
    """Two block decoder that classifies the value at every position from causal context"""

    def __init__(self,mq,mechanism):
        super().__init__()
        m = mq["model"]
        self.embed = nn.Embedding(mq["vocab"],m["dim"])
        self.blocks = nn.ModuleList([Block(m["dim"],m["heads"],m["mlp_ratio"],mechanism,mq) for _ in range(m["blocks"])])
        self.norm = nn.LayerNorm(m["dim"])
        self.head = nn.Linear(m["dim"],mq["values"])

    def forward(self,tokens=None,hidden=None,test_window=None):
        """Value logits from token ids or from given input embeddings"""
        x = self.embed(tokens) if hidden is None else hidden
        x = x+sinusoid(x.shape[1],x.shape[2],x.device)
        for block in self.blocks:
            x = block(x,test_window)
        return self.head(self.norm(x))


def loss_and_hits(logits,labels,offset):
    """Cross entropy and correctness over the supervised query positions only"""
    keep = labels >= 0
    target = labels[keep]-offset
    picked = logits[keep]
    return F.cross_entropy(picked,target),picked.argmax(-1) == target
