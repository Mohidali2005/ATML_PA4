import math
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache,Qwen2Config,Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from common.utils import load_json
from task1_compress.quantize import pack_bits,unpack_bits

CODE = re.compile(r"(?<!\d)\d{5}(?!\d)")


def device():
    """Pick the gpu when one is present"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(cfg,tiny=False):
    """Load the pinned instruct tokenizer and model or a tiny random model with the same tokenizer"""
    rev = load_json(cfg["paths"]["lock"])["revisions"]["model_target"]
    kv = cfg["kv"]
    tok = AutoTokenizer.from_pretrained(kv["model"],revision=rev)
    if tiny:
        torch.manual_seed(cfg["seed"])
        config = Qwen2Config(vocab_size=len(tok),hidden_size=256,intermediate_size=512,num_hidden_layers=2,
                             num_attention_heads=4,num_key_value_heads=2,tie_word_embeddings=True,
                             max_position_embeddings=8192,attn_implementation=kv["attn"])
        # using half precision like the real model so the byte accounting is tested as it will run
        model = Qwen2ForCausalLM(config).half()
    else:
        model = AutoModelForCausalLM.from_pretrained(kv["model"],revision=rev,dtype=getattr(torch,kv["dtype"]),
                                                     attn_implementation=kv["attn"])
    return tok,model.to(device()).eval().requires_grad_(False),rev


def build_prompt(tok,case,marker):
    """Chat template prompt split into context and question suffix with token level evidence positions"""
    text = tok.apply_chat_template([{"role":"user","content":case["user_text"]}],tokenize=False,add_generation_prompt=True)
    enc = tok(text,return_offsets_mapping=True,add_special_tokens=False)
    ids,offs = enc["input_ids"],enc["offset_mapping"]
    base = text.index(case["user_text"])
    cut = base+case["user_text"].index(marker)
    # the context ends with the last token that finishes before the question marker begins
    split = next(i for i,(a,b) in enumerate(offs) if b > cut)
    lo,hi = base+case["evidence_char_start"],base+case["evidence_char_end"]
    evidence = [i for i,(a,b) in enumerate(offs) if a < hi and b > lo]
    gold = tok(case["code"],add_special_tokens=False)["input_ids"]
    return {"ids":ids,"split":split,"evidence":evidence,"gold":gold,"boundary_exact":offs[split][0] == cut,
            "evidence_text_ok":case["evidence_text"] in text[offs[evidence[0]][0]:offs[evidence[-1]][1]]}


def as_cache(kv):
    """Build a fresh cache object from per layer key and value tensors without copying them"""
    cache = DynamicCache()
    for i,(k,v) in enumerate(kv):
        cache.update(k,v,i)
    return cache


def tensors(cache):
    """Per layer key and value tensors held by a cache"""
    return [(layer.keys,layer.values) for layer in cache.layers]


@torch.inference_mode()
def prefill(model,ids):
    """Run the context once in full precision and return its cache"""
    out = model(input_ids=ids,use_cache=True,logits_to_keep=1)
    return tensors(out.past_key_values)


@torch.inference_mode()
def step(model,cache,ids,start):
    """Feed new tokens through an edited cache at their original absolute positions"""
    n = ids.shape[1]
    phys = cache.get_seq_length()
    pos = torch.arange(start,start+n,device=ids.device)[None]
    mask = torch.ones(1,phys+n,dtype=torch.long,device=ids.device)
    out = model(input_ids=ids,past_key_values=cache,use_cache=True,position_ids=pos,
                cache_position=torch.arange(phys,phys+n,device=ids.device),attention_mask=mask)
    return out.logits[0].float()


@torch.inference_mode()
def question_scores(model,kv,suffix,start):
    """Attention mass each context position receives from the question suffix per layer and key value head

    queries and keys are rebuilt from the projections with their rotary positions and scored in fp32"""
    cfg = model.config
    heads,kvh = cfg.num_attention_heads,cfg.num_key_value_heads
    hd = cfg.hidden_size//heads
    ctx = kv[0][0].shape[-2]
    scores = torch.zeros(len(kv),kvh,ctx,device=suffix.device)
    hooks = []

    def make(i,attn):
        """Hook that scores one layer before the layer appends the suffix keys"""
        def hook(module,args,kwargs):
            h = kwargs["hidden_states"]
            cos,sin = kwargs["position_embeddings"]
            n = h.shape[1]
            q = attn.q_proj(h).view(1,n,heads,hd).transpose(1,2)
            k = attn.k_proj(h).view(1,n,kvh,hd).transpose(1,2)
            q,k = apply_rotary_pos_emb(q,k,cos,sin)
            keys = torch.cat([kv[i][0],k],-2).float().repeat_interleave(heads//kvh,1)
            s = torch.matmul(q.float(),keys.transpose(-1,-2))/math.sqrt(hd)
            future = torch.ones(n,n,dtype=torch.bool,device=h.device).triu(1)
            s[...,ctx:] = s[...,ctx:].masked_fill(future,float("-inf"))
            p = torch.softmax(s,-1)[...,:ctx]
            scores[i] = p.sum(2).view(kvh,heads//kvh,ctx).sum(1)
        return hook
    for i,layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(make(i,layer.self_attn),with_kwargs=True))
    step(model,as_cache(kv),suffix,start)
    for hook in hooks:
        hook.remove()
    return scores


def pooled(scores,width):
    """Smooth scores with a running maximum over neighbouring positions"""
    return F.max_pool1d(scores,width,stride=1,padding=width//2)


def guard(ctx,sink,recent):
    """Positions every eviction rule always keeps"""
    keep = torch.zeros(ctx,dtype=torch.bool)
    keep[:sink] = True
    keep[ctx-recent:] = True
    return keep


def sink_recent(ctx,n,sink):
    """Keep the first sink positions and the most recent remaining ones up to n"""
    return torch.cat([torch.arange(sink),torch.arange(ctx-(n-sink),ctx)])


def top_positions(scores,count,sink,recent):
    """Per layer and head the guard positions plus the highest scoring others in position order"""
    layers,kvh,ctx = scores.shape
    s = scores.clone()
    g = guard(ctx,sink,recent).to(s.device)
    s[...,g] = float("-inf")
    pick = s.topk(count,dim=-1).indices
    base = torch.nonzero(g).flatten()
    full = torch.cat([base.expand(layers,kvh,-1),pick],-1)
    return full.sort(-1).values


def gather(kv,idx):
    """Keep only the chosen positions of every layer and head"""
    out = []
    for i,(k,v) in enumerate(kv):
        ix = idx[i].to(k.device)[None,:,:,None].expand(k.shape[0],-1,-1,k.shape[-1])
        out.append((k.gather(2,ix),v.gather(2,ix)))
    return out


def quantize(x,bits,floor=1e-5):
    """Asymmetric min max quantization of the last axis with an fp16 scale and fp16 offset per group"""
    qmax = 2**bits-1
    lo,hi = x.float().amin(-1,keepdim=True),x.float().amax(-1,keepdim=True)
    # storing the scale and offset in half precision and quantizing with the stored values
    scale = ((hi-lo)/qmax).clamp(min=floor).half()
    offset = lo.half()
    codes = torch.round((x.float()-offset.float())/scale.float()).clamp(0,qmax).to(torch.uint8)
    return codes,scale,offset


def dequantize(codes,scale,offset):
    """Turn codes back into half precision values"""
    return (codes.float()*scale.float()+offset.float()).half()


def store(x,bits,group,axis):
    """Quantize groups of a tensor along tokens or channels and return its reconstruction and packed parts

    x has tokens on the second last axis and channels on the last axis and the token count is a whole number of groups"""
    if axis == "channel":
        g = x.transpose(-1,-2).reshape(*x.shape[:-2],x.shape[-1],-1,group)
    else:
        g = x.reshape(*x.shape[:-1],-1,group)
    codes,scale,offset = quantize(g,bits)
    rec = dequantize(codes,scale,offset)
    rec = rec.reshape(*x.shape[:-2],x.shape[-1],-1).transpose(-1,-2) if axis == "channel" else rec.reshape(x.shape)
    parts = {"codes":pack_bits(codes,bits),"scale":scale,"offset":offset}
    return rec.to(x.dtype),parts


def nbytes(parts):
    """Serialized bytes of a list of stored tensors"""
    return sum(t.numel()*t.element_size() for t in parts)


def restore(parts,bits,group,axis,shape):
    """Unpack stored codes and rebuild the half precision tensor of the given token by channel shape"""
    if axis == "channel":
        grouped = (*shape[:-2],shape[-1],shape[-2]//group,group)
    else:
        grouped = (*shape[:-1],shape[-1]//group,group)
    codes = unpack_bits(parts["codes"],bits,math.prod(grouped)).view(grouped)
    rec = dequantize(codes,parts["scale"],parts["offset"])
    return rec.reshape(*shape[:-2],shape[-1],-1).transpose(-1,-2) if axis == "channel" else rec.reshape(shape)


def int_cache(kv,bits,group,key_axis):
    """Quantize every complete token group of the context and keep the trailing tokens in fp16

    returns the reconstructed cache the serialized byte count and the number of quantized tokens"""
    out,stored = [],[]
    ctx = kv[0][0].shape[-2]
    full = ctx//group*group
    for k,v in kv:
        rk,pk = store(k[...,:full,:],bits,group,key_axis)
        rv,pv = store(v[...,:full,:],bits,group,"token")
        out.append((torch.cat([rk,k[...,full:,:]],-2),torch.cat([rv,v[...,full:,:]],-2)))
        stored += list(pk.values())+list(pv.values())+[k[...,full:,:],v[...,full:,:]]
    return out,nbytes(stored),full


def formula_bytes(cfg_model,ctx,bits,group):
    """Manual byte formula for a uniform integer cache with an fp16 trailing remainder"""
    channels = 2*cfg_model["layers"]*cfg_model["kv_heads"]*cfg_model["head_dim"]
    return channels*(ctx//group*(group*bits//8+4)+2*(ctx % group))


def fp16_bytes(shape,ctx):
    """Bytes of the full fp16 cache"""
    return 4*shape["layers"]*shape["kv_heads"]*shape["head_dim"]*ctx


def eviction_count(shape,ctx,budget,indexed,sink,recent,index_bytes):
    """Largest number of fp16 positions per head whose bytes stay within the budget"""
    units = shape["layers"]*shape["kv_heads"]
    per = 4*shape["head_dim"]
    best = sink+recent
    for n in range(sink+recent,ctx+1):
        cost = units*(per*n+(index_bytes*(n-sink-recent) if indexed else 0))
        if cost > budget:
            break
        best = n
    return best


def hybrid_plan(shape,ctx,budget,sink,recent,bits,group,index_bytes):
    """Largest number of selected fp16 positions with the rest in complete low bit groups

    every head keeps the guard positions in fp16 plus m selected ones and the leftover of the low bit set"""
    units = shape["layers"]*shape["kv_heads"]
    hd = shape["head_dim"]
    per_group = 2*hd*(group*bits//8+4)
    best = None
    for m in range(0,ctx-sink-recent+1):
        low = ctx-sink-recent-m
        groups,rest = low//group,low % group
        cost = units*(4*hd*(sink+recent+m+rest)+per_group*groups+index_bytes*m)
        if cost <= budget:
            best = {"selected":m,"groups":groups,"fp16_rest":rest,"bytes":cost}
    return best


def hybrid_cache(kv,scores,plan,sink,recent,bits,group):
    """Keep guard and top scoring positions in fp16 and store the rest in low bit groups per head"""
    ctx = kv[0][0].shape[-2]
    fp_idx = top_positions(scores,plan["selected"],sink,recent)
    g = guard(ctx,sink,recent)
    out,stored,kept = [],[],[]
    for i,(k,v) in enumerate(kv):
        hk,hv = k.clone(),v.clone()
        layer_kept = []
        for h in range(k.shape[1]):
            keep = torch.zeros(ctx,dtype=torch.bool,device=k.device)
            keep[fp_idx[i,h].to(k.device)] = True
            # storing only the selected positions since the guard and the low bit set follow from them
            sel = fp_idx[i,h].cpu()
            stored.append(sel[~g[sel]].to(torch.int16))
            low = torch.nonzero(~keep).flatten()
            # leaving the most recent incomplete group of the low set in fp16
            q_idx = low[:plan["groups"]*group]
            keep[low[plan["groups"]*group:]] = True
            if len(q_idx):
                rk,pk = store(k[:,h:h+1,q_idx,:],bits,group,"channel")
                rv,pv = store(v[:,h:h+1,q_idx,:],bits,group,"token")
                hk[:,h:h+1,q_idx,:],hv[:,h:h+1,q_idx,:] = rk,rv
                stored += list(pk.values())+list(pv.values())
            stored += [k[:,h,keep,:],v[:,h,keep,:]]
            layer_kept.append(keep)
        out.append((hk,hv))
        kept.append(torch.stack(layer_kept))
    return out,nbytes(stored),torch.stack(kept)


class Requantizer:
    """Quantize each newly completed token group of a uniform integer cache during decoding"""

    def __init__(self,start,bits,group,key_axis):
        self.done,self.bits,self.group,self.axis = start,bits,group,key_axis
        self.added = 0

    def __call__(self,cache):
        """Replace every complete fp16 group after the last quantized position by its reconstruction"""
        length = cache.get_seq_length()
        while length-self.done >= self.group:
            a,b = self.done,self.done+self.group
            for layer in cache.layers:
                rk,pk = store(layer.keys[...,a:b,:],self.bits,self.group,self.axis)
                rv,pv = store(layer.values[...,a:b,:],self.bits,self.group,"token")
                layer.keys = torch.cat([layer.keys[...,:a,:],rk,layer.keys[...,b:,:]],-2)
                layer.values = torch.cat([layer.values[...,:a,:],rv,layer.values[...,b:,:]],-2)
            self.done = b
            self.added += 1


def extract(text):
    """First standalone five digit number in a generated answer"""
    m = CODE.search(text)
    return m.group(0) if m else None
