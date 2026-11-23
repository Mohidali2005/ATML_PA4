import os
import statistics
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from task1_compress.model import eligible,projection,unique_parameters
from task1_compress.quantize import dequantize,pack_bits,unpack_bits


def sync():
    """Wait for queued gpu work so timers measure finished work"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn,warmup,reps):
    """Run a function after warmup calls and return the mean and spread of its wall time in milliseconds"""
    times = []
    for r in range(warmup+reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        if r >= warmup:
            times.append(1000*(time.perf_counter()-start))
    return statistics.mean(times),statistics.stdev(times) if len(times) > 1 else 0.0


def storage(model,cfg,kind,sizes=None,o_scales=False):
    """Byte count of the dense half precision model and of the compressed representation

    kind is dense or sparse or quant and sizes holds the packed bytes of every projection for quant"""
    mods = eligible(model,cfg)
    total = sum(m.weight.numel() for _,_,m in mods)
    other = unique_parameters(model)-total
    out = {"eligible_weights":total,"other_parameters":other,"dense_fp16_bytes":2*unique_parameters(model)}
    if kind == "sparse":
        nnz = sum(int((m.weight != 0).sum()) for _,_,m in mods)
        out["sparse_format_bytes"] = sum(-(-m.weight.numel()//8) for _,_,m in mods)+2*nnz+2*other
        out["nominal_bits"] = 8*(out["sparse_format_bytes"]-2*other)/total
    if kind == "quant":
        extra = 2*len(model.model.layers)*model.config.hidden_size if o_scales else 0
        out["packed_bytes"] = sum(sizes.values())+extra+2*other
        out["o_scale_bytes"] = extra
        out["nominal_bits"] = 8*(sum(sizes.values()))/total
    if kind == "dense":
        out["nominal_bits"] = 16.0
    return out


@torch.inference_mode()
def latency(model,ids,cfg,warmup,reps,decode_tokens):
    """Prefill and decode latency at batch one plus resident and peak allocated memory"""
    t = cfg["timing"]
    out = {}
    for length in t["prefill"]:
        x = ids[:,:length]
        out[f"prefill_{length}_ms"],out[f"prefill_{length}_std"] = timed(lambda:model(input_ids=x,logits_to_keep=1),warmup,reps)
    if torch.cuda.is_available():
        # reading the memory after the first calls so the matmul workspace counts the same for every model
        torch.cuda.empty_cache()
        out["resident_gb"] = torch.cuda.memory_allocated()/1e9
    prompt = ids[:,:t["decode_prompt"]]
    times = []
    for r in range(warmup+reps):
        res = model(input_ids=prompt,use_cache=True,logits_to_keep=1)
        cache,nxt = res.past_key_values,res.logits[:,-1:].argmax(-1)
        sync()
        start = time.perf_counter()
        for _ in range(decode_tokens):
            res = model(input_ids=nxt,past_key_values=cache,use_cache=True)
            cache,nxt = res.past_key_values,res.logits[:,-1:].argmax(-1)
        sync()
        if r >= warmup:
            times.append(1000*(time.perf_counter()-start)/decode_tokens)
    out["decode_ms_per_token"] = statistics.mean(times)
    out["decode_std"] = statistics.stdev(times) if len(times) > 1 else 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model(input_ids=ids[:,:max(t["prefill"])],logits_to_keep=1)
        out["peak_prefill_gb"] = torch.cuda.max_memory_allocated()/1e9
    return out


class PackedLinear(torch.nn.Module):
    """Linear layer that keeps packed integer codes and unpacks them on every call"""

    def __init__(self,codes,scale,zero,bits,group,bias,in_scale):
        super().__init__()
        self.shape,self.bits,self.group = tuple(codes.shape),bits,group
        self.register_buffer("packed",pack_bits(codes,bits))
        self.register_buffer("scale",scale)
        self.register_buffer("zero",zero)
        self.register_buffer("in_scale",in_scale)
        self.bias = bias

    def forward(self,x):
        """Rebuild the half precision weight and apply it"""
        codes = unpack_bits(self.packed,self.bits,self.shape[0]*self.shape[1]).view(self.shape)
        if self.in_scale is not None:
            x = x/self.in_scale.to(x.dtype)
        return F.linear(x,dequantize(codes,self.scale,self.zero,self.group).to(x.dtype),self.bias)


class SparseLinear(torch.nn.Module):
    """Linear layer that keeps a one bit mask and the nonzero values and rebuilds the weight on every call"""

    def __init__(self,weight,bias):
        super().__init__()
        keep = weight != 0
        self.shape = tuple(weight.shape)
        self.register_buffer("mask",pack_bits(keep.to(torch.uint8),1))
        self.register_buffer("values",weight[keep].clone())
        self.bias = bias

    def forward(self,x):
        """Scatter the stored values into a dense weight and apply it"""
        keep = unpack_bits(self.mask,1,self.shape[0]*self.shape[1]).view(self.shape).bool()
        w = torch.zeros(self.shape,dtype=self.values.dtype,device=self.values.device)
        w[keep] = self.values
        return F.linear(x,w,self.bias)


def swap_projection(layer,name,new):
    """Put a new module in place of one projection"""
    parent = layer.self_attn if hasattr(layer.self_attn,name) else layer.mlp
    setattr(parent,name,new)


def file_bytes(tensors,path):
    """Write tensors to a safetensors file and return its size on disk"""
    save_file({k:v.contiguous() for k,v in tensors.items()},str(path))
    size = os.path.getsize(path)
    os.remove(path)
    return size


def packed_tensors(model):
    """Collect the tensors that a packed checkpoint would store with shared tensors kept once"""
    out,seen = {},set()
    for name,t in list(model.named_parameters())+list(model.named_buffers()):
        if t.data_ptr() in seen or name.endswith("inv_freq"):
            continue
        seen.add(t.data_ptr())
        out[name] = t.detach()
    return out
