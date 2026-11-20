import torch
import torch.nn.functional as F

from task1_compress.prune import lowest_mask,magnitude_score,two_four_mask
from task1_compress.quantize import dequantize,pack_bits,quantize,unpack_bits
from task1_compress.resources import timed

SHAPES = [(1536,1536),(8960,1536),(1536,8960)]


def kernel_probe(cfg,warmup,reps,tokens=(1,2048)):
    """Time the kernels that are really available for each projection shape

    dense half and full precision matmuls are compared with a sparse csr matmul at two
    sparsities and with unpacking four bit codes before a dense matmul
    semi structured sparsity only runs when the gpu supports it"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    capability = torch.cuda.get_device_capability() if dev == "cuda" else (0,0)
    semi = dev == "cuda" and capability >= (8,0)
    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    rows = []
    for out_f,in_f in SHAPES:
        w = (torch.randn(out_f,in_f,generator=gen)*0.02).to(dev)
        w16 = w.half()
        codes,scale,zero = quantize(w16,4,cfg["quant"]["group"],cfg["quant"]["scale_floor"])
        packed = pack_bits(codes,4)
        for m in tokens:
            x = torch.randn(m,in_f,generator=gen).to(dev)
            x16 = x.half()
            base = {"out":out_f,"in":in_f,"tokens":m,"gpu_capability":f"{capability[0]}.{capability[1]}"}

            def add(kernel,fn):
                mean,std = timed(fn,warmup,reps)
                rows.append({**base,"kernel":kernel,"ms":mean,"std":std})

            add("dense_fp16",lambda:F.linear(x16,w16))
            add("dense_fp32",lambda:F.linear(x,w))
            for s in (0.5,0.75):
                ws = w.clone()
                ws[lowest_mask(magnitude_score(ws),round(s*ws.numel()))] = 0
                csr = ws.to_sparse_csr()
                add(f"csr_fp32_s{round(100*s)}",lambda:torch.sparse.mm(csr,x.T))
            add("int4_unpack_then_dense_fp16",
                lambda:F.linear(x16,dequantize(unpack_bits(packed,4,codes.numel()).view(codes.shape),scale,zero,
                                                cfg["quant"]["group"])))
            if semi:
                w24 = w16.clone()
                w24[two_four_mask(magnitude_score(w24))] = 0
                sparse24 = torch.sparse.to_sparse_semi_structured(w24)
                add("two_four_semi_structured_fp16",lambda:F.linear(x16,sparse24))
            else:
                rows.append({**base,"kernel":"two_four_semi_structured_fp16","ms":None,"std":None,
                             "note":"needs compute capability 8.0 or newer"})
    return rows
