import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from importlib.metadata import version

import pandas as pd
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from common.utils import load_json,load_yaml,repo_path,save_json
from task2_attention.ops import dense_attention,sdpa_attention,tiled_attention
from task2_attention.timing import peak,sync,timed

DTYPES = {"float16":torch.float16,"float32":torch.float32}
BACKENDS = {"flash":SDPBackend.FLASH_ATTENTION,"efficient":SDPBackend.EFFICIENT_ATTENTION,
            "cudnn":SDPBackend.CUDNN_ATTENTION,"math":SDPBackend.MATH}
IMPLS = ["dense","tiled","sdpa","sdpa_efficient","sdpa_math"]


def device():
    """Pick the gpu when one is present"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def tensors(cfg,length,seed):
    """Fixed fp32 queries keys values and upstream gradient for one sequence length"""
    e = cfg["exact"]
    g = torch.Generator(device()).manual_seed(seed+length)
    shape = (e["batch"],e["heads"],length,e["head_dim"])
    return [torch.randn(shape,generator=g,device=device()) for _ in range(4)]


def runner(name,tile):
    """Return the attention function of one implementation"""
    if name == "dense":
        return dense_attention
    if name == "tiled":
        return lambda q,k,v:tiled_attention(q,k,v,tile=tile)
    if name == "sdpa":
        return sdpa_attention
    backend = BACKENDS[name.split("_")[1]]

    def forced(q,k,v):
        with sdpa_kernel(backend):
            return sdpa_attention(q,k,v)
    return forced


def error(out,ref):
    """Largest and relative error of a result against the reference"""
    diff = (out.double()-ref.double())
    return diff.abs().max().item(),(diff.norm()/ref.double().norm()).item()


def with_grads(fn,q,k,v,dout,dtype):
    """Run one implementation in a precision and return its output and input gradients"""
    leaves = [x.to(dtype).detach().requires_grad_(True) for x in (q,k,v)]
    out = fn(*leaves)
    out.backward(dout.to(dtype))
    return out.detach(),[x.grad for x in leaves]


def free_gb():
    """Free gpu memory in gigabytes or a large number on the cpu"""
    return torch.cuda.mem_get_info()[0]/1e9 if torch.cuda.is_available() else 1e9


def fidelity(cfg,lengths,impls):
    """Output and gradient errors of every implementation against dense fp32 attention"""
    e = cfg["exact"]
    rows = []
    for length in lengths:
        q,k,v,dout = tensors(cfg,length,cfg["seed"])
        grads = length <= e["grad_max_len"]
        # estimating dense fp32 memory from its score sized tensors before running it
        need = (6 if grads else 3)*e["batch"]*e["heads"]*length*length*4/1e9
        if need > free_gb()-e["memory_margin_gb"]:
            rows.append({"length":length,"status":f"skipped needing about {need:.1f} GB"})
            continue
        if grads:
            ref,ref_g = with_grads(dense_attention,q,k,v,dout,torch.float32)
        else:
            with torch.no_grad():
                ref = dense_attention(q,k,v)
        for dtype in e["dtypes"]:
            for name in impls:
                fn = runner(name,e["tile"])
                row = {"length":length,"dtype":dtype,"impl":name,"status":"ok"}
                if not supported(name,q.to(DTYPES[dtype])):
                    rows.append({**row,"status":"unsupported"})
                    continue
                if grads:
                    out,g = with_grads(fn,q,k,v,dout,DTYPES[dtype])
                    for label,a,b in zip(("dq","dk","dv"),g,ref_g):
                        row[f"{label}_max_abs"],row[f"{label}_rel"] = error(a,b)
                else:
                    with torch.no_grad():
                        out = fn(q.to(DTYPES[dtype]),k.to(DTYPES[dtype]),v.to(DTYPES[dtype]))
                row["out_max_abs"],row["out_rel"] = error(out,ref)
                row["finite"] = bool(torch.isfinite(out).all())
                rows.append(row)
        del ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


@torch.no_grad()
def tile_sweep(cfg,lengths):
    """Output error of the tiled forward for every tile size and precision"""
    e = cfg["exact"]
    rows = []
    for length in [x for x in lengths if x <= e["tile_sweep_max_len"]]:
        q,k,v,_ = tensors(cfg,length,cfg["seed"])
        ref = dense_attention(q,k,v)
        for dtype in e["dtypes"]:
            for tile in e["tile_sweep"]:
                out = tiled_attention(q.to(DTYPES[dtype]),k.to(DTYPES[dtype]),v.to(DTYPES[dtype]),tile=tile)
                mx,rel = error(out,ref)
                rows.append({"length":length,"dtype":dtype,"tile":tile,"out_max_abs":mx,"out_rel":rel,
                             "tile_scores_bytes":e["batch"]*e["heads"]*tile*tile*4})
    return pd.DataFrame(rows)


@torch.no_grad()
def causality(cfg,length):
    """Change every key and value after a cut and check that earlier outputs do not move at all"""
    e = cfg["exact"]
    q,k,v,_ = tensors(cfg,length,cfg["seed"])
    cut = length//2
    k2,v2 = k.clone(),v.clone()
    k2[...,cut+1:,:] = torch.randn_like(k2[...,cut+1:,:])*10
    v2[...,cut+1:,:] = torch.randn_like(v2[...,cut+1:,:])*10
    rows = []
    for dtype in e["dtypes"]:
        c = lambda x:x.to(DTYPES[dtype])
        for name in [n for n in IMPLS if supported(n,c(q))]:
            fn = runner(name,e["tile"])
            a = fn(c(q),c(k),c(v))[...,:cut+1,:]
            b = fn(c(q),c(k2),c(v2))[...,:cut+1,:]
            after = (fn(c(q),c(k),c(v))[...,cut+1:,:]-fn(c(q),c(k2),c(v2))[...,cut+1:,:]).abs().max().item()
            rows.append({"length":length,"dtype":dtype,"impl":name,"cut":cut,
                         "max_change_before_cut":(a-b).abs().max().item(),"max_change_after_cut":after})
    return pd.DataFrame(rows)


@torch.no_grad()
def edge_cases(cfg,length):
    """A query row with every key masked and scores pushed to extreme size"""
    e = cfg["exact"]
    q,k,v,_ = tensors(cfg,length,cfg["seed"])
    rows = []
    row = length//3
    mask = torch.ones(length,length,dtype=torch.bool,device=q.device)
    mask[row] = False
    big = q*e["extreme_scale"]
    ref = dense_attention(big.double(),k.double(),v.double())
    for dtype in e["dtypes"]:
        c = lambda x:x.to(DTYPES[dtype])
        cases = {"dense":lambda m:dense_attention(c(q),c(k),c(v),mask=m),
                 "tiled":lambda m:tiled_attention(c(q),c(k),c(v),tile=e["tile"],mask=m),
                 "sdpa":lambda m:sdpa_attention(c(q),c(k),c(v),mask=m)}
        for name,fn in cases.items():
            out = fn(mask)[...,row,:]
            rows.append({"case":"all_masked_row","dtype":dtype,"impl":name,"nan":int(out.isnan().sum()),
                         "elements":out.numel(),"max_abs_output":out.nan_to_num(0).abs().max().item()})
        extreme = {"dense_stable":lambda:dense_attention(c(big),c(k),c(v)),
                   "dense_unstable":lambda:dense_attention(c(big),c(k),c(v),stable=False),
                   "tiled":lambda:tiled_attention(c(big),c(k),c(v),tile=e["tile"]),
                   "sdpa":lambda:sdpa_attention(c(big),c(k),c(v))}
        for name,fn in extreme.items():
            out = fn()
            finite = bool(torch.isfinite(out).all())
            mx,rel = error(out.nan_to_num(0),ref)
            rows.append({"case":f"scores_times_{e['extreme_scale']:g}","dtype":dtype,"impl":name,"nan":int(out.isnan().sum()),
                         "elements":out.numel(),"finite":finite,"out_max_abs":mx if finite else None,
                         "out_rel":rel if finite else None})
    return pd.DataFrame(rows)


def reasons(check,params):
    """Ask torch whether a backend can run and keep the warning text that explains a refusal"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = bool(check(params,True))
    return ok,"" if ok else " | ".join(str(w.message).strip() for w in caught)[:400]


def supported(name,q):
    """Whether an implementation can run on these tensors"""
    if name in ("dense","tiled","sdpa","sdpa_math"):
        return True
    if not torch.cuda.is_available():
        return False
    check = {"efficient":torch.backends.cuda.can_use_efficient_attention,"flash":torch.backends.cuda.can_use_flash_attention,
             "cudnn":torch.backends.cuda.can_use_cudnn_attention}[name.split("_")[1]]
    return reasons(check,torch.backends.cuda.SDPAParams(q,q,q,None,0.0,True,False))[0]


def backend_table(cfg,length):
    """Which sdpa backends this gpu supports per precision and which kernels the default dispatch launches"""
    rows = []
    kernels = {}
    q,k,v,dout = tensors(cfg,length,cfg["seed"])
    for dtype in cfg["exact"]["dtypes"]:
        c = q.to(DTYPES[dtype])
        for name in BACKENDS:
            if name == "math":
                rows.append({"dtype":dtype,"backend":name,"supported":True,"reason":""})
                continue
            if not torch.cuda.is_available():
                rows.append({"dtype":dtype,"backend":name,"supported":False,"reason":"no cuda device"})
                continue
            check = {"efficient":torch.backends.cuda.can_use_efficient_attention,
                     "flash":torch.backends.cuda.can_use_flash_attention,
                     "cudnn":torch.backends.cuda.can_use_cudnn_attention}[name]
            ok,why = reasons(check,torch.backends.cuda.SDPAParams(c,c,c,None,0.0,True,False))
            rows.append({"dtype":dtype,"backend":name,"supported":ok,"reason":why})
        activity = [torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        leaves = [x.to(DTYPES[dtype]).requires_grad_(True) for x in (q,k,v)]
        with torch.profiler.profile(activities=activity) as prof:
            out = sdpa_attention(*leaves)
            out.backward(dout.to(DTYPES[dtype]))
            sync()
        names = sorted({ev.key for ev in prof.key_averages() if "attention" in ev.key.lower() or "fmha" in ev.key.lower()
                        or "flash" in ev.key.lower() or "efficient" in ev.key.lower()})
        kernels[dtype] = {"length":length,"kernels":names}
    return pd.DataFrame(rows),kernels


def bench_one(cfg,name,dtype,length,smoke):
    """Forward and forward plus backward time and peak memory of one implementation in this process"""
    e = cfg["exact"]
    warm,reps = (1,cfg["smoke"]["exact_reps"]) if smoke else (e["warmup"],e["reps"])
    q,k,v,dout = [x.to(DTYPES[dtype]) for x in tensors(cfg,length,cfg["seed"])]
    fn = runner(name,e["tile"])
    out = {"impl":name,"dtype":dtype,"length":length,"status":"ok","reps":reps,"warmup":warm}
    if not supported(name,q):
        out["status"] = "unsupported"
        return out
    with torch.no_grad():
        out["forward_ms"],out["forward_std"] = timed(lambda:fn(q,k,v),warm,reps)
        out["forward_peak_gb"],out["forward_reserved_gb"] = peak(lambda:fn(q,k,v))
    leaves = [x.detach().requires_grad_(True) for x in (q,k,v)]

    def step():
        """One forward and backward pass that drops the gradients afterwards"""
        fn(*leaves).backward(dout)
        for x in leaves:
            x.grad = None
    out["forward_backward_ms"],out["forward_backward_std"] = timed(step,warm,reps)
    out["forward_backward_peak_gb"],out["forward_backward_reserved_gb"] = peak(step)
    out["input_gb"] = 4*q.numel()*q.element_size()/1e9
    out["dense_scores_gb"] = e["batch"]*e["heads"]*length*length*q.element_size()/1e9
    return out


def bench_all(cfg,lengths,smoke,config_path):
    """Time every implementation precision and length each in a fresh process so a real out of memory is recorded"""
    rows = []
    for length in lengths:
        for dtype in cfg["exact"]["dtypes"]:
            for name in IMPLS:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/"row.json"
                    cmd = [sys.executable,"-m","task2_attention.exact","--config",config_path,"--bench-one",name,dtype,
                           str(length),"--out",str(path)]+(["--smoke"] if smoke else [])
                    done = subprocess.run(cmd,capture_output=True,text=True)
                    if done.returncode == 0:
                        rows.append(load_json(path))
                        continue
                    oom = "OutOfMemoryError" in done.stderr or "out of memory" in done.stderr
                    rows.append({"impl":name,"dtype":dtype,"length":length,"status":"oom" if oom else "error",
                                 "detail":done.stderr.strip().splitlines()[-1][:300] if done.stderr.strip() else ""})
    return pd.DataFrame(rows)


def environment():
    """Software and device versions recorded with every result"""
    out = {"torch":torch.__version__,"transformers":version("transformers"),"cuda":torch.version.cuda,
           "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if torch.cuda.is_available():
        out["capability"] = ".".join(map(str,torch.cuda.get_device_capability(0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--bench-one",nargs=3,default=None)
    ap.add_argument("--out",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.bench_one:
        name,dtype,length = args.bench_one
        save_json(args.out,bench_one(cfg,name,dtype,int(length),args.smoke))
        return
    out = repo_path(args.results or cfg["paths"]["results_dir"])/"exact"
    lengths = cfg["smoke"]["exact_lengths"] if args.smoke else cfg["exact"]["lengths"]
    if args.smoke:
        cfg["exact"]["grad_max_len"] = max(lengths)
        cfg["exact"]["tile"] = 32
        cfg["exact"]["tile_sweep"] = [16,32]
        cfg["exact"]["tile_sweep_max_len"] = max(lengths)
    save_json(out/"environment.json",environment())
    causality(cfg,lengths[-1] if args.smoke else 512).to_csv(out/"causality.csv",index=False)
    edge_cases(cfg,lengths[-1] if args.smoke else 256).to_csv(out/"edge_cases.csv",index=False)
    table,kernels = backend_table(cfg,lengths[-1] if args.smoke else 1024)
    table.to_csv(out/"backends.csv",index=False)
    save_json(out/"sdpa_kernels.json",kernels)
    tile_sweep(cfg,lengths).to_csv(out/"tile_sweep.csv",index=False)
    fidelity(cfg,lengths,IMPLS).to_csv(out/"fidelity.csv",index=False)
    print("exact attention correctness written")
    bench_all(cfg,lengths,args.smoke,args.config).to_csv(out/"timing.csv",index=False)
    print("exact attention timing written")


if __name__ == "__main__":
    main()
