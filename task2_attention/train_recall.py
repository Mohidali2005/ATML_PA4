import argparse
import gzip
import time

import numpy as np
import pandas as pd
import torch

from common.utils import load_yaml,repo_path,save_json,set_seed
from task2_attention.mqar import build_stream,check,eval_sets,load_stream
from task2_attention.ops import dilated_mask,window_mask
from task2_attention.recall_model import RecallModel,loss_and_hits
from task2_attention.timing import peak,sync,timed


def device():
    """Pick the gpu when one is present"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def factor(tr):
    """Learning rate multiplier with linear warmup then linear decay to the final fraction"""
    def f(u):
        if u < tr["warmup"]:
            return (u+1)/tr["warmup"]
        return 1-(1-tr["final_lr_fraction"])*(u-tr["warmup"])/max(1,tr["updates"]-tr["warmup"])
    return f


def optimizer(model,tr):
    """AdamW with weight decay on matrices only"""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params":decay,"weight_decay":tr["weight_decay"]},{"params":rest,"weight_decay":0.0}],
                             lr=tr["lr"])


def parameters(model):
    """Number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model,sets,mq,batch,test_window=None):
    """Per query correctness and mean loss on every evaluation set"""
    model.eval()
    out = {}
    for key,s in sets.items():
        rows,loss,count = [],0.0,0
        for a in range(0,len(s["tokens"]),batch):
            x = s["tokens"][a:a+batch].to(device())
            y = s["labels"][a:a+batch].to(device())
            logits = model(x,test_window=test_window)
            l,_ = loss_and_hits(logits,y,mq["value_offset"])
            n = int((y >= 0).sum())
            loss,count = loss+l.item()*n,count+n
            hits = (logits.argmax(-1) == (y-mq["value_offset"])) & (y >= 0)
            # reading each row in query position order to line up with the stored distances
            rows += [h[m].cpu().numpy() for h,m in zip(hits,y >= 0)]
        out[key] = {"rows":rows,"loss":loss/count}
    model.train()
    return out


def summarize(result,sets,mq,meta):
    """Accuracy per evaluation set and per distance bin"""
    set_rows,bin_rows = [],[]
    edges = mq["bins"]
    for (k,regime),r in result.items():
        hits = np.concatenate(r["rows"])
        dist = np.concatenate([np.array(row["distances"]) for row in sets[(k,regime)]["rows"]])
        set_rows.append({**meta,"test_k":k,"test_regime":regime,"queries":len(hits),"correct":int(hits.sum()),
                         "accuracy":hits.mean(),"loss":r["loss"]})
        for lo,hi in zip(edges[:-1],edges[1:]):
            sel = (dist >= lo) & (dist < hi)
            if sel.any():
                bin_rows.append({**meta,"test_k":k,"test_regime":regime,"bin":f"[{lo},{hi})","queries":int(sel.sum()),
                                 "correct":int(hits[sel].sum()),"accuracy":hits[sel].mean()})
    return set_rows,bin_rows


def query_table(result,sets,mask):
    """One row per test sequence with its query correctness as a string of zeros and ones"""
    rows = []
    for key,r in result.items():
        for rec,h in zip(sets[key]["rows"],r["rows"]):
            rows.append({"id":rec["id"],"mask":mask,"correct":"".join("1" if x else "0" for x in h)})
    return rows


def run_one(cfg,run,seed,out_root,dev_sets,test_sets,smoke):
    """Train one mechanism at one load and regime with one seed and evaluate it"""
    mq,tr = cfg["mqar"],dict(cfg["mqar"]["train"])
    if smoke:
        tr["updates"],tr["eval_at"] = cfg["smoke"]["mqar_updates"],cfg["smoke"]["mqar_eval_at"]
    out = out_root/f"{run['name']}_s{seed}"
    if (out/"summary.json").exists():
        return
    set_seed(seed)
    model = RecallModel(mq,run["mechanism"]).to(device())
    opt = optimizer(model,tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt,factor(tr))
    count = tr["updates"]*tr["batch"]
    tokens,labels = load_stream(cfg,run["k"],run["regime"],count)
    tokens,labels = tokens.to(device()),labels.to(device())
    # the seed fixes both the initial weights and the order of the shared fixed stream
    order = torch.from_numpy(np.random.default_rng(seed).permutation(count)).to(device())
    log,curve = [],[]
    train_seconds = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    for u in range(tr["updates"]):
        idx = order[u*tr["batch"]:(u+1)*tr["batch"]]
        x,y = tokens[idx].long(),labels[idx].long()
        loss,hits = loss_and_hits(model(x),y,mq["value_offset"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),tr["clip"])
        opt.step()
        sched.step()
        done = u+1
        if done % tr["log_every"] == 0 or done == 1:
            log.append({"update":done,"loss":loss.item(),"train_accuracy":hits.float().mean().item(),
                        "grad_norm":norm.item(),"lr":sched.get_last_lr()[0]})
        if done in tr["eval_at"]:
            sync()
            train_seconds += time.perf_counter()-start
            res = evaluate(model,dev_sets,mq,tr["eval_batch"])
            for (k,regime),r in res.items():
                hit = np.concatenate(r["rows"])
                curve.append({"update":done,"k":k,"regime":regime,"accuracy":hit.mean(),"loss":r["loss"]})
            sync()
            start = time.perf_counter()
    sync()
    train_seconds += time.perf_counter()-start
    meta = {"run":run["name"],"seed":seed,"mechanism":run["mechanism"],"train_k":run["k"],"train_regime":run["regime"]}
    result = evaluate(model,test_sets,mq,tr["eval_batch"])
    set_rows,bin_rows = summarize(result,test_sets,mq,{**meta,"mask":"trained"})
    queries = query_table(result,test_sets,"trained")
    if run["name"] in mq["test_window_runs"]:
        windowed = evaluate(model,test_sets,mq,tr["eval_batch"],test_window=mq["window"])
        a,b = summarize(windowed,test_sets,mq,{**meta,"mask":"test_window"})
        set_rows,bin_rows = set_rows+a,bin_rows+b
        queries += query_table(windowed,test_sets,"test_window")
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(log).to_csv(out/"train_log.csv",index=False)
    pd.DataFrame(curve).to_csv(out/"curve.csv",index=False)
    pd.DataFrame(set_rows).to_csv(out/"test_sets.csv",index=False)
    pd.DataFrame(bin_rows).to_csv(out/"test_bins.csv",index=False)
    with gzip.open(out/"test_queries.csv.gz","wt",encoding="utf-8") as f:
        pd.DataFrame(queries).to_csv(f,index=False)
    ckpt = repo_path(cfg["paths"]["outputs_dir"])/"checkpoints"/f"{run['name']}_s{seed}.pt"
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.half() for k,v in model.state_dict().items()},ckpt)
    save_json(out/"summary.json",{**meta,"parameters":parameters(model),"updates":tr["updates"],
              "train_seconds":train_seconds,"ms_per_update":1000*train_seconds/tr["updates"],
              "peak_train_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None,
              "final_train_loss":log[-1]["loss"],"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})


def theory(cfg):
    """Attended pairs candidate positions and persistent state of every mechanism at the training length"""
    mq = cfg["mqar"]
    t,heads,dim = mq["length"],mq["model"]["heads"],mq["model"]["dim"]
    hd = dim//heads
    causal = torch.tril(torch.ones(t,t,dtype=torch.bool))
    local = window_mask(t,mq["window"],"cpu")
    spaced = dilated_mask(t,mq["dilation"],mq["dilated_positions"],"cpu")
    masks = {"dense":[causal]*heads,"window":[local]*heads,"dilated":[local]*(heads//2)+[spaced]*(heads-heads//2)}
    offset = torch.arange(t)[:,None]-torch.arange(t)[None,:]
    far = torch.arange(128,251)
    rows = []
    for name,ms in masks.items():
        pairs = sum(int(m.sum()) for m in ms)
        union = torch.stack(ms).any(0).float()
        # a second layer reads positions that the first layer already gathered so reach composes
        two = (union@union) > 0
        last = two[-1].flip(0)
        # a decoder must keep every position that any later query of a head may still read
        buffer = sum(min(t,int(offset[m].max())+1) for m in ms)
        rows.append({"mechanism":name,"pairs_per_layer":pairs,"mean_positions_per_query":pairs/(heads*t),
                     "max_offset_one_layer":int(offset[union.bool()].max()),"max_offset_two_layers":int(offset[two].max()),
                     "far_offsets_reachable_two_layers":float(last[far].float().mean()),
                     "decode_state_floats_per_layer":2*hd*buffer,"decode_state_bytes_fp32":4*2*hd*buffer})
    rows.append({"mechanism":"linear","pairs_per_layer":None,"mean_positions_per_query":None,
                 "max_offset_one_layer":t-1,"max_offset_two_layers":t-1,"far_offsets_reachable_two_layers":1.0,
                 "decode_state_floats_per_layer":heads*(hd*hd+hd),"decode_state_bytes_fp32":4*heads*(hd*hd+hd)})
    return pd.DataFrame(rows)


@torch.no_grad()
def sweep(cfg,mechanisms,lengths,smoke):
    """Forward only time and memory of every mechanism on random input embeddings at growing lengths"""
    mq,sw = cfg["mqar"],cfg["mqar"]["sweep"]
    warm,reps = (1,2) if smoke else (sw["warmup"],sw["reps"])
    rows = []
    for mech in mechanisms:
        set_seed(cfg["seed"])
        model = RecallModel(mq,mech).to(device()).eval()
        for t in lengths:
            hidden = torch.randn(sw["batch"],t,mq["model"]["dim"],device=device())
            heads = mq["model"]["heads"]
            need = 4*sw["batch"]*heads*t*t*4/1e9 if mech != "linear" else 3*sw["batch"]*heads*t*(mq["model"]["dim"]//heads)**2*4/1e9
            free = torch.cuda.mem_get_info()[0]/1e9 if torch.cuda.is_available() else 1e9
            row = {"mechanism":mech,"length":t,"batch":sw["batch"],"reps":reps,"estimated_gb":need,
                   "implementation":"materialized prefix state" if mech == "linear" else
                   ("dense scores with a dense boolean mask" if mech != "dense" else "dense scores with a causal mask")}
            if need > free-1.0:
                rows.append({**row,"status":"skipped over the free memory"})
                continue
            ms,sd = timed(lambda:model(hidden=hidden),warm,reps)
            gb,reserved = peak(lambda:model(hidden=hidden))
            rows.append({**row,"status":"ok","forward_ms":ms,"forward_std":sd,"peak_gb":gb,"reserved_gb":reserved})
            del hidden
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--stage",choices=["data","train","sweep","all"],default="all")
    ap.add_argument("--runs",nargs="*",default=None)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--results",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    mq = cfg["mqar"]
    res = repo_path(args.results or cfg["paths"]["results_dir"])
    cfg["paths"]["results_dir"] = str(res)
    runs = [r for r in mq["runs"] if args.runs is None or r["name"] in args.runs]
    count = (cfg["smoke"]["mqar_updates"] if args.smoke else mq["train"]["updates"])*mq["train"]["batch"]
    seeds = mq["seeds"][:1] if args.smoke else mq["seeds"]
    if args.stage in ("data","all"):
        streams = {f"k{k}_{reg}":build_stream(cfg,k,reg,count) for k,reg in sorted({(r["k"],r["regime"]) for r in runs})}
        check(cfg,streams)
        print("mqar data checked for",len(streams),"training streams")
    if args.stage in ("train","all"):
        per_set = cfg["smoke"]["mqar_test_per_set"] if args.smoke else None
        dev_sets = eval_sets(cfg,"development",per_set)
        test_sets = eval_sets(cfg,"test",per_set)
        for run in runs:
            for seed in seeds:
                run_one(cfg,run,seed,res/"recall"/"runs",dev_sets,test_sets,args.smoke)
                print("finished",run["name"],"seed",seed)
    if args.stage in ("sweep","all"):
        theory(cfg).to_csv(res/"recall"/"theory.csv",index=False)
        mechs = sorted({r["mechanism"] for r in runs},key=["dense","window","dilated","linear"].index)
        lengths = cfg["smoke"]["sweep_lengths"] if args.smoke else mq["sweep"]["lengths"]
        sweep(cfg,mechs,lengths,args.smoke).to_csv(res/"recall"/"sweep.csv",index=False)
        print("mqar resource sweep written")


if __name__ == "__main__":
    main()
