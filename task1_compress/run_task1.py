import argparse
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from common.utils import load_json,load_yaml,repo_path,save_json,set_seed
from task1_compress import checks
from task1_compress.allocate import interpolate,knapsack,owl
from task1_compress.calibrate import input_stats,walk
from task1_compress.data import hellaswag,sequences,windows
from task1_compress.evaluate import hellaswag as score_hellaswag
from task1_compress.evaluate import hellaswag_scores,perplexity,pooled,sequence_windows
from task1_compress.kernel_probe import kernel_probe
from task1_compress.model import device,eligible,fingerprint,load_model,load_tokenizer,projection,revision
from task1_compress.prune import lowest_mask,magnitude_score,projection_totals,two_four_mask,wanda_score
from task1_compress.quantize import GROUPS,apply_scale,pack_bits,quantize,quantize_model,unpack_bits
from task1_compress.resources import PackedLinear,SparseLinear,file_bytes,latency,packed_tensors,storage,swap_projection
from task1_compress.sensitivity import mixed_scales,pruning_walk,quant_walk


class Run:
    """The config the fixed data and the output folders shared by every stage"""

    def __init__(self,cfg,smoke,tiny,artifacts):
        self.cfg,self.smoke,self.tiny = cfg,smoke,tiny
        small = cfg["smoke"] if smoke else {}
        tag = "_smoke" if smoke else ""
        self.results = repo_path(cfg["paths"]["results_dir"]+tag)
        self.artifacts = Path(artifacts) if artifacts else repo_path(cfg["paths"]["artifacts_dir"]+tag)
        self.calib = sequences(cfg,"c4_calibration",small.get("calib_sequences")).to(device())
        dev_ids = sequences(cfg,"c4_development",small.get("dev_sequences"))
        self.dev_wins = sequence_windows(dev_ids)
        self.time_ids = dev_ids.flatten()[None,:max(cfg["timing"]["prefill"])].to(device())
        self.wins = {s:windows(cfg,s,small.get("windows")) for s in ("wikitext","c4")}
        self.docs = hellaswag(cfg,small.get("hellaswag_rows"))
        self.tok = load_tokenizer(cfg)
        self.grid = small.get("awq_grid",cfg["quant"]["awq_grid"])
        self.reps = small.get("timing_reps",cfg["timing"]["reps"])
        self.originals = None

    def fresh(self,dtype=None):
        """Load an unchanged copy of the model and keep the original eligible weights the first time"""
        model = load_model(self.cfg,self.tiny,dtype)
        if self.originals is None:
            self.originals = {f"{i}.{n}":m.weight.detach().cpu().clone() for i,n,m in eligible(model,self.cfg)}
        return model

    def done(self,name):
        """Whether a condition already has saved scores"""
        return (self.results/"conditions"/name/"scores.json").exists()


def begin():
    """Start timing a build and reset the peak memory counter"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.time()


def finish(start):
    """Wall time and peak allocated memory since begin"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else 0.0
    return {"compress_seconds":time.time()-start,"compress_peak_gb":peak}


def to_csv(rows,path):
    """Save a list of dictionaries as a csv file"""
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def evaluate(run,name,model,info):
    """Score one model on every fixed set and save the per window and per row results"""
    out = run.results/"conditions"/name
    if run.done(name):
        return load_json(out/"scores.json")
    cfg = run.cfg
    start = time.time()
    batch = cfg["eval"]["window_batch"]
    calib_nll,errors = walk(model,cfg,run.calib,originals=run.originals)
    rows = {"dev":perplexity(model,run.dev_wins,batch),"wikitext":perplexity(model,run.wins["wikitext"],batch),
            "c4":perplexity(model,run.wins["c4"],batch)}
    count = cfg["eval"]["hellaswag_full"] if name in cfg["eval"]["full_hellaswag"] else cfg["eval"]["hellaswag_subset"]
    hs = score_hellaswag(model,run.tok,run.docs[:count],cfg["eval"]["hellaswag_batch"])
    mods = eligible(model,cfg)
    zeros = sum(int((m.weight == 0).sum()) for _,_,m in mods)
    total = sum(m.weight.numel() for _,_,m in mods)
    to_csv(errors,out/"calib_errors.csv")
    for k,r in rows.items():
        to_csv(r,out/f"windows_{k}.csv")
    to_csv([{**{k:v for k,v in r.items() if not isinstance(v,list)},
             **{f"{k}{j}":r[k][j] for k in ("ll","chars","tokens") for j in range(4)}} for r in hs],out/"hellaswag.csv")
    scores = {"condition":name,"calib_nll":calib_nll,**{k:pooled(r) for k,r in rows.items()},
              "hellaswag":hellaswag_scores(hs),"eligible_zeros":zeros,"eligible_weights":total,
              "sparsity":zeros/total,"effective_nonzero_fraction":1-zeros/total,"eval_seconds":time.time()-start,**info}
    save_json(out/"scores.json",scores)
    save_json(run.artifacts/name/"fingerprint.json",fingerprint(model,cfg))
    print(f"{name} wikitext {scores['wikitext']['ppl']:.3f} c4 {scores['c4']['ppl']:.3f} "
          f"hellaswag {scores['hellaswag']['acc_norm']:.4f} on {scores['hellaswag']['rows']} rows",flush=True)
    return scores


def save_masks(run,name,masks):
    """Store the packed removal masks of a pruned model"""
    (run.artifacts/name).mkdir(parents=True,exist_ok=True)
    torch.save(masks,run.artifacts/name/"masks.pt")


def apply_masks(model,cfg,masks):
    """Zero the stored removal masks in a fresh model"""
    for i,n,m in eligible(model,cfg):
        removed = unpack_bits(masks[f"{i}.{n}"].to(m.weight.device),1,m.weight.numel()).view(m.weight.shape).bool()
        m.weight.data[removed] = 0


def magnitude(model,cfg,s):
    """Remove the smallest magnitude weights of every row of every eligible projection"""
    masks = {}
    for i,n,m in eligible(model,cfg):
        mask = lowest_mask(magnitude_score(m.weight),round(s*m.weight.numel()))
        m.weight.data[mask] = 0
        masks[f"{i}.{n}"] = pack_bits(mask.to(torch.uint8),1).cpu()
    return masks


def wanda(run,model,totals=None,two_four=False,agreement=None):
    """Prune block by block with Wanda scores taken from the inputs of the already pruned blocks before it"""
    cfg = run.cfg
    names,micro = cfg["eligible"],cfg["calib"]["micro_batch"]
    masks = {}

    def before(i,layer,hidden,rope):
        stats = input_stats(layer,hidden,rope,micro,names)
        for n in names:
            w = projection(layer,n).weight
            score = wanda_score(w,stats[n]["sumsq"])
            mask = two_four_mask(score) if two_four else lowest_mask(score,totals[f"{i}.{n}"])
            if agreement is not None:
                mag = lowest_mask(magnitude_score(w),totals[f"{i}.{n}"])
                norms = stats[n]["sumsq"].sqrt()
                agreement.append({"block":i,"proj":n,"overlap":int((mask & mag).sum())/int(mask.sum()),
                                  "norm_max_over_median":(norms.max()/norms.median()).item()})
            w.data[mask] = 0
            masks[f"{i}.{n}"] = pack_bits(mask.to(torch.uint8),1).cpu()

    walk(model,cfg,run.calib,before=before)
    return masks


def uniform_totals(run,s):
    """Exact number of removed weights per projection at one sparsity"""
    return {k:round(s*w.numel()) for k,w in run.originals.items()}


def numels(run):
    """Eligible projection keys in model order with their sizes"""
    keys = list(run.originals)
    return keys,[run.originals[k].numel() for k in keys]


def block_totals(run,block_s,target):
    """Exact number of removed weights per projection for one sparsity per block"""
    keys,sizes = numels(run)
    per_block = len(run.cfg["eligible"])
    unit = [block_s[k//per_block] for k in range(len(keys))]
    return dict(zip(keys,projection_totals(unit,sizes,round(target*sum(sizes)))))


def environment(run):
    """Record the software hardware data and model versions used"""
    cfg = run.cfg
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return {"python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__,
            "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"device":gpu,
            "capability":list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            "model":cfg["model"]["name"],"model_revision":revision(cfg),"dtype":cfg["model"]["dtype"],
            "attention":cfg["model"]["attn"],"seed":cfg["seed"],"tiny":run.tiny,"smoke":run.smoke,
            "source_revisions":load_json(cfg["paths"]["lock"])["revisions"],"data_sha256":load_json(cfg["paths"]["checksums"])}


def data_checks(run):
    """Count windows scored tokens and sequence lengths of the fixed sets"""
    out = {}
    for s,wins in run.wins.items():
        out[s] = {"windows":len(wins),"scored_tokens":sum(w["end"]-w["start"] for w in wins),
                  "documents":len({w["source_row"] for w in wins}),"max_length":max(len(w["ids"]) for w in wins)}
    out["calibration"] = list(run.calib.shape)
    out["development"] = [len(run.dev_wins),len(run.dev_wins[0]["ids"])]
    out["hellaswag_rows"] = len(run.docs)
    if not run.smoke:
        assert all(out[s]["scored_tokens"] == 50000 for s in run.wins)
    return out


def baseline(run):
    """Run every correctness check on the unchanged model and evaluate it"""
    cfg = run.cfg
    if run.done("fp16"):
        return
    model = run.fresh()
    out = run.results/"checks"
    save_json(out/"environment.json",environment(run))
    save_json(out/"data.json",data_checks(run))
    save_json(out/"structure.json",checks.structure(model,cfg))
    save_json(out/"runner.json",checks.runner(model,run.calib[:2]))
    save_json(out/"batching.json",checks.batching(model,run.wins["wikitext"][:16]+run.wins["c4"][:8]))
    save_json(out/"quantizer.json",checks.quantizer())
    save_json(out/"harness.json",checks.harness(model,run.tok,cfg,run.docs,min(64,len(run.docs))))
    if not run.tiny:
        model32 = load_model(cfg,dtype=torch.float32)
        save_json(out/"precision.json",checks.precision(model32,model,run.wins["wikitext"][:16]+run.wins["c4"][:8]))
        del model32
        torch.cuda.empty_cache()
    evaluate(run,"fp16",model,storage(model,cfg,"dense"))


def pruning_conditions(run):
    """Magnitude and Wanda pruning at every uniform sparsity plus the two of four pattern"""
    cfg = run.cfg
    for s in cfg["prune"]["sparsities"]:
        name = f"p1_s{round(100*s)}"
        if not run.done(name):
            model = run.fresh()
            start = begin()
            masks = magnitude(model,cfg,s)
            info = finish(start)
            save_masks(run,name,masks)
            evaluate(run,name,model,{**info,**storage(model,cfg,"sparse"),"target_sparsity":s})
    for s in cfg["prune"]["sparsities"]:
        name = f"p2_s{round(100*s)}"
        if not run.done(name):
            model = run.fresh()
            agreement = []
            start = begin()
            masks = wanda(run,model,uniform_totals(run,s),agreement=agreement)
            info = finish(start)
            save_masks(run,name,masks)
            to_csv(agreement,run.results/"conditions"/name/"mask_agreement.csv")
            evaluate(run,name,model,{**info,**storage(model,cfg,"sparse"),"target_sparsity":s})
    if not run.done("p2_24"):
        model = run.fresh()
        start = begin()
        masks = wanda(run,model,two_four=True)
        info = finish(start)
        save_masks(run,"p2_24",masks)
        evaluate(run,"p2_24",model,{**info,**storage(model,cfg,"sparse"),"target_sparsity":0.5})


def select(rows,order,path):
    """Pick the candidate with the lowest development nll and write the choice before any test scoring"""
    if path.exists():
        return load_json(path)
    choice = {}
    for target in sorted({r["target"] for r in rows}):
        cands = sorted([r for r in rows if r["target"] == target],key=lambda r:(r["dev_nll"],order.index(r["design"])))
        choice[str(target)] = {"selected":cands[0]["design"],"dev_nll":{r["design"]:r["dev_nll"] for r in cands}}
    choice["criterion"] = "lowest development pooled nll with ties going to the earlier candidate"
    choice["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(path,choice)
    return choice


def audit(run,name,model,design,target):
    """Score a candidate that was not selected on the two held out perplexity sets"""
    batch = run.cfg["eval"]["window_batch"]
    return {"target":target,"design":design,"name":name,
            "wikitext":pooled(perplexity(model,run.wins["wikitext"],batch))["ppl"],
            "c4":pooled(perplexity(model,run.wins["c4"],batch))["ppl"]}


def adaptive_pruning(run):
    """Measure block sensitivity build every allocation candidate select on the development set and evaluate"""
    cfg,p = run.cfg,run.cfg["p3"]
    sens = run.results/"sensitivity"
    if not (sens/"p3_block_probes.csv").exists():
        model = run.fresh()
        start = begin()
        base,probes,owl_rows,stats_rows = pruning_walk(model,cfg,run.calib,p["probe_levels"])
        info = finish(start)
        to_csv(owl_rows,sens/"owl_ratios.csv")
        to_csv(stats_rows,sens/"wanda_stats.csv")
        save_json(sens/"p3_walk.json",{"dense_calib_nll":base,"probes":len(probes),**info})
        to_csv(probes,sens/"p3_block_probes.csv")
        del model
    probes = pd.read_csv(sens/"p3_block_probes.csv")
    ratios = pd.read_csv(sens/"owl_ratios.csv").sort_values("block")["outlier_ratio"].tolist()
    blocks = sorted(probes["block"].unique())
    grid = np.round(np.arange(p["min"],p["max"]+1e-9,p["grid_step"]),4)
    dev_rows,candidates = [],{}
    for target in p["targets"]:
        curves = [interpolate(p["probe_levels"],probes[probes["block"] == b].sort_values("sparsity")["delta"].values,grid)
                  for b in blocks]
        steps = [round(g/p["grid_step"]) for g in grid]
        choice = knapsack(curves,[steps]*len(blocks),round(len(blocks)*target/p["grid_step"]),exact=True)
        cands = {"measured":[round(float(grid[c]),6) for c in choice]}
        for lam in p["owl_lambdas"]:
            cands[f"owl_{lam}"] = [round(float(x),6) for x in owl(ratios,target,lam,p["min"],p["max"])]
        for design,block_s in cands.items():
            key = f"p3_s{round(100*target)}_{design}"
            art = run.artifacts/key
            candidates[(target,design)] = key
            if not (art/"dev.json").exists():
                model = run.fresh()
                totals = block_totals(run,block_s,target)
                start = begin()
                masks = wanda(run,model,totals)
                info = finish(start)
                dev = pooled(perplexity(model,run.dev_wins,cfg["eval"]["window_batch"]))
                save_masks(run,key,masks)
                save_json(art/"dev.json",{"dev_nll":dev["nll"],"block_sparsity":block_s,
                                          "eligible_zeros":sum(totals.values()),**info})
                del model
            rec = load_json(art/"dev.json")
            dev_rows.append({"target":target,"design":design,"dev_nll":rec["dev_nll"],"dev_ppl":math.exp(rec["dev_nll"]),
                             "build_seconds":rec["compress_seconds"],"eligible_zeros":rec["eligible_zeros"],
                             **{f"block{b}":s for b,s in enumerate(rec["block_sparsity"])}})
    to_csv(dev_rows,run.results/"selection"/"p3_candidates_dev.csv")
    choice = select(dev_rows,p["candidates"],run.results/"selection"/"p3_selection.json")
    walk_info = load_json(sens/"p3_walk.json")
    audits = []
    for target in p["targets"]:
        picked = choice[str(target)]["selected"]
        name = f"p3_s{round(100*target)}"
        rec = load_json(run.artifacts/candidates[(target,picked)]/"dev.json")
        if not run.done(name):
            model = run.fresh()
            masks = torch.load(run.artifacts/candidates[(target,picked)]/"masks.pt")
            apply_masks(model,cfg,masks)
            save_masks(run,name,masks)
            save_json(run.results/"conditions"/name/"allocation.json",{"design":picked,"unit":"block",
                      "block_sparsity":rec["block_sparsity"],"eligible_zeros":rec["eligible_zeros"]})
            evaluate(run,name,model,{**storage(model,cfg,"sparse"),"target_sparsity":target,"design":picked,
                                     "compress_seconds":rec["compress_seconds"],"compress_peak_gb":rec["compress_peak_gb"],
                                     "sensitivity_seconds":walk_info["compress_seconds"]})
        for design in p["candidates"]:
            if design != picked:
                model = run.fresh()
                apply_masks(model,cfg,torch.load(run.artifacts/candidates[(target,design)]/"masks.pt"))
                audits.append(audit(run,candidates[(target,design)],model,design,target))
                del model
    to_csv(audits,run.results/"selection"/"p3_audit_test.csv")


def uniform_bits(run,b):
    """The same bit width for every eligible projection"""
    return {(int(k.split(".")[0]),k.split(".")[1]):b for k in run.originals}


def quant_condition(run,name,bits_map,scales,extra):
    """Build and evaluate one quantized model and store what is needed to rebuild it"""
    cfg = run.cfg
    model = run.fresh()
    start = begin()
    sizes = quantize_model(model,cfg,bits_map,scales)
    info = finish(start)
    (run.artifacts/name).mkdir(parents=True,exist_ok=True)
    torch.save({"bits":bits_map,"scales":scales},run.artifacts/name/"quant.pt")
    return evaluate(run,name,model,{**info,**storage(model,cfg,"quant",sizes,o_scales=scales is not None),**extra})


def rtn_conditions(run):
    """Plain round to nearest quantization at every bit width"""
    for b in run.cfg["quant"]["bits"]:
        quant_condition(run,f"q1_b{b}",uniform_bits(run,b),None,{"bits":b})


def awq_conditions(run):
    """Search activation scales and probe every projection then evaluate the uniform activation aware models"""
    cfg = run.cfg
    sens = run.results/"sensitivity"
    path = run.artifacts/"awq_scales.pt"
    if not (sens/"q3_projection_probes.csv").exists():
        model = run.fresh()
        start = begin()
        base,scales,search_rows,local_rows,probes,timer = quant_walk(model,cfg,run.calib,cfg["q3"]["bits"],run.grid)
        info = finish(start)
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(scales,path)
        to_csv(search_rows,sens/"awq_search.csv")
        to_csv(local_rows,sens/"awq_local_errors.csv")
        save_json(sens/"q3_walk.json",{"dense_calib_nll":base,"probes":len(probes),
                                       "search_seconds":{str(b):t for b,t in timer.items()},**info})
        to_csv(probes,sens/"q3_projection_probes.csv")
        del model
        # checking that folding the scales alone leaves the function unchanged before any rounding
        model = run.fresh()
        with torch.no_grad():
            ids = run.calib[:2]
            ref = model(input_ids=ids,labels=ids)
            for i,layer in enumerate(model.model.layers):
                for g in GROUPS:
                    apply_scale(layer,g,scales[(i,g,4)])
            new = model(input_ids=ids,labels=ids)
            diff = (ref.logits.float()-new.logits.float()).abs()
            save_json(run.results/"checks"/"awq_fold.json",{"max_abs_logit_diff":diff.max().item(),
                      "mean_abs_logit_diff":diff.mean().item(),"nll_before":ref.loss.item(),"nll_after":new.loss.item()})
        del model
    scales = torch.load(path)
    walk_info = load_json(sens/"q3_walk.json")
    for b in cfg["quant"]["bits"]:
        chosen = {(i,g):scales[(i,g,b)].float() for (i,g,bb) in scales if bb == b}
        quant_condition(run,f"q2_b{b}",uniform_bits(run,b),chosen,
                        {"bits":b,"search_seconds":walk_info["search_seconds"][str(b)]})
    return scales


def mixed_precision(run,scales):
    """Allocate bit widths from the projection probes or the local errors select on the development set and evaluate"""
    cfg,q = run.cfg,run.cfg["q3"]
    sens = run.results/"sensitivity"
    probes = pd.read_csv(sens/"q3_projection_probes.csv")
    local = pd.read_csv(sens/"awq_local_errors.csv")
    keys,sizes = numels(run)
    unit = math.gcd(*sizes)
    units = [s//unit for s in sizes]
    pairs = [(int(k.split(".")[0]),k.split(".")[1]) for k in keys]
    value = {"measured":probes.set_index(["block","proj","bits"])["delta"],
             "local":local.set_index(["block","proj","bits"])["rel_error"]}
    dev_rows,candidates = [],{}
    walk_info = load_json(sens/"q3_walk.json")
    for budget in q["budgets"]:
        limit = load_json(run.results/"conditions"/f"q2_b{budget}"/"scores.json")["packed_bytes"]
        for design in q["candidates"]:
            key = f"q3_int{budget}_{design}"
            art = run.artifacts/key
            candidates[(budget,design)] = key
            if not (art/"dev.json").exists():
                costs = [[float(value[design][(i,n,b)]) for b in q["bits"]] for i,n in pairs]
                weights = [[u*b for b in q["bits"]] for u in units]
                choice = knapsack(costs,weights,sum(units)*budget)
                bits_map = {pr:q["bits"][c] for pr,c in zip(pairs,choice)}
                model = run.fresh()
                start = begin()
                mixed = mixed_scales(model,cfg,run.calib,bits_map,scales,run.grid)
                mixed = {k:v.float() for k,v in mixed.items()}
                sizes_q = quantize_model(model,cfg,bits_map,mixed)
                info = finish(start)
                packed = storage(model,cfg,"quant",sizes_q,o_scales=True)["packed_bytes"]
                assert packed <= limit,f"{key} uses {packed} bytes above the budget {limit}"
                dev = pooled(perplexity(model,run.dev_wins,cfg["eval"]["window_batch"]))
                art.mkdir(parents=True,exist_ok=True)
                torch.save({"bits":bits_map,"scales":mixed},art/"quant.pt")
                save_json(art/"dev.json",{"dev_nll":dev["nll"],"packed_bytes":packed,"budget_bytes":limit,
                                          "bits":{f"{i}.{n}":b for (i,n),b in bits_map.items()},**info})
                del model
            rec = load_json(art/"dev.json")
            counts = pd.Series(rec["bits"]).value_counts().to_dict()
            dev_rows.append({"target":budget,"design":design,"dev_nll":rec["dev_nll"],"dev_ppl":math.exp(rec["dev_nll"]),
                             "packed_bytes":rec["packed_bytes"],"budget_bytes":rec["budget_bytes"],
                             "build_seconds":rec["compress_seconds"],**{f"n_b{b}":counts.get(b,0) for b in q["bits"]}})
    to_csv(dev_rows,run.results/"selection"/"q3_candidates_dev.csv")
    choice = select(dev_rows,q["candidates"],run.results/"selection"/"q3_selection.json")
    probe_seconds = float(probes["seconds"].sum())
    audits = []
    for budget in q["budgets"]:
        picked = choice[str(budget)]["selected"]
        name = f"q3_int{budget}"
        art = run.artifacts/candidates[(budget,picked)]
        rec = load_json(art/"dev.json")
        if not run.done(name):
            saved = torch.load(art/"quant.pt")
            model = run.fresh()
            sizes_q = quantize_model(model,cfg,saved["bits"],saved["scales"])
            (run.artifacts/name).mkdir(parents=True,exist_ok=True)
            torch.save(saved,run.artifacts/name/"quant.pt")
            save_json(run.results/"conditions"/name/"allocation.json",{"design":picked,"unit":"projection",
                      "bits":rec["bits"],"packed_bytes":rec["packed_bytes"],"budget_bytes":rec["budget_bytes"]})
            evaluate(run,name,model,{**storage(model,cfg,"quant",sizes_q,o_scales=True),"budget_bits":budget,
                                     "design":picked,"compress_seconds":rec["compress_seconds"],
                                     "compress_peak_gb":rec["compress_peak_gb"],"probe_seconds":probe_seconds,
                                     "walk_seconds":walk_info["compress_seconds"]})
        for design in q["candidates"]:
            if design != picked:
                saved = torch.load(run.artifacts/candidates[(budget,design)]/"quant.pt")
                model = run.fresh()
                quantize_model(model,cfg,saved["bits"],saved["scales"])
                audits.append(audit(run,candidates[(budget,design)],model,design,budget))
                del model
    to_csv(audits,run.results/"selection"/"q3_audit_test.csv")


def track_a(run):
    """Baseline checks uniform and adaptive pruning and plain quantization"""
    baseline(run)
    pruning_conditions(run)
    adaptive_pruning(run)
    rtn_conditions(run)
    print("finished track a",flush=True)


def track_b(run):
    """Activation aware quantization and mixed precision"""
    scales = awq_conditions(run)
    mixed_precision(run,scales)
    print("finished track b",flush=True)


def rebuild_all(run):
    """Recreate the saved form of every evaluated condition from its stored rule or allocation

    every rebuilt model is walked over the calibration set again so its nll can be compared
    with the evaluated one and its calibration output errors are written again"""
    cfg = run.cfg
    cond = run.results/"conditions"
    rows = []

    def store(name,model,masks=None,quant=None):
        """Save what the timing stage needs and compare the rebuilt model with the evaluated one"""
        art = run.artifacts/name
        art.mkdir(parents=True,exist_ok=True)
        if masks is not None:
            torch.save(masks,art/"masks.pt")
        if quant is not None:
            torch.save(quant,art/"quant.pt")
        save_json(art/"fingerprint.json",fingerprint(model,cfg))
        nll,errors = walk(model,cfg,run.calib,originals=run.originals)
        both = pd.read_csv(cond/name/"calib_errors.csv").merge(pd.DataFrame(errors),on=["block","proj"],
                                                                suffixes=("_old","_new"))
        gap = (both["rel_error_old"]-both["rel_error_new"]).abs()
        up = both["proj"] == "up_proj"
        rows.append({"condition":name,"calib_nll_evaluated":load_json(cond/name/"scores.json")["calib_nll"],
                     "calib_nll_rebuilt":nll,"max_error_change_up":gap[up].max(),"max_error_change_rest":gap[~up].max()})
        to_csv(errors,cond/name/"calib_errors.csv")
        print("rebuilt",name,flush=True)

    store("fp16",run.fresh())
    for s in cfg["prune"]["sparsities"]:
        model = run.fresh()
        store(f"p1_s{round(100*s)}",model,masks=magnitude(model,cfg,s))
        model = run.fresh()
        store(f"p2_s{round(100*s)}",model,masks=wanda(run,model,uniform_totals(run,s)))
    model = run.fresh()
    store("p2_24",model,masks=wanda(run,model,two_four=True))
    for target in cfg["p3"]["targets"]:
        name = f"p3_s{round(100*target)}"
        model = run.fresh()
        totals = block_totals(run,load_json(cond/name/"allocation.json")["block_sparsity"],target)
        store(name,model,masks=wanda(run,model,totals))
    for b in cfg["quant"]["bits"]:
        model = run.fresh()
        bits = uniform_bits(run,b)
        quantize_model(model,cfg,bits,None)
        store(f"q1_b{b}",model,quant={"bits":bits,"scales":None})
    del model
    model = run.fresh()
    _,scales,_,_,_,_ = quant_walk(model,cfg,run.calib,cfg["q3"]["bits"],run.grid,probe=False)
    for b in cfg["quant"]["bits"]:
        chosen = {(i,g):scales[(i,g,bb)].float() for (i,g,bb) in scales if bb == b}
        model = run.fresh()
        bits = uniform_bits(run,b)
        quantize_model(model,cfg,bits,chosen)
        store(f"q2_b{b}",model,quant={"bits":bits,"scales":chosen})
    for budget in cfg["q3"]["budgets"]:
        name = f"q3_int{budget}"
        bits = {(int(k.split(".")[0]),k.split(".")[1]):b for k,b in load_json(cond/name/"allocation.json")["bits"].items()}
        model = run.fresh()
        mixed = {k:v.float() for k,v in mixed_scales(model,cfg,run.calib,bits,scales,run.grid).items()}
        model = run.fresh()
        quantize_model(model,cfg,bits,mixed)
        store(name,model,quant={"bits":bits,"scales":mixed})
    to_csv(rows,run.results/"checks"/"rebuild.csv")
    print("finished rebuild",flush=True)


def rebuild(run,name):
    """Rebuild a saved condition in a fresh model and confirm it matches the evaluated weights"""
    model = run.fresh()
    art = run.artifacts/name
    saved = None
    if (art/"masks.pt").exists():
        apply_masks(model,run.cfg,torch.load(art/"masks.pt"))
    if (art/"quant.pt").exists():
        saved = torch.load(art/"quant.pt")
        quantize_model(model,run.cfg,saved["bits"],saved["scales"])
    stored = load_json(art/"fingerprint.json")
    now = fingerprint(model,run.cfg)
    assert all(math.isclose(now[k][j],stored[k][j],rel_tol=1e-12,abs_tol=1e-9) for k in stored for j in (0,1)),name
    return model,saved


def packed_model(run,model,saved):
    """Replace every eligible projection with a layer that keeps its compressed form on the gpu"""
    cfg = run.cfg
    for i,layer in enumerate(model.model.layers):
        for n in cfg["eligible"]:
            m = projection(layer,n)
            if saved is None:
                new = SparseLinear(m.weight.data,m.bias)
            else:
                codes,scale,zero = quantize(m.weight.data,saved["bits"][(i,n)],cfg["quant"]["group"],cfg["quant"]["scale_floor"])
                in_scale = None
                if n == "o_proj" and saved["scales"] is not None:
                    in_scale = saved["scales"][(i,"o")].half().to(m.weight.device)
                new = PackedLinear(codes,scale,zero,saved["bits"][(i,n)],cfg["quant"]["group"],m.bias,in_scale)
            swap_projection(layer,n,new)
    return model


@torch.no_grad()
def final(run):
    """Time every condition alone on one gpu and measure the packed execution path and the kernels"""
    cfg,t = run.cfg,run.cfg["timing"]
    names = sorted([p.name for p in (run.results/"conditions").iterdir() if (p/"scores.json").exists()])
    if run.smoke:
        # one dense one sparse and two quantized paths are enough to exercise every branch
        names = [n for n in names if n in ("fp16","p2_s50","q2_b4","q3_int3")]
    out = run.results/"resources"
    timing,packed,texts = [],[],[]
    warmup = t["warmup"] if not run.smoke else 1
    # running the unchanged model once untimed so the gpu clocks settle before the first measured condition
    model,_ = rebuild(run,"fp16")
    latency(model,run.time_ids,cfg,warmup,run.reps,t["decode_tokens"])
    del model
    for name in names:
        model,saved = rebuild(run,name)
        timing.append({"condition":name,**latency(model,run.time_ids,cfg,warmup,run.reps,t["decode_tokens"])})
        if name in cfg["qualitative"]["continuation_models"]:
            for prompt in cfg["qualitative"]["prompts"]:
                ids = run.tok(prompt,return_tensors="pt").input_ids.to(device())
                gen = model.generate(ids,max_new_tokens=cfg["qualitative"]["new_tokens"],do_sample=False,
                                     pad_token_id=run.tok.eos_token_id)
                texts.append({"condition":name,"prompt":prompt,"continuation":run.tok.decode(gen[0,ids.shape[1]:])})
        if name in t["packed_conditions"]:
            ref = model(input_ids=run.calib[:1]).logits.float()
            if saved is not None:
                del model
                model = run.fresh()
                if saved["scales"] is not None:
                    for i,layer in enumerate(model.model.layers):
                        for g in GROUPS:
                            apply_scale(layer,g,saved["scales"][(i,g)])
            model = packed_model(run,model,saved)
            diff = (ref-model(input_ids=run.calib[:1]).logits.float()).abs().max().item()
            # freeing the reference logits so the packed memory reading holds only the model
            del ref
            packed.append({"condition":name,"max_abs_logit_diff":diff,
                           **latency(model,run.time_ids,cfg,t["packed_warmup"],t["packed_reps"] if not run.smoke else 2,
                                     t["packed_decode_tokens"]),
                           "file_bytes":file_bytes(packed_tensors(model),run.artifacts/"packed.safetensors")})
        if name == "fp16":
            packed.append({"condition":name,"max_abs_logit_diff":0.0,
                           "file_bytes":file_bytes(packed_tensors(model),run.artifacts/"dense.safetensors")})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("timed",name,flush=True)
    # timing the unchanged model again at the end to show how far the gpu clocks drifted over the stage
    model,_ = rebuild(run,"fp16")
    timing.append({"condition":"fp16_end",**latency(model,run.time_ids,cfg,warmup,run.reps,t["decode_tokens"])})
    del model
    to_csv(timing,out/"timing.csv")
    to_csv(packed,out/"packed_execution.csv")
    save_json(run.results/"qualitative"/"continuations.json",texts)
    tokens = (1,64) if run.tiny else (1,max(t["prefill"]))
    to_csv(kernel_probe(cfg,1 if run.smoke else t["warmup"],run.reps,tokens),out/"kernel_probe.csv")
    print("finished final stage",flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--stage",choices=["track_a","track_b","final","all","rebuild"],required=True)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--artifacts")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.smoke:
        cfg["prune"]["sparsities"] = [0.5]
        cfg["quant"]["bits"] = [4,3]
    set_seed(cfg["seed"])
    run = Run(cfg,args.smoke,args.tiny,args.artifacts)
    if args.stage in ("track_a","all"):
        track_a(run)
    if args.stage in ("track_b","all"):
        track_b(run)
    if args.stage == "rebuild":
        rebuild_all(run)
    if args.stage in ("final","all"):
        final(run)


if __name__ == "__main__":
    main()
