import argparse
import time

import pandas as pd
import torch
from transformers import DynamicCache

from common.utils import load_yaml,repo_path,save_json,set_seed
from task2_attention.timing import timed
from task3_speculative.data import chat_ids,fixed_subset,test_prompts,train_prompts,write_gz
from task3_speculative.lora import add_lora,load_factors
from task3_speculative.models import DRAFTS,artifacts,device,load_model,load_tokenizer
from task3_speculative.sampler import HFLM,now,speculative,target_only

SPEC_KEYS = ("rounds","proposed","accepted","examined","target_passes","draft_passes","draft_fed","target_fed","eos_route",
             "target_prefill_ms","draft_prefill_ms","decode_ms","eos","truncated")


def reset_peak():
    """Start a fresh peak memory reading"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    """Peak allocated memory since the last reset"""
    return torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None


def allocated_gb():
    """Memory held by tensors right now"""
    return torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else None


def seeds(cfg,p,rep):
    """Sampling seeds of one prompt that are shared by every draft and proposal length"""
    base = cfg["seed"]+cfg["banks"]["seed_step"]*p["index"]+7919*(rep+1)
    return base,base+1


def pair(cfg,target,draft,p,ids,gamma,cap,rep,gen,ignore,target_first):
    """Time the speculative sampler and an adjacent target only run on one prompt"""
    eos = set(cfg["models"]["eos"])
    s_target,s_spec = seeds(cfg,p,rep)
    res = {}
    for who in (("target","spec") if target_first else ("spec","target")):
        reset_peak()
        gen.manual_seed(s_target if who == "target" else s_spec)
        if who == "target":
            r = target_only(target,ids,cap,eos,gen,ignore)
        else:
            r = speculative(target,draft,ids,gamma,cap,eos,gen,ignore)
        r["peak_gb"] = peak_gb()
        res[who] = r
    t,s = res["target"],res["spec"]
    row = {"prompt_id":p["id"],"split":p["split"],"prompt_tokens":len(ids),"gamma":gamma,"rep":rep,"target_first":target_first,
           "target_tokens":len(t["tokens"]),"target_only_prefill_ms":t["target_prefill_ms"],"target_only_decode_ms":t["decode_ms"],
           "target_only_passes":t["target_passes"],"target_only_eos":t["eos"],"target_only_truncated":t["truncated"],
           "target_only_peak_gb":t["peak_gb"],"spec_tokens":len(s["tokens"]),"spec_peak_gb":s["peak_gb"],
           **{f"spec_{k}":s[k] for k in SPEC_KEYS}}
    return row,{"target_tokens":t["tokens"],"spec_tokens":s["tokens"],"accepted_per_round":s["accepted_per_round"],
                "proposed_per_round":s["proposed_per_round"]}


def main_stage(cfg,args,out):
    """End to end benchmark of the original and the four merged drafts at both proposal lengths"""
    b = cfg["benchmark"]
    tok = load_tokenizer(cfg)
    prompts = test_prompts(cfg,cfg["smoke"]["test_prompts_per_split"] if args.smoke else None)
    fixed = fixed_subset(prompts,cfg["smoke"]["fixed_prompts"] if args.smoke else b["fixed_prompts"],cfg["seed"])
    cap = cfg["smoke"]["bench_cap"] if args.smoke else b["cap"]
    ids = {p["id"]:chat_ids(tok,p["text"]) for p in prompts}
    gen = torch.Generator(device=device())
    t0 = time.perf_counter()
    target = HFLM(load_model(cfg,"target",args.tiny))
    target_load = time.perf_counter()-t0
    target_resident = allocated_gb()
    rows,tokens,timed_rows,blocks = [],[],[],[]
    folder = out/"benchmark"
    folder.mkdir(parents=True,exist_ok=True)
    eos = set(cfg["models"]["eos"])
    for rep in range(b["reps"]):
        order = DRAFTS if rep == 0 else DRAFTS[::-1]
        for name in order:
            t0 = time.perf_counter()
            draft = HFLM(load_model(cfg,name,args.tiny,args.artifacts))
            load_s = time.perf_counter()-t0
            t0 = now()
            for p in prompts[:b["warmup_prompts"]]:
                gen.manual_seed(cfg["seed"])
                target_only(target,ids[p["id"]],cap,eos,gen,True)
                speculative(target,draft,ids[p["id"]],max(b["gammas"]),cap,eos,gen,True)
            blocks.append({"rep":rep,"draft":name,"draft_load_seconds":load_s,"warmup_ms":now()-t0,"resident_gb":allocated_gb(),
                           "target_resident_gb":target_resident,"target_load_seconds":target_load})
            for gamma in b["gammas"] if rep == 0 else b["gammas"][::-1]:
                for kind,group,ignore in (("eos",prompts,False),("fixed",fixed,True)):
                    for p in group:
                        # alternating which of the two runs goes first over prompts and repetitions
                        first = (p["index"]+rep)%2 == 0
                        row,toks = pair(cfg,target,draft,p,ids[p["id"]],gamma,cap if kind == "eos" else
                                        (cap if args.smoke else b["fixed_tokens"]),rep,gen,ignore,first)
                        rows.append({"kind":kind,"draft":name,**row})
                        tokens.append({"kind":kind,"draft":name,"gamma":gamma,"rep":rep,"prompt_id":p["id"],"split":p["split"],**toks})
                if rep == 0:
                    # repeating the same runs with synchronized timers around the four parts of every round
                    plain = {t["prompt_id"]:t["spec_tokens"] for t in tokens
                             if (t["kind"],t["draft"],t["gamma"],t["rep"]) == ("eos",name,gamma,rep)}
                    base = {r["prompt_id"]:r["spec_decode_ms"] for r in rows
                            if (r["kind"],r["draft"],r["gamma"],r["rep"]) == ("eos",name,gamma,rep)}
                    for p in prompts:
                        gen.manual_seed(seeds(cfg,p,rep)[1])
                        r = speculative(target,draft,ids[p["id"]],gamma,cap,eos,gen,timers=True)
                        timed_rows.append({"draft":name,"gamma":gamma,"prompt_id":p["id"],"split":p["split"],"tokens":len(r["tokens"]),
                                           "same_tokens":r["tokens"] == plain[p["id"]],"untimed_decode_ms":base[p["id"]],
                                           **{k:r[k] for k in ("rounds","proposed","accepted","draft_passes","draft_ms","verify_ms",
                                                               "accept_ms","rollback_ms","host_ms","decode_ms","target_prefill_ms",
                                                               "draft_prefill_ms")}})
                print("benchmark",rep,name,gamma,len(rows),"rows",flush=True)
            pd.DataFrame(rows).to_csv(folder/"runs.csv",index=False)
            pd.DataFrame(timed_rows).to_csv(folder/"rounds_timed.csv",index=False)
            write_gz(folder/"tokens.jsonl.gz",tokens)
            save_json(folder/"blocks.json",blocks)
            del draft
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def filler(cfg,tok,count):
    """A fixed run of ordinary text tokens from the training questions"""
    text = " ".join(p["text"] for p in train_prompts(cfg)["math"][:60])
    return tok(text)["input_ids"][:count]


def isolated_rows(name,lm,tokens,lengths,gammas,warm,reps):
    """Cost of single passes of one model at several context lengths"""
    rows = []
    for n in lengths:
        def prefill():
            """Process the whole context from an empty cache"""
            lm.reset()
            lm.prefill(tokens[:n])
        ms,sd = timed(prefill,warm,reps)
        rows.append({"model":name,"context":n,"pass":"prefill","tokens":n,"ms":ms,"std":sd,"reps":reps})
        prefill()
        for width in sorted({1,2}|{g+1 for g in gammas}):
            def step():
                """Feed new tokens after the context and roll back"""
                lm.feed(tokens[n:n+width])
                lm.crop(n)
            ms,sd = timed(step,warm,reps)
            rows.append({"model":name,"context":n,"pass":"step","tokens":width,"ms":ms,"std":sd,"reps":reps})
    return rows


@torch.inference_mode()
def batch_rows(name,model,tokens,length,batches,gammas,warm,reps):
    """Cost of one decode pass of one model when several sequences share it"""
    rows = []
    for n in batches:
        ids = torch.tensor([tokens[:length]]*n,device=model.device)
        cache = DynamicCache()
        model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
        for width in sorted({1}|{g+1 for g in gammas}):
            new = torch.tensor([tokens[length:length+width]]*n,device=model.device)

            def step():
                """One pass for all sequences followed by a rollback"""
                model(input_ids=new,past_key_values=cache,use_cache=True)
                cache.crop(length)
            ms,sd = timed(step,warm,reps)
            rows.append({"model":name,"batch":n,"context":length,"tokens":width,"ms":ms,"std":sd,"reps":reps})
    return rows


def isolated_stage(cfg,args,out):
    """Isolated pass costs the batch size probe target only memory and the unmerged adapter"""
    b = cfg["benchmark"]
    tok = load_tokenizer(cfg)
    lengths = cfg["smoke"]["isolated_lengths"] if args.smoke else b["isolated_lengths"]
    reps = cfg["smoke"]["isolated_reps"] if args.smoke else b["isolated_reps"]
    warm,probe = b["isolated_warmup"],b["batch_probe"]
    length = lengths[0] if args.smoke else probe["length"]
    tokens = filler(cfg,tok,max(lengths)+16)
    folder = out/"benchmark"
    folder.mkdir(parents=True,exist_ok=True)
    prompts = test_prompts(cfg,cfg["smoke"]["test_prompts_per_split"] if args.smoke else None)
    fixed = fixed_subset(prompts,cfg["smoke"]["fixed_prompts"] if args.smoke else b["fixed_prompts"],cfg["seed"])
    cap = cfg["smoke"]["bench_cap"] if args.smoke else b["cap"]
    eos,gen = set(cfg["models"]["eos"]),torch.Generator(device=device())
    rows,batch,memory = [],[],{}
    target = HFLM(load_model(cfg,"target",args.tiny))
    memory["target_resident_gb"] = allocated_gb()
    reset_peak()
    for p in fixed:
        gen.manual_seed(cfg["seed"])
        target_only(target,chat_ids(tok,p["text"]),cap,eos,gen,True)
    memory["target_only_peak_gb"] = peak_gb()
    rows += isolated_rows("target",target,tokens,lengths,b["gammas"],warm,reps)
    batch += batch_rows("target",target.model,tokens,length,probe["batches"],b["gammas"],warm,reps)
    models = {}
    for name in ("original",b["unmerged"]):
        models[name] = HFLM(load_model(cfg,name,args.tiny,args.artifacts))
        rows += isolated_rows(name,models[name],tokens,lengths,b["gammas"],warm,reps)
        batch += batch_rows(name,models[name].model,tokens,length,probe["batches"],b["gammas"],warm,reps)
    # wrapping a fresh draft with the same adapter and running it through the training precision rule
    wrapped = load_model(cfg,"draft",args.tiny)
    add_lora(wrapped,cfg,0)
    load_factors(wrapped,torch.load(artifacts(cfg,args.artifacts)/"adapters"/f"{b['unmerged']}.pt"))
    models["unmerged"] = HFLM(wrapped,amp=True)
    rows += isolated_rows("unmerged",models["unmerged"],tokens,lengths,b["gammas"],warm,reps)
    runs = []
    for p in fixed:
        ids = chat_ids(tok,p["text"])
        for gamma in b["gammas"]:
            names = (b["unmerged"],"unmerged") if p["index"]%2 == 0 else ("unmerged",b["unmerged"])
            for name in names:
                gen.manual_seed(seeds(cfg,p,0)[1])
                r = speculative(target,models[name],ids,gamma,cap,eos,gen,True)
                runs.append({"draft":"unmerged" if name == "unmerged" else "merged","gamma":gamma,"prompt_id":p["id"],
                             "tokens":len(r["tokens"]),"decode_ms":r["decode_ms"],"draft_prefill_ms":r["draft_prefill_ms"],
                             "target_prefill_ms":r["target_prefill_ms"],"accepted":r["accepted"],"examined":r["examined"],
                             "rounds":r["rounds"],"draft_passes":r["draft_passes"]})
    pd.DataFrame(rows).to_csv(folder/"isolated.csv",index=False)
    pd.DataFrame(batch).to_csv(folder/"batch_probe.csv",index=False)
    pd.DataFrame(runs).to_csv(folder/"unmerged_runs.csv",index=False)
    save_json(folder/"memory.json",memory)
    print("isolated costs written",flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--stage",choices=["main","isolated"],required=True)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--artifacts",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    out = repo_path(args.results or cfg["paths"]["results_dir"])
    {"main":main_stage,"isolated":isolated_stage}[args.stage](cfg,args,out)


if __name__ == "__main__":
    main()
