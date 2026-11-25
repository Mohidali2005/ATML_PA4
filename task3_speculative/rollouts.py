import argparse
import random
import time

import torch

from common.utils import load_yaml,repo_path,save_json,set_seed
from task3_speculative.data import chat_ids,read_gz,test_prompts,train_prompts,write_gz
from task3_speculative.models import STUDENTS,device,load_model,load_tokenizer
from task3_speculative.sampler import HFLM,speculative,target_only

SHARDS = 2


def saved_positions(cfg,pid,bank,length):
    """Response positions of one rollout that enter the prefix bank chosen by the fixed per bin rule"""
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{bank}")
    out = []
    for lo,hi,n in cfg["banks"]["bins"]:
        candidates = list(range(lo,min(hi,length)+1))
        out.extend(sorted(rng.sample(candidates,min(n,len(candidates)))))
    return out


def prefix_records(cfg,bank,rows):
    """Prefix bank entries of the rollouts in the released record format"""
    out = []
    for r in rows:
        for pos in saved_positions(cfg,r["prompt_id"],bank,len(r["tokens"])):
            out.append({"prompt_id":r["prompt_id"],"domain":r["domain"],"split":r["split"],"bank":bank,"position":pos,
                        "response_prefix_token_ids":r["tokens"][:pos-1],"source_length":len(r["tokens"]),
                        "source_generation_seed":r["seed"]})
    return out


def pool(cfg,args,out):
    """One frozen teacher response for every training prompt of this shard"""
    tok = load_tokenizer(cfg)
    t0 = time.perf_counter()
    lm = HFLM(load_model(cfg,"target",args.tiny))
    load_s = time.perf_counter()-t0
    eos,gen = set(cfg["models"]["eos"]),torch.Generator(device=device())
    caps = cfg["smoke"]["cap"] if args.smoke else cfg["distill"]["cap"]
    groups = train_prompts(cfg,cfg["smoke"]["train_prompts"] if args.smoke else None)
    rows,cost = [],{}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for category,prompts in groups.items():
        t0 = time.perf_counter()
        n = 0
        for i,p in list(enumerate(prompts))[args.shard::SHARDS]:
            seed = cfg["seed"]+50021*i+(7 if category == "short" else 0)
            gen.manual_seed(seed)
            r = target_only(lm,chat_ids(tok,p["text"]),caps[category],eos,gen)
            rows.append({"id":p["id"],"category":category,"seed":seed,"tokens":r["tokens"],
                         "termination":"eos" if r["eos"] else "cap","ms":r["target_prefill_ms"]+r["decode_ms"]})
            n += len(r["tokens"])
        cost[category] = {"responses":len(prompts[args.shard::SHARDS]),"tokens":n,"seconds":time.perf_counter()-t0}
        print("pool",category,cost[category],flush=True)
    write_gz(out/"rollouts"/f"pool_shard{args.shard}.jsonl.gz",rows)
    save_json(out/"distill"/f"pool_shard{args.shard}.json",
              {"load_seconds":load_s,"peak_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None,**cost})


def read_pool(out):
    """Teacher responses of both shards by prompt id"""
    return {r["id"]:r for s in range(SHARDS) for r in read_gz(out/"rollouts"/f"pool_shard{s}.jsonl.gz")}


def bank(cfg,args,out,name,sample):
    """Generate one rollout per test prompt and save it with its prefix records"""
    tok = load_tokenizer(cfg)
    b = cfg["banks"]
    cap = cfg["smoke"]["bank_cap"] if args.smoke else b["cap"]
    gen = torch.Generator(device=device())
    rows = []
    t0 = time.perf_counter()
    for p in test_prompts(cfg,cfg["smoke"]["test_prompts_per_split"] if args.smoke else None):
        seed = cfg["seed"]+b["seed_step"]*p["index"]+b["seed_offset"][name]
        gen.manual_seed(seed)
        r = sample(chat_ids(tok,p["text"]),cap,gen)
        row = {"prompt_id":p["id"],"domain":p["domain"],"split":p["split"],"bank":name,"seed":seed,"tokens":r["tokens"],
               "termination":"eos" if r["eos"] else "cap"}
        if "rounds" in r:
            row.update({k:r[k] for k in ("rounds","proposed","accepted","examined","target_passes","draft_passes")})
        rows.append(row)
    write_gz(out/"rollouts"/f"{name}.jsonl.gz",rows)
    records = prefix_records(cfg,name,rows)
    write_gz(out/"banks"/f"{name}.jsonl.gz",records)
    print("bank",name,len(rows),"rollouts",len(records),"prefixes",f"{time.perf_counter()-t0:.0f} s",flush=True)


def shared_banks(cfg,args,out):
    """Target original draft and exact speculative prefix banks"""
    eos = set(cfg["models"]["eos"])
    target,draft = HFLM(load_model(cfg,"target",args.tiny)),HFLM(load_model(cfg,"draft",args.tiny))
    bank(cfg,args,out,"teacher",lambda ids,cap,gen:target_only(target,ids,cap,eos,gen))
    bank(cfg,args,out,"original_draft",lambda ids,cap,gen:target_only(draft,ids,cap,eos,gen))
    bank(cfg,args,out,"exact_spec_original_gamma4",
         lambda ids,cap,gen:speculative(target,draft,ids,cfg["banks"]["gamma"],cap,eos,gen))


def own_banks(cfg,args,out):
    """Prefix banks sampled from the policy of every merged student"""
    eos = set(cfg["models"]["eos"])
    for name in STUDENTS:
        lm = HFLM(load_model(cfg,name,args.tiny,args.artifacts))
        bank(cfg,args,out,name,lambda ids,cap,gen:target_only(lm,ids,cap,eos,gen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--stage",choices=["pool","shared","own"],required=True)
    ap.add_argument("--shard",type=int,default=0)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--artifacts",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    out = repo_path(args.results or cfg["paths"]["results_dir"])
    {"pool":pool,"shared":shared_banks,"own":own_banks}[args.stage](cfg,args,out)


if __name__ == "__main__":
    main()
