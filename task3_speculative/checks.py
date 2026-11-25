import argparse
import itertools
from collections import Counter
from multiprocessing import Pool

import pandas as pd
import torch
from scipy.stats import chisquare

from common.utils import load_yaml,repo_path,save_json,set_seed
from task3_speculative.data import chat_ids,test_prompts
from task3_speculative.models import load_model,load_tokenizer
from task3_speculative.sampler import HFLM,MarkovLM,sample_batch,speculative,target_only


def joint(table,length,eos=None):
    """Exact probability of every continuation from symbol zero that stops at the cap or at the end symbol"""
    out = {}
    for n in range(1,length+1):
        for s in itertools.product(range(3),repeat=n):
            stops = [i for i,x in enumerate(s) if x == eos]
            if (n < length and (not stops or stops[0] != n-1)) or (n == length and stops and stops[0] != n-1):
                continue
            prob,prev = 1.0,0
            for x in s:
                prob *= table[prev][x]
                prev = x
            out[s] = prob
    return out


def empirical(name,table,draft,gamma,length,eos,samples,seed,wrong=False):
    """Sample continuations with the speculative sampler or with the target alone and count them"""
    gen = torch.Generator().manual_seed(seed)
    t,d = MarkovLM(table),MarkovLM(draft)
    stops = {eos} if eos is not None else set()
    counts,routes,full,rounds,prop,acc = Counter(),Counter(),0,0,0,0
    for _ in range(samples):
        if name == "target":
            r = target_only(t,[0],length,stops,gen)
        else:
            r = speculative(t,d,[0],gamma,length,stops,gen,wrong=wrong)
            routes[r["eos_route"]] += 1
            rounds += r["rounds"]
            prop += r["proposed"]
            acc += r["accepted"]
            full += sum(a == k and k > 0 for a,k in zip(r["accepted_per_round"],r["proposed_per_round"]))
            # the caches must hold the returned continuation without its last symbol
            assert t.tokens == ([0]+r["tokens"])[:-1] and d.tokens == t.tokens[:len(d.tokens)]
        counts[tuple(r["tokens"])] += 1
    info = {"rounds":rounds,"proposed":prop,"accepted":acc,"fully_accepted_rounds":full,
            "eos_routes":{str(k):v for k,v in routes.items()}}
    return counts,info


def run_job(job):
    """Draw all samples of one toy condition in a worker process"""
    # one thread is faster for the many tiny tensors of the chains
    torch.set_num_threads(1)
    kind,table,draft,gamma,length,eos,samples,seed,wrong = job
    return empirical(kind,table,draft,gamma,length,eos,samples,seed,wrong)


def toy_tests(cfg,out,samples,workers):
    """Goodness of fit of the joint sequence distribution on the small chains with a wrong sampler as control"""
    c = cfg["checks"]
    toy,eos_toy,long_toy = c["toy"],c["toy_eos"],c["toy_long"]
    conditions = []
    for g in c["gammas"]:
        conditions.append(("length3",toy["target"],toy["draft"],g,toy["length"],None,1))
        conditions.append(("eos4",toy["target"],toy["draft"],g,eos_toy["length"],eos_toy["eos"],1))
    # the long chain is repeated over several seeds because it is the only one with full rounds of four proposals
    conditions.append(("length6",long_toy["target"],long_toy["draft"],long_toy["gamma"],long_toy["length"],None,long_toy["repeats"]))
    labels,jobs,references = [],[],set()
    for name,tab,dra,gamma,length,eos,repeats in conditions:
        variants = [("speculative",gamma,False,repeats),("wrong_residual",gamma,True,1)]
        if name not in references:
            variants.append(("target",0,False,repeats))
            references.add(name)
        for kind,g,wrong,n in variants:
            for rep in range(n):
                seed = (cfg["seed"]+1000*gamma+{"length3":1,"eos4":2,"length6":3}[name]+(50 if wrong else 0)+
                        (100 if kind == "target" else 0)+100000*rep)
                labels.append((name,kind,g,rep,n,joint(tab,length,eos)))
                jobs.append((kind,tab,dra,gamma,length,eos,samples,seed,wrong))
    with Pool(workers) as pool:
        answers = list(pool.imap(run_job,jobs))
    results,tables,pooled = [],{},{}

    def record(name,kind,g,rep,exact,observed,total,info):
        """Add one goodness of fit test"""
        expected = [total*exact[k] for k in sorted(exact)]
        stat,p = chisquare(observed,expected)
        results.append({"toy":name,"sampler":kind,"gamma":g,"repetition":rep,"samples":total,"categories":len(observed),
                        "min_expected":min(expected),"chi2":float(stat),"dof":len(observed)-1,"p_value":float(p),
                        "rejected":bool(p < c["alpha"]),
                        "max_abs_prob_error":max(abs(o-e)/total for o,e in zip(observed,expected)),**info})
        print(name,kind,g,rep,f"chi2 {stat:.1f} p {p:.3g}",flush=True)

    for (name,kind,g,rep,n,exact),(counts,info) in zip(labels,answers):
        keys = sorted(exact)
        assert abs(sum(exact.values())-1) < 1e-9
        observed = [counts[k] for k in keys]
        assert sum(observed) == samples
        record(name,kind,g,rep,exact,observed,samples,info)
        rows = tables.setdefault(name,{"sequence":["".join(map(str,k)) for k in keys],"target_probability":[exact[k] for k in keys]})
        rows[f"{kind}_gamma{g}"+(f"_rep{rep}" if n > 1 else "")] = observed
        if n > 1:
            old = pooled.get((name,kind,g),(exact,[0]*len(keys),0))
            pooled[name,kind,g] = (exact,[x+y for x,y in zip(old[1],observed)],old[2]+samples)
    for (name,kind,g),(exact,observed,total) in pooled.items():
        record(name,kind,g,"pooled",exact,observed,total,{})
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(out/f"toy_joint_{name}.csv",index=False)
    save_json(out/"toy_tests.json",results)
    for r in results:
        assert r["min_expected"] >= c["min_expected"] or samples < c["samples"]
    return results


def max_diff(a,b):
    """Largest absolute difference and argmax agreement of two sets of distributions"""
    return {"max_abs_prob":float((a-b).abs().max()),"max_total_variation":float(0.5*(a-b).abs().sum(-1).max()),
            "same_argmax":float((a.argmax(-1) == b.argmax(-1)).float().mean())}


@torch.inference_mode()
def full_probs(model,ids,keep):
    """Distributions after the last tokens of one full forward pass without a cache"""
    out = model(input_ids=torch.tensor([ids],device=model.device),logits_to_keep=keep)
    return torch.softmax(out.logits[0].float(),-1)


@torch.inference_mode()
def real_checks(cfg,target,draft,tok,prompts,cap):
    """Split prefill rollback padding stopping and length checks on the pretrained models"""
    eos,gen = set(cfg["models"]["eos"]),torch.Generator(device=target.device)
    rows = []
    for p in prompts:
        ids = chat_ids(tok,p["text"])
        row = {"id":p["id"],"split":p["split"],"prompt_tokens":len(ids)}
        for name,model in (("target",target),("draft",draft)):
            lm = HFLM(model)
            gen.manual_seed(cfg["seed"]+p["index"])
            cont = target_only(lm,ids,8,set(),gen)["tokens"]
            full = full_probs(model,ids+cont,len(cont)+1)
            # feeding the same tokens once in a block and once one by one after a split prefill
            lm.reset()
            lm.prefill(ids[:-1])
            row[f"{name}_split_block"] = max_diff(lm.feed([ids[-1]]+cont),full)
            lm.reset()
            lm.prefill(ids[:-1])
            single = torch.stack([lm.feed([t])[-1] for t in [ids[-1]]+cont])
            row[f"{name}_split_single"] = max_diff(single,full)
            # overwriting the cache with unrelated tokens then rolling back before feeding the real ones
            lm.reset()
            lm.prefill(ids[:-1])
            lm.feed([ids[-1]]+[t+1 for t in cont])
            lm.crop(len(ids)-1)
            row[f"{name}_rollback_length_ok"] = lm.length == len(ids)-1
            again = torch.stack([lm.feed([t])[-1] for t in [ids[-1]]+cont])
            row[f"{name}_rollback_max_abs_prob"] = float((again-single).abs().max())
            # skipping the rollback must be visible or the check above would have no power
            lm.reset()
            lm.prefill(ids[:-1])
            lm.feed([ids[-1]]+[t+1 for t in cont])
            stale = torch.stack([lm.feed([t])[-1] for t in [ids[-1]]+cont])
            row[f"{name}_no_rollback_max_abs_prob"] = float((stale-single).abs().max())
        t,d = HFLM(target),HFLM(draft)
        for gamma in (1,2,4):
            gen.manual_seed(cfg["seed"]+10*gamma+p["index"])
            r = speculative(t,d,ids,gamma,cap,eos,gen)
            seq = ids+r["tokens"]
            # after the run both caches hold the text without its last token and must give the same next distribution
            row[f"gamma{gamma}_cache_lengths_ok"] = t.length == len(seq)-1 and d.length <= len(seq)-1
            row[f"gamma{gamma}_target_after_run"] = max_diff(t.feed([seq[-1]])[-1:],full_probs(target,seq,1))
            row[f"gamma{gamma}_draft_after_run"] = max_diff(d.feed(seq[d.length:])[-1:],full_probs(draft,seq,1))
            row[f"gamma{gamma}_run"] = {k:r[k] for k in ("rounds","proposed","accepted","examined","eos","truncated","eos_route")}
            row[f"gamma{gamma}_token_count_ok"] = len(r["tokens"]) == r["rounds"]+r["accepted"]-(r["eos_route"] == "accepted")
        for limit in (1,2,3):
            gen.manual_seed(cfg["seed"]+p["index"])
            r = speculative(t,d,ids,4,limit,set(),gen)
            row[f"cap{limit}_tokens"] = len(r["tokens"])
            row[f"cap{limit}_proposed"] = r["proposed"]
        # the target drafting for itself accepts almost every proposal up to half precision noise
        gen.manual_seed(cfg["seed"]+p["index"])
        r = speculative(t,HFLM(target),ids,4,cap,eos,gen)
        row["self_draft_accepted_over_examined"] = r["accepted"]/max(1,r["examined"])
        row["self_draft_full_rounds"] = sum(a == k for a,k in zip(r["accepted_per_round"],r["proposed_per_round"]))
        row["self_draft_rounds"] = r["rounds"]
        # treating the first two sampled tokens as end tokens so that a run must stop on each route it meets
        gen.manual_seed(cfg["seed"]+p["index"])
        free = speculative(t,d,ids,4,cap,set(),gen)["tokens"]
        gen.manual_seed(cfg["seed"]+p["index"])
        stop = speculative(t,d,ids,4,cap,{free[min(5,len(free)-1)]},gen)
        first = stop["tokens"].index(stop["tokens"][-1])
        row["forced_eos"] = {"stopped":stop["eos"],"route":stop["eos_route"],
                             "end_token_only_at_end":first == len(stop["tokens"])-1 if stop["eos"] else None}
        rows.append(row)
    # left padded batch against unpadded single prompts
    texts = [chat_ids(tok,p["text"]) for p in prompts]
    width = max(map(len,texts))
    pad = []
    for name,model in (("target",target),("draft",draft)):
        ids = torch.tensor([[cfg["models"]["eos"][1]]*(width-len(x))+x for x in texts],device=model.device)
        mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in texts],device=model.device)
        logits = model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp_min(0),logits_to_keep=1).logits[:,-1]
        single = torch.cat([full_probs(model,x,1) for x in texts])
        pad.append({"model":name,"prompts":len(texts),"pad_width":width,**max_diff(torch.softmax(logits.float(),-1),single)})
    gen.manual_seed(cfg["seed"])
    batch = sample_batch(target,texts,cap,list(eos),gen,cfg["models"]["eos"][1])
    ok = all(len(b["tokens"]) <= cap and (b["eos"] != b["truncated"]) for b in batch)
    return {"prompts":rows,"padding":pad,"batched_sampling_ok":ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--part",choices=["toy","real"],required=True)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--artifacts",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    out = repo_path(args.results or cfg["paths"]["results_dir"])/"checks"
    out.mkdir(parents=True,exist_ok=True)
    if args.part == "toy":
        res = toy_tests(cfg,out,cfg["smoke"]["samples"] if args.smoke else cfg["checks"]["samples"],args.workers)
        print("toy tests written with",sum(r["rejected"] for r in res),"rejections out of",len(res))
        return
    tok = load_tokenizer(cfg)
    target,draft = load_model(cfg,"target",args.tiny),load_model(cfg,"draft",args.tiny)
    prompts = test_prompts(cfg,cfg["smoke"]["test_prompts_per_split"] if args.smoke else cfg["checks"]["real_prompts_per_split"])
    save_json(out/"real_checks.json",real_checks(cfg,target,draft,tok,prompts,8 if args.smoke else cfg["checks"]["real_tokens"]))
    print("real model checks written for",len(prompts),"prompts")


if __name__ == "__main__":
    main()
