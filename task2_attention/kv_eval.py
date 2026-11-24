import argparse
import copy
import gzip
import json
import time

import pandas as pd
import torch
import torch.nn.functional as F

from common.utils import load_yaml,repo_path,save_json,set_seed
from task2_attention.kv_cache import (Requantizer,as_cache,build_prompt,eviction_count,extract,formula_bytes,fp16_bytes,
                                      gather,hybrid_cache,hybrid_plan,int_cache,load,nbytes,pooled,prefill,
                                      question_scores,restore,sink_recent,step,store,tensors,top_positions)
from task2_attention.timing import sync,timed


def cases(cfg,limit=None):
    """The frozen retrieval prompts in file order"""
    with gzip.open(repo_path(cfg["paths"]["kv_cases"]),"rt",encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit]


def shape_of(model):
    """Cache shape numbers of the model"""
    c = model.config
    return {"layers":c.num_hidden_layers,"kv_heads":c.num_key_value_heads,"head_dim":c.hidden_size//c.num_attention_heads}


def clock():
    """Synchronized wall clock in milliseconds"""
    sync()
    return 1000*time.perf_counter()


def kl(p,q):
    """Kullback leibler divergence of the second distribution from the first for each row of logits"""
    lp,lq = F.log_softmax(p.float(),-1),F.log_softmax(q.float(),-1)
    return (lp.exp()*(lp-lq)).sum(-1)


def respond(model,tok,kv,suffix,start,gold,max_new,requant=None):
    """Prefill the question through an edited cache then score the gold answer and decode greedily"""
    stops = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id,list)
                else [model.generation_config.eos_token_id])
    cache = as_cache(kv)
    t0 = clock()
    post = step(model,cache,suffix,start)[-1]
    if requant:
        requant(cache)
    t1 = clock()
    pos = start+suffix.shape[1]
    # teacher forcing the gold code on a branch so the greedy answer starts from the same state
    branch,bq = as_cache(tensors(cache)),copy.copy(requant)
    forced = [post]
    for t,g in enumerate(gold[:-1]):
        forced.append(step(model,branch,torch.tensor([[g]],device=suffix.device),pos+t)[-1])
        if bq:
            bq(branch)
    forced = torch.stack(forced)
    gen,calls = [],0
    nxt = int(post.argmax())
    t2 = clock()
    for i in range(max_new):
        gen.append(nxt)
        if nxt in stops:
            break
        logits = step(model,cache,torch.tensor([[nxt]],device=suffix.device),pos+i)
        calls += 1
        if requant:
            requant(cache)
        nxt = int(logits[-1].argmax())
    t3 = clock()
    nll = F.cross_entropy(forced,torch.tensor(gold,device=forced.device)).item()
    text = tok.decode(gen,skip_special_tokens=True)
    return {"post":post,"forced":forced,"tokens":gen,"text":text,"nll":nll,"truncated":gen[-1] not in stops,
            "suffix_ms":t1-t0,"decode_ms_per_token":(t3-t2)/calls if calls else None,"generated":len(gen),
            "finite":bool(torch.isfinite(forced).all()),"requantized_groups":requant.added if requant else 0}


def recon_error(kv,rec):
    """Relative reconstruction error of keys and of values over all layers"""
    out = []
    for j in (0,1):
        num = sum(((a[j].float()-b[j].float())**2).sum() for a,b in zip(kv,rec))
        den = sum((a[j].float()**2).sum() for a in kv)
        out.append(float((num/den).sqrt()))
    return out


def retained(units_keep,positions):
    """Share of the given positions kept per layer and head averaged over all of them"""
    if not positions:
        return None
    return float(units_keep[...,positions].float().mean())


def code_positions(p):
    """Token positions of the gold code digits inside the evidence record"""
    ids,gold = p["ids"],p["gold"]
    for a in p["evidence"]:
        if ids[a:a+len(gold)] == gold:
            return list(range(a,a+len(gold)))
    return []


def evaluate_case(model,tok,cfg,case,shape):
    """Run the full reference and the five compressed caches on one prompt"""
    kv_cfg = cfg["kv"]
    p = build_prompt(tok,case,kv_cfg["question_marker"])
    ids = torch.tensor([p["ids"]],device=model.device)
    ctx,suffix = ids[:,:p["split"]],ids[:,p["split"]:]
    c = ctx.shape[1]
    sink,recent,group = kv_cfg["sink"],kv_cfg["recent"],kv_cfg["group"]
    units = shape["layers"]*shape["kv_heads"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = clock()
    kv = prefill(model,ctx)
    prefill_ms = clock()-t0
    prefill_peak = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None
    evidence = [i for i in p["evidence"] if i < c]
    digits = [i for i in code_positions(p) if i < c]
    full = fp16_bytes(shape,c)
    budget = formula_bytes(shape,c,kv_cfg["bits"],group)
    t0 = clock()
    scores = pooled(question_scores(model,kv,suffix,c),kv_cfg["pool"])
    score_ms = clock()-t0
    meta = {"id":case["id"],"target_tokens":case["target_tokens"],"position":case["position"],
            "distractors":case["distractors"],"code":case["code"],"prompt_tokens":len(p["ids"]),"context_tokens":c,
            "suffix_tokens":suffix.shape[1],"evidence_tokens":len(evidence),"budget_bytes":budget,"fp16_bytes":full,
            "prefill_ms":prefill_ms,"prefill_peak_gb":prefill_peak,"boundary_exact":p["boundary_exact"]}
    rows,ref,keeps = [],None,{}
    for method in kv_cfg["methods"]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = clock()
        requant,keep,err,extra = None,None,(None,None),{}
        if method == "fp16":
            edited,stored = kv,full
        elif method == "sink_recent":
            n = eviction_count(shape,c,budget,False,sink,recent,kv_cfg["index_bytes"])
            idx = sink_recent(c,n,sink).expand(shape["layers"],shape["kv_heads"],-1)
            edited,stored = gather(kv,idx),units*4*shape["head_dim"]*n
            extra = {"fp16_positions":n}
        elif method == "question_aware":
            n = eviction_count(shape,c,budget,True,sink,recent,kv_cfg["index_bytes"])
            idx = top_positions(scores,n-sink-recent,sink,recent)
            edited = gather(kv,idx)
            # the sorted rows hold the sink first and the recent positions last so the middle is the selected set
            stored = units*4*shape["head_dim"]*n+nbytes([idx[...,sink:n-recent].to(torch.int16)])
            extra = {"fp16_positions":n}
        elif method.startswith("int4"):
            axis = method.split("_")[1]
            edited,stored,done = int_cache(kv,kv_cfg["bits"],group,axis)
            requant = Requantizer(done,kv_cfg["bits"],group,axis)
            err = recon_error(kv,edited)
            extra = {"quantized_tokens":done,"formula_bytes":formula_bytes(shape,c,kv_cfg["bits"],group)}
        else:
            plan = hybrid_plan(shape,c,budget,sink,recent,kv_cfg["hybrid_bits"],group,kv_cfg["index_bytes"])
            edited,stored,keep = hybrid_cache(kv,scores,plan,sink,recent,kv_cfg["hybrid_bits"],group)
            err = recon_error(kv,edited)
            extra = {"fp16_positions":sink+recent+plan["selected"]+plan["fp16_rest"],"selected":plan["selected"],
                     "int2_groups":plan["groups"],"int2_positions":plan["groups"]*group,"plan_bytes":plan["bytes"]}
        encode_ms = clock()-t0
        if method in ("sink_recent","question_aware"):
            keep = torch.zeros(shape["layers"],shape["kv_heads"],c,dtype=torch.bool)
            keep.scatter_(2,idx.cpu(),True)
        r = respond(model,tok,edited,suffix,c,p["gold"],kv_cfg["max_new_tokens"],requant)
        if method == "fp16":
            ref = r
        got = extract(r["text"])
        row = {**meta,"method":method,"answer":got,"correct":got == case["code"],"text":r["text"],"truncated":r["truncated"],
               "generated":r["generated"],"nll":r["nll"],"kl_post":float(kl(ref["post"][None],r["post"][None])[0]),
               "kl_answer":float(kl(ref["forced"],r["forced"]).mean()),"finite":r["finite"],"stored_bytes":stored,
               "fraction_of_fp16":stored/full,"within_budget":stored <= budget or method == "fp16",
               "evidence_retained":retained(keep,evidence) if keep is not None else None,
               "code_retained":retained(keep,digits) if keep is not None else None,
               "k_rel_error":err[0],"v_rel_error":err[1],"encode_ms":encode_ms,
               "score_ms":score_ms if method in ("question_aware","hybrid") else 0.0,
               "suffix_ms":r["suffix_ms"],"decode_ms_per_token":r["decode_ms_per_token"],
               "peak_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None,
               "requantized_groups":r["requantized_groups"],**extra}
        rows.append(row)
        if keep is not None:
            keeps[method] = keep
    return rows,keeps,p


@torch.inference_mode()
def validate(model,tok,cfg,case,shape):
    """Check split prefill against one full forward and an untouched edited cache against ordinary generation"""
    kv_cfg = cfg["kv"]
    p = build_prompt(tok,case,kv_cfg["question_marker"])
    ids = torch.tensor([p["ids"]],device=model.device)
    c = p["split"]
    ctx,suffix = ids[:,:c],ids[:,c:]
    n = ids.shape[1]
    full = model(input_ids=ids,logits_to_keep=n-c).logits[0].float()
    kv = prefill(model,ctx)
    split = step(model,as_cache(kv),suffix,c)
    idx = torch.arange(c).expand(shape["layers"],shape["kv_heads"],-1)
    kept = gather(kv,idx)
    ours = respond(model,tok,kept,suffix,c,p["gold"],kv_cfg["max_new_tokens"])
    plain = respond(model,tok,kv,suffix,c,p["gold"],kv_cfg["max_new_tokens"])
    # turning off the sampling and repetition settings of the shipped generation config for plain greedy decoding
    gen = model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=kv_cfg["max_new_tokens"],
                         do_sample=False,temperature=None,top_p=None,top_k=None,repetition_penalty=1.0)[0,n:].tolist()
    # dropping a shared random third of the free context and comparing with a masked full forward
    g = torch.Generator().manual_seed(cfg["seed"])
    free = torch.arange(kv_cfg["sink"],c-kv_cfg["recent"])
    drop = free[torch.randperm(len(free),generator=g)[:len(free)//3]]
    keep = torch.ones(c,dtype=torch.bool)
    keep[drop] = False
    sub = torch.nonzero(keep).flatten().expand(shape["layers"],shape["kv_heads"],-1)
    evicted = step(model,as_cache(gather(kv,sub)),suffix,c)
    mask = torch.ones(n,n,dtype=torch.bool,device=ids.device).tril()
    mask[c:,drop.to(ids.device)] = False
    masked = model(input_ids=ids,attention_mask=mask[None,None],logits_to_keep=n-c).logits[0].float()
    stops = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id,list)
                else [model.generation_config.eos_token_id])
    trimmed = gen[:next((i+1 for i,t in enumerate(gen) if t in stops),len(gen))]
    return {"id":case["id"],"context_tokens":c,"suffix_tokens":suffix.shape[1],
            "split_vs_full_max_abs_logit":float((split-full).abs().max()),
            "split_vs_full_same_argmax":bool((split.argmax(-1) == full.argmax(-1)).all()),
            "full_retention_tokens_equal_generate":ours["tokens"] == trimmed,
            "full_retention_max_abs_logit_vs_plain_cache":float((ours["forced"]-plain["forced"]).abs().max()),
            "evicted_vs_masked_full_max_abs_logit":float((evicted-masked).abs().max()),
            "evicted_vs_masked_same_argmax":bool((evicted.argmax(-1) == masked.argmax(-1)).all()),
            "dropped_positions":len(drop),"all_finite":bool(torch.isfinite(full).all() and torch.isfinite(split).all()),
            "generate_text":tok.decode(trimmed,skip_special_tokens=True),"edited_text":ours["text"]}


@torch.inference_mode()
def bench(model,cfg,case_rows,tok,shape,smoke):
    """Packing unpacking and decode step time of quantized caches against fp16 at fixed shapes"""
    b = cfg["kv"]["bench"]
    warm,reps = (1,2) if smoke else (b["warmup"],b["reps"])
    group = cfg["kv"]["group"]
    layouts = [("int4_token",cfg["kv"]["bits"],"token"),("int4_channel",cfg["kv"]["bits"],"channel"),
               ("int2_channel",cfg["kv"]["hybrid_bits"],"channel")]
    rows = []
    for length in b["lengths"]:
        case = next(x for x in case_rows if x["target_tokens"] == length)
        p = build_prompt(tok,case,cfg["kv"]["question_marker"])
        c = p["split"]//group*group
        base = prefill(model,torch.tensor([p["ids"][:c]],device=model.device))
        for batch in b["batches"]:
            kv = [(k.expand(batch,-1,-1,-1).contiguous(),v.expand(batch,-1,-1,-1).contiguous()) for k,v in base]
            nxt = torch.full((batch,1),p["ids"][c],device=model.device)
            pos = torch.full((batch,1),c,device=model.device)

            def decode_step(cache_kv):
                """One new token through a cache built from the given tensors"""
                cache = as_cache(cache_kv)
                return model(input_ids=nxt,past_key_values=cache,use_cache=True,position_ids=pos,
                             cache_position=torch.tensor([c],device=model.device),
                             attention_mask=torch.ones(batch,c+1,dtype=torch.long,device=model.device))
            ms,sd = timed(lambda:decode_step(kv),warm,reps)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            decode_step(kv)
            rows.append({"layout":"fp16","context":c,"batch":batch,"serialized_bytes":nbytes([t for pair in kv for t in pair]),
                         "decode_step_ms":ms,"decode_step_std":sd,"reps":reps,
                         "peak_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None})
            for name,bits,axis in layouts:
                def encode():
                    """Quantize and pack every layer"""
                    return [(store(k,bits,group,axis)[1],store(v,bits,group,"token")[1]) for k,v in kv]
                enc_ms,enc_sd = timed(encode,warm,reps)
                parts = encode()
                size = sum(nbytes(list(pk.values())+list(pv.values())) for pk,pv in parts)

                def decode():
                    """Unpack and dequantize every layer back to fp16"""
                    return [(restore(pk,bits,group,axis,k.shape),restore(pv,bits,group,"token",v.shape))
                            for (pk,pv),(k,v) in zip(parts,kv)]
                dec_ms,dec_sd = timed(decode,warm,reps)
                rebuilt = decode()
                err = max(float((r[0].float()-store(k,bits,group,axis)[0].float()).abs().max()) for r,(k,v) in zip(rebuilt,kv))
                step_ms,step_sd = timed(lambda:decode_step(decode()),warm,reps)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                decode_step(decode())
                rows.append({"layout":name,"context":c,"batch":batch,"serialized_bytes":size,"encode_ms":enc_ms,
                             "encode_std":enc_sd,"unpack_ms":dec_ms,"unpack_std":dec_sd,"decode_step_ms":step_ms,
                             "decode_step_std":step_sd,"reps":reps,"restore_matches_store":err == 0.0,
                             "peak_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None})
            del kv
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--results",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    out = repo_path(args.results or cfg["paths"]["results_dir"])/"kv"
    out.mkdir(parents=True,exist_ok=True)
    tok,model,rev = load(cfg,args.tiny)
    shape = shape_of(model)
    rows_all = cases(cfg)
    chosen = rows_all[:cfg["smoke"]["kv_cases"]] if args.smoke else rows_all
    checks = [validate(model,tok,cfg,next(x for x in rows_all if x["target_tokens"] == t),shape)
              for t in sorted({x["target_tokens"] for x in rows_all})[:1 if args.smoke else None]]
    save_json(out/"validation.json",{"model_revision":rev,"cache_shape":shape,"checks":checks})
    print("cache validation written")
    q = cfg["kv"]["qualitative"]
    rows,qual = [],[]
    for case in chosen:
        r,keeps,p = evaluate_case(model,tok,cfg,case,shape)
        rows += r
        pick = case["target_tokens"] == q["target_tokens"] and case["distractors"] == q["distractors"] and \
            case["id"].endswith(f"-{q['case']:02d}")
        if pick or (args.smoke and not qual):
            ev = [i for i in p["evidence"] if i < p["split"]]
            lo,hi = max(0,ev[0]-48),min(p["split"],ev[-1]+48)
            qual.append({"id":case["id"],"position":case["position"],"evidence_tokens":ev,"window":[lo,hi],
                         "window_text":[tok.decode([t]) for t in p["ids"][lo:hi]],
                         "generations":{x["method"]:x["text"] for x in r},
                         "masks":{m:["".join("1" if b else "0" for b in unit[lo:hi].tolist())
                                     for unit in keep.flatten(0,1)] for m,keep in keeps.items()},
                         "kept_share":{m:keep.float().mean((0,1)).tolist() for m,keep in keeps.items()}})
    pd.DataFrame(rows).to_csv(out/"per_prompt.csv",index=False)
    save_json(out/"qualitative.json",qual)
    print("kv evaluation written for",len(chosen),"prompts")
    bench(model,cfg,rows_all,tok,shape,args.smoke).to_csv(out/"bench.csv",index=False)
    print("kv benchmark written")


if __name__ == "__main__":
    main()
