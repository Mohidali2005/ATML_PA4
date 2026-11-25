import argparse
import math
import time

import torch
import torch.nn.functional as F

from common.utils import load_yaml,repo_path,save_json,set_seed
from task3_speculative.data import chat_ids,schedule,train_prompts,write_gz
from task3_speculative.lora import add_lora,checksum,export_factors
from task3_speculative.models import artifacts,count_parameters,device,load_model,load_tokenizer
from task3_speculative.rollouts import read_pool
from task3_speculative.sampler import HFLM,now,target_only

CATEGORIES = ("math","short")


class Stream:
    """Responses of one category handed out in pieces of supervised tokens"""

    def __init__(self,category,prompts,order,quota,draw):
        self.category,self.prompts,self.order,self.left,self.draw = category,prompts,order,quota,draw
        self.current,self.count,self.done = None,0,[]

    def take(self,n,update):
        """Return pieces that hold exactly n tokens and draw new responses when the current one is used up"""
        pieces = []
        while n > 0:
            if self.current is None:
                p = self.prompts[self.order[self.count%len(self.order)]]
                self.current = {**self.draw(p,self.count),"id":p["id"],"category":self.category,"presentation":self.count,
                                "scored":0,"drawn_at_update":update,"last_update":update,"carried":0}
                # the teacher only scores the part of the response that fits into the remaining quota
                self.current["limit"] = min(len(self.current["tokens"]),self.left)
                self.count += 1
            c = self.current
            take = min(n,c["limit"]-c["scored"])
            pieces.append((c,c["scored"],c["scored"]+take))
            c["scored"] += take
            c["last_update"] = update
            c["carried"] += take if update > c["drawn_at_update"] else 0
            n -= take
            self.left -= take
            if c["scored"] == c["limit"]:
                self.done.append(c)
                self.current = None
        return pieces


def log_row(c):
    """Record of one presentation for the training log"""
    return {"id":c["id"],"category":c["category"],"presentation":c["presentation"],"rollout_seed":c["seed"],
            "response_token_ids":c["tokens"],"termination":c["termination"],"response_length":len(c["tokens"]),
            "teacher_scored_tokens":c["scored"],"quota_truncated":c["scored"] < len(c["tokens"]),
            "drawn_at_update":c["drawn_at_update"],"last_update":c["last_update"],"tokens_carried_over":c["carried"]}


def train(cfg,args,method,seed,out):
    """Distill one student with one prefix source and one seed under the fixed token quota"""
    d,s = cfg["distill"],cfg["smoke"]
    name = f"{method}_{seed}"
    quota,per,caps = (s["quota"],s["per_update"],s["cap"]) if args.smoke else (d["quota"],d["per_update"],d["cap"])
    eos,dev = set(cfg["models"]["eos"]),device()
    tok = load_tokenizer(cfg)
    set_seed(seed)
    teacher = load_model(cfg,"target",args.tiny)
    student = load_model(cfg,"draft",args.tiny)
    backbone = count_parameters(student)
    params = add_lora(student,cfg,seed)
    init = checksum(student)
    opt = torch.optim.AdamW(params,lr=d["lr"],weight_decay=d["weight_decay"])
    scaler = torch.amp.GradScaler(dev,init_scale=d["init_scale"])
    policy = HFLM(student,amp=True)
    gen = torch.Generator(device=dev)
    clock = {k:0.0 for k in ("rollout","teacher_scoring","forward_backward","optimizer")}
    groups = train_prompts(cfg,s["train_prompts"] if args.smoke else None)
    frozen = read_pool(out) if method == "off" else None

    def draw(p,i):
        """One response for a presentation from the frozen collection or from the current student"""
        if method == "off":
            r = frozen[p["id"]]
            return {"tokens":r["tokens"],"termination":r["termination"],"seed":r["seed"]}
        roll = seed+50021*i+(7 if p["category"] == "short" else 0)
        gen.manual_seed(roll)
        t = now()
        r = target_only(policy,chat_ids(tok,p["text"]),caps[p["category"]],eos,gen)
        clock["rollout"] += now()-t
        return {"tokens":r["tokens"],"termination":"eos" if r["eos"] else "cap","seed":roll}

    streams = {c:Stream(c,groups[c],schedule(groups[c],seed,c),quota[c],draw) for c in CATEGORIES}
    texts = {p["id"]:chat_ids(tok,p["text"]) for g in groups.values() for p in g}
    updates = math.ceil(max(quota[c]/per[c] for c in CATEGORIES))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log,skipped = [],0
    begin = time.perf_counter()
    for u in range(1,updates+1):
        pieces = {c:streams[c].take(min(per[c],streams[c].left),u) for c in CATEGORIES}
        total = sum(b-a for c in CATEGORIES for _,a,b in pieces[c])
        sums,counts = {c:0.0 for c in CATEGORIES},{c:0 for c in CATEGORIES}
        for c in CATEGORIES:
            for item,a,b in pieces[c]:
                prompt = texts[item["id"]]
                t = now()
                if "teacher" not in item:
                    # scoring each response once and keeping its logits until its last piece has been used
                    ids = torch.tensor([prompt+item["tokens"][:item["limit"]-1]],device=dev)
                    with torch.inference_mode():
                        item["teacher"] = teacher(input_ids=ids,logits_to_keep=item["limit"]).logits[0].clone()
                t1 = now()
                clock["teacher_scoring"] += t1-t
                ids = torch.tensor([prompt+item["tokens"][:b-1]],device=dev)
                with torch.autocast(dev,dtype=torch.float16):
                    logits = student(input_ids=ids,logits_to_keep=b-a).logits[0]
                lt = F.log_softmax(item["teacher"][a:b].clone().float(),-1)
                ls = F.log_softmax(logits.float(),-1)
                kl = (lt.exp()*(lt-ls)).sum()
                scaler.scale(kl/total).backward()
                sums[c] += kl.item()
                counts[c] += b-a
                clock["forward_backward"] += now()-t1
                if b == item["limit"]:
                    del item["teacher"]
        t = now()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(params,d["clip"]).item()
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        skip = scaler.get_scale() < before
        skipped += skip
        clock["optimizer"] += now()-t
        log.append({"update":u,"tokens":total,"scored_math":quota["math"]-streams["math"].left,
                    "scored_short":quota["short"]-streams["short"].left,"kl":sum(sums.values())/total,
                    "kl_math":sums["math"]/max(1,counts["math"]),"kl_short":sums["short"]/max(1,counts["short"]),
                    "lr":opt.param_groups[0]["lr"],"grad_norm":norm,"scale":before,"skipped":bool(skip),
                    "presentations":sum(x.count for x in streams.values()),"seconds":time.perf_counter()-begin,
                    "peak_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None})
        if u%10 == 0 or u == updates:
            print(name,f"update {u} kl {log[-1]['kl']:.4f} norm {norm:.2f} time {log[-1]['seconds']:.0f} s",flush=True)
    wall = time.perf_counter()-begin
    rows = [log_row(c) for x in streams.values() for c in x.done+([x.current] if x.current else [])]
    scored = {c:sum(r["teacher_scored_tokens"] for r in rows if r["category"] == c) for c in CATEGORIES}
    assert scored == {c:quota[c] for c in CATEGORIES},scored
    folder = out/"distill"/name
    write_gz(folder/"presentations.jsonl.gz",rows)
    save_json(folder/"log.json",log)
    used = {c:{"presentations":len(x),"unique_prompts":len({r["id"] for r in x}),
               "mean_response_length":sum(r["response_length"] for r in x)/len(x),
               "ended_with_eos":sum(r["termination"] == "eos" for r in x)/len(x),
               "quota_truncated":sum(r["quota_truncated"] for r in x),
               "generated_tokens":sum(r["response_length"] for r in x),
               "tokens_carried_over":sum(r["tokens_carried_over"] for r in x)}
            for c in CATEGORIES for x in [[r for r in rows if r["category"] == c]]}
    save_json(folder/"summary.json",{"name":name,"method":method,"seed":seed,"updates":updates,"skipped_updates":skipped,
                                     "scored_tokens":scored,"wall_seconds":wall,
                                     "seconds":{k:v/1000 for k,v in clock.items()},"responses":used,
                                     "backbone_parameters":backbone,"adapter_parameters":sum(p.numel() for p in params),
                                     "lora_init":init,"lora_final":checksum(student),
                                     "first_kl":log[0]["kl"],"last_kl":log[-1]["kl"],
                                     "peak_gb":torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None})
    path = artifacts(cfg,args.artifacts)/"adapters"
    path.mkdir(parents=True,exist_ok=True)
    torch.save(export_factors(student),path/f"{name}.pt")
    print(name,"finished with",updates,"updates",skipped,"skipped",f"{wall:.0f} s",flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--method",choices=["off","on"],default=None)
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--artifacts",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(args.results or cfg["paths"]["results_dir"])
    for method in [args.method] if args.method else cfg["distill"]["methods"]:
        train(cfg,args,method,args.seed,out)


if __name__ == "__main__":
    main()
