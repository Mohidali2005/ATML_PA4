import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from common.utils import load_yaml,repo_path,save_json,set_seed
from task3_speculative.data import SPLITS,chat_ids,read_gz,test_prompts
from task3_speculative.models import DRAFTS,STUDENTS,load_model,load_tokenizer

SHARED = ("teacher","original_draft","exact_spec_original_gamma4")


@torch.inference_mode()
def next_logits(model,prompt,tokens):
    """Logits that predict every response token with the given prompt in one full forward pass"""
    ids = torch.tensor([prompt+tokens[:-1]],device=model.device)
    return model(input_ids=ids,logits_to_keep=len(tokens)).logits[0].float()


def surprisal(lp,tokens):
    """Summed surprisal minus entropy of the emitted tokens and its conditional variance

    the difference has mean zero at every step when the tokens follow the given distributions"""
    p = lp.exp()
    h = -(p*lp).sum(-1)
    s = -lp[torch.arange(len(tokens),device=lp.device),torch.tensor(tokens,device=lp.device)]
    var = (p*lp**2).sum(-1)-h**2
    return {"tokens":len(tokens),"excess":float((s-h).double().sum()),"variance":float(var.double().sum()),
            "surprisal":float(s.double().sum()),"entropy":float(h.double().sum())}


def prefix_metrics(lp,lq):
    """Divergences overlap and greedy agreement between target and draft at every saved prefix"""
    p,q = lp.exp(),lq.exp()
    tv = 0.5*(p-q).abs().sum(-1)
    return {"forward_kl":(p*(lp-lq)).sum(-1),"reverse_kl":(q*(lq-lp)).sum(-1),"tv":tv,
            "greedy_agree":(lp.argmax(-1) == lq.argmax(-1)).float(),"expected_accept":1-tv,
            "target_entropy":-(p*lp).sum(-1)}


def align(cfg,args,out):
    """Evaluate every draft on the same saved prefixes of every bank"""
    tok = load_tokenizer(cfg)
    prompts = {p["id"]:p for p in test_prompts(cfg)}
    target = load_model(cfg,"target",args.tiny)
    drafts = {n:load_model(cfg,n,args.tiny,args.artifacts) for n in DRAFTS}
    rows,seq = [],[]
    for bank in SHARED+STUDENTS:
        rolls = read_gz(out/"rollouts"/f"{bank}.jsonl.gz")
        saved = {}
        for r in read_gz(out/"banks"/f"{bank}.jsonl.gz"):
            saved.setdefault(r["prompt_id"],[]).append(r)
        for r in rolls:
            entries = saved[r["prompt_id"]]
            # every saved prefix must be the start of the stored rollout
            assert all(e["response_prefix_token_ids"] == r["tokens"][:e["position"]-1] for e in entries)
            pos = torch.tensor([e["position"]-1 for e in entries],device=target.device)
            prompt = chat_ids(tok,prompts[r["prompt_id"]]["text"])
            lp = F.log_softmax(next_logits(target,prompt,r["tokens"]),-1)
            seq.append({"source":bank,"prompt_id":r["prompt_id"],"split":r["split"],"termination":r["termination"],
                        **surprisal(lp,r["tokens"])})
            for name,model in drafts.items():
                lq = F.log_softmax(next_logits(model,prompt,r["tokens"])[pos],-1)
                m = {k:v.tolist() for k,v in prefix_metrics(lp[pos],lq).items()}
                for i,e in enumerate(entries):
                    rows.append({"bank":bank,"prompt_id":e["prompt_id"],"split":e["split"],"position":e["position"],"draft":name,
                                 **{k:round(v[i],7) for k,v in m.items()}})
        print("bank",bank,len(rows),"rows",flush=True)
    df = pd.DataFrame(rows)
    folder = out/"alignment"
    folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/"prefix_metrics.csv.gz",index=False)
    pd.DataFrame(seq).to_csv(folder/"rollout_surprisal.csv",index=False)
    save_json(folder/"disagreements.json",disagreements(cfg,out,tok,prompts,target,drafts,df))


def disagreements(cfg,out,tok,prompts,target,drafts,df):
    """Aligned prefixes where the two students of the first seed differ most from each other in both directions"""
    seed = cfg["distill"]["seeds"][0]
    a,b = f"off_{seed}",f"on_{seed}"
    wide = df[df["bank"] == "teacher"].pivot_table(index=["prompt_id","split","position"],columns="draft",values="tv").reset_index()
    wide["gap"] = wide[a]-wide[b]
    rolls = {r["prompt_id"]:r["tokens"] for r in read_gz(out/"rollouts"/"teacher.jsonl.gz")}
    cases = []
    for s in SPLITS:
        part = wide[wide["split"] == s].sort_values("gap")
        for label,row in (("off_policy_closer",part.iloc[0]),("on_policy_closer",part.iloc[-1])):
            tokens = rolls[row["prompt_id"]]
            i = int(row["position"])-1
            prompt = chat_ids(tok,prompts[row["prompt_id"]]["text"])
            case = {"split":s,"rule":label,"prompt_id":row["prompt_id"],"position":i+1,"question":prompts[row["prompt_id"]]["text"],
                    "prefix_end":tok.decode(tokens[max(0,i-40):i]),"next_teacher_token":tok.decode([tokens[i]]),"models":{}}
            for name,model in [("target",target)]+list(drafts.items()):
                p = torch.softmax(next_logits(model,prompt,tokens[:i+1])[-1],-1)
                top = p.topk(4)
                case["models"][name] = {"tv":None if name == "target" else float(row[name]),
                                        "top":[[tok.decode([t]),round(v,4)] for v,t in zip(top.values.tolist(),top.indices.tolist())]}
            cases.append(case)
    return cases


def benchmark_surprisal(cfg,args,out):
    """Surprisal test of every emitted sequence of the benchmark under the target"""
    tok = load_tokenizer(cfg)
    prompts = {p["id"]:p for p in test_prompts(cfg)}
    target = load_model(cfg,"target",args.tiny)
    rows,seen = [],{}
    for r in read_gz(out/"benchmark"/"tokens.jsonl.gz"):
        if r["kind"] != "eos":
            continue
        prompt = chat_ids(tok,prompts[r["prompt_id"]]["text"])
        for who in ("target","spec"):
            key = (r["prompt_id"],tuple(r[f"{who}_tokens"]))
            if key not in seen:
                seen[key] = surprisal(F.log_softmax(next_logits(target,prompt,r[f"{who}_tokens"]),-1),r[f"{who}_tokens"])
            rows.append({"who":who,"draft":r["draft"],"gamma":r["gamma"],"rep":r["rep"],"prompt_id":r["prompt_id"],"split":r["split"],
                         **seen[key]})
    folder = out/"alignment"
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/"benchmark_surprisal.csv",index=False)
    print("benchmark surprisal written for",len(rows),"sequences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--stage",choices=["prefixes","benchmark"],required=True)
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--artifacts",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    out = repo_path(args.results or cfg["paths"]["results_dir"])
    {"prefixes":align,"benchmark":benchmark_surprisal}[args.stage](cfg,args,out)


if __name__ == "__main__":
    main()
