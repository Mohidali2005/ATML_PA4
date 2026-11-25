import argparse
import re
import time

import pandas as pd
import torch

from common.utils import load_yaml,repo_path,save_json,set_seed
from task3_speculative.data import chat_ids,test_prompts,write_gz
from task3_speculative.models import DRAFTS,device,load_model,load_tokenizer
from task3_speculative.sampler import sample_batch

NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# the only text a compliant short answer may hold for every template family
FORMAT = {0:re.compile(rf"({MONTHS})"),1:re.compile(r"\d+"),2:re.compile(r"[A-Z]{2}")}


def number(text):
    """Read a number from text or return nothing"""
    m = NUMBER.findall(text)
    return float(m[-1].replace(",","").replace("$","")) if m else None


def math_answer(text):
    """Last boxed number of a solution or the last number when nothing is boxed"""
    boxed = [number(b) for b in BOXED.findall(text)]
    boxed = [b for b in boxed if b is not None]
    return boxed[-1] if boxed else number(text)


def score(p,text):
    """Correctness extraction and format of one response"""
    if p["domain"] == "math":
        got = math_answer(text)
        return {"extracted":got is not None,"compliant":None,"correct":got is not None and abs(got-float(p["answer"])) < 1e-6,
                "contains_answer":None,"answer":got}
    clean = text.strip()
    ok = FORMAT[p["family"]].fullmatch(clean) is not None
    return {"extracted":None,"compliant":ok,"correct":clean == p["answer"],
            "contains_answer":re.search(rf"(?<!\w){re.escape(p['answer'])}(?!\w)",text) is not None,"answer":clean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--artifacts",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    q = cfg["quality"]
    caps = cfg["smoke"]["quality_cap"] if args.smoke else q["cap"]
    tok = load_tokenizer(cfg)
    prompts = test_prompts(cfg,cfg["smoke"]["test_prompts_per_split"] if args.smoke else None)
    gen = torch.Generator(device=device())
    eos = cfg["models"]["eos"]
    rows,texts,times = [],[],{}
    for name in ("target",)+DRAFTS:
        model = load_model(cfg,name,args.tiny,args.artifacts)
        t0 = time.perf_counter()
        for rep in range(q["reps"]):
            for domain in ("math","short"):
                group = [p for p in prompts if p["domain"] == domain]
                for a in range(0,len(group),q["batch"]):
                    part = group[a:a+q["batch"]]
                    gen.manual_seed(cfg["seed"]+7919*(rep+1)+part[0]["index"])
                    out = sample_batch(model,[chat_ids(tok,p["text"]) for p in part],caps[domain],eos,gen,eos[1])
                    for p,r in zip(part,out):
                        text = tok.decode(r["tokens"],skip_special_tokens=True)
                        rows.append({"model":name,"rep":rep,"prompt_id":p["id"],"split":p["split"],"tokens":len(r["tokens"]),
                                     "truncated":r["truncated"],"gold":p["answer"],**score(p,text)})
                        texts.append({"model":name,"rep":rep,"prompt_id":p["id"],"text":text})
        times[name] = time.perf_counter()-t0
        print("quality",name,f"{times[name]:.0f} s",flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    folder = repo_path(args.results or cfg["paths"]["results_dir"])/"quality"
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/"responses.csv",index=False)
    write_gz(folder/"texts.jsonl.gz",texts)
    save_json(folder/"seconds.json",times)


if __name__ == "__main__":
    main()
