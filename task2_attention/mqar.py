import argparse
import gzip
import importlib.util
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch

from common.utils import load_json,load_yaml,repo_path,save_json
from task1_compress.data import content_hash

SPLITS = {"development":"mqar_development","test":"mqar_test"}
REGIMES = ("near","far","mixed")
KS = (4,8,16)


def generator(cfg):
    """Import the course generator module straight from its file"""
    path = repo_path(cfg["paths"]["course_dir"])/"scripts"/"mqar_generator.py"
    spec = importlib.util.spec_from_file_location("mqar_generator",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen(cfg,split):
    """Read the frozen development or test records"""
    path = repo_path(cfg["paths"]["mqar_dir"])/f"{SPLITS[split]}.jsonl.gz"
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pack(rows,length):
    """Turn records into token and label arrays with the label stored only at query positions"""
    tokens = np.array([r["tokens"] for r in rows],dtype=np.uint8)
    labels = np.full((len(rows),length),-1,dtype=np.int16)
    for i,r in enumerate(rows):
        labels[i,r["query_positions"]] = r["answer_ids"]
    return tokens,labels


def make_chunk(args):
    """Generate one chunk of training examples in a worker process"""
    path,k,regime,start,stop = args
    spec = importlib.util.spec_from_file_location("mqar_generator",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [module.make_example("train",k,regime,i) for i in range(start,stop)]


def stream_path(cfg,k,regime,count):
    """Location of a cached training stream"""
    return repo_path(cfg["paths"]["outputs_dir"])/"streams"/f"train_k{k}_{regime}_{count}.npz"


def build_stream(cfg,k,regime,count):
    """Generate the fixed training stream of one load and regime once and cache it"""
    path = stream_path(cfg,k,regime,count)
    if path.exists():
        return path
    src = str(repo_path(cfg["paths"]["course_dir"])/"scripts"/"mqar_generator.py")
    step = 2000
    jobs = [(src,k,regime,a,min(a+step,count)) for a in range(0,count,step)]
    with Pool(os.cpu_count()) as pool:
        rows = [r for chunk in pool.map(make_chunk,jobs) for r in chunk]
    tokens,labels = pack(rows,cfg["mqar"]["length"])
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,tokens=tokens,labels=labels)
    return path


def load_stream(cfg,k,regime,count):
    """Load a cached training stream as compact tensors that batches turn into long ids"""
    data = np.load(build_stream(cfg,k,regime,count))
    return torch.from_numpy(data["tokens"]),torch.from_numpy(data["labels"])


def eval_sets(cfg,split,per_set=None):
    """Group the frozen records by load and regime and pack each group"""
    rows = frozen(cfg,split)
    out = {}
    for k in KS:
        for regime in REGIMES:
            group = [r for r in rows if r["K"] == k and r["regime"] == regime][:per_set]
            tokens,labels = pack(group,cfg["mqar"]["length"])
            out[(k,regime)] = {"rows":group,"tokens":torch.from_numpy(tokens.astype(np.int64)),
                               "labels":torch.from_numpy(labels.astype(np.int64))}
    return out


def check(cfg,streams):
    """Check the generator invariants regime limits and split disjointness and save the result"""
    gen = generator(cfg)
    out = {"splits":{}}
    seen = {}
    for split in SPLITS:
        rows = frozen(cfg,split)
        for r in rows:
            gen.validate(r)
            # rebuilding each frozen record from its own seed to prove the generator matches the files
            k,regime,index = r["K"],r["regime"],int(r["id"].split("-")[-1])
            assert gen.make_example(split,k,regime,index)["tokens"] == r["tokens"]
        counts = Counter(f"K{r['K']}_{r['regime']}" for r in rows)
        dist = {reg:[min(d for r in rows if r["regime"] == reg for d in r["distances"]),
                     max(d for r in rows if r["regime"] == reg for d in r["distances"])] for reg in REGIMES}
        far_mod4 = sum(d % 4 != 0 for r in rows if r["regime"] == "far" for d in r["distances"])
        out["splits"][split] = {"records":len(rows),"per_set":dict(counts),"distance_range":dist,
                                "far_queries_not_multiple_of_4":far_mod4,
                                "queries":sum(r["K"] for r in rows)}
        seen[split] = {bytes(r["tokens"]) for r in rows}
    assert not seen["development"] & seen["test"]
    overlap = {}
    for name,path in streams.items():
        tokens = np.load(path)["tokens"]
        hashes = {t.tobytes() for t in tokens}
        overlap[name] = {"examples":len(tokens),"unique":len(hashes),
                         "equal_to_development":len(hashes & seen["development"]),
                         "equal_to_test":len(hashes & seen["test"])}
        assert overlap[name]["equal_to_test"] == 0 and overlap[name]["equal_to_development"] == 0
    out["train_streams"] = overlap
    out["development_test_overlap"] = 0
    save_json(repo_path(cfg["paths"]["results_dir"])/"checks"/"mqar_checks.json",out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task2.yaml")
    ap.add_argument("--write-lock",action="store_true")
    ap.add_argument("--verify",action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    files = {"mqar_development":repo_path(cfg["paths"]["mqar_dir"])/"mqar_development.jsonl.gz",
             "mqar_test":repo_path(cfg["paths"]["mqar_dir"])/"mqar_test.jsonl.gz",
             "kv_retrieval_cases":repo_path(cfg["paths"]["kv_cases"])}
    sums = {name:content_hash(path) for name,path in files.items()}
    if args.write_lock:
        save_json(cfg["paths"]["checksums"],sums)
        print("wrote the task2 data lock for",len(sums),"files")
    if args.verify:
        expected = load_json(cfg["paths"]["checksums"])
        bad = [name for name in sums if sums[name] != expected[name]]
        assert not bad,f"frozen records differ from the committed hashes {bad}"
        print("all",len(sums),"frozen task2 files hold the committed records")


if __name__ == "__main__":
    main()
