import argparse
import gzip
import hashlib
import json
import re
import shutil

import torch

from common.utils import load_json,load_yaml,repo_path,save_json

LOCKED = ["c4_calibration","c4_development","c4_validation_windows","wikitext2_test_windows",
          "hellaswag_val","gsm8k_train","gsm8k_test"]


def read_locked(cfg,name):
    """Read one locked data file as a list of records"""
    path = repo_path(cfg["paths"]["data_dir"])/f"{name}_locked.jsonl.gz"
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sequences(cfg,name,count=None):
    """Load the fixed calibration or development token sequences as one tensor"""
    rows = read_locked(cfg,name)[:count]
    return torch.tensor([r["token_ids"] for r in rows],dtype=torch.long)


def windows(cfg,suite,count=None):
    """Load the fixed evaluation windows of one perplexity suite"""
    name = {"wikitext":"wikitext2_test_windows","c4":"c4_validation_windows"}[suite]
    rows = read_locked(cfg,name)[:count]
    return [{"window":i,"source_row":r["source_row"],"ids":r["token_ids"],
             "start":r["score_target_offsets"][0],"end":r["score_target_offsets"][1]} for i,r in enumerate(rows)]


def preprocess(text):
    """Clean one hellaswag string the same way the standard evaluation harness does"""
    text = text.strip()
    text = text.replace(" [title]",". ")
    text = re.sub("\\[.*?\\]","",text)
    text = text.replace("  "," ")
    return text


def hellaswag(cfg,count=None):
    """Build the hellaswag rows in manifest order with their query endings and gold label"""
    rows = sorted(read_locked(cfg,"hellaswag_val"),key=lambda r:r["order"])[:count]
    out = []
    for r in rows:
        ex = r["example"]
        ctx = ex["ctx_a"]+" "+ex["ctx_b"].capitalize()
        out.append({"order":r["order"],"source_row":r["source_row_index"],
                    "query":preprocess(ex["activity_label"]+": "+ctx),
                    "choices":[preprocess(e) for e in ex["endings"]],"gold":int(ex["label"])})
    return out


def sha256(path):
    """Hash a file in chunks"""
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task1.yaml")
    ap.add_argument("--write-lock",action="store_true")
    ap.add_argument("--verify",action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    data_dir = repo_path(cfg["paths"]["data_dir"])
    sums = {name:sha256(data_dir/f"{name}_locked.jsonl.gz") for name in LOCKED}
    if args.write_lock:
        shutil.copy(data_dir.parent/"local_source_revisions.json",repo_path(cfg["paths"]["lock"]))
        save_json(cfg["paths"]["checksums"],sums)
        print("wrote the data lock for",len(sums),"files")
    if args.verify:
        expected = load_json(cfg["paths"]["checksums"])
        bad = [name for name in LOCKED if sums[name] != expected[name]]
        assert not bad,f"locked files differ from the committed checksums {bad}"
        print("all",len(sums),"locked files match the committed checksums")


if __name__ == "__main__":
    main()
