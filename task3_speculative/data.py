import argparse
import gzip
import hashlib
import json
import random

from common.utils import load_json,load_yaml,read_jsonl,repo_path,save_json
from task3_speculative.models import load_tokenizer

SPLITS = ("math","familiar","shifted")


def read_locked(cfg,name):
    """Read one locked gsm8k file in manifest order"""
    path = repo_path(cfg["paths"]["course_dir"])/"data"/"public_locked"/f"{name}_locked.jsonl.gz"
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return sorted((json.loads(line) for line in f if line.strip()),key=lambda r:r["order"])


def short_form(cfg,name):
    """Read one short form file"""
    return read_jsonl(repo_path(cfg["paths"]["course_dir"])/"data"/"short_form"/f"{name}.jsonl")


def final_number(answer):
    """The gold number after the four hash marks of a gsm8k answer"""
    return answer.split("####")[-1].strip().replace(",","")


def train_prompts(cfg,limit=None):
    """The 800 training prompts grouped by quota category"""
    math = [{"id":f"GSM8K-train-{r['order']:04d}","category":"math","text":r["example"]["question"]}
            for r in read_locked(cfg,"gsm8k_train")]
    short = [{"id":r["id"],"category":"short","text":r["input"]} for r in short_form(cfg,"short_train")]
    return {"math":math[:limit],"short":short[:limit]}


def test_prompts(cfg,per_split=None):
    """The 120 test prompts in the fixed order used for every seed with their split and gold answer"""
    rows = [{"id":f"GSM8K-{r['order']:04d}","domain":"math","split":"math","family":None,"text":r["example"]["question"],
             "answer":final_number(r["example"]["answer"])} for r in read_locked(cfg,"gsm8k_test")]
    rows += [{"id":r["id"],"domain":"short","split":r["split"],"family":r["family"],"text":r["input"],"answer":r["answer"]}
             for r in short_form(cfg,"short_test")]
    for j,r in enumerate(rows):
        r["index"] = j
    if per_split:
        rows = [r for s in SPLITS for r in [x for x in rows if x["split"] == s][:per_split]]
    return rows


def chat_ids(tok,text):
    """Token ids of the chat prompt with a single user message"""
    return tok.apply_chat_template([{"role":"user","content":text}],add_generation_prompt=True)


def schedule(prompts,seed,category):
    """Fixed seeded order in which the prompts of one category are visited again and again"""
    order = list(range(len(prompts)))
    random.Random(f"task3:{seed}:{category}").shuffle(order)
    return order


def fixed_subset(rows,count,seed):
    """Seeded choice of mathematics prompts for the fixed length timing diagnostic"""
    math = [r for r in rows if r["split"] == "math"]
    return sorted(random.Random(f"task3:fixed:{seed}").sample(math,min(count,len(math))),key=lambda r:r["index"])


def write_gz(path,rows):
    """Write records to a gzipped jsonl file and create the folder when needed"""
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")


def read_gz(path):
    """Read records from a gzipped jsonl file"""
    with gzip.open(path,"rt",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_hash(rows):
    """Hash records so that line endings do not matter"""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r,sort_keys=True,ensure_ascii=False).encode("utf-8")+b"\n")
    return h.hexdigest()


def tokenizer_agreement(cfg):
    """Compare token ids and chat templates of both tokenizers on every prompt of the study"""
    a,b = load_tokenizer(cfg,"target"),load_tokenizer(cfg,"draft")
    texts = [p["text"] for group in train_prompts(cfg).values() for p in group]+[p["text"] for p in test_prompts(cfg)]
    plain = sum(a(t)["input_ids"] == b(t)["input_ids"] for t in texts)
    chat = sum(chat_ids(a,t) == chat_ids(b,t) for t in texts)
    lengths = [len(chat_ids(a,t)) for t in texts]
    return {"prompts":len(texts),"plain_token_ids_equal":plain,"chat_token_ids_equal":chat,
            "chat_template_equal":a.chat_template == b.chat_template,"vocab_equal":a.get_vocab() == b.get_vocab(),
            "eos_token":a.eos_token,"eos_token_id":a.eos_token_id,"draft_eos_token_id":b.eos_token_id,
            "tokenizer_vocab":len(a),"max_prompt_tokens":max(lengths),"min_prompt_tokens":min(lengths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--write-lock",action="store_true")
    ap.add_argument("--verify",action="store_true")
    ap.add_argument("--tokenizers",action="store_true")
    ap.add_argument("--results",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = {"gsm8k_train":read_locked(cfg,"gsm8k_train"),"gsm8k_test":read_locked(cfg,"gsm8k_test"),
              "short_train":short_form(cfg,"short_train"),"short_test":short_form(cfg,"short_test")}
    sums = {k:record_hash(v) for k,v in groups.items()}
    if args.write_lock:
        save_json(cfg["paths"]["checksums"],sums)
        print("wrote the data lock for",len(sums),"files")
    if args.verify:
        expected = load_json(cfg["paths"]["checksums"])
        bad = [k for k in sums if sums[k] != expected[k]]
        assert not bad,f"records differ from the committed hashes {bad}"
        train = {p["text"] for g in train_prompts(cfg).values() for p in g}
        test = test_prompts(cfg)
        overlap = sorted(train&{p["text"] for p in test})
        assert not overlap,"a test prompt appears in the training pool"
        counts = {s:sum(p["split"] == s for p in test) for s in SPLITS}
        print("all",len(sums),"files hold the committed records and no test prompt is in training",counts)
    if args.tokenizers:
        out = tokenizer_agreement(cfg)
        assert out["plain_token_ids_equal"] == out["chat_token_ids_equal"] == out["prompts"],out
        save_json(repo_path(args.results or cfg["paths"]["results_dir"])/"checks"/"tokenizers.json",out)
        print("tokenizers agree on",out["prompts"],"prompts")


if __name__ == "__main__":
    main()
