import argparse
import time

import torch
import torch.nn.functional as F

from common.utils import load_yaml,repo_path,save_json,set_seed
from task3_speculative.data import chat_ids,test_prompts
from task3_speculative.lora import add_lora,load_factors,merge_factors
from task3_speculative.models import STUDENTS,artifacts,device,load_model,load_tokenizer


@torch.inference_mode()
def prompt_logits(model,ids,amp):
    """Logits at every position of one prompt with or without the precision rule used in training"""
    with torch.autocast(device(),dtype=torch.float16,enabled=amp):
        return model(input_ids=torch.tensor([ids],device=model.device)).logits[0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/task3.yaml")
    ap.add_argument("--tiny",action="store_true")
    ap.add_argument("--results",default=None)
    ap.add_argument("--artifacts",default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    art = artifacts(cfg,args.artifacts)
    tok = load_tokenizer(cfg)
    prompts = [chat_ids(tok,p["text"]) for p in test_prompts(cfg,4)]
    scale = cfg["distill"]["lora"]["alpha"]/cfg["distill"]["lora"]["rank"]
    rows = []
    for name in STUDENTS:
        state = torch.load(art/"adapters"/f"{name}.pt")
        t0 = time.perf_counter()
        merged = load_model(cfg,"draft",args.tiny)
        t1 = time.perf_counter()
        merge_factors(merged,state,scale)
        t2 = time.perf_counter()
        merged.save_pretrained(art/"merged"/name)
        t3 = time.perf_counter()
        wrapped = load_model(cfg,"draft",args.tiny)
        add_lora(wrapped,cfg,0)
        load_factors(wrapped,state)
        errors,kls,same,count = [],[],0,0
        for ids in prompts:
            a,b = prompt_logits(wrapped,ids,True),prompt_logits(merged,ids,False)
            errors.append(float((a-b).abs().max()))
            lp,lq = F.log_softmax(a,-1),F.log_softmax(b,-1)
            kls.append((lp.exp()*(lp-lq)).sum(-1))
            same += int((a.argmax(-1) == b.argmax(-1)).sum())
            count += len(ids)
        kl = torch.cat(kls)
        rows.append({"name":name,"prompts":len(prompts),"positions":count,"max_abs_logit_error":max(errors),
                     "mean_kl":float(kl.mean()),"max_kl":float(kl.max()),"argmax_agreement":same/count,
                     "load_seconds":t1-t0,"merge_seconds":t2-t1,"save_seconds":t3-t2,
                     "adapter_bytes":sum(4*(a.numel()+b.numel()) for a,b in state.values())})
        print("merged",rows[-1],flush=True)
        del merged,wrapped
    save_json(repo_path(args.results or cfg["paths"]["results_dir"])/"distill"/"merge.json",rows)


if __name__ == "__main__":
    main()
