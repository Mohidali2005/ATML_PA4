import math

import torch
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks.hellaswag.utils import preprocess as reference

from task1_compress.data import preprocess,read_locked
from task1_compress.evaluate import hellaswag,perplexity
from task1_compress.model import eligible,embed,run_blocks,unique_parameters
from task1_compress.quantize import fake_quant,pack_bits,unpack_bits


def structure(model,cfg):
    """Count the eligible projections and check that the output head shares the embedding"""
    mods = eligible(model,cfg)
    c = model.config
    per_block = (2*c.hidden_size*c.hidden_size+2*c.hidden_size*c.num_key_value_heads*(c.hidden_size//c.num_attention_heads)
                 +3*c.hidden_size*c.intermediate_size)
    total = sum(m.weight.numel() for _,_,m in mods)
    out = {"eligible_modules":len(mods),"eligible_weights":total,"expected_eligible_weights":per_block*c.num_hidden_layers,
           "tied_head":model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(),
           "unique_parameters":unique_parameters(model),"other_parameters":unique_parameters(model)-total,
           "head_in_eligible":any(m is model.lm_head for _,_,m in mods)}
    assert out["eligible_modules"] == 7*c.num_hidden_layers and total == out["expected_eligible_weights"]
    assert out["tied_head"] and not out["head_in_eligible"]
    return out


@torch.no_grad()
def runner(model,ids):
    """Check that running the blocks one by one gives the same logits as the full model"""
    hidden,rope = embed(model,ids)
    mine = model.lm_head(model.model.norm(run_blocks(model,hidden,rope)))
    return {"max_abs_logit_diff":(mine-model(input_ids=ids).logits).abs().max().item()}


def batching(model,wins):
    """Check that batched and one at a time window scoring agree"""
    a = perplexity(model,wins,8)
    b = perplexity(model,wins,1)
    return {"windows":len(wins),"max_abs_nll_diff":max(abs(x["nll_sum"]-y["nll_sum"]) for x,y in zip(a,b))}


def precision(model32,model16,wins):
    """Compare the half precision model with a full precision copy on a few windows"""
    a = perplexity(model16,wins,4)
    b = perplexity(model32,wins,4)
    tok = sum(r["tokens"] for r in a)
    return {"windows":len(wins),"tokens":tok,"nll_fp16":sum(r["nll_sum"] for r in a)/tok,
            "nll_fp32":sum(r["nll_sum"] for r in b)/tok,
            "max_window_nll_diff":max(abs(x["nll_sum"]-y["nll_sum"])/x["tokens"] for x,y in zip(a,b))}


def harness(model,tok,cfg,docs,count):
    """Compare our hellaswag formatting and log likelihoods with the standard evaluation harness"""
    raw = sorted(read_locked(cfg,"hellaswag_val"),key=lambda r:r["order"])
    same = all(preprocess(e) == reference(e) for r in raw for e in r["example"]["endings"])
    lm = HFLM(pretrained=model,tokenizer=tok,batch_size=8)
    reqs = [Instance(request_type="loglikelihood",doc={},arguments=(d["query"]," "+c),idx=k)
            for d in docs[:count] for k,c in enumerate(d["choices"])]
    theirs = [ll for ll,_ in lm.loglikelihood(reqs)]
    mine = [x for r in hellaswag(model,tok,docs[:count],8) for x in r["ll"]]
    picks = [max(range(4),key=lambda i:v[4*k+i]) for v in (theirs,mine) for k in range(count)]
    return {"preprocess_identical":same,"rows":count,"max_abs_ll_diff":max(abs(a-b) for a,b in zip(theirs,mine)),
            "same_raw_pick":picks[:count] == picks[count:]}


def quantizer():
    """Check the quantizer against the written formula and check that packing round trips"""
    torch.manual_seed(0)
    w = torch.randn(64,256)*0.05
    out = {}
    for bits in (2,3,4,8):
        g = w.view(64,2,128)
        s = ((g.amax(-1,keepdim=True)-g.amin(-1,keepdim=True))/(2**bits-1)).clamp(min=1e-5).half().float()
        z = (-torch.round(g.amin(-1,keepdim=True)/s)).clamp(0,2**bits-1)
        ref = (s*(torch.clamp(torch.round(g/s)+z,0,2**bits-1)-z)).view(64,256).half()
        out[f"formula_diff_b{bits}"] = (fake_quant(w,bits,128,1e-5).half().float()-ref.float()).abs().max().item()
    for bits in (1,2,3,4,8):
        codes = torch.randint(0,2**bits,(1000,),dtype=torch.uint8)
        out[f"roundtrip_b{bits}"] = bool(torch.equal(unpack_bits(pack_bits(codes,bits),bits,1000),codes))
        out[f"bytes_b{bits}"] = pack_bits(codes,bits).numel()
    assert all(v for k,v in out.items() if k.startswith("roundtrip"))
    assert all(math.isclose(v,0.0,abs_tol=0) for k,v in out.items() if k.startswith("formula"))
    return out
