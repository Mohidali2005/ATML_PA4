import math

import torch
import torch.nn.functional as F


def padded(seqs,dev):
    """Right pad token lists into an id tensor and an attention mask"""
    width = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs),width,dtype=torch.long)
    mask = torch.zeros(len(seqs),width,dtype=torch.long)
    for j,s in enumerate(seqs):
        ids[j,:len(s)] = torch.tensor(s)
        mask[j,:len(s)] = 1
    return ids.to(dev),mask.to(dev)


@torch.no_grad()
def perplexity(model,wins,batch):
    """Score every window once and return its summed nll and number of scored tokens"""
    dev = next(model.parameters()).device
    # sorting by length so each batch carries little padding
    order = sorted(range(len(wins)),key=lambda i:-len(wins[i]["ids"]))
    rows = {}
    for a in range(0,len(order),batch):
        chunk = [wins[i] for i in order[a:a+batch]]
        ids,mask = padded([w["ids"] for w in chunk],dev)
        logits = model(input_ids=ids,attention_mask=mask).logits
        for j,w in enumerate(chunk):
            s,e = w["start"],w["end"]
            nll = F.cross_entropy(logits[j,s-1:e-1].float(),ids[j,s:e],reduction="sum").item()
            assert math.isfinite(nll),f"non finite nll in window {w['window']}"
            rows[w["window"]] = {"window":w["window"],"source_row":w["source_row"],"nll_sum":nll,"tokens":e-s}
    return [rows[w["window"]] for w in wins]


def pooled(rows):
    """Pool window rows into one mean token nll and perplexity"""
    tokens = sum(r["tokens"] for r in rows)
    nll = sum(r["nll_sum"] for r in rows)/tokens
    return {"tokens":tokens,"nll":nll,"ppl":math.exp(nll)}


def sequence_windows(ids):
    """Treat each fixed sequence as one window that scores every token after the first"""
    return [{"window":i,"source_row":i,"ids":s.tolist(),"start":1,"end":len(s)} for i,s in enumerate(ids)]


def encode_pair(tok,context,continuation):
    """Split a context and continuation into token ids the same way the standard evaluation harness does"""
    spaces = len(context)-len(context.rstrip())
    if spaces:
        continuation = context[-spaces:]+continuation
        context = context[:-spaces]
    whole = tok.encode(context+continuation,add_special_tokens=False)
    ctx = tok.encode(context,add_special_tokens=False)
    return ctx,whole[len(ctx):]


@torch.no_grad()
def hellaswag(model,tok,docs,batch):
    """Score the four endings of every row by their summed log likelihood"""
    dev = next(model.parameters()).device
    rows = []
    for a in range(0,len(docs),batch):
        chunk = docs[a:a+batch]
        pairs = [encode_pair(tok,d["query"]," "+c) for d in chunk for c in d["choices"]]
        ids,mask = padded([c+x for c,x in pairs],dev)
        logits = model(input_ids=ids,attention_mask=mask).logits
        lls = []
        for j,(c,x) in enumerate(pairs):
            n,m = len(c),len(x)
            logp = logits[j,n-1:n+m-1].float().log_softmax(-1)
            lls.append(logp.gather(1,ids[j,n:n+m,None]).sum().item())
        for k,d in enumerate(chunk):
            ll = lls[4*k:4*k+4]
            chars = [len(c) for c in d["choices"]]
            norm = [l/n for l,n in zip(ll,chars)]
            rows.append({"order":d["order"],"source_row":d["source_row"],"gold":d["gold"],
                         "ll":ll,"chars":chars,"tokens":[len(pairs[4*k+i][1]) for i in range(4)],
                         "pred":max(range(4),key=lambda i:ll[i]),"pred_norm":max(range(4),key=lambda i:norm[i])})
    return rows


def hellaswag_scores(rows):
    """Raw and character length normalized accuracy"""
    n = len(rows)
    return {"rows":n,"acc":sum(r["pred"] == r["gold"] for r in rows)/n,
            "acc_norm":sum(r["pred_norm"] == r["gold"] for r in rows)/n}
