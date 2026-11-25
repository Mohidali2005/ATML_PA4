import contextlib
import time

import torch
from transformers import DynamicCache

from task2_attention.timing import sync


def now():
    """Wall clock in milliseconds after all queued gpu work has finished"""
    sync()
    return 1000*time.perf_counter()


class HFLM:
    """A causal language model with a rollback cache that counts its forward passes"""

    def __init__(self,model,amp=False):
        self.model = model
        self.device = model.device
        self.amp = amp
        self.reset()

    def reset(self):
        """Start from an empty cache"""
        self.cache = DynamicCache()
        self.passes,self.fed = 0,0

    @property
    def length(self):
        """Number of tokens held by the cache"""
        return self.cache.get_seq_length()

    def context(self):
        """Autocast for a model that still carries fp32 adapters"""
        return torch.autocast(self.device.type,dtype=torch.float16) if self.amp else contextlib.nullcontext()

    def prefill(self,tokens):
        """Run the prompt through the model and keep only the cache"""
        with self.context():
            self.model(input_ids=torch.tensor([tokens],device=self.device),past_key_values=self.cache,use_cache=True,
                       logits_to_keep=1)

    def feed(self,tokens):
        """Append tokens to the cache and return the next token distribution after each of them in fp32"""
        with self.context():
            out = self.model(input_ids=torch.tensor([tokens],device=self.device),past_key_values=self.cache,use_cache=True)
        self.passes += 1
        self.fed += len(tokens)
        return torch.softmax(out.logits[0].float(),-1)

    def crop(self,n):
        """Drop everything after the first n tokens"""
        if self.length > n:
            self.cache.crop(n)


class MarkovLM:
    """A small chain with the same interface whose next symbol depends only on the previous one"""

    def __init__(self,table):
        self.table = torch.tensor(table,dtype=torch.float64)
        self.device = torch.device("cpu")
        self.reset()

    def reset(self):
        """Start from an empty history"""
        self.tokens = []
        self.passes,self.fed = 0,0

    @property
    def length(self):
        """Number of symbols seen so far"""
        return len(self.tokens)

    def prefill(self,tokens):
        """Append symbols without reading the distribution"""
        self.tokens += tokens

    def feed(self,tokens):
        """Append symbols and return the next symbol distribution after each of them"""
        self.tokens += tokens
        self.passes += 1
        self.fed += len(tokens)
        return self.table[tokens]

    def crop(self,n):
        """Drop everything after the first n symbols"""
        del self.tokens[n:]


def pick(probs,gen):
    """Draw one token from a probability vector"""
    return int(torch.multinomial(probs,1,generator=gen))


def residual(p,q,wrong=False):
    """Distribution used after a rejection

    the wrong variant samples the target itself and exists only as a negative control"""
    if wrong:
        return p
    r = (p-q).clamp_min(0)
    total = r.sum()
    return r/total if total > 0 else p


def start(models,prompt):
    """Reset the models and prefill the prompt without its last token"""
    times = []
    for m in models:
        m.reset()
        t = now()
        if len(prompt) > 1:
            m.prefill(prompt[:-1])
        times.append(now()-t)
    return times


@torch.inference_mode()
def target_only(target,prompt,max_new,eos,gen,ignore_eos=False):
    """Sample a continuation one token per forward pass"""
    prefill = start([target],prompt)
    seq,n0 = list(prompt),len(prompt)
    t0 = now()
    done = False
    while len(seq)-n0 < max_new and not done:
        x = pick(target.feed([seq[-1]])[-1],gen)
        seq.append(x)
        done = x in eos and not ignore_eos
    t1 = now()
    return {"tokens":seq[n0:],"eos":done,"truncated":not done,"target_prefill_ms":prefill[0],"decode_ms":t1-t0,
            "target_passes":target.passes}


@torch.inference_mode()
def speculative(target,draft,prompt,gamma,max_new,eos,gen,ignore_eos=False,wrong=False,timers=False):
    """Exact speculative sampling with a draft proposal one target verification and cache rollback

    with timers each round is split into synchronized draft verification acceptance and rollback times"""
    prefill = start([target,draft],prompt)
    seq,n0 = list(prompt),len(prompt)
    rounds,parts,route = [],[0.0,0.0,0.0,0.0],None
    clock = now if timers else (lambda:0.0)
    begin = now()
    while route is None and len(seq)-n0 < max_new:
        a = clock()
        # leaving room for the token that the target always adds at the end of the round
        k = min(gamma,max_new-(len(seq)-n0)-1)
        props,qs = [],[]
        for _ in range(k):
            q = draft.feed((seq+props)[draft.length:])[-1]
            x = pick(q,gen)
            props.append(x)
            qs.append(q)
            if x in eos and not ignore_eos:
                break
        b = clock()
        p = target.feed([seq[-1]]+props)
        c = clock()
        accepted = 0
        if props:
            idx = torch.tensor(props,device=p.device)
            row = torch.arange(len(props),device=p.device)
            ratio = p[row,idx]/torch.stack([q[x] for q,x in zip(qs,props)])
            ok = (torch.rand(len(props),generator=gen,device=p.device,dtype=p.dtype) < ratio).tolist()
            accepted = ok.index(False) if False in ok else len(props)
        new = props[:accepted]
        ended = bool(new) and new[-1] in eos and not ignore_eos
        if accepted < len(props):
            x = pick(residual(p[accepted],qs[accepted],wrong),gen)
            new.append(x)
            route = "residual" if x in eos and not ignore_eos else None
        elif not ended:
            x = pick(p[len(props)],gen)
            new.append(x)
            route = "bonus" if x in eos and not ignore_eos else None
        else:
            route = "accepted"
        d = clock()
        seq += new
        # keeping exactly the returned continuation without its last token in both caches
        target.crop(len(seq)-1)
        draft.crop(len(seq)-1)
        e = clock()
        rounds.append((len(props),accepted,min(accepted+1,len(props))))
        if timers:
            for i,t in enumerate((b-a,c-b,d-c,e-d)):
                parts[i] += t
    end = now()
    out = {"tokens":seq[n0:],"eos":route is not None,"truncated":route is None,"eos_route":route,"rounds":len(rounds),
           "proposed":sum(r[0] for r in rounds),"accepted":sum(r[1] for r in rounds),"examined":sum(r[2] for r in rounds),
           "accepted_per_round":[r[1] for r in rounds],"proposed_per_round":[r[0] for r in rounds],
           "target_passes":target.passes,"draft_passes":draft.passes,"draft_fed":draft.fed,"target_fed":target.fed,
           "target_prefill_ms":prefill[0],"draft_prefill_ms":prefill[1],"decode_ms":end-begin}
    if timers:
        out.update(zip(("draft_ms","verify_ms","accept_ms","rollback_ms"),parts))
        out["host_ms"] = out["decode_ms"]-sum(parts)
    return out


@torch.inference_mode()
def sample_batch(model,prompts,max_new,eos,gen,pad):
    """Sample continuations of several prompts together with left padding"""
    width,n = max(len(p) for p in prompts),len(prompts)
    dev = model.device
    ids = torch.tensor([[pad]*(width-len(p))+p for p in prompts],device=dev)
    mask = torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prompts],device=dev)
    pos = (mask.cumsum(-1)-1).clamp_min(0)
    cache = DynamicCache()
    stops = torch.tensor(eos,device=dev)
    done = torch.zeros(n,dtype=torch.bool,device=dev)
    out = []
    for step in range(max_new):
        logits = model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,
                       logits_to_keep=1).logits[:,-1].float()
        x = torch.multinomial(torch.softmax(logits,-1),1,generator=gen)
        x = torch.where(done[:,None],torch.full_like(x,pad),x)
        out.append(x)
        done = done|torch.isin(x[:,0],stops)
        if bool(done.all()):
            break
        ids,pos = x,pos[:,-1:]+1
        mask = torch.cat([mask,torch.ones(n,1,dtype=mask.dtype,device=dev)],1)
    rows = torch.cat(out,1).tolist()
    final = []
    for r in rows:
        cut = next((i+1 for i,t in enumerate(r) if t in eos),len(r))
        final.append({"tokens":r[:cut],"eos":r[cut-1] in eos,"truncated":r[cut-1] not in eos})
    return final
