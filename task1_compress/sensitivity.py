import time

import torch
import torch.nn.functional as F

from task1_compress.calibrate import input_stats,walk
from task1_compress.model import nll_from_block,projection
from task1_compress.prune import lowest_mask,wanda_score
from task1_compress.quantize import GROUP_OF,GROUPS,apply_scale,clear_marks,fake_quant,group_inputs,search_scale

QUANTILES = [0.01,0.1,0.25,0.5,0.75,0.9,0.99]
KEEP = ["q_proj","o_proj","gate_proj","down_proj"]


def block_state(layer):
    """Copy every tensor of a block so it can be put back after a probe"""
    return {k:v.clone() for k,v in layer.state_dict().items()}


def output_error(fn,x,ref,micro):
    """Mean squared difference between a function of the stored inputs and the stored reference output"""
    total = 0.0
    for a in range(0,x.shape[0],micro):
        total += (fn(x[a:a+micro])-ref[a:a+micro]).float().pow(2).sum().item()
    return total/ref.numel()


def reference(fn,x,micro):
    """Apply a function to the stored inputs in micro batches"""
    return torch.cat([fn(x[a:a+micro]) for a in range(0,x.shape[0],micro)])


def pruning_walk(model,cfg,ids,levels):
    """Walk the dense model once to collect Wanda statistics outlier ratios and block pruning probes"""
    names,micro,m = cfg["eligible"],cfg["calib"]["micro_batch"],cfg["p3"]["owl_m"]
    base,_ = walk(model,cfg,ids)
    probes,owl,stats_rows = [],[],[]

    def before(i,layer,hidden,rope):
        stats = input_stats(layer,hidden,rope,micro,names)
        scores = {n:wanda_score(projection(layer,n).weight,stats[n]["sumsq"]) for n in names}
        count = sum(s.numel() for s in scores.values())
        mean = sum(s.sum() for s in scores.values())/count
        outliers = sum((s > m*mean).sum() for s in scores.values())
        owl.append({"block":i,"outlier_ratio":(outliers/count).item()})
        for n,s in scores.items():
            norms = stats[n]["sumsq"].sqrt()
            q = torch.quantile(s.flatten(),torch.tensor(QUANTILES,device=s.device))
            stats_rows.append({"block":i,"proj":n,"norm_max_over_median":(norms.max()/norms.median()).item(),
                               "norm_cv":(norms.std()/norms.mean()).item(),"score_mean":s.mean().item(),
                               **{f"score_q{round(100*a)}":v.item() for a,v in zip(QUANTILES,q)}})
        saved = block_state(layer)
        for level in levels:
            start = time.time()
            for n in names:
                w = projection(layer,n).weight
                w.data[lowest_mask(scores[n],round(level*w.numel()))] = 0
            nll = nll_from_block(model,hidden,rope,ids,i,micro)
            layer.load_state_dict(saved)
            probes.append({"block":i,"sparsity":level,"nll":nll,"delta":nll-base,"seconds":time.time()-start})

    walk(model,cfg,ids,before=before)
    return base,probes,owl,stats_rows


def quant_walk(model,cfg,ids,bits_list,grid,probe=True):
    """Walk the dense model once to search activation scales at every bit width and probe every projection

    when probe is false only the scale search and the local errors are computed
    returns the dense calibration nll the scales the search rows the local errors the probe rows
    and the search time spent on each bit width"""
    names,micro = cfg["eligible"],cfg["calib"]["micro_batch"]
    group,floor = cfg["quant"]["group"],cfg["quant"]["scale_floor"]
    base,_ = walk(model,cfg,ids)
    scales,search_rows,local_rows,probes = {},[],[],[]
    timer = {b:0.0 for b in bits_list}

    def before(i,layer,hidden,rope):
        inputs = group_inputs(layer,input_stats(layer,hidden,rope,micro,KEEP,keep=KEEP),rope)
        refs = {g:reference(fn,x,micro) for g,(x,_,fn) in inputs.items()}
        for b in bits_list:
            start = time.time()
            for g,(x,absmean,fn) in inputs.items():
                mods = [projection(layer,n) for n in GROUPS[g]]
                (_,alpha,s),errors = search_scale(mods,[b]*len(mods),absmean,
                                                  lambda:output_error(fn,x,refs[g],micro),cfg,grid)
                scales[(i,g,b)] = s.half().cpu()
                search_rows.extend([{"block":i,"group":g,"bits":b,"alpha":k/grid,"mse":e,"chosen":k/grid == alpha}
                                    for k,e in enumerate(errors)])
                # measuring each projection alone with the chosen scale as the cheap proxy
                for n,mod in zip(GROUPS[g],mods):
                    w = mod.weight.data
                    w_eff = (fake_quant(w.float()*s[None,:],b,group,floor).float()/s[None,:]).to(w.dtype)
                    ref = reference(lambda t:F.linear(t,w),x,micro)
                    err = output_error(lambda t:F.linear(t,w_eff),x,ref,micro)
                    local_rows.append({"block":i,"proj":n,"bits":b,"alpha":alpha,
                                       "rel_error":err*ref.numel()/ref.float().pow(2).sum().item()})
            timer[b] += time.time()-start
        if not probe:
            return
        saved = block_state(layer)
        for n in names:
            for b in bits_list:
                start = time.time()
                handle = apply_scale(layer,GROUP_OF[n],scales[(i,GROUP_OF[n],b)].to(hidden.device).float())
                mod = projection(layer,n)
                mod.weight.data = fake_quant(mod.weight.data,b,group,floor)
                nll = nll_from_block(model,hidden,rope,ids,i,micro)
                layer.load_state_dict(saved)
                clear_marks(layer)
                if handle is not None:
                    handle.remove()
                probes.append({"block":i,"proj":n,"bits":b,"nll":nll,"delta":nll-base,"seconds":time.time()-start})

    walk(model,cfg,ids,before=before)
    return base,scales,search_rows,local_rows,probes,timer


def mixed_scales(model,cfg,ids,bits_map,cache,grid):
    """Search activation scales for a mixed bit assignment

    a group whose projections share one bit width reuses the uniform search at that width"""
    micro = cfg["calib"]["micro_batch"]
    scales = {}

    def before(i,layer,hidden,rope):
        for g in GROUPS:
            key = {bits_map[(i,n)] for n in GROUPS[g]}
            if len(key) == 1 and (i,g,key.pop()) in cache:
                scales[(i,g)] = cache[(i,g,bits_map[(i,GROUPS[g][0])])]
        need = [g for g in GROUPS if (i,g) not in scales]
        if not need:
            return
        inputs = group_inputs(layer,input_stats(layer,hidden,rope,micro,KEEP,keep=KEEP),rope)
        for g in need:
            x,absmean,fn = inputs[g]
            ref = reference(fn,x,micro)
            mods = [projection(layer,n) for n in GROUPS[g]]
            (_,_,s),_ = search_scale(mods,[bits_map[(i,n)] for n in GROUPS[g]],absmean,
                                     lambda:output_error(fn,x,ref,micro),cfg,grid)
            scales[(i,g)] = s.half().cpu()

    walk(model,cfg,ids,before=before)
    return scales
