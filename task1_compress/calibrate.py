import torch
import torch.nn.functional as F

from task1_compress.model import embed,projection,sequence_nll


def forward_block(layer,inputs,rope,micro):
    """Run one block over stored inputs in micro batches"""
    return torch.cat([layer(inputs[a:a+micro],position_embeddings=rope) for a in range(0,inputs.shape[0],micro)])


def input_stats(layer,inputs,rope,micro,names,keep=()):
    """Run one block and gather per channel input statistics of the chosen projections

    sumsq holds the summed square of every input channel and absmean its mean absolute value
    while the projections listed in keep also return their full inputs"""
    stats = {n:{"sumsq":0.0,"absmean":0.0,"inputs":[]} for n in names}
    hooks = []
    for n in names:
        def hook(module,args,n=n):
            x = args[0].detach().reshape(-1,args[0].shape[-1]).float()
            stats[n]["sumsq"] = stats[n]["sumsq"]+x.pow(2).sum(0)
            stats[n]["absmean"] = stats[n]["absmean"]+x.abs().sum(0)
            if n in keep:
                stats[n]["inputs"].append(args[0].detach())
        hooks.append(projection(layer,n).register_forward_pre_hook(hook))
    forward_block(layer,inputs,rope,micro)
    for h in hooks:
        h.remove()
    tokens = inputs.shape[0]*inputs.shape[1]
    for n in names:
        stats[n]["absmean"] = stats[n]["absmean"]/tokens
        stats[n]["inputs"] = torch.cat(stats[n]["inputs"]) if n in keep else None
    return stats


@torch.no_grad()
def walk(model,cfg,ids,before=None,originals=None):
    """Run calibration sequences through the blocks one at a time

    before may read or change each block from its input before the block is applied
    and when original weights are given the relative output error of every projection
    is measured on the inputs it really receives
    returns the pooled calibration nll and the error rows"""
    names,micro = cfg["eligible"],cfg["calib"]["micro_batch"]
    hidden,rope = embed(model,ids)
    errors = []
    for i,layer in enumerate(model.model.layers):
        if before is not None:
            before(i,layer,hidden,rope)
        acc = {n:[0.0,0.0] for n in names}
        hooks = []
        if originals is not None:
            for n in names:
                w0 = originals[f"{i}.{n}"].to(hidden.device)
                def hook(module,args,out,n=n,w0=w0):
                    x = args[0]
                    # undoing the activation scale so the reference sees the original input
                    if hasattr(module,"awq_scale"):
                        x = x*module.awq_scale.to(x.dtype)
                    # undoing the down scale folded into the up rows so both outputs share one scale
                    if hasattr(module,"awq_out_scale"):
                        out = out.float()*module.awq_out_scale.float()
                    ref = F.linear(x,w0,module.bias)
                    acc[n][0] += (out-ref).float().pow(2).sum().item()
                    acc[n][1] += ref.float().pow(2).sum().item()
                hooks.append(projection(layer,n).register_forward_hook(hook))
        hidden = forward_block(layer,hidden,rope,micro)
        for h in hooks:
            h.remove()
        if originals is not None:
            errors += [{"block":i,"proj":n,"rel_error":acc[n][0]/acc[n][1]} for n in names]
    nll = torch.cat([sequence_nll(model,hidden[a:a+micro],ids[a:a+micro]) for a in range(0,ids.shape[0],micro)])
    return nll.sum().item()/(ids.shape[0]*(ids.shape[1]-1)),errors
