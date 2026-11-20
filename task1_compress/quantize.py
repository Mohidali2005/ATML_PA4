import torch

from task1_compress.model import projection

# projections that share one input and therefore one activation scale
GROUPS = {"qkv":["q_proj","k_proj","v_proj"],"o":["o_proj"],"gate_up":["gate_proj","up_proj"],"down":["down_proj"]}
GROUP_OF = {n:g for g,names in GROUPS.items() for n in names}


def quantize(w,bits,group,floor):
    """Quantize a weight matrix group by group into codes with one scale and one zero point per group"""
    rows,cols = w.shape
    g = w.float().view(rows,cols//group,group)
    qmax = 2**bits-1
    wmin,wmax = g.amin(-1,keepdim=True),g.amax(-1,keepdim=True)
    # rounding the scale to half precision first so the stored value is the one used
    scale = ((wmax-wmin)/qmax).clamp(min=floor).half().float()
    zero = (-torch.round(wmin/scale)).clamp(0,qmax)
    codes = (torch.round(g/scale)+zero).clamp(0,qmax)
    return codes.to(torch.uint8).view(rows,cols),scale.half().view(rows,-1),zero.to(torch.uint8).view(rows,-1)


def dequantize(codes,scale,zero,group):
    """Turn codes back into half precision weights"""
    rows,cols = codes.shape
    g = codes.float().view(rows,cols//group,group)
    return ((g-zero.float()[...,None])*scale.float()[...,None]).view(rows,cols).half()


def fake_quant(w,bits,group,floor):
    """Quantize and dequantize a weight matrix in one step"""
    return dequantize(*quantize(w,bits,group,floor),group).to(w.dtype)


def pack_bits(codes,bits):
    """Pack codes lowest bit first so that every eight codes fill exactly bits bytes"""
    flat = codes.flatten()
    if bits == 8:
        return flat.to(torch.uint8)
    flat = torch.cat([flat,flat.new_zeros(-len(flat)%8)]).view(-1,8).long()
    word = (flat << (bits*torch.arange(8,device=flat.device))).sum(-1)
    return ((word[:,None] >> (8*torch.arange(bits,device=flat.device))) & 255).flatten().to(torch.uint8)


def unpack_bits(packed,bits,count):
    """Recover count codes from a packed byte array"""
    if bits == 8:
        return packed[:count]
    by = packed.view(-1,bits).long()
    word = (by << (8*torch.arange(bits,device=packed.device))).sum(-1)
    codes = (word[:,None] >> (bits*torch.arange(8,device=packed.device))) & (2**bits-1)
    return codes.flatten()[:count].to(torch.uint8)


def group_modules(layer,group):
    """Return the projections of one scale group"""
    return [projection(layer,n) for n in GROUPS[group]]


def search_scale(mods,bits,absmean,error_fn,cfg,grid):
    """Try every alpha on the grid and keep the channel scale that best preserves the inspected output

    the weights are swapped for their quantized versions in the original basis while the
    error is measured and are put back afterwards"""
    group,floor = cfg["quant"]["group"],cfg["quant"]["scale_floor"]
    originals = [m.weight.data.clone() for m in mods]
    errors,best = [],None
    for k in range(grid):
        alpha = k/grid
        s = absmean.pow(alpha).clamp(min=1e-4)
        s = (s/(s.max()*s.min()).sqrt()).half().float()
        for m,w,b in zip(mods,originals,bits):
            m.weight.data = (fake_quant(w.float()*s[None,:],b,group,floor).float()/s[None,:]).to(w.dtype)
        err = error_fn()
        errors.append(err)
        if best is None or err < best[0]:
            best = (err,alpha,s)
    for m,w in zip(mods,originals):
        m.weight.data = w
    return best,errors


def divide_input(module,args):
    """Divide the projection input by its stored channel scale"""
    return (args[0]/module.awq_scale.to(args[0].dtype),)


def apply_scale(layer,group,s):
    """Fold a channel scale into the block so the full precision function is unchanged

    returns a hook handle when the scale has to stay as a separate input division"""
    handle = None
    s = s.to(layer.input_layernorm.weight.device).float()
    mods = group_modules(layer,group)
    if group == "qkv":
        norm = layer.input_layernorm.weight
        norm.data = (norm.float()/s).to(norm.dtype)
    elif group == "gate_up":
        norm = layer.post_attention_layernorm.weight
        norm.data = (norm.float()/s).to(norm.dtype)
    elif group == "down":
        up = layer.mlp.up_proj
        up.weight.data = (up.weight.float()/s[:,None]).to(up.weight.dtype)
        up.awq_out_scale = s
    else:
        # keeping o_proj scales as stored half values since the grouped value heads block folding
        handle = mods[0].register_forward_pre_hook(divide_input)
    for m in mods:
        m.weight.data = (m.weight.float()*s[None,:]).to(m.weight.dtype)
        m.awq_scale = s
    return handle


def clear_marks(layer):
    """Remove the scale marks that the error measurement reads from every projection of a block"""
    for names in GROUPS.values():
        for n in names:
            m = projection(layer,n)
            for attr in ("awq_scale","awq_out_scale"):
                if hasattr(m,attr):
                    delattr(m,attr)


def group_inputs(layer,stats,rope):
    """Return the stored input statistic and output function of every scale group of a block"""
    first = {"qkv":"q_proj","o":"o_proj","gate_up":"gate_proj","down":"down_proj"}
    fns = {"qkv":lambda t:layer.self_attn(hidden_states=t,position_embeddings=rope,attention_mask=None)[0],
           "o":layer.self_attn.o_proj,"gate_up":layer.mlp,"down":layer.mlp.down_proj}
    return {g:(stats[first[g]]["inputs"],stats[first[g]]["absmean"],fns[g]) for g in GROUPS}


def quantize_model(model,cfg,bits_map,scales=None):
    """Fold the given scales into the model and fake quantize every eligible projection at its bit width

    returns the packed byte count of every projection"""
    group,floor = cfg["quant"]["group"],cfg["quant"]["scale_floor"]
    sizes = {}
    for i,layer in enumerate(model.model.layers):
        if scales is not None:
            for g in GROUPS:
                apply_scale(layer,g,scales[(i,g)])
        for name in cfg["eligible"]:
            m = projection(layer,name)
            bits = bits_map[(i,name)]
            codes,scale,zero = quantize(m.weight.data,bits,group,floor)
            sizes[(i,name)] = pack_bits(codes,bits).numel()+scale.numel()*2+zero.numel()
            m.weight.data = dequantize(codes,scale,zero,group).to(m.weight.dtype)
    return sizes
