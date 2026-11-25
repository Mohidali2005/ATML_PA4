import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low rank correction"""

    def __init__(self,base,rank,alpha):
        super().__init__()
        self.base = base
        self.scale = alpha/rank
        device = base.weight.device
        # keeping the adapters in fp32 while the frozen weight stays in half precision
        self.A = nn.Parameter(torch.empty(rank,base.in_features,device=device,dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features,rank,device=device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))

    def forward(self,x):
        update = F.linear(F.linear(x.to(self.A.dtype),self.A),self.B)
        return self.base(x)+self.scale*update.to(x.dtype)


def add_lora(model,cfg,seed):
    """Freeze the model and wrap every target projection using one seed for the initial factors"""
    spec = cfg["distill"]["lora"]
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)
    for layer in model.model.layers:
        for name in spec["targets"]:
            setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name),spec["rank"],spec["alpha"]))
    return lora_parameters(model)


def lora_modules(model):
    """Name and module of every wrapped projection"""
    return [(n,m) for n,m in model.named_modules() if isinstance(m,LoRALinear)]


def lora_parameters(model):
    """All trainable factors"""
    return [p for _,m in lora_modules(model) for p in (m.A,m.B)]


def checksum(model):
    """Exact sums that identify the current factors"""
    mods = lora_modules(model)
    return {"a_sum":sum(m.A.detach().double().sum().item() for _,m in mods),
            "a_abs_sum":sum(m.A.detach().double().abs().sum().item() for _,m in mods),
            "b_abs_sum":sum(m.B.detach().double().abs().sum().item() for _,m in mods)}


def export_factors(model):
    """Copy the factors to the cpu"""
    return {n:(m.A.detach().cpu().clone(),m.B.detach().cpu().clone()) for n,m in lora_modules(model)}


def load_factors(model,state):
    """Write saved factors into a wrapped model"""
    for n,m in lora_modules(model):
        m.A.data.copy_(state[n][0])
        m.B.data.copy_(state[n][1])


def merge_factors(model,state,scale):
    """Add the saved corrections into the matching weights of a plain model"""
    modules = dict(model.named_modules())
    for name,(a,b) in state.items():
        w = modules[name].weight
        delta = scale*(b.to(w.device).float()@a.to(w.device).float())
        w.data = (w.data.float()+delta).to(w.dtype)
