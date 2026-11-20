import torch


def lowest_mask(score,total):
    """Mark the lowest scoring weights of every row so that exactly total weights are marked

    rows share the total as evenly as possible and ties keep their column order"""
    rows,cols = score.shape
    counts = torch.full((rows,),total//rows,dtype=torch.long,device=score.device)
    counts[:total%rows] += 1
    order = torch.argsort(score,dim=1,stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1,order,torch.arange(cols,device=score.device).expand(rows,cols).contiguous())
    mask = rank < counts[:,None]
    assert int(mask.sum()) == total
    return mask


def two_four_mask(score):
    """Mark the two lowest scoring weights in every run of four input channels"""
    rows,cols = score.shape
    order = torch.argsort(score.view(rows,cols//4,4),dim=-1,stable=True)
    mask = torch.zeros(rows,cols//4,4,dtype=torch.bool,device=score.device)
    mask.scatter_(-1,order[...,:2],True)
    return mask.view(rows,cols)


def wanda_score(weight,sumsq):
    """Weight magnitude times the norm of its input channel"""
    return weight.abs().float()*sumsq.sqrt()[None,:]


def magnitude_score(weight):
    """Plain weight magnitude"""
    return weight.abs().float()


def projection_totals(unit_sparsity,numels,total):
    """Turn a sparsity per projection into whole weight counts that add up to the global total

    the targets are first rescaled to sum to the total and the fractional parts are then
    handed out by the largest remainder rule"""
    raw = [s*n for s,n in zip(unit_sparsity,numels)]
    raw = [r*total/sum(raw) for r in raw]
    counts = [int(r) for r in raw]
    left = total-sum(counts)
    assert 0 <= left <= len(raw)
    order = sorted(range(len(raw)),key=lambda i:(-(raw[i]-counts[i]),i))
    for i in order[:left]:
        counts[i] += 1
    return counts
