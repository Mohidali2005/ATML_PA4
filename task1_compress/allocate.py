import numpy as np


def knapsack(costs,weights,capacity,exact=False):
    """Pick one option per item to minimise the total cost under a total weight limit

    costs and weights hold one list per item with one entry per option and the weights are whole numbers
    with exact set the total weight has to equal the capacity
    returns the chosen option of every item"""
    best = np.full(capacity+1,np.inf)
    best[0] = 0.0
    picks = np.zeros((len(costs),capacity+1),dtype=np.int16)
    for i,(cost,weight) in enumerate(zip(costs,weights)):
        nxt = np.full(capacity+1,np.inf)
        for j,(c,w) in enumerate(zip(cost,weight)):
            cand = np.full(capacity+1,np.inf)
            cand[w:] = best[:capacity+1-w]+c
            better = cand < nxt
            nxt[better] = cand[better]
            picks[i,better] = j
        best = nxt
    end = capacity if exact else int(np.argmin(best))
    assert np.isfinite(best[end]),"no allocation fits the budget"
    choice = []
    for i in range(len(costs)-1,-1,-1):
        j = int(picks[i,end])
        choice.append(j)
        end -= weights[i][j]
    return choice[::-1]


def rescale(s,target,lo,hi):
    """Shift every value by one amount so the clipped mean hits the target"""
    s = np.asarray(s,dtype=float)
    a,b = -1.0,1.0
    for _ in range(100):
        mid = (a+b)/2
        if np.clip(s+mid,lo,hi).mean() < target:
            a = mid
        else:
            b = mid
    return np.clip(s+(a+b)/2,lo,hi)


def owl(ratios,target,lam,lo,hi):
    """Map block outlier ratios to sparsities so blocks with more outliers keep more weights"""
    r = np.asarray(ratios,dtype=float)
    r = (r-r.min())/(r.max()-r.min())*lam*2
    density = r-r.mean()+(1-target)
    return rescale(1-density,target,lo,hi)


def interpolate(levels,deltas,grid):
    """Read a measured loss curve at every grid point by straight lines between measured levels"""
    return np.interp(grid,levels,deltas)
