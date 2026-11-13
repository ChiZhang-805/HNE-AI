"""Windowed DNB: SD inside times absolute PCC inside divided by PCC outside."""
from functools import lru_cache
from itertools import combinations

import numpy as np


@lru_cache(None)
def subsets(n,size):
    return np.array(list(combinations(range(n),size)),dtype=int)


def score_window(values,minimum=12):
    valid=np.isfinite(values)
    # A fixed coverage-only cap makes the combinatorial module search practical.
    order=np.argsort(-valid.sum(0),kind='stable')[:10]
    values=values[:,order];valid=valid[:,order]
    n=values.shape[1]
    selected=None
    for size in range(min(n,5),2,-1):
        candidates=subsets(n,size)
        masks=valid[:,candidates].all(2)
        support=masks.sum(0)
        for index in np.argsort(-support,kind='stable'):
            if support[index]<minimum:break
            data=values[masks[:,index]][:,candidates[index]]
            sd=data.std(0,ddof=1)
            if (sd>1e-8).all():
                selected=(data,sd);break
        if selected is not None:break
    if selected is None:return np.nan
    data,sd=selected
    corr=np.corrcoef(data,rowvar=False);best=0.
    for size in (2,3):
        if size>=len(sd):continue
        for module in combinations(range(len(sd)),size):
            outside=[j for j in range(len(sd)) if j not in module]
            inside=list(combinations(module,2))
            pin=np.mean([abs(corr[a,b]) for a,b in inside])
            pout=np.mean(abs(corr[np.ix_(module,outside)]))
            value=sd[list(module)].mean()*pin/(pout+1e-6)
            best=max(best,float(value))
    return best
