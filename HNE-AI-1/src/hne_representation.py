"""One HNE representation for arbitrary state/context counts and sampling rates."""
from dataclasses import dataclass

import numpy as np

from .hne_features import series_stats, graph_stats, pool_nodes, correlation_matrix


def select_reference_channels(states,names,reference=56,budget=32,minimum=28):
    """A common reference-only coordinate budget, independent of future outcomes."""
    if states.shape[1]<=budget:return np.arange(states.shape[1])
    prefix=np.asarray(states[:reference],dtype=float)
    counts=np.isfinite(prefix).sum(0)
    eligible=np.flatnonzero(counts>=minimum)
    eligible=sorted(eligible,key=lambda j:str(names[j]))
    rho,_=correlation_matrix(prefix,prefix,minimum=12)
    similarity=np.where(np.isfinite(rho),abs(rho),1.)
    selected=[]
    while eligible and len(selected)<budget:
        value={j:counts[j]/reference*(1-max(similarity[j,selected]) if selected else 1.) for j in eligible}
        chosen=max(eligible,key=lambda j:value[j])
        selected.append(chosen);eligible.remove(chosen)
    return np.asarray(selected,dtype=int)


@dataclass(frozen=True)
class Sampling:
    reference: int
    scales: tuple[int,int,int]
    lags: tuple[int,int,int]
    memories: tuple[int,int]
    block: int
    minimum: int
    reference_minimum: int
    pair_minimum: int

    @classmethod
    def hourly(cls):
        return cls(672,(24,72,168),(0,6,24),(1,24),168,8,24,12)

    @classmethod
    def daily(cls):
        return cls(56,(8,16,32),(0,1,7),(1,7),14,4,14,6)


def lag_context(context, lags):
    columns=[]
    for lag in lags:
        current=np.full_like(context,np.nan)
        if lag:current[lag:]=context[:-lag]
        else:current[:]=context
        columns.append(current)
    return np.concatenate(columns,axis=1)


def context_residuals(states,context,spec,ridge=10.):
    """Masked reference ridge; the same calculation is used for one or many E."""
    lagged=lag_context(context,spec.lags)
    support=np.isfinite(lagged[:spec.reference]).sum(0)>=4
    observed=np.isfinite(lagged)&support[None]
    values=np.where(observed,np.clip(lagged,-8,8),0.)
    design=np.column_stack((np.ones(len(states)),values,observed.astype(float)))
    any_context=observed.any(1)
    reference_design=design[:spec.reference]
    adjusted=states.copy()
    effect=np.zeros((states.shape[1],2),np.float32)
    # Ridge is expressed per observation so a higher sampling rate does not
    # silently weaken the context prior.
    penalty=np.eye(design.shape[1])*ridge
    penalty[0,0]=0

    def fit(mask,target):
        u=reference_design[mask]
        x=u[:,1:]; y=target[mask]
        xm=x.mean(0);ym=y.mean();xc=x-xm;yc=y-ym
        if x.shape[1]>len(x):
            slope=xc.T@np.linalg.solve(xc@xc.T+len(x)*ridge*np.eye(len(x)),yc)
        else:
            slope=np.linalg.solve(xc.T@xc+len(x)*ridge*np.eye(x.shape[1]),xc.T@yc)
        return np.r_[ym-xm@slope,slope]

    for j in range(states.shape[1]):
        observed_state=np.isfinite(states[:spec.reference,j])
        valid=observed_state & any_context[:spec.reference]
        if valid.sum()<spec.reference_minimum:
            continue
        target=np.clip(states[:spec.reference,j],-12,12)
        beta=fit(valid,target)
        adjusted[any_context,j]-=design[any_context]@beta
        for start in range(0,spec.reference,spec.block):
            held=np.arange(spec.reference)//spec.block==start//spec.block
            train=valid&~held
            rows=held & any_context[:spec.reference]
            fold_beta=fit(train,target) if train.sum()>=spec.reference_minimum else np.zeros_like(beta)
            adjusted[:spec.reference,j][rows]=states[:spec.reference,j][rows]-reference_design[rows]@fold_beta
        effect[j]=[np.linalg.norm(beta[1:1+lagged.shape[1]])/np.sqrt(max(1,support.sum())),valid.mean()]
    return adjusted,effect


def encode_person(states,contexts,cutoffs,spec:Sampling,context_ridge=10.,conditioned=None):
    """Cutoff is exclusive; future samples never enter features or reference fits."""
    states=np.asarray(states,dtype=np.float64)
    contexts=np.asarray(contexts,dtype=np.float64)
    adjusted,effect=context_residuals(states,contexts,spec,ridge=context_ridge) if conditioned is None else conditioned
    raw_ref=states[:spec.reference];adj_ref=adjusted[:spec.reference]
    kwargs=dict(memory_lags=spec.memories,minimum=spec.minimum,reference_minimum=spec.reference_minimum,
                pair_minimum=spec.pair_minimum)
    all_nodes,all_pooled=[],[]
    for cutoff in cutoffs:
        parts,raw_graph,adj_graph,context_scales=[],[],[],[]
        for length in spec.scales:
            raw=states[cutoff-length:cutoff];adj=adjusted[cutoff-length:cutoff]
            parts.extend([np.stack([series_stats(raw[:,j],raw_ref[:,j],**kwargs) for j in range(states.shape[1])]),
                          np.stack([series_stats(adj[:,j],adj_ref[:,j],**kwargs) for j in range(states.shape[1])])])
            raw_graph.extend(graph_stats(raw,raw_ref,lags=spec.lags,pair_minimum=spec.pair_minimum))
            adj_graph.extend(graph_stats(adj,adj_ref,lags=spec.lags,pair_minimum=spec.pair_minimum))
            context_scales.append(np.stack([series_stats(contexts[cutoff-length:cutoff,j],
                contexts[:spec.reference,j],**kwargs) for j in range(contexts.shape[1])]))
        nodes=np.column_stack((*parts,effect))
        h=pool_nodes(nodes[:,np.r_[0:8,16:24,32:40]])
        environment_nodes=pool_nodes(nodes[:,np.r_[8:16,24:32,40:50]])
        # Shared coordinates retain their meaning as the number of contexts varies.
        contexts_by_channel=np.concatenate(context_scales,axis=1)
        valid=np.isfinite(contexts_by_channel)
        mean_context=np.divide(np.where(valid,contexts_by_channel,0).sum(0),valid.sum(0),
            out=np.full(24,np.nan),where=valid.sum(0)>0)
        e=np.r_[environment_nodes,mean_context]
        all_nodes.append(nodes)
        all_pooled.append(np.r_[h,raw_graph,e,adj_graph])
    groups=np.array(['H']*96+['N']*36+['E']*128+['NE']*36)
    return np.asarray(all_nodes,np.float32),np.asarray(all_pooled,np.float32),groups
