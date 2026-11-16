"""Jointly trained evidence prompts and state-dependent H/N/E computations."""
from pathlib import Path
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer,StandardScaler
from chronos.chronos2.model import Chronos2EncoderBlock,Chronos2Encoder
from chronos.chronos2.config import Chronos2CoreConfig
from .hne_routed import LoRALinear

ROOT=Path(__file__).resolve().parents[1]
PATHS=np.array(list(itertools.permutations(range(3))))


class EvidenceBasis:
    def fit(self,x,groups):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler=StandardScaler();z=np.clip(self.scaler.fit_transform(self.imputer.fit_transform(x)),-8,8)
        self.spline=SplineTransformer(n_knots=3,degree=2,include_bias=False,extrapolation='linear')
        b=self.spline.fit_transform(z);self.basis_scaler=StandardScaler().fit(b)
        self.repeats=b.shape[1]//x.shape[1]
        selected=[groups=='H',groups=='N',np.isin(groups,['E','NE'])]
        self.indices=[np.flatnonzero(np.repeat(mask,self.repeats)) for mask in selected]
        self.widths=[len(index) for index in self.indices]
        return self
    def transform(self,x):
        z=np.clip(self.scaler.transform(self.imputer.transform(x)),-8,8)
        b=self.basis_scaler.transform(self.spline.transform(z))
        return [np.asarray(b[:,index]/np.sqrt(len(index)),np.float32) for index in self.indices]


class HNEJoint(nn.Module):
    def __init__(self,config,widths,variant='joint'):
        super().__init__();self.variant=variant;self.config=config;d=config['hidden_width']
        payload=torch.load(ROOT/'tools/models/chronos-2/tail.pt',weights_only=False,map_location='cpu')
        core=Chronos2CoreConfig(**payload['config']);core._attn_implementation='sdpa'
        self.tail=Chronos2EncoderBlock(core);self.tail.load_state_dict(payload['block'])
        for p in self.tail.parameters():p.requires_grad_(False)
        for layer in self.tail.layer[:2]:
            for name in ['q','v']:setattr(layer.self_attention,name,LoRALinear(getattr(layer.self_attention,name),config['lora_rank']))
        self.evidence=nn.ModuleList([nn.Sequential(nn.Linear(w,d),nn.GELU()) for w in widths])
        self.evidence_coefficients=nn.ParameterList([nn.Parameter(torch.zeros(w)) for w in widths])
        self.gates=nn.ModuleList([nn.Linear(d,w//3) for w in widths])
        for gate in self.gates:nn.init.zeros_(gate.weight);nn.init.zeros_(gate.bias)
        self.prompt_up=nn.Linear(d,768,bias=False)
        self.prompt_types=nn.Parameter(torch.randn(3,768)*.01)
        self.readout=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,d),nn.GELU())
        self.root=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.LayerNorm(d))
        self.recurrence=nn.GRUCell(d,d)
        self.router=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,3))
        self.risk=nn.Linear(d,1);self.future=nn.Linear(d,10)
        self.register_buffer('paths',torch.tensor(PATHS,dtype=torch.long))

    def encode(self,x,mask,reference,reference_mask,basis):
        numeric=torch.stack([encoder(b) for encoder,b in zip(self.evidence,basis)],1)
        if self.variant=='coordinates_only':
            root=self.root(torch.cat([numeric[:,0],numeric[:,1],numeric[:,2]],-1))
            return root,numeric
        b,c,t,_=x.shape
        prompts=self.prompt_up(numeric)+self.prompt_types
        full=torch.cat([x,prompts[:,None].expand(-1,c,-1,-1)],2);length=t+3
        observed=torch.cat([mask,torch.ones((b,c,3),dtype=torch.bool,device=x.device)],2)
        flat=full.reshape(b*c,length,768);fm=observed.reshape(b*c,length).to(flat.dtype)
        groups=torch.arange(b,device=x.device).repeat_interleave(c)
        time_mask=Chronos2Encoder._expand_and_invert_time_attention_mask(fm,flat.dtype)
        group_mask=Chronos2Encoder._construct_and_invert_group_time_mask(groups,fm,flat.dtype)
        tail=self.tail(flat,position_ids=torch.arange(length,device=x.device)[None],attention_mask=time_mask,group_time_mask=group_mask).hidden_states.reshape(b,c,length,768)
        current_mask=mask[:,:-1,:,None]
        reference_mask=reference_mask[:,:-1,:,None]
        context_mask=mask[:,-1,:,None]
        current=(self.readout(tail[:,:-1,:t])*current_mask).sum((1,2))/current_mask.sum((1,2))
        remembered=(self.readout(reference[:,:-1])*reference_mask).sum((1,2))/reference_mask.sum((1,2))
        context=(self.readout(tail[:,-1,:t])*context_mask).sum(1)/context_mask.sum(1)
        enriched=self.readout(tail[:,:-1,-3:]).mean(1)+numeric
        return self.root(torch.cat([current,remembered,context],-1)),enriched

    def forward(self,x,mask,reference,reference_mask,stats,basis):
        if self.variant=='no_coordinates':basis=[torch.zeros_like(v) for v in basis]
        root,evidence=self.encode(x,mask,reference,reference_mask,basis)
        # All six orders share the same encoded input and deterministic decision nodes.
        with torch.autocast(x.device.type,enabled=False):
            evidence=evidence.float();state=root.float()[:,None].expand(-1,6,-1)
            b=len(state);log_policy=torch.zeros((b,6),device=x.device);accumulated=torch.zeros_like(log_policy)
            choices=[];used=torch.zeros((6,3),dtype=torch.bool,device=x.device)
            global_context=evidence.mean(1)[:,None].expand(-1,6,-1)
            for step in range(3):
                logits=self.router(torch.cat([state,global_context],-1)).masked_fill(used[None],-1e4)
                action=self.paths[:,step];choices.append(logits.argmax(-1))
                log_policy=log_policy+logits.log_softmax(-1).gather(-1,action[None,:,None].expand(b,-1,1)).squeeze(-1)
                increment=torch.zeros_like(accumulated)
                for a in range(3):
                    gate=1+self.config['gate_amplitude']*torch.tanh(self.gates[a](state))
                    score=(basis[a].float()[:,None]*gate.repeat_interleave(3,-1)*self.evidence_coefficients[a]).sum(-1)
                    increment=increment+score*(action==a)[None]
                accumulated=accumulated+increment
                evidence_at_step=evidence[:,action]
                state=self.recurrence(evidence_at_step.reshape(-1,evidence.shape[-1]),state.reshape(-1,state.shape[-1])).reshape(b,6,-1)
                used=used.clone();used.scatter_(1,action[:,None],True)
            candidate_logits=accumulated+self.risk(state).squeeze(-1)
            decisions=torch.stack(choices,-1)
            selected=(decisions==self.paths[None]).all(-1).float().argmax(-1)
            if self.variant=='fixed':selected=torch.zeros_like(selected)
            ix=torch.arange(b,device=x.device);policy=log_policy.exp()
            return dict(logits=candidate_logits[ix,selected],all_logits=candidate_logits,policy=policy,
                log_policy=log_policy,selected=selected,routes=self.paths[selected],future=self.future(state).reshape(b,6,5,2))


def joint_objective(model,out,y,weights,person,future,config):
    losses=F.binary_cross_entropy_with_logits(out['all_logits'],y[:,None].expand(-1,6),reduction='none')
    if model.variant=='fixed':
        path_weights=F.one_hot(torch.zeros(len(y),dtype=torch.long,device=y.device),6).float();routing=losses.sum()*0
    else:
        path_weights=(1-config['path_exploration_mass'])*out['policy']+config['path_exploration_mass']/6
        relative=losses.detach()-losses.detach().mean(1,keepdim=True)
        scale=losses.detach().std(1,keepdim=True,unbiased=False).clamp_min(config['path_loss_scale_floor'])
        teacher=(-relative/scale).softmax(-1)
        routing=(-(teacher*out['log_policy']).sum(1)*weights).mean()
    event=((path_weights*losses).sum(1)*weights).mean()
    risk=(path_weights*out['all_logits']).sum(1)
    pairs=(person[:,None]==person[None,:])&(y[:,None]>0)&(y[None,:]==0)
    ranking=(F.softplus(-(risk[:,None]-risk[None,:]))*pairs).sum()/pairs.sum().clamp_min(1)
    observed=torch.isfinite(future);target=torch.nan_to_num(future)[:,None].expand(-1,6,-1,-1)
    future_loss=F.smooth_l1_loss(out['future'],target,reduction='none')*observed[:,None]
    future_loss=future_loss.sum((2,3))/observed.sum((1,2)).clamp_min(1)[:,None]
    prediction=((path_weights*future_loss).sum(1)*weights).mean()
    penalty=sum((v*v).sum() for v in model.evidence_coefficients)
    return event+config['path_supervision_weight']*routing+config['within_person_rank_weight']*ranking+config['future_loss_weight']*prediction+config['evidence_l2']*penalty
