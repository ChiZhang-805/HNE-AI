"""HNE inside a pretrained Chronos-2 encoder: learned H/N/E computation paths."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from chronos.chronos2.model import Chronos2EncoderBlock, Chronos2Encoder
from chronos.chronos2.config import Chronos2CoreConfig

ROOT=Path(__file__).resolve().parents[1]


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base=base
        self.a=nn.Parameter(torch.empty(rank,base.in_features))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.a),self.b)


def mlp(nin,nout,width=96):
    return nn.Sequential(nn.Linear(nin,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,nout))


class HNERouted(nn.Module):
    def __init__(self,config,variant='learned'):
        super().__init__()
        self.variant='plain' if variant=='native' else variant
        self.unique_actions=False
        self.float32_router=False
        self.evidence_intervention=None
        self.steps=config['reasoning_steps']
        self.temperature=config['router_temperature']
        payload=torch.load(ROOT/'tools/models/chronos-2/tail.pt',weights_only=False,map_location='cpu')
        core=Chronos2CoreConfig(**payload['config'])
        core._attn_implementation='sdpa'
        self.tail=Chronos2EncoderBlock(core)
        self.tail.load_state_dict(payload['block'])
        for p in self.tail.parameters():p.requires_grad_(False)
        for layer in self.tail.layer[:2]:
            for name in ['q','v']:
                linear=getattr(layer.self_attention,name)
                setattr(layer.self_attention,name,LoRALinear(linear,config['lora_rank']))
        d=config['hidden_width']
        self.norm=nn.LayerNorm(768)
        self.down=nn.Linear(768,d)
        self.stats=nn.Linear(3,d)
        self.h_attn=nn.MultiheadAttention(d,4,batch_first=True,dropout=.1)
        self.n_attn=nn.MultiheadAttention(d,4,batch_first=True,dropout=.1)
        self.e_attn=nn.MultiheadAttention(d,4,batch_first=True,dropout=.1)
        self.h_update=mlp(3*d,d,d)
        self.n_update=mlp(2*d,d,d)
        self.e_update=mlp(2*d,d,d)
        self.step_norm=nn.LayerNorm(d)
        self.router=mlp(3*d,3,d)
        self.step_embedding=nn.Parameter(torch.randn(self.steps,d)*.01)
        self.up=nn.Linear(d,768,bias=False)
        nn.init.normal_(self.up.weight,std=.002)
        self.event_head=mlp(2*d,1,d)
        self.path_head=mlp(2*d,1,d)
        self.future_head=mlp(d,2,d)
        self.readout=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,d),nn.GELU())

    @staticmethod
    def pool(z):
        # Channel permutation invariant, shared across source and target domains.
        v=z.mean(-2)
        return torch.cat([v.mean(-2),v.amax(-2)],-1)

    def forward(self,x,mask,reference,reference_mask,stats,epoch=20,force_path=None):
        b,c,t,_=x.shape
        state_count=c-1
        if self.variant=='no_environment' or self.evidence_intervention=='context_zero':
            x=x.clone();x[:,-1]=0
            reference=reference.clone();reference[:,-1]=0
            stats=stats.clone();stats[:,-1]=0
        z0=self.down(self.norm(x))+self.stats(stats).unsqueeze(-2)
        memory=self.down(self.norm(reference))
        if self.variant=='no_history' or self.evidence_intervention=='reference_zero':memory=torch.zeros_like(memory)
        z=z0[:,:state_count]
        env=z0[:,-1]
        if self.evidence_intervention=='context_zero':env=torch.zeros_like(env)
        ref=memory[:,:state_count]
        env_summary=env.mean(1)
        ref_summary=ref.mean((1,2))
        root_context=torch.cat([(z+self.step_embedding[0]).mean((1,2)),ref_summary,env_summary],-1)
        routes=[];route_logits=[];candidate_logits=[];stage_logits=[];route_features=[]
        if self.variant!='plain':
            for step in range(self.steps):
                q=z+self.step_embedding[step]
                # A forced evaluation path needs only its selected update.
                action=None if force_path is None else int(force_path[step])
                h=n=e=torch.zeros_like(z)
                if action is None or action==0:
                    hz=self.h_attn(q.flatten(0,1),ref.flatten(0,1),ref.flatten(0,1),
                                   key_padding_mask=~reference_mask[:,:state_count].flatten(0,1),need_weights=False)[0].reshape_as(z)
                    h=self.h_update(torch.cat([z,hz,z-hz],-1))
                if action is None or action==1:
                    nq=q.permute(0,2,1,3).reshape(b*t,state_count,-1)
                    nv=self.n_attn(nq,nq,nq,need_weights=False)[0].reshape(b,t,state_count,-1).permute(0,2,1,3)
                    n=self.n_update(torch.cat([z,nv],-1))
                if action is None or action==2:
                    ev=self.e_attn(q.flatten(1,2),env,env,key_padding_mask=~mask[:,-1],need_weights=False)[0].reshape_as(z)
                    e=self.e_update(torch.cat([z,ev],-1))
                updates=torch.stack([h,n,e],1)
                candidates=self.step_norm(z[:,None]+.25*updates)
                if self.evidence_intervention=='network_cut':
                    candidates=candidates.clone();candidates[:,1]=z
                route_feature=torch.cat([q.mean((1,2)),ref_summary,env_summary],-1)
                route_features.append(route_feature)
                if self.float32_router:
                    with torch.autocast(x.device.type,enabled=False):logits=self.router(route_feature.float())
                else:logits=self.router(route_feature)
                if self.unique_actions and routes:
                    logits=logits.clone()
                    for previous in routes:logits.scatter_(-1,previous[:,None],-1e4)
                disabled={'no_history':0,'no_network':1,'no_environment':2}.get(self.variant)
                if disabled is not None:
                    logits=logits.clone();logits[:,disabled]=-1e4
                if force_path is not None:
                    chosen=F.one_hot(torch.full((b,),int(force_path[step]),device=x.device),3).float()
                elif self.variant=='fixed':
                    chosen=F.one_hot(torch.full((b,),step%3,device=x.device),3).float()
                elif self.variant=='random':
                    chosen=F.one_hot(torch.randint(3,(b,),device=x.device),3).float()
                elif self.training:
                    if epoch<3:
                        chosen=F.one_hot(torch.randint(3,(b,),device=x.device),3).float()
                    else:
                        chosen=F.gumbel_softmax(logits,tau=self.temperature,hard=True,dim=-1)
                else:
                    chosen=F.one_hot(logits.argmax(-1),3).float()
                cp=self.pool(candidates)
                candidate_logits.append(self.path_head(cp).squeeze(-1))
                z=(candidates*chosen[:,:,None,None,None]).sum(1)
                routes.append(chosen.argmax(-1));route_logits.append(logits)
                stage_logits.append(self.path_head(self.pool(z)).squeeze(-1))
            correction=torch.cat([self.up(z-z0[:,:state_count]),torch.zeros_like(x[:,-1:])],1)
            x=x+correction
        else:
            z=z0[:,:state_count]
        flat=x.reshape(b*c,t,768)
        fm=mask.reshape(b*c,t).to(flat.dtype)
        groups=torch.arange(b,device=x.device).repeat_interleave(c)
        if self.variant=='no_network' or self.evidence_intervention=='network_cut':groups=torch.arange(b*c,device=x.device)
        time_mask=Chronos2Encoder._expand_and_invert_time_attention_mask(fm,flat.dtype)
        group_mask=Chronos2Encoder._construct_and_invert_group_time_mask(groups,fm,flat.dtype)
        tail=self.tail(flat,position_ids=torch.arange(t,device=x.device)[None],attention_mask=time_mask,
                       group_time_mask=group_mask).hidden_states.reshape(b,c,t,768)
        representation=self.readout(tail[:,:state_count])+z
        pooled=self.pool(representation)
        result={'logits':self.event_head(pooled).squeeze(-1),
                'future':self.future_head(representation.mean(-2)),
                'readout_features':pooled,'routing_context':root_context}
        if routes:
            result.update(routes=torch.stack(routes,1),route_logits=torch.stack(route_logits,1),
                          candidate_logits=torch.stack(candidate_logits,1),stage_logits=torch.stack(stage_logits,1),
                          route_features=torch.stack(route_features,1))
        return result


class HNEContextPaths(HNERouted):
    """Preserve reference context and learn the order of all three H/N/E updates."""
    def __init__(self,config,variant='context_path'):
        super().__init__(config,'plain' if variant=='context_plain' else 'fixed' if variant=='context_fixed' else 'learned')
        self.context_variant=variant;self.unique_actions=True;self.float32_router=True
        d=config['hidden_width']
        self.context_head=mlp(3*d,1,2*d if variant=='context_plain' else d)
        self.context_correction=nn.Linear(2*d,1,bias=False)
        nn.init.zeros_(self.context_correction.weight)

    def forward(self,*args,**kwargs):
        output=super().forward(*args,**kwargs)
        with torch.autocast(output['routing_context'].device.type,enabled=False):
            logits=self.context_head(output['routing_context'].float()).squeeze(-1)
            if self.context_variant!='context_plain':
                logits=logits+self.context_correction(output['readout_features'].float()).squeeze(-1)
        output['logits']=logits
        return output


class HNEContextFree(HNERouted):
    """Retain reference context while learning both update order and repetition."""
    def __init__(self,config,variant='context_free'):
        super().__init__(config,'plain' if variant=='context_free_plain' else 'fixed' if variant=='context_free_fixed' else 'learned')
        self.context_variant=variant;self.float32_router=True
        d=config['hidden_width']
        self.context_head=mlp(3*d,1,2*d if variant=='context_free_plain' else d)
        self.risk_scale=nn.Parameter(torch.zeros(()))

    def forward(self,*args,**kwargs):
        output=super().forward(*args,**kwargs)
        with torch.autocast(output['routing_context'].device.type,enabled=False):
            logits=self.context_head(output['routing_context'].float()).squeeze(-1)
            if self.context_variant!='context_free_plain':
                risk=torch.logit(output['logits'].float().sigmoid().clamp(1e-6,1-1e-6))
                logits=logits+self.risk_scale*risk
        output['logits']=logits
        return output


def make_hne(config,variant='learned'):
    if variant.startswith('context_free'):return HNEContextFree(config,variant)
    return HNEContextPaths(config,variant) if variant.startswith('context_') else HNERouted(config,variant)


def objective(output,y,weight,target,config,variant):
    event=F.binary_cross_entropy_with_logits(output['logits'],y,reduction='none')
    observed=torch.isfinite(target)
    delta=F.smooth_l1_loss(output['future'],torch.nan_to_num(target),reduction='none')
    future=(delta*observed).sum((1,2))/observed.sum((1,2)).clamp_min(1)
    total=event+config['future_loss_weight']*future
    if 'candidate_logits' in output:
        cy=y[:,None,None].expand_as(output['candidate_logits'])
        candidate=F.binary_cross_entropy_with_logits(output['candidate_logits'],cy,reduction='none')
        total=total+config['auxiliary_event_weight']*candidate.mean((1,2))
        if variant not in ['fixed','random','no_path_loss']:
            # Training labels supervise local action utility; inference uses only the router.
            utility=-candidate.detach()/.25
            disabled={'no_history':0,'no_network':1,'no_environment':2}.get(variant)
            if disabled is not None:utility[:,:,disabled]=-1e4
            teacher=utility.softmax(-1)
            path=-(teacher*output['route_logits'].log_softmax(-1)).sum(-1).mean(-1)
            total=total+config['path_loss_weight']*path
    return (total*weight).mean()


class RoutedData:
    def __init__(self,dataset,native=False):
        folder=ROOT/'data/hne_routed'/dataset
        self.arrays=np.load(folder/'arrays.npz')
        self.history=np.load(folder/('history_native_prefix.npy' if native else 'history_prefix.npy'),mmap_mode='r')
        self.reference=np.load(folder/'reference_prefix.npy',mmap_mode='r')
        self.mask=np.load(folder/'history_mask.npy')
        self.rmask=np.load(folder/'reference_mask.npy')
    def batch(self,ids,device='cuda'):
        pi=self.arrays['person_index'][ids]
        return [torch.as_tensor(np.asarray(v).copy(),device=device) for v in
                (self.history[ids].astype(np.float32),self.mask[ids],self.reference[pi].astype(np.float32),
                 self.rmask[pi],self.arrays['stats'][ids])]
