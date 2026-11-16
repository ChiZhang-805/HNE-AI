"""Bounded CPU checks of design-critical interfaces; no model training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
import sys, json, time, hashlib, inspect
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.layers import MHA
from precursor_guidance import ConditionalPredictor, EvidenceBias, UtilityGate, innovation_features, bounded_centered_bias, observed_utility
from chronos_bridge import encode_packed_segments, preserve_mask_add_bias
from event_readout import FamilyEventHead

start=time.monotonic(); torch.set_num_threads(2); torch.manual_seed(20261114)
checks={}
def check(name, condition):
    checks[name]=bool(condition)
    if not condition: raise AssertionError(name)
def count(m): return sum(p.numel() for p in m.parameters())
s=ConditionalPredictor().eval()
v=torch.randn(2,4,8,16); obs=torch.rand_like(v)>.12; clock=torch.randn(2,8,4)
state=torch.tensor([[1,1,1,0]]*2,dtype=torch.bool); family=torch.tensor([[0,0,1,2]]*2)
pred=s(v,obs,clock,state,family)
changed=v.clone(); changed[:,:,5:]=torch.randn_like(changed[:,:,5:])*50
after=s(changed,obs,clock,state,family)
check('S_strict_prediction_causality',all(torch.allclose(pred['views'][k][0][:,:,:6],after['views'][k][0][:,:,:6],atol=2e-6) for k in ('H','HN','HNE')))
perm=torch.tensor([3,1,0,2]); swap=s(v[:,perm],obs[:,perm],clock,state[:,perm],family[:,perm])
check('S_channel_permutation',all(torch.allclose(pred['views'][k][0][:,perm],swap['views'][k][0],atol=2e-6) for k in ('H','HN','HNE')))
junk=s(v.masked_fill(~obs,1e5),obs,clock,state,family)
check('S_missing_fill_invariance',torch.equal(pred['views']['HNE'][0],junk['views']['HNE'][0]))
sparse=obs.clone(); sparse[:,3]=False; sparse[:,3,:,0]=True
sp=s(v,sparse,clock,state,family)
check('sparse_observed_context_is_usable',sp['pair_allowed'][:,1:,0,3].all())
node,valid,edge,edge_valid=innovation_features(pred,v,obs)
check('S_evidence_detached',not node.requires_grad and not edge.requires_grad)
raw=torch.randn(3,5,5)*100; allowed=torch.rand(3,5,5)>.3; allowed[0,0]=False
b=bounded_centered_bias(raw,allowed)
check('centered_bias_bound_and_empty_row',b.abs().max()<=2.000001 and torch.isfinite(b).all() and b[0,0].abs().sum()==0)
cfg=Chronos2CoreConfig(d_model=32,d_kv=8,d_ff=64,num_layers=2,num_heads=4,dropout_rate=0.,attn_implementation='sdpa',reg_token_id=1,chronos_config=dict(context_length=256,input_patch_size=16,input_patch_stride=16,output_patch_size=16,quantiles=[.1,.5,.9],use_reg_token=True,max_output_patches=12,time_encoding_scale=256))
m=Chronos2Model(cfg).eval()
for p in m.parameters(): p.requires_grad_(False)
x=torch.randn(5,64); mask=torch.ones_like(x,dtype=torch.bool)
offset=torch.arange(-64,0)[None].expand(5,-1); groups=torch.tensor([0,0,1,1,1])
def run(z=x,ma=mask,pos=offset,tb=None,gb=None,gate=1.):
    return encode_packed_segments(m,z,ma,pos,groups,2,tb,gb,(1,),gate)
base=run()[0]; native=m.encode(x,group_ids=groups,num_output_patches=2)[0].last_hidden_state
check('packed_matches_native_contiguous',torch.allclose(base,native,atol=2e-6,rtol=2e-6))
n,t,_=base.shape; bt=torch.randn(n,1,t,t)*.2; bg=torch.randn(t,1,n,n)*.2
check('gate_zero_exact_identity',torch.equal(run(tb=bt,gb=bg,gate=0)[0],base))
changed=x.clone(); changed[2:]=torch.randn_like(changed[2:])*20
check('people_isolation_with_guidance',torch.allclose(run(changed,tb=bt,gb=bg)[0][:2],run(tb=bt,gb=bg)[0][:2],atol=2e-6,rtol=2e-6))
mask2=mask.clone(); mask2[0,3:8]=False
check('native_scaling_ignores_missing_fill',torch.equal(run(x.masked_fill(~mask2,0),mask2)[0],run(x.masked_fill(~mask2,1e8),mask2)[0]))
gaps=offset.clone(); gaps[:,:32]-=24; out=run(pos=gaps)
check('real_gap_position_and_embedding',torch.equal(out[2][:,:2],run()[2][:,:2]-1.5) and not torch.allclose(out[0],base))
nm=torch.tensor([0.,torch.finfo(torch.float32).min])
check('guidance_cannot_unmask',preserve_mask_add_bias(nm,torch.tensor([1.,1e30]))[1]==nm[1])
g=EvidenceBias(); ta=torch.ones(n,t,t,dtype=torch.bool); ga=(groups[:,None]==groups[None,:])[None].expand(t,-1,-1)
tb,gb=g(torch.randn(n,t,t,19),torch.randn(t,n,n,4),ta,ga)
check('guide_zero_initialization',tb.abs().sum()==0 and gb.abs().sum()==0)
active_g=EvidenceBias()
for last in (active_g.time[-1],active_g.group[-1]): torch.nn.init.normal_(last.weight)
tactive=ta.clone(); tactive[:,0]=False
gactive=ga.clone(); gactive[0]=False
at,ag=active_g(torch.randn(n,t,t,19),torch.randn(t,n,n,4),ta,ga,tactive,gactive)
check('inactive_query_guidance_zero',at[:,0].abs().sum()==0 and ag[0].abs().sum()==0 and at.abs().sum()>0)
out=run(tb=tb[:,None],gb=gb[:,None])[0]; (out*torch.randn_like(out)).sum().backward()
check('guide_gradient_through_frozen_C2',any(p.grad is not None and p.grad.abs().sum()>0 for p in g.parameters()))
check('no_gradients_into_S_or_C2',all(p.grad is None for p in list(s.parameters())+list(m.parameters())))
check('hooks_removed',all(len(mod._forward_pre_hooks)==0 for mod in m.modules()))
check('utility_direction',observed_utility(torch.tensor([0.]),torch.tensor([2.]),torch.tensor([1.]))[0]>0)
h=FamilyEventHead(32).eval(); z=torch.randn(1,3,t,32); mk=torch.ones(1,3,t,dtype=torch.bool)
w=torch.zeros(3,t); w[0,:2]=1; w[1,2:4]=1; w[2,-2:]=1
a=h(z,mk,torch.ones(1,3,dtype=torch.bool),torch.tensor([[0,1,2]]),w)
r=torch.tensor([0,0,1,2]); b=h(z[:,r],mk[:,r],torch.ones(1,4,dtype=torch.bool),torch.tensor([[0,0,1,2]]),w)
check('equal_family_readout',torch.allclose(a,b,atol=1e-6))
_,features=h(z,mk,torch.ones(1,3,dtype=torch.bool),torch.tensor([[0,1,2]]),w,torch.tensor([4]),True)
check('dropped_family_remains_in_coverage_denominator',features[0,-2]==.75)
source=Path(inspect.getfile(MHA))
report=dict(status='passed',scope='CPU synthetic interfaces only; no training/task evaluation',checks=checks,parameter_counts=dict(small_predictor=count(s),guidance_heads=count(g),event_head_for_768_width=count(FamilyEventHead()),binary_utility_gate=count(UtilityGate()),planned_lora_qv_last4_time_and_group=4*2*2*8*(768+768)),torch=torch.__version__,installed_chronos_layers=str(source),installed_layers_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),elapsed_seconds=round(time.monotonic()-start,3),gpu_used=False)
report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('src/precursor_guidance.py','src/chronos_bridge.py','src/event_readout.py','scripts/verify_design_core.py')}
path=ROOT/'artifacts/verification/cpu_verification.json'; path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False))
