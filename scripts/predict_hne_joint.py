"""Predict from observed hourly and daily histories with a trained HNE model.

Input NPZ: observed_hourly [complete_observed_days*24,6].
The last channel is observed context; all rows precede the issue time.
"""
from pathlib import Path
import argparse,json,sys,warnings
import joblib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.daily_protocol import robust_reference,normalize
from src.hne_representation import Sampling,encode_person
from src.hne_joint import HNEJoint


def predict_observations(observed_hourly,checkpoint,basis_file):
    values=np.asarray(observed_hourly,dtype=np.float64);days=len(values)//24
    states=np.full((days,10),np.nan);contexts=np.full((days,1),np.nan)
    for day,block in enumerate(values.reshape(days,24,6)):
        for j in range(5):
            valid=block[:,j][np.isfinite(block[:,j])]
            if len(valid)>=4:states[day,2*j:2*j+2]=[valid.mean(),valid.std()]
        valid=block[:,5][np.isfinite(block[:,5])]
        if len(valid):contexts[day,0]=valid.mean()
    return inference(values[-32*24:],values[:56*24],states,contexts,checkpoint,basis_file)


def inference(hourly_history,hourly_reference,daily_states,daily_contexts,checkpoint,basis_file,variant='joint'):
    from chronos import Chronos2Pipeline
    config=json.loads((ROOT/'configs/hne_joint.json').read_text(encoding='utf8'))
    normalized=[]
    for values in [daily_states,daily_contexts]:
        center,scale,_=robust_reference(values,56,28)
        normalized.append(normalize(values,center,scale).astype(np.float32))
    _,numerical,_=encode_person(*normalized,[len(daily_states)],Sampling.daily())
    processor=joblib.load(basis_file);basis=[torch.tensor(v,device='cuda') for v in processor.transform(numerical)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        h=np.nanmean(hourly_history.reshape(-1,3,6),1);r=np.nanmean(hourly_reference.reshape(-1,3,6),1)
        center,scale,_=robust_reference(r,len(r),32)
        h=np.clip(normalize(h,center,scale),-12,12).T.astype(np.float32)
        r=np.clip(normalize(r,center,scale),-12,12).T.astype(np.float32)
    backbone=Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'),device_map='cuda').model.eval()
    captured={}
    def hook(module,args,kwargs):
        captured['x']=args[0].detach().half().float()
        captured['mask']=kwargs['attention_mask'][:,0,0,:]==0
    handle=backbone.encoder.block[-1].register_forward_pre_hook(hook,with_kwargs=True);inputs=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for array in [h,r]:
            backbone.encode(torch.tensor(array,device='cuda'),group_ids=torch.arange(6,device='cuda'))
            inputs.append((captured['x'].unsqueeze(0),captured['mask'].unsqueeze(0)))
    handle.remove();del backbone;torch.cuda.empty_cache()
    model=HNEJoint(config,processor.widths,variant).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(inputs[0][0],inputs[0][1],inputs[1][0],inputs[1][1],None,basis)
    return dict(risk_score=float(out['logits'].sigmoid()[0]),path=['HNE'[i] for i in out['routes'][0].tolist()],
        six_path_probabilities=out['all_logits'][0].sigmoid().cpu().tolist(),
        path_probabilities=out['policy'][0].cpu().tolist())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path)
    parser.add_argument('--dataset');parser.add_argument('--person');parser.add_argument('--day',type=int)
    parser.add_argument('--checkpoint',required=True,type=Path);parser.add_argument('--basis',required=True,type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args();torch.set_num_threads(2)
    if args.input:
        hourly=np.load(args.input)['observed_hourly']
    else:
        hourly=np.load(ROOT/f'data/development/{args.dataset}/{args.person}_hourly.npz')['values']
        hourly=hourly[:(args.day+1)*24]
    result=predict_observations(hourly,args.checkpoint,args.basis)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
