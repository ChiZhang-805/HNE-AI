"""Execute selected checkpoints from raw hourly inputs in both cohorts."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from predict_hne_routed import inference


def run():
    torch.set_num_threads(2);config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    variant=json.loads((ROOT/'results/hne_routed/architecture_selection.json').read_text())['primary_variant'];results=[]
    for ds in config['datasets']:
        row=pd.read_csv(ROOT/f'data/hne_routed/{ds}/samples.csv',dtype={'person_id':str}).iloc[0]
        k=int(row.fold);selection=json.loads((ROOT/f'results/hne_routed/{ds}/{variant}/fold{k}/selection.json').read_text())
        checkpoint=ROOT/selection['runs'][0]/'model.pt'
        values=np.load(ROOT/f'data/development/{ds}/{row.person_id}_hourly.npz')['values'];end=(int(row.origin_day)+1)*24
        history=values[end-32*24:end];reference=values[:56*24]
        example=ROOT/f'data/hne_routed/{ds}/example_input.npz';np.savez_compressed(example,history=history,reference=reference)
        output=inference(history,reference,checkpoint,config,variant)
        saved=pd.read_csv(checkpoint.parent/'held_person.csv',dtype={'person_id':str})
        expected=saved[saved.person_id.eq(row.person_id)&saved.origin_day.eq(row.origin_day)].iloc[0]
        expected_path=['HNE'[int(expected[f'route_{step}'])] for step in [1,2,3]]
        results.append(dict(dataset=ds,variant=variant,input=str(example.relative_to(ROOT)),checkpoint=str(checkpoint.relative_to(ROOT)),
            cached_risk=float(expected.probability),absolute_difference=abs(float(expected.probability)-output['risk_score']),
            cached_path=expected_path,path_matches=expected_path==output['path'],**output))
        print(json.dumps(results[-1]),flush=True)
    (ROOT/'results/hne_routed/summary/raw_inference.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':run()
