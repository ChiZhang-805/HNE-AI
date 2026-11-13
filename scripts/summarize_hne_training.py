"""Measured weight updates and fixed-computation checks for selected models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]


def run():
    torch.set_num_threads(2);rows=[]
    for ds in ['RESILIENT','BrisT1D-Open']:
        for k in range(5):
            selection=json.loads((ROOT/f'results/hne_routed/{ds}/context_free/fold{k}/selection.json').read_text())
            for checkpoint in selection['runs']:
                folder=ROOT/checkpoint;fit=json.loads((folder/'result.json').read_text());source_folder=ROOT/fit['source_checkpoint']
                source=torch.load(source_folder/'model.pt',weights_only=True,map_location='cpu')
                trained=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')
                fixed_keys=[key for key in source if not key.startswith('router.')]
                fixed=all(torch.equal(source[key],trained[key]) for key in fixed_keys)
                router_delta=max(float((source[key]-trained[key]).abs().max()) for key in source if key.startswith('router.'))
                first=json.loads((source_folder/'result.json').read_text())
                rows.append(dict(dataset=ds,fold=k,seed=fit['seed'],first_stage_trainable_parameters=first['trainable_parameters'],
                    second_stage_active_parameters=fit['active_trainable_parameters'],computation_weights_identical=fixed,
                    router_max_change=router_delta,risk_scale=fit['risk_scale'],
                    maximum_lora_B_update=max(float(value.abs().max()) for name,value in source.items() if name.startswith('tail.') and name.endswith('.b'))))
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'results/hne_routed/summary/training_summary.csv',index=False)
    print(frame.groupby('dataset')[['computation_weights_identical','router_max_change','risk_scale','maximum_lora_B_update']].agg(['min','mean','max']).to_string())


if __name__=='__main__':run()
