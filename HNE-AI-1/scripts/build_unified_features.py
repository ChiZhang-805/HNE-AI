from pathlib import Path
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.hne_representation import Sampling,encode_person,select_reference_channels
from src.hne_features import normalized_series


def run(frequency):
    spec=Sampling.hourly() if frequency=='hourly' else Sampling.daily()
    source=ROOT/'data'/('development' if frequency=='hourly' else 'daily_study')
    output=ROOT/f'data/features/unified_{frequency}';output.mkdir(parents=True,exist_ok=True)
    for folder in sorted(source.iterdir()):
        if not (folder/'samples.csv').exists():continue
        target=output/f'{folder.name}.npz'
        if target.exists():continue
        meta=pd.read_csv(folder/'samples.csv',dtype={'person_id':str})
        if not len(meta):continue
        refs=json.loads((folder/'references.json').read_text()) if frequency=='hourly' else None
        all_nodes,all_pooled,all_windows,indices=[],[],[],[]
        for person,frame in meta.groupby('person_id',sort=False):
            if frequency=='hourly':
                values=normalized_series(folder,person,refs[person]);states=values[:,:-1];contexts=values[:,-1:]
                cutoffs=frame.issue_time_hours.to_numpy()
            else:
                data=np.load(folder/f'{person}_daily.npz')
                states=data['normalized'];contexts=data['normalized_contexts'];cutoffs=frame.origin_day.to_numpy()+1
                selected=select_reference_channels(states,data['state_names'],reference=spec.reference)
                states=states[:,selected]
            nodes,pooled,groups=encode_person(states,contexts,cutoffs,spec)
            windows=np.stack([np.column_stack((states,contexts))[t-spec.scales[-1]:t] for t in cutoffs])
            all_nodes.extend(nodes);all_pooled.extend(pooled);all_windows.extend(windows);indices.extend(frame.index)
        order=np.argsort(indices)
        payload=dict(nodes=np.stack(all_nodes)[order],pooled=np.stack(all_pooled)[order],windows=np.stack(all_windows)[order],
                     groups=groups,y=meta.y.to_numpy(),fold=meta.fold.to_numpy())
        if 'split' in meta:payload['split']=meta.split.astype(str).to_numpy()
        np.savez_compressed(target,**payload)
        meta.to_csv(output/f'{folder.name}.csv',index=False)
        print(json.dumps(dict(frequency=frequency,dataset=folder.name,n=len(meta),positives=int(meta.y.sum()),
                              node_shape=payload['nodes'].shape,window_shape=payload['windows'].shape)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--frequency',choices=['hourly','daily'],required=True)
    args=parser.parse_args()
    warnings.filterwarnings('ignore',message='All-NaN slice encountered')
    with threadpool_limits(limits=2):run(args.frequency)
