"""Apply the existing daily endpoint and HNE encoder to the new cohort."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_daily_study import write_person,summarize_people
from src.hne_representation import Sampling,encode_person,select_reference_channels


def main(splits):
    source=ROOT/'data/new_cohorts/human-rhythms'
    assignment=pd.read_csv(source/'subject_split.csv',dtype={'person_id':str})
    config_bytes=(ROOT/'configs/daily_protocol.json').read_bytes()
    config=json.loads(config_bytes);digest=hashlib.sha256(config_bytes).hexdigest()
    output=ROOT/'data/human_rhythms_study';features=ROOT/'data/features/human_rhythms'
    output.mkdir(parents=True,exist_ok=True);features.mkdir(parents=True,exist_ok=True)
    spec=Sampling.daily()
    for split in splits:
        destination=features/f'{split}.npz'
        if destination.exists():
            print(split,'features already complete',flush=True);continue
        folder=output/split;folder.mkdir(exist_ok=True)
        all_people=[];all_origins=[];all_samples=[]
        node_rows=[];pooled_rows=[];window_rows=[];future_rows=[];future_available=[]
        for row in assignment[assignment.split.eq(split)].itertuples():
            person=row.person_id
            with np.load(source/f'daily/{person}.npz') as archive:
                arrays={key:archive[key] for key in archive.files}
            origins,samples,person_summary=write_person(folder,'Human-Rhythms',person,split,0,arrays,config,digest)
            all_people.append(person_summary);all_origins.extend(origins);all_samples.extend(samples)
            if not samples:continue
            data=np.load(folder/f'{person}_daily.npz')
            selected=select_reference_channels(data['normalized'],data['state_names'],reference=56)
            states=data['normalized'][:,selected];contexts=data['normalized_contexts']
            cutoffs=np.array([sample['origin_day']+1 for sample in samples])
            nodes,pooled,groups=encode_person(states,contexts,cutoffs,spec)
            node_rows.extend(nodes);pooled_rows.extend(pooled)
            trajectory=np.column_stack((states,contexts)).astype(np.float32)
            for cutoff in cutoffs:
                window_rows.append(trajectory[cutoff-32:cutoff])
                complete=cutoff+32<=len(trajectory)
                future_available.append(complete)
                if split!='test':
                    future_rows.append(trajectory[cutoff:cutoff+32] if complete else np.full((32,trajectory.shape[1]),np.nan,np.float32))
            print(split,person,len(samples),'origins',flush=True)
        meta=pd.DataFrame(all_samples)
        persons=pd.DataFrame(all_people)
        meta.to_csv(folder/'samples.csv',index=False);meta.to_csv(features/f'{split}.csv',index=False)
        persons.to_csv(folder/'people.csv',index=False)
        pd.DataFrame(all_origins).to_csv(folder/'origins.csv',index=False)
        payload=dict(nodes=np.stack(node_rows),pooled=np.stack(pooled_rows),groups=groups,
            windows=np.stack(window_rows),y=meta.y.to_numpy(),fold=meta.fold.to_numpy(),
            future_available=np.asarray(future_available,bool))
        if split!='test':payload['future']=np.stack(future_rows)
        np.savez_compressed(destination,**payload)
        summary=summarize_people(persons)
        summary.update(split=split,protocol_sha256=digest,state_dimension=payload['nodes'].shape[1],
            context_dimension=payload['windows'].shape[-1]-payload['nodes'].shape[1],
            complete32day_future=int(sum(future_available)))
        (folder/'summary.json').write_text(json.dumps(summary,indent=2))
        print('COMPLETE',json.dumps(summary),flush=True)
    (features/'protocol.json').write_text(json.dumps(dict(event=config,sha256=digest,
        source='Human Rhythms official native daily fields; schema fixed before outcomes',
        representation='src.hne_representation.encode_person; unchanged daily configuration',
        participant_split='100train/33validation/33test by fixed ID hash',
        test_trajectories_exported=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train','validation'])
    args=parser.parse_args()
    if 'test' in args.splits and not (ROOT/'results/human_rhythms/frozen_models.json').exists():
        raise RuntimeError('Finish and freeze the training/validation model selection before constructing test outcomes.')
    warnings.filterwarnings('ignore',message='All-NaN slice encountered')
    warnings.filterwarnings('ignore',message='Mean of empty slice')
    warnings.filterwarnings('ignore',message='Degrees of freedom <= 0 for slice')
    with threadpool_limits(limits=1):main(args.splits)
