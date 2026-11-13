"""Aggregate original author scores and supervised calibration separately."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def run():
    config=json.loads((ROOT/'configs/hne_routed.json').read_text())
    for ds in config['datasets']:
        for method in ['DNB','FCM','CATCH']:
            seeds=config['seeds'][:1] if method=='DNB' else config['seeds']
            for k in config['folds']:
                folder=ROOT/f'results/hne_routed/{ds}/{method}/fold{k}'
                for part in ['validation','held_person']:
                    frames=[pd.read_csv(folder/f'seed{seed}/{part}.csv',dtype={'person_id':str}) for seed in seeds]
                    frame=frames[0].copy()
                    for col in ['raw_score','probability','calibrated_probability']:frame[col]=np.mean([f[col].to_numpy() for f in frames],axis=0)
                    frame['rank_probability']=frame['probability'];frame['probability']=frame['raw_score']
                    frame.to_csv(folder/f'{part}.csv',index=False)
    print('Aggregated raw scores and calibrated probabilities for all baseline runs.')


if __name__=='__main__':run()
