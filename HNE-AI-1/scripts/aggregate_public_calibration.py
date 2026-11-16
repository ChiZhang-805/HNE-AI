"""Three-seed ensembles of label-adapted official FCM/CATCH scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
seeds = [20261110, 20261111, 20261112]
sources = [ROOT/'results/unified_daily']+[ROOT/'results/public_daily_repeats'/f'seed{s}' for s in seeds[1:]]
for dataset in ['RESILIENT', 'BrisT1D-Open']:
    for method in ['FCM-score-LR', 'CATCH-score-LR']:
        for fold in range(5):
            folder = ROOT/'results/public_daily_repeats/ensemble'/dataset/method/f'fold{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            for part in ['train', 'validation', 'held_person']:
                frames = [pd.read_csv(p/dataset/method/f'fold{fold}'/f'{part}.csv', dtype={'person_id': str}) for p in sources]
                result = frames[0].copy(); index = pd.MultiIndex.from_frame(result[['person_id', 'origin_day']])
                aligned = [f.set_index(['person_id', 'origin_day']).reindex(index) for f in frames]
                assert all(np.array_equal(f.y.to_numpy(), result.y.to_numpy()) for f in aligned)
                result['score'] = np.mean([f.score.to_numpy() for f in aligned], axis=0)
                result.to_csv(folder/f'{part}.csv', index=False)
            (folder/'metadata.json').write_text(json.dumps(dict(seeds=seeds, weights=[1/3]*3,
                source='official FCM/CATCH score; training-partition single-score LR; validation-C selection',
                aggregation='equal mean calibrated probabilities'), indent=2), encoding='utf8')
        print(dataset, method, 'five-fold three-seed ensemble complete', flush=True)
