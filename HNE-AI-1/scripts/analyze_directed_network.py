"""Paired participant comparisons for the fixed directed-network experiment."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_development import fold_auc, bootstrap_aucs
from src.hne_directed import VARIANTS


def main():
    output = ROOT/'results/directed_network'
    rows, differences = [], []
    controls = ['DNB', 'FCM-ensemble', 'CATCH-ensemble', 'History-LR', 'History-HGB',
                'HNE-HGB', 'JAPE-event', 'JAPE-event-AUC']
    for dataset in ['RESILIENT', 'BrisT1D-Open']:
        frame = pd.read_csv(ROOT/f'results/unified_daily/summary/{dataset}_predictions.csv', dtype={'person_id': str})
        for name in (*VARIANTS, 'CNM-GC-past32-LR'):
            score = pd.read_csv(output/dataset/name/'held_person.csv', dtype={'person_id': str})
            frame = frame.merge(score[['person_id', 'origin_day', 'probability']].rename(columns={'probability': name}),
                                on=['person_id', 'origin_day'], validate='one_to_one')
        native = pd.read_csv(ROOT/f'data/features/directed_network/{dataset}/cnm_scores.csv', dtype={'person_id': str})
        frame = frame.merge(native[['person_id', 'origin_day', 'score']].rename(columns={'score': 'CNM-GC-past32'}),
                            on=['person_id', 'origin_day'], validate='one_to_one')
        frame.to_csv(output/f'{dataset}_predictions.csv', index=False)
        for population in ['all', 'primary-common', 'cnm-common']:
            sub = frame if population == 'all' else frame[frame.primary_common]
            if population == 'cnm-common':
                sub = sub[np.isfinite(sub['CNM-GC-past32'])]
            people, inverse = np.unique(sub.person_id, return_inverse=True)
            weights = np.random.default_rng(20261110).multinomial(len(people), np.full(len(people), 1/len(people)), size=2000)
            methods = [*VARIANTS, 'CNM-GC-past32', 'CNM-GC-past32-LR', 'HNE-additive', *controls]
            boots = {m: bootstrap_aucs(sub, m, inverse, weights) for m in methods}
            for method in methods:
                valid = sub[np.isfinite(sub[method])]
                within = [roc_auc_score(g.y, g[method]) for _, g in valid.groupby('person_id') if g.y.nunique() == 2]
                lo, hi = np.nanquantile(boots[method], [.025, .975])
                rows.append(dict(dataset=dataset, population=population, method=method, origins=len(valid),
                                 people=valid.person_id.nunique(), auroc=fold_auc(valid, method), ci_low=lo, ci_high=hi,
                                 within_person_auroc=np.mean(within)))
            comparators = controls+['CNM-GC-past32', 'CNM-GC-past32-LR'] if population == 'cnm-common' else controls
            for target in VARIANTS:
                for comparator in ['HNE-additive', 'strongest-control']:
                    strongest = max(comparators, key=lambda m: fold_auc(sub, m)) if comparator == 'strongest-control' else comparator
                    compare = np.nanmax(np.stack([boots[m] for m in comparators]), axis=0) if comparator == 'strongest-control' else boots[strongest]
                    lo, hi = np.nanquantile(boots[target]-compare, [.025, .975])
                    differences.append(dict(dataset=dataset, population=population, target=target, comparator=comparator,
                                            strongest=strongest, delta=fold_auc(sub, target)-fold_auc(sub, strongest), ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(output/'paired_differences.csv', index=False)
    print(pd.DataFrame(rows).query("population == 'primary-common' and method in ['HNE-GC', 'HNE-GC-null', 'HNE-additive', 'CATCH-ensemble']").to_string(index=False))
    print(pd.DataFrame(differences).query("population == 'primary-common'").to_string(index=False))


if __name__ == '__main__':
    main()
