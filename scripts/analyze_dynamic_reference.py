"""Compare warning performance and matched-date normal-reference predictions."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_development import fold_auc, bootstrap_aucs
from src.hne_dynamic_reference import VARIANTS


def main(experiment='dynamic_reference'):
    output = ROOT/'results'/experiment
    cfg = json.loads((ROOT/'configs'/f'{experiment}.json').read_text())
    variants = cfg['variants']
    rows, differences, reference_rows = [], [], []
    controls = [m for m in cfg['comparison'] if m != 'HNE-additive']
    for dataset in cfg['datasets']:
        frame = pd.read_csv(ROOT/f'results/unified_daily/summary/{dataset}_predictions.csv', dtype={'person_id': str})
        for name in variants:
            score = pd.read_csv(output/dataset/name/'held_person.csv', dtype={'person_id': str})
            frame = frame.merge(score[['person_id', 'origin_day', 'probability']].rename(columns={'probability': name}),
                                on=['person_id', 'origin_day'], validate='one_to_one')
        frame.to_csv(output/f'{dataset}_predictions.csv', index=False)
        for population in ['all', 'primary-common']:
            sub = frame if population == 'all' else frame[frame.primary_common]
            people, inverse = np.unique(sub.person_id, return_inverse=True)
            weights = np.random.default_rng(20261110).multinomial(len(people), np.full(len(people), 1/len(people)), size=2000)
            methods = [*variants, *cfg['comparison']]
            boots = {m: bootstrap_aucs(sub, m, inverse, weights) for m in methods}
            for method in methods:
                valid = sub[np.isfinite(sub[method])]
                within = [roc_auc_score(g.y, g[method]) for _, g in valid.groupby('person_id') if g.y.nunique() == 2]
                lo, hi = np.nanquantile(boots[method], [.025, .975])
                rows.append(dict(dataset=dataset, population=population, method=method, origins=len(valid),
                    people=valid.person_id.nunique(), auroc=fold_auc(valid, method), ci_low=lo, ci_high=hi,
                    within_person_auroc=float(np.mean(within))))
            for target in variants:
                for comparator in ['HNE-additive', 'strongest-control']:
                    strongest = max(controls, key=lambda m: fold_auc(sub, m)) if comparator == 'strongest-control' else comparator
                    reference = np.nanmax(np.stack([boots[m] for m in controls]), axis=0) if comparator == 'strongest-control' else boots[strongest]
                    lo, hi = np.nanquantile(boots[target]-reference, [.025, .975])
                    differences.append(dict(dataset=dataset, population=population, target=target, comparator=comparator,
                        strongest=strongest, delta=fold_auc(sub, target)-fold_auc(sub, strongest), ci_low=lo, ci_high=hi))
        diagnostic = pd.read_csv(ROOT/f'data/features/{experiment}/{dataset}/reference_prediction.csv', dtype={'person_id': str})
        diagnostic['relative_mse_change'] = (diagnostic.full_mse-diagnostic.own_mse)/(diagnostic.own_mse+1e-12)
        person = diagnostic.groupby('person_id').relative_mse_change.mean()
        samples = np.random.default_rng(20261110).choice(person.to_numpy(), size=(2000, len(person)), replace=True).mean(1)
        lo, hi = np.quantile(samples, [.025, .975])
        reference_rows.append(dict(dataset=dataset, people=len(person), nodes=len(diagnostic),
            observations=int(diagnostic.observations.sum()), equal_person_relative_mse_change=float(person.mean()), ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(output/'paired_differences.csv', index=False)
    pd.DataFrame(reference_rows).to_csv(output/'reference_prediction.csv', index=False)
    table = pd.DataFrame(rows)
    print(table[table.population.eq('primary-common') & table.method.isin([*variants, 'HNE-additive', 'CATCH-ensemble'])].to_string(index=False))
    print(pd.DataFrame(differences).query("population == 'primary-common'").to_string(index=False))
    print(pd.DataFrame(reference_rows).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['dynamic_reference', 'shared_reference'], default='dynamic_reference')
    main(parser.parse_args().experiment)
