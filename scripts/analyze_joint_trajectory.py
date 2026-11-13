"""Participant-paired comparisons of the completed fixed trajectory experiment."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_development import fold_auc, bootstrap_aucs


def run():
    output = ROOT/'results/joint_trajectory'
    rows, differences, training = [], [], []
    controls = ['DNB', 'FCM-ensemble', 'CATCH-ensemble', 'History-LR', 'History-HGB',
                'HNE-HGB', 'JAPE-event', 'JAPE-event-AUC']
    for dataset in ['RESILIENT', 'BrisT1D-Open']:
        frame = pd.read_csv(ROOT/f'results/unified_daily/summary/{dataset}_predictions.csv', dtype={'person_id': str})
        for condition in ['diagonal', 'joint']:
            folders = sorted((output/dataset/condition).glob('fold*'))
            if len(folders) != 5 or not all((f/'held_person.csv').exists() for f in folders):
                raise ValueError('All five prespecified development folds are required')
            parts = [pd.read_csv(f/'held_person.csv', dtype={'person_id': str}) for f in folders]
            score = pd.concat(parts)[['person_id', 'origin_day', 'probability']]
            frame = frame.merge(score.rename(columns={'probability': condition}), on=['person_id', 'origin_day'], validate='one_to_one')
            for folder in folders:
                for path in sorted(folder.glob('seed*/metadata.json')):
                    info = json.loads(path.read_text())
                    training.append(dict(dataset=dataset, condition=condition, fold=folder.name, seed=info['seed'],
                                         epochs=info['epochs'], best_epoch=info['best_epoch'],
                                         validation_nll=info['best_validation_loss'], seconds=info['seconds']))
        frame.to_csv(output/f'{dataset}_predictions.csv', index=False)
        for population in ['all', 'common']:
            sub = frame if population == 'all' else frame[frame.primary_common]
            people, inverse = np.unique(sub.person_id, return_inverse=True)
            weights = np.random.default_rng(20261110).multinomial(len(people), np.full(len(people), 1/len(people)), size=2000)
            methods = ['diagonal', 'joint', 'HNE-additive'] + controls
            boot = {m: bootstrap_aucs(sub, m, inverse, weights) for m in methods}
            for method in methods:
                valid = sub[np.isfinite(sub[method])]
                within = [roc_auc_score(g.y, g[method]) for _, g in valid.groupby('person_id') if g.y.nunique() == 2]
                low, high = np.nanquantile(boot[method], [.025, .975])
                rows.append(dict(dataset=dataset, population=population, method=method, origins=len(valid),
                                 people=valid.person_id.nunique(), auroc=fold_auc(valid, method), ci_low=low, ci_high=high,
                                 within_person_auroc=np.mean(within),
                                 brier=brier_score_loss(valid.y, valid[method]) if method in ['diagonal', 'joint', 'HNE-additive'] else None))
            for target in ['joint', 'diagonal']:
                for comparator in ['diagonal', 'HNE-additive', 'strongest-control']:
                    if target == comparator:
                        continue
                    best = max(controls, key=lambda m: fold_auc(sub, m)) if comparator == 'strongest-control' else comparator
                    comparison = np.max(np.stack([boot[m] for m in controls]), 0) if comparator == 'strongest-control' else boot[best]
                    low, high = np.nanquantile(boot[target]-comparison, [.025, .975])
                    differences.append(dict(dataset=dataset, population=population, target=target, comparator=comparator,
                                            strongest=best, delta=fold_auc(sub, target)-fold_auc(sub, best), ci_low=low, ci_high=high))
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(output/'paired_differences.csv', index=False)
    pd.DataFrame(training).to_csv(output/'training_summary.csv', index=False)
    print(pd.DataFrame(rows).query("population == 'common' and method in ['joint', 'diagonal', 'HNE-additive', 'CATCH-ensemble']").to_string(index=False))
    print(pd.DataFrame(differences).query("population == 'common'").to_string(index=False))
    print('Completed fits:', len(training), 'epochs:', sum(x['epochs'] for x in training))


if __name__ == '__main__':
    run()
