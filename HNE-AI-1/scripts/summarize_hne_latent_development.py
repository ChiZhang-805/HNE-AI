"""Summarize completed inner-person comparisons without mixing in outer scores."""
from pathlib import Path
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


def summarize(folder):
    runs, curves = [], []
    for path in sorted(folder.glob('*/*/fold*/selection.json')):
        dataset, variant, fold = path.parts[-4:-1]
        selection = json.loads(path.read_text(encoding='utf8'))
        first, last = selection['history'][0], selection['history'][-1]
        row = dict(dataset=dataset, variant=variant, outer_fold=int(fold[4:]),
                   training_people=len(selection['training_people']),
                   inner_fits=selection['inner_training_fits'],
                   selected_epoch=selection['selected_epoch'],
                   completed_epochs=last['epoch'],
                   inner_auroc=selection['inner_auroc'], inner_bce=selection['inner_bce'],
                   first_train_objective=first['train_objective'],
                   last_train_objective=last['train_objective'],
                   first_inner_bce=first['inner_bce'], last_inner_bce=last['inner_bce'])
        runs.append(row)
        with (path.parent/'inner_curves.csv').open(newline='', encoding='utf8') as handle:
            curves.extend(dict(dataset=dataset, variant=variant, outer_fold=row['outer_fold'], **r)
                          for r in csv.DictReader(handle))
    output = folder/'summary'
    output.mkdir(exist_ok=True)
    for name, records in [('development_metrics', runs), ('development_curves', curves)]:
        with (output/f'{name}.csv').open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    comparisons = []
    for row in runs:
        if row['variant'] != 'recurrent':
            continue
        for control in runs:
            if (control['dataset'], control['outer_fold']) != (row['dataset'], row['outer_fold']):
                continue
            if control['variant'] == 'recurrent':
                continue
            comparisons.append(dict(dataset=row['dataset'], outer_fold=row['outer_fold'],
                control=control['variant'], inner_auroc_difference=row['inner_auroc']-control['inner_auroc']))
    report = dict(evaluation_scope='Three participant folds within the original training subset; each condition selects its epoch on these same inner folds.',
                  score_role='Development and model selection, not outer-fold test performance.',
                  completed_conditions=len(runs), completed_inner_fits=sum(r['inner_fits'] for r in runs),
                  runs=runs, comparisons=comparisons)
    (output/'development_summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/hne_latent_grounded')
    args = parser.parse_args()
    summarize(ROOT/args.output)
