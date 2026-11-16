"""Record the selected models before constructing external test outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/human_rhythms/frozen_models.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        print('External study is already frozen.');return
    selections=[]
    for method in ['FCM','CATCH']:
        for seed in [20261110,20261111,20261112]:
            selections.append(ROOT/f'results/human_rhythms/{method}/seed{seed}/selection.json')
            selections.append(ROOT/f'results/human_rhythms/{method}-score-LR/seed{seed}/selection.json')
        selections.append(ROOT/f'results/human_rhythms/{method}/ensemble/selection.json')
    selections += [ROOT/'results/human_rhythms/freeze_manifest.json',
        ROOT/'results/human_rhythms_transfer/freeze_manifest.json',
        ROOT/'results/context_grid/Human-Rhythms/freeze_manifest.json',
        ROOT/'results/multidomain_hne/Human-Rhythms/selection.json']
    for path in selections:json.loads(path.read_text())
    if (ROOT/'data/features/human_rhythms/test.npz').exists():
        raise RuntimeError('Test outcomes were constructed before study freezing.')
    models=[]
    for folder in ['results/human_rhythms','results/human_rhythms_transfer',
                   'results/context_grid/Human-Rhythms','results/multidomain_hne/Human-Rhythms']:
        for path in (ROOT/folder).rglob('*'):
            if path.suffix in ['.joblib','.pt','.pkl']:
                models.append(path)
    fixed=[ROOT/'configs/external_evaluation.json',ROOT/'configs/daily_protocol.json',
        ROOT/'data/new_cohorts/human-rhythms/subject_split.csv',
        ROOT/'data/new_cohorts/human-rhythms/daily/schema_mapping.json']
    record=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        test_outcomes_constructed=False,
        files={str(path.relative_to(ROOT)).replace('\\','/'):digest(path) for path in selections+models+fixed},
        plan=json.loads((ROOT/'configs/external_evaluation.json').read_text()))
    OUT.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Frozen',len(selections),'selection records and',len(models),'models before external test outcomes.')


if __name__=='__main__':main()
