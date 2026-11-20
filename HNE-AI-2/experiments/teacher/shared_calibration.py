"""Fixed selected-teacher control: use H's personal adapter for every view."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
import psutil
from run import ROOT, reference_people, audit_person, write_csv, summarize, teacher_from_checkpoint, sha, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-run-key', required=True)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    source = (ROOT / 'artifacts/experiments' / args.teacher_run_key).resolve()
    if not source.is_relative_to((ROOT / 'artifacts/experiments/teacher').resolve()):
        raise ValueError('Expected a teacher run')
    parent = json.loads((source / 'manifest.json').read_text())
    selected = json.loads((source / 'selection.json').read_text())
    config = dict(parent_teacher_run=args.teacher_run_key, parent_selection_sha256=sha(source / 'selection.json'),
                  calibration_policy='shared_H', bootstrap_repetitions=2000, model_selection_changed=False)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    study = parent['config']
    key = f"teacher/{study['dataset']}/fold{study['fold']}/seed{study['seed']}/shared_calibration/{digest[:16]}"
    output = ROOT / 'results/experiments' / key
    artifact = ROOT / 'artifacts/experiments' / key
    output.mkdir(parents=True, exist_ok=True); artifact.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    people, _ = reference_people(study['dataset'], study['fold'])
    for group in people.values():
        for person in group:
            assert all(sha(path) == parent['data_sha256'][path] for path in person.source_files)
    models = {}
    for entry in selected['selected']:
        path = ROOT / entry['path']; assert sha(path) == entry['sha256']
        saved = torch.load(path, weights_only=False)
        models[entry['component']] = (teacher_from_checkpoint(saved), saved)
    start = time.perf_counter(); rows = []
    for split in ('train', 'validation'):
        for person in people[split]:
            component = parent['small_component'][person.person]
            model, saved = models[component]
            assert person.person not in saved['report']['fit_people']
            assert person.person not in saved['report']['selection_holdout']
            rows.extend(audit_person(model, person, saved['architecture'], component,
                                    'source_oof' if split == 'train' else 'source_validation', study, 'shared_H'))
    write_csv(output / 'person_metrics.csv', rows)
    result = dict(status='completed', run_key=key, config=config, config_sha256=digest,
                  code_commit=args.code_commit, source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('run.py')]},
                  protocol_sha256=parent['protocol_sha256'], seconds=time.perf_counter() - start,
                  conditional_gains=summarize(rows, study['seed'], 2000), event_labels_loaded=False,
                  warning='Post-audit development control; selected models and the ongoing core evidence are unchanged.')
    save_json(output / 'summary.json', result); save_json(artifact / 'receipt.json', result)
    print(json.dumps(dict(run_key=key, seconds=result['seconds'], gains=[r for r in result['conditional_gains']
                      if r['calibrated'] and r['mode']=='aligned'])))


if __name__ == '__main__':
    main()
