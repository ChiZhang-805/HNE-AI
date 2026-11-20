"""Fit M13 against a finished source-only M04 and its cached features."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time

import psutil
import torch

from run import (ROOT, load_people, metrics, paired_bootstrap, read_json, resource_check,
                 same_evidence, save_json, selected_origins, sha, train_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor-root', type=Path, default=ROOT)
    parser.add_argument('--anchor-key', required=True)
    parser.add_argument('--expect-digest', required=True)
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'core/(BrisT1D-Open|RESILIENT)/fold[0-4]/seed\d+/M04_M10_M12/[a-f0-9]{16}', args.anchor_key):
        raise ValueError('Expected a registered M04/M10/M12 source-only anchor key')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu')
    anchor = args.anchor_root / 'artifacts/experiments' / args.anchor_key
    manifest = read_json(anchor / 'manifest.json')
    config = manifest['config']
    if config['include_evaluation'] or config['device'] != 'cpu':
        raise ValueError('Cached M13 expects a source-only CPU anchor')
    paths = {name: anchor / name for name in ('manifest.json', 'baseline/candidate0.pt',
                                              'baseline_features.pt', 'evidence_receipts.json')}
    inputs = {name: sha(path) for name, path in paths.items()}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if digest != args.expect_digest:
        raise ValueError('Anchor artifacts differ from the registered input digest')
    key = f"core/{config['dataset']}/fold{config['fold']}/seed{config['seed']}/M13/{digest[:16]}"
    directory = args.anchor_root / 'artifacts/experiments' / key
    output = args.anchor_root / 'results/experiments' / key
    provenance = dict(run_key=key, config=config, anchor_key=args.anchor_key, inputs=inputs,
                      config_sha256=digest, protocol_sha256=manifest['protocol_sha256'], code_commit=args.code_commit,
                      source_sha256={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('run.py'))},
                      scope='Matched source validation development with frozen M04; no outer evaluation')
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with (directory / 'execution.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(directory, output, anchor, paths, provenance, manifest)


def execute(directory, output, anchor, paths, provenance, manifest):
    provenance_path = directory / 'manifest.json'
    if provenance_path.exists() and read_json(provenance_path) != provenance:
        raise ValueError('Existing M13 run has different provenance')
    status = directory / 'status.json'
    if status.exists() and read_json(status)['status'] == 'completed':
        print(json.dumps(dict(status='completed', run_key=provenance['run_key'], reused=True)))
        return
    save_json(provenance_path, provenance)
    save_json(status, dict(status='running', pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat()))
    started = time.perf_counter()
    config = provenance['config']

    def log(event, **value):
        print(json.dumps(dict(event=event, **value)), flush=True)

    try:
        features = torch.load(paths['baseline_features.pt'], weights_only=True, map_location='cpu')
        receipts = {(r['person'], r['origin']): r for r in read_json(paths['evidence_receipts.json'])}
        splits = load_people(config['dataset'], config['fold'], include_evaluation=False)
        if {k: [p.person for p in v] for k, v in splits.items()} != manifest['split_people']:
            raise ValueError('Source split differs from the original M04')
        items = {'train': [], 'validation': []}
        seen = set()
        for split, rows in items.items():
            for person in splits[split]:
                for path in person.source_files:
                    relative = str(path).replace('\\', '/').split('/data/', 1)[1]
                    if sha(path) != manifest['data_sha256'][relative]:
                        raise ValueError('Source input differs from the original M04')
                cal = anchor / 'evidence' / person.person / 'calibration.pt'
                calibration_hash = sha(cal)
                for origin, label in selected_origins(person, config[f'{split}_origins_per_person']):
                    evidence_path = cal.with_name(f'origin{origin}.pt')
                    receipt = receipts[person.person, origin]
                    if sha(evidence_path) != receipt['sha256'] or calibration_hash != receipt['calibration_sha256']:
                        raise ValueError('Source evidence differs from the original M04 run')
                    feature_key = f'{person.person}/{origin}'
                    seen.add(feature_key)
                    rows.append(dict(person=person.person, origin=origin, label=label, source=person,
                                     baseline=features[feature_key], evidence=same_evidence(
                                         torch.load(evidence_path, weights_only=True, map_location='cpu'))))
        if seen != set(features) or len(seen) != len(receipts):
            raise ValueError('Anchor features, evidence and origin sets do not match')
        base = [dict(person=x['person'], origin=x['origin'], label=x['label'],
                     logit=x['baseline']['logit'].item(), probability=x['baseline']['logit'].reshape(1).sigmoid().item())
                for x in items['validation']]
        candidates = []
        for candidate, (lr, decay) in enumerate(itertools.product(config['learning_rates'], config['weight_decays'])):
            _, done, evaluate, restore = train_stage('logit_fusion', candidate, lr, decay, config, directory,
                                                     items, None, None, log)
            selected = evaluate(items['validation'])
            restore(done['trained_state'])
            always_on = evaluate(items['validation'])
            if evaluate(items['validation'], True) != base:
                raise ValueError('Disabled M13 diverges from the canonical M04')
            candidates.append((done, selected, always_on))
        sel = min(range(len(candidates)), key=lambda i: (candidates[i][0]['report']['selected_nll'], i))
        on = min(range(len(candidates)), key=lambda i: (candidates[i][0]['report']['always_on_nll'], i))
        torch.save(dict(state=candidates[sel][0]['state'], trained_state=candidates[on][0]['trained_state'],
                        selected_report=candidates[sel][0]['report'], always_on_report=candidates[on][0]['report'],
                        base_checkpoint_sha256=provenance['inputs']['baseline/candidate0.pt'],
                        evidence_receipts_sha256=provenance['inputs']['evidence_receipts.json']), directory / 'M13.pt')
        outputs = {'M04_C2_LoRA': {'validation': base}, 'M13_selected': {'validation': candidates[sel][1]},
                   'M13_always_on': {'validation': candidates[on][2]}}
        with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['method', 'split', 'person', 'origin', 'label', 'logit', 'probability'])
            writer.writeheader()
            for method, splits in outputs.items():
                writer.writerows(dict(method=method, split='validation', **row) for row in splits['validation'])
        save_json(output / 'paired.json', paired_bootstrap(outputs, config['seed'], config['bootstrap_repetitions']))
        result = dict(status='completed', run_key=provenance['run_key'], scope=provenance['scope'],
                      metrics={k: metrics(v['validation']) for k, v in outputs.items()},
                      candidates=[c[0]['report'] for c in candidates], selected_candidate=sel, always_on_candidate=on,
                      checkpoint_sha256=sha(directory / 'M13.pt'), seconds=time.perf_counter() - started,
                      off_exact=True, train_origins=len(items['train']), validation_origins=len(base),
                      anchor_inputs=provenance['inputs'], gpu_used=False)
        if any(sha(path) != provenance['inputs'][name] for name, path in paths.items()):
            raise ValueError('Anchor changed during M13 run')
        save_json(output / 'summary.json', result)
        save_json(status, dict(status='completed', seconds=result['seconds']))
        log('completed', run_key=provenance['run_key'], metrics=result['metrics'], seconds=result['seconds'])
    except Exception as error:
        save_json(status, dict(status='failed', error=str(error), seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
