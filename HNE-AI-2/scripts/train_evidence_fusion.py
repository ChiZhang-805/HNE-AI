"""Add a full-evidence late-fusion control to a completed key comparison."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import psutil
import torch
from torch import nn
from training_data import SOURCE, load_people, selected_origins
from train_backbone import load_backbone, load_adapter, encoder_tail, read_event
from train_comparison import metrics, save_json, sha, paired_bootstrap, resource_check
from fusion.evidence import FullEvidenceFusion

METHOD = 'M12_full_evidence_fusion'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='cpu_pilot')
    parser.add_argument('--dataset', default='BrisT1D-Open')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261114)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu')
    directory = ROOT / 'artifacts/key_comparison' / args.profile / args.dataset / f'fold{args.fold}' / f'seed{args.seed}'
    out = ROOT / 'results/key_comparison' / args.profile / args.dataset
    prefix = f'fold{args.fold}_seed{args.seed}'
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'completed':
        raise RuntimeError('First complete the shared baseline and key comparisons')
    config = status['config']
    completed_path = directory / 'full_evidence_fusion_status.json'
    if completed_path.exists() and json.loads(completed_path.read_text())['status'] == 'completed':
        recorded = json.loads((directory / 'full_evidence_fusion_manifest.json').read_text())
        if recorded['config'] != config or recorded['baseline_sha256'] != sha(directory / 'M04_C2_LoRA.pt'):
            raise RuntimeError('Completed full-evidence comparison has different inputs; preserve its results')
        print(json.dumps(dict(status='completed', action='no_op_existing_run_preserved',
                              directory=str(directory), provenance='See the original full-evidence manifest and sources.zip')), flush=True)
        return
    path = directory / f'{METHOD}.pt'
    started = time.perf_counter()
    people = load_people(args.dataset, args.fold, config['include_evaluation'])
    records = {'train': [], 'validation': [], 'evaluation': []}
    feature_file = directory / 'baseline_features.pt'
    features = torch.load(feature_file, weights_only=True) if feature_file.exists() else {}
    model = head = None
    with torch.no_grad():
        for split, persons in people.items():
            for person in persons:
                maximum = config['train_origins_per_person'] if split == 'train' else config['validation_origins_per_person'] if split == 'validation' else 0
                for origin, label in selected_origins(person, maximum):
                    key = f'{person.person}/{origin}'
                    if key not in features:
                        if model is None:
                            model, head = load_backbone(SOURCE / 'tools/models/chronos-2')
                            saved = torch.load(directory / 'M04_C2_LoRA.pt', weights_only=False)['state']
                            load_adapter(model, saved['lora']); head.load_state_dict(saved['head'])
                            model.requires_grad_(False); head.requires_grad_(False)
                        cached = torch.load(directory / 'prefix' / f'{person.person}_{origin}.pt', weights_only=True)
                        _, state, family, _ = person.large_window(origin)
                        output = encoder_tail(model, cached)
                        z, feature = read_event(head, output, cached, state, family, person.dataset, True)
                        features[key] = dict(logit=z[0], feature=feature[0])
                    evidence = torch.load(directory / 'evidence' / person.person / f'origin{origin}.pt', weights_only=True)
                    records[split].append(dict(person=person.person, origin=origin, label=label,
                                               family=person.family[0], evidence=evidence, **features[key]))
                print(json.dumps(dict(event='full_fusion_features', split=split, person=person.person, count=len(records[split]))), flush=True)
            torch.save(features, feature_file)
    feature_seconds = time.perf_counter() - started
    del model, head
    torch.manual_seed(args.seed + 4)
    module = FullEvidenceFusion()
    assert sum(p.numel() for p in module.parameters()) == 897
    def forward(row):
        return module(row['logit'], row['feature'], row['evidence'], row['family'])
    with torch.no_grad():
        if not torch.equal(forward(records['train'][0]), records['train'][0]['logit'].reshape(1)):
            raise AssertionError('Zero fusion must equal the frozen baseline')
    provenance = dict(baseline_sha256=sha(directory / 'M04_C2_LoRA.pt'),
                      source_sha256=sha(ROOT / 'src/fusion/evidence.py'),
                      trainer_sha256=sha(Path(__file__)), config=config,
                      inputs='Complete calibrated node/edge arrays, masks, age, same 12 S summaries, and frozen L feature')
    save_json(directory / 'full_evidence_fusion_manifest.json', provenance)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.001, weight_decay=.0001)
    counts = {p.person: sum(r['person'] == p.person for r in records['train']) for p in people['train']}
    n_train, n_people = len(records['train']), len(counts)
    def evaluate(split):
        rows = []
        with torch.no_grad():
            for row in records[split]:
                z = forward(row)
                rows.append(dict(person=row['person'], origin=row['origin'], label=row['label'],
                                 logit=float(z), probability=float(z.sigmoid())))
        return rows
    train_start = time.perf_counter()
    curve = []; best = float('inf'); best_state = None; selected = None
    if path.exists():
        saved = torch.load(path, weights_only=False)
        if saved['provenance'] != provenance:
            raise RuntimeError('Existing full-evidence fusion has different provenance; preserve it')
        module.load_state_dict(saved['state']); report = saved['report']
    else:
        for epoch in range(1, config['event_epochs'] + 1):
            epoch_start = time.perf_counter(); optimizer.zero_grad(); losses = []
            for step, index in enumerate(np.random.default_rng(args.seed + epoch).permutation(n_train)):
                row = records['train'][index]
                loss = nn.functional.binary_cross_entropy_with_logits(forward(row), torch.tensor([row['label']]))
                weight = n_train / (n_people * counts[row['person']])
                remaining = n_train - (step // config['effective_batch']) * config['effective_batch']
                (loss * weight / min(config['effective_batch'], remaining)).backward()
                losses.append(float(loss.detach()) * weight)
                if (step + 1) % config['effective_batch'] == 0 or step + 1 == n_train:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1)
                    optimizer.step(); optimizer.zero_grad()
            val = metrics(evaluate('validation'))
            curve.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation=val,
                              elapsed_seconds=time.perf_counter() - epoch_start))
            if epoch >= config['minimum_selection_epoch'] and val['nll'] < best:
                best, selected = val['nll'], epoch
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            print(json.dumps(dict(event='epoch_complete', method=METHOD, **curve[-1])), flush=True)
        if best_state is None:
            raise RuntimeError('No eligible checkpoint')
        module.load_state_dict(best_state)
        report = dict(selected_epoch=selected, selection_metric='validation_NLL', curve=curve,
                      training_and_validation_seconds=time.perf_counter() - train_start,
                      additional_feature_cache_seconds=feature_seconds, trainable_parameters=897)
        torch.save(dict(state=best_state, report=report, provenance=provenance), path)
    predictions = {}
    for split in ('validation', 'evaluation'):
        if records[split]:
            t = time.perf_counter(); predictions[split] = evaluate(split)
            report[split] = metrics(predictions[split]); report[f'{split}_inference_seconds'] = time.perf_counter() - t
    results_path = out / f'{prefix}_metrics.json'
    results = json.loads(results_path.read_text()); results[METHOD] = report; save_json(results_path, results)
    prediction_path = out / f'{prefix}_predictions.csv'
    with prediction_path.open(newline='', encoding='utf8') as f:
        previous = list(csv.DictReader(f))
    previous = [r for r in previous if r['method'] != METHOD]
    with prediction_path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'split', 'person', 'origin', 'label', 'logit', 'probability'])
        writer.writeheader(); writer.writerows(previous)
        for split, rows in predictions.items():
            writer.writerows(dict(method=METHOD, split=split, **row) for row in rows)
    base = {}
    for split in predictions:
        base[split] = [dict(person=r['person'], origin=int(r['origin']), label=float(r['label']),
                            probability=float(r['probability'])) for r in previous if r['method'] == 'M04_C2_LoRA' and r['split'] == split]
    paired_path = out / f'{prefix}_paired.json'
    paired = json.loads(paired_path.read_text())
    paired.update(paired_bootstrap({'M04_C2_LoRA': base, METHOD: predictions}, args.seed, config['bootstrap_repetitions']))
    save_json(paired_path, paired)
    timing_path = out / f'{prefix}_timing.json'
    timing = json.loads(timing_path.read_text())
    timing['full_evidence_fusion_additional_wall_seconds'] = time.perf_counter() - started
    timing['full_evidence_fusion_feature_seconds'] = feature_seconds
    memory = psutil.Process().memory_info()
    timing['full_evidence_fusion_peak_rss_bytes'] = getattr(memory, 'peak_wset', memory.rss)
    timing['method_training_seconds'][METHOD] = report['training_and_validation_seconds']
    save_json(timing_path, timing)
    save_json(directory / 'full_evidence_fusion_status.json', dict(status='completed', report=report,
                                                                 elapsed_seconds=time.perf_counter() - started))
    print(json.dumps(dict(event='full_evidence_fusion_complete', metrics=report['validation'],
                          elapsed_seconds=time.perf_counter() - started)), flush=True)


if __name__ == '__main__':
    main()
