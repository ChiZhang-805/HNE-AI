"""A-side matched M04/M10/M12 development training with six residual candidates.

Private artifacts are immutable by config/source/input provenance. The original
trainer remains available; scientific kernels are imported from shared modules.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import psutil
import torch
from torch import nn
from conditional_teacher import teacher_from_checkpoint
from training_data import load_people, selected_origins, SOURCE
from train_evidence import calibrate_person, origin_evidence
from train_backbone import load_backbone, capture_prefix, encoder_tail, read_event, adapter_state, load_adapter
from train_comparison import save_json, sha, metrics, resource_check, paired_bootstrap
from precursor_guidance import EvidenceBias
from pack_guidance import pack_bias
from fusion.evidence import FullEvidenceFusion


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def copied_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def same_tensors(left, right):
    return left.keys() == right.keys() and all(torch.equal(v, right[k]) for k, v in left.items())


def data_key(path):
    return str(path).replace('\\', '/').split('/data/', 1)[1]


def same_evidence(evidence):
    """One view used by both consumers; quality summaries are not extra inputs."""
    return dict(node=evidence['node'], node_valid=evidence['node_valid'],
                edge=evidence['edge'], edge_valid=evidence['edge_valid'],
                summary=torch.zeros_like(evidence['summary']))


def train_stage(mode, candidate, lr, decay, config, directory, items, model, head, log):
    torch.manual_seed(config['seed'] + {'baseline': 0, 'guidance': 1, 'fusion': 2}[mode])
    module = head if mode == 'baseline' else EvidenceBias() if mode == 'guidance' else FullEvidenceFusion('hidden')
    groups = ([dict(params=[p for p in model.parameters() if p.requires_grad], lr=config['baseline_lora_lr']),
               dict(params=head.parameters(), lr=config['baseline_head_lr'])] if mode == 'baseline'
              else [dict(params=module.parameters(), lr=lr)])
    optimizer = torch.optim.AdamW(groups, weight_decay=decay)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    target = directory / mode / f'candidate{candidate}.pt'
    resume = target.with_name(target.stem + '_training.pt')
    target.parent.mkdir(parents=True, exist_ok=True)

    def snapshot():
        return dict(lora=adapter_state(model), head=copied_state(head)) if mode == 'baseline' else copied_state(module)

    def restore(state):
        if mode == 'baseline':
            load_adapter(model, state['lora']); head.load_state_dict(state['head'])
        else:
            module.load_state_dict(state)

    def forward(item, off=False):
        if off:
            return item['baseline']['logit'].reshape(1), torch.tensor(0.)
        if mode == 'fusion':
            base = item['baseline']
            return module(base['logit'], base['feature'], item['evidence'], item['source'].family[0]), torch.tensor(0.)
        cached = torch.load(item['prefix_path'], weights_only=True)
        tb = gb = None
        penalty = torch.tensor(0.)
        if mode == 'guidance':
            tb, gb, penalty = pack_bias(module, item['evidence'], cached, item['keep'])
        hidden = encoder_tail(model, cached, tb, gb)
        return read_event(head, hidden, cached, item['state'], item['family'], config['dataset']), penalty

    def evaluate(rows, off=False):
        with torch.no_grad():
            output = []
            for item in rows:
                z, _ = forward(item, off)
                output.append(dict(person=item['person'], origin=item['origin'], label=item['label'],
                                   logit=float(z.item()), probability=float(z.sigmoid().item())))
            return output

    if target.exists():
        done = torch.load(target, weights_only=False)
        restore(done['state'])
        return module, done, evaluate, restore
    started = time.perf_counter()
    curve, first_epoch, elapsed_prior = [], 1, 0.
    best, best_state, best_epoch = float('inf'), None, None
    trained_best, trained_state, trained_epoch = float('inf'), None, None
    if mode != 'baseline':
        initial = evaluate(items['validation'])
        off = evaluate(items['validation'], True)
        assert all(a['logit'] == b['logit'] for a, b in zip(initial, off)), 'Residual epoch zero differs from common M04'
        best, best_state, best_epoch = metrics(initial)['nll'], snapshot(), 0
        curve.append(dict(epoch=0, validation=metrics(initial), seconds=0.))
    if resume.exists():
        saved = torch.load(resume, weights_only=False)
        restore(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        curve, best, best_state, best_epoch, trained_best, trained_state, trained_epoch = (
            saved[k] for k in ('curve', 'best', 'best_state', 'best_epoch', 'trained_best', 'trained_state', 'trained_epoch'))
        first_epoch, elapsed_prior = saved['epoch'] + 1, saved['seconds']
    counts = {p: sum(r['person'] == p for r in items['train']) for p in {r['person'] for r in items['train']}}
    n_train = len(items['train'])
    for epoch in range(first_epoch, config['event_epochs'] + 1):
        resource_check('cpu')
        epoch_start = time.perf_counter(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, index in enumerate(np.random.default_rng(config['seed'] + epoch).permutation(n_train)):
            item = items['train'][index]
            z, penalty = forward(item)
            loss = nn.functional.binary_cross_entropy_with_logits(z, torch.tensor([item['label']]))
            loss = loss + config['guidance_penalty'] * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            weight = n_train / (len(counts) * counts[item['person']])
            denom = min(config['effective_batch'], n_train - (step // config['effective_batch']) * config['effective_batch'])
            (loss * weight / denom).backward(); losses.append(float(loss.detach()) * weight)
            if (step + 1) % config['effective_batch'] == 0 or step + 1 == n_train:
                nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                resource_check('cpu')
        validation = metrics(evaluate(items['validation']))
        curve.append(dict(epoch=epoch, validation=validation, train_loss=float(np.mean(losses)), seconds=time.perf_counter() - epoch_start))
        if validation['nll'] < trained_best:
            trained_best, trained_state, trained_epoch = validation['nll'], snapshot(), epoch
        if validation['nll'] < best:
            best, best_state, best_epoch = validation['nll'], snapshot(), epoch
        torch.save(dict(state=snapshot(), optimizer=optimizer.state_dict(), epoch=epoch, curve=curve,
                        best=best, best_state=best_state, best_epoch=best_epoch, trained_best=trained_best,
                        trained_state=trained_state, trained_epoch=trained_epoch,
                        seconds=elapsed_prior + time.perf_counter() - started), resume)
        log('epoch_complete', method=mode, candidate=candidate,
            **{('epoch_seconds' if k == 'seconds' else k): v for k, v in curve[-1].items()})
    done = dict(state=best_state, trained_state=trained_state, report=dict(candidate=candidate, lr=lr, weight_decay=decay,
                selected_epoch=best_epoch, selected_nll=best, always_on_epoch=trained_epoch, always_on_nll=trained_best,
                epochs_completed=config['event_epochs'], curve=curve, trainable_parameters=sum(p.numel() for p in parameters),
                training_validation_seconds=elapsed_prior + time.perf_counter() - started))
    torch.save(done, target); restore(best_state)
    return module, done, evaluate, restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/core/config.json')
    parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config['device'] != 'cpu' or config['threads'] != 1 or config['include_evaluation']:
        raise ValueError('This development entry is CPU one thread and source-only')
    if config['event_epochs'] < 6 or config['fusion_evidence'] != 'node_edge_only':
        raise ValueError('Expected six or more training epochs and matched node/edge evidence')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu')
    digest = sha(args.config)
    key = f"core/{config['dataset']}/fold{config['fold']}/seed{config['seed']}/M04_M10_M12/{digest[:16]}"
    directory = ROOT / 'artifacts/experiments' / key
    output = ROOT / 'results/experiments' / key
    directory.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    teacher_root = ROOT / 'artifacts/experiments' / config['teacher_run']
    teacher_manifest = read_json(teacher_root / 'manifest.json')
    selection = read_json(teacher_root / 'selection.json')
    assert read_json(teacher_root / 'status.json')['status'] == 'completed'
    assert selection['manifest_sha256'] == sha(teacher_root / 'manifest.json')
    assert teacher_manifest['config']['dataset'] == config['dataset'] and teacher_manifest['config']['fold'] == config['fold']
    assert teacher_manifest['config']['seed'] == config['seed'] and not teacher_manifest['event_labels_loaded']
    people = load_people(config['dataset'], config['fold'], False)
    data = {data_key(p): sha(p) for group in people.values() for person in group for p in person.source_files}
    teacher_data = {data_key(p): h for p, h in teacher_manifest['data_sha256'].items()}
    assert all(teacher_data[p] == h for p, h in data.items())
    sources = [Path(__file__).resolve(), args.config.resolve(), *sorted((ROOT / 'src').rglob('*.py'))]
    manifest = dict(config=config, config_sha256=digest, code_commit=args.code_commit,
                    protocol_sha256=sha(ROOT / 'configs/daily_protocol.json'), data_sha256=data,
                    model_sha256=sha(SOURCE / 'tools/models/chronos-2/model.safetensors'),
                    source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                    teacher_selection_sha256=sha(teacher_root / 'selection.json'), selected_teachers=selection['selected'],
                    split_people={k: [p.person for p in v] for k, v in people.items()},
                    scope=config['scope'])
    assert manifest['protocol_sha256'] == teacher_manifest['protocol_sha256']
    prior_seconds = 0.
    if (directory / 'manifest.json').exists():
        if read_json(directory / 'manifest.json') != manifest:
            continuation = read_json(directory / 'continuation.json') if (directory / 'continuation.json').exists() else {}
            if (continuation.get('original_manifest_sha256') != sha(directory / 'manifest.json')
                    or continuation.get('runtime_manifest') != manifest):
                raise RuntimeError('Run provenance differs; existing outputs preserved')
            prior_seconds = continuation['prior_invocation_seconds']
    else:
        save_json(directory / 'manifest.json', manifest)
    if (directory / 'status.json').exists() and read_json(directory / 'status.json')['status'] == 'completed':
        print(json.dumps(dict(status='completed', directory=str(directory)))); return
    if not (directory / 'sources.zip').exists():
        with zipfile.ZipFile(directory / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sources:
                archive.write(file, file.relative_to(ROOT).as_posix())
    started = time.perf_counter()
    def log(event, **values):
        row = dict(event=event, seconds=time.perf_counter() - started, **values)
        with (directory / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    save_json(directory / 'status.json', dict(status='running', pid=os.getpid(), run_key=key))
    try:
        items = {k: [] for k in people}; evidence_receipts = []
        teachers = {}
        for chosen in selection['selected']:
            path = ROOT / chosen['path']; assert sha(path) == chosen['sha256']
            checkpoint = torch.load(path, weights_only=False)
            teachers[chosen['component']] = (teacher_from_checkpoint(checkpoint), checkpoint)
        evidence_start = time.perf_counter()
        for split, persons in people.items():
            for person in persons:
                component = teacher_manifest['small_component'][person.person]
                teacher, checkpoint = teachers[component]
                assert person.person not in checkpoint['report']['fit_people']
                assert person.person not in checkpoint['report']['selection_holdout']
                personal_dir = directory / 'evidence' / person.person; personal_dir.mkdir(parents=True, exist_ok=True)
                cal_path = personal_dir / 'calibration.pt'
                if not cal_path.exists():
                    torch.save(calibrate_person(teacher, person, config['personal_steps']), cal_path)
                calibration = torch.load(cal_path, weights_only=False)
                maximum = config['train_origins_per_person'] if split == 'train' else config['validation_origins_per_person']
                for origin, label in selected_origins(person, maximum):
                    path = personal_dir / f'origin{origin}.pt'
                    if not path.exists():
                        evidence = origin_evidence(teacher, person, calibration, origin)
                        if checkpoint['architecture'] == 'history':
                            evidence['edge'].zero_(); evidence['edge_valid'].zero_()
                        torch.save(evidence, path)
                    evidence = torch.load(path, weights_only=True)
                    assert evidence['node'].shape == (6, 21, 9) and evidence['edge'].shape == (21, 6, 6, 4)
                    evidence_receipts.append(dict(person=person.person, origin=origin, component=component,
                                                  sha256=sha(path), calibration_sha256=sha(cal_path)))
                    items[split].append(dict(person=person.person, origin=origin, label=label,
                                            evidence=same_evidence(evidence), evidence_path=path, source=person))
        save_json(directory / 'evidence_receipts.json', evidence_receipts)
        evidence_seconds = time.perf_counter() - evidence_start
        del teachers
        torch.manual_seed(config['seed'])
        model, head = load_backbone(SOURCE / 'tools/models/chronos-2')
        # Reuse only frozen pretrained prefixes, and recompute every one for exact
        # equality. Historical LoRA, teachers and event heads are never imported.
        prefix_start = time.perf_counter(); prefix_receipts = []
        prefix_reference = ROOT / config['prefix_reference'] / 'prefix'
        own_prefix = directory / 'prefix'; own_prefix.mkdir(exist_ok=True)
        previous_prefixes = {Path(r['path']).name: r for r in read_json(directory / 'prefix_receipts.json')} if (directory / 'prefix_receipts.json').exists() else {}
        for split, rows in items.items():
            for item in rows:
                name = f"{item['person']}_{item['origin']}.pt"
                path = prefix_reference / name
                packed, state, family, keep = item['source'].large_window(item['origin'])
                if name in previous_prefixes:
                    receipt = previous_prefixes[name]
                    path = ROOT / receipt['path']
                    assert receipt['exact_recompute'] and sha(path) == receipt['sha256']
                    item.update(prefix_path=path, state=state, family=family, keep=torch.from_numpy(keep))
                    prefix_receipts.append(receipt)
                    resource_check('cpu')
                    continue
                fresh = capture_prefix(model, packed)
                borrowed_exact = path.exists() and same_tensors(fresh, torch.load(path, weights_only=True))
                rejected_hash = sha(path) if path.exists() and not borrowed_exact else None
                if not borrowed_exact:
                    path = own_prefix / name
                    if path.exists():
                        assert same_tensors(fresh, torch.load(path, weights_only=True))
                    else:
                        torch.save(fresh, path)
                item.update(prefix_path=path, state=state, family=family, keep=torch.from_numpy(keep))
                prefix_receipts.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), exact_recompute=True,
                                            borrowed_exact=borrowed_exact, rejected_historical_sha256=rejected_hash))
                resource_check('cpu')
        save_json(directory / 'prefix_receipts.json', prefix_receipts)
        prefix_seconds = time.perf_counter() - prefix_start
        log('data_ready', counts={k: len(v) for k, v in items.items()}, evidence_seconds=evidence_seconds, prefix_seconds=prefix_seconds)
        module, base, evaluate, _ = train_stage('baseline', 0, config['baseline_head_lr'], config['baseline_weight_decay'],
                                               config, directory, items, model, head, log)
        outputs = {'M04_C2_LoRA': {'validation': evaluate(items['validation'])}}
        model.requires_grad_(False); head.requires_grad_(False)
        model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        frozen = dict(lora=adapter_state(model), head=copied_state(head))
        feature_start = time.perf_counter()
        with torch.no_grad():
            for rows in items.values():
                for item in rows:
                    cached = torch.load(item['prefix_path'], weights_only=True)
                    z, feature = read_event(head, encoder_tail(model, cached), cached, item['state'], item['family'], config['dataset'], True)
                    item['baseline'] = dict(logit=z[0], feature=feature[0])
        torch.save({f"{x['person']}/{x['origin']}": x['baseline'] for rows in items.values() for x in rows}, directory / 'baseline_features.pt')
        feature_seconds = time.perf_counter() - feature_start
        method_reports, verification = {'M04': base['report']}, dict(prefix_exact_count=len(prefix_receipts), same_evidence_summary_zero=True)
        for mode, method in [('guidance', 'M10'), ('fusion', 'M12')]:
            candidates = []
            for candidate, (lr, decay) in enumerate(itertools.product(config['learning_rates'], config['weight_decays'])):
                module, done, evaluate, restore = train_stage(mode, candidate, lr, decay, config, directory, items, model, head, log)
                selected = evaluate(items['validation'])
                restore(done['trained_state']); always_on = evaluate(items['validation'])
                off = evaluate(items['validation'], True)
                assert all(a['logit'] == b['logit'] for a, b in zip(off, outputs['M04_C2_LoRA']['validation']))
                changes = np.asarray([a['logit'] - b['logit'] for a, b in zip(always_on, off)])
                done['report']['intervention'] = dict(mean_abs_logit_change=float(np.abs(changes).mean()),
                                                      max_abs_logit_change=float(np.abs(changes).max()))
                if mode == 'guidance':
                    example = items['validation'][0]
                    cached = torch.load(example['prefix_path'], weights_only=True)
                    with torch.no_grad():
                        tb, gb, _ = pack_bias(module, example['evidence'], cached, example['keep'])
                        hidden_delta = encoder_tail(model, cached, tb, gb) - encoder_tail(model, cached)
                    done['report']['intervention'].update(time_bias_max=float(tb.abs().max()),
                                                         group_bias_max=float(gb.abs().max()),
                                                         example_hidden_delta_max=float(hidden_delta.abs().max()))
                assert same_tensors(frozen['lora'], adapter_state(model)) and same_tensors(frozen['head'], copied_state(head))
                assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in head.parameters())
                candidates.append(dict(report=done['report'], selected=selected, always_on=always_on))
                log('candidate_complete', method=method, candidate=candidate,
                    selected_epoch=done['report']['selected_epoch'], selected_nll=done['report']['selected_nll'])
            selected_index = min(range(len(candidates)), key=lambda i: (candidates[i]['report']['selected_nll'], i))
            on_index = min(range(len(candidates)), key=lambda i: (candidates[i]['report']['always_on_nll'], i))
            outputs[f'{method}_selected'] = {'validation': candidates[selected_index]['selected']}
            outputs[f'{method}_always_on'] = {'validation': candidates[on_index]['always_on']}
            method_reports[method] = dict(selected_candidate=selected_index, always_on_candidate=on_index,
                                          candidates=[c['report'] for c in candidates])
            selected_checkpoint = torch.load(directory / mode / f'candidate{selected_index}.pt', weights_only=False)
            on_checkpoint = torch.load(directory / mode / f'candidate{on_index}.pt', weights_only=False)
            torch.save(dict(state=selected_checkpoint['state'], trained_state=on_checkpoint['trained_state'],
                            report=method_reports[method], base_checkpoint_sha256=sha(directory / 'baseline/candidate0.pt'),
                            evidence_receipts_sha256=sha(directory / 'evidence_receipts.json')),
                       directory / f'{method}.pt')
            verification[method] = dict(baseline_unchanged=True, frozen_gradients_absent=True, off_exact=True, epoch_zero_exact=True)
            save_json(output / 'reports.json', method_reports)
        prediction_path = output / 'predictions.csv'
        with prediction_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['method', 'split', 'person', 'origin', 'label', 'logit', 'probability'])
            writer.writeheader()
            for method, splits in outputs.items():
                for split, rows in splits.items():
                    for row in rows:
                        writer.writerow(dict(method=method, split=split, **row))
        save_json(output / 'paired.json', paired_bootstrap(outputs, config['seed'], config['bootstrap_repetitions']))
        result = dict(status='completed', run_key=key, scope=config['scope'], seconds=prior_seconds + time.perf_counter() - started,
                      current_invocation_seconds=time.perf_counter() - started, prior_invocation_seconds=prior_seconds,
                      evidence_seconds=evidence_seconds, prefix_verify_seconds=prefix_seconds, feature_seconds=feature_seconds,
                      metrics={name: metrics(splits['validation']) for name, splits in outputs.items()},
                      reports=method_reports, verification=verification, gpu_used=False,
                      base_checkpoint_sha256=sha(directory / 'baseline/candidate0.pt'),
                      evidence_receipts_sha256=sha(directory / 'evidence_receipts.json'),
                      peak_rss_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss))
        save_json(output / 'summary.json', result)
        save_json(directory / 'status.json', dict(status='completed', seconds=result['seconds']))
        log('completed', metrics=result['metrics'])
    except BaseException as error:
        save_json(directory / 'status.json', dict(status='failed', seconds=time.perf_counter() - started, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
