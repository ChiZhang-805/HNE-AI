"""Matched baseline, attention guidance, logit and hidden fusion training."""
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from train_backbone import (load_backbone, capture_prefix, encoder_tail, read_event,
                            adapter_state, load_adapter)
from train_evidence import fit_small_components, calibrate_person, origin_evidence
from training_data import load_people, component_assignments, selected_origins, SOURCE
from precursor_guidance import EvidenceBias
from pack_guidance import pack_bias
from fusion.evidence import FullEvidenceFusion, InputEvidenceResidual


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def resource_check(device):
    if psutil.virtual_memory().available < 2 * 1024 ** 3:
        raise RuntimeError('Less than 2 GiB free RAM; preserve other experiments and resume later')
    if device == 'cuda':
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                           '--format=csv,noheader'], text=True)
        foreign = [line for line in output.splitlines() if line.strip()
                   and int(line.split(',')[0]) != os.getpid()]
        if foreign:
            raise RuntimeError('GPU is occupied by another process; use CPU or resume after it finishes')


def metrics(records):
    y = np.asarray([r['label'] for r in records])
    p = np.asarray([r['probability'] for r in records])
    two = len(np.unique(y)) == 2
    return dict(origins=len(y), people=len(set(r['person'] for r in records)), positives=int(y.sum()),
                auroc=float(roc_auc_score(y, p)) if two else None,
                auprc=float(average_precision_score(y, p)) if y.sum() else None,
                brier=float(brier_score_loss(y, p)), nll=float(log_loss(y, p, labels=[0, 1])))


class ResidualFusion(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        size, width = (13, 57) if mode == 'logit' else (142, 5)
        self.head = nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, logit, feature, evidence):
        x = torch.cat((logit.reshape(1) if self.mode == 'logit' else feature, evidence))
        return logit.reshape(1) + self.head(x).reshape(1)


def execute(root, config):
    root = Path(root)
    directory = root / 'artifacts/key_comparison' / config['run_name'] / config['dataset'] / f"fold{config['fold']}" / f"seed{config['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    result_dir = root / 'results/key_comparison' / config['run_name'] / config['dataset']
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = directory / 'progress.jsonl'
    start = time.perf_counter()
    def log(event, **data):
        row = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   invocation_seconds=time.perf_counter() - start)
        row.update(data)
        with log_path.open('a', encoding='utf8') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    device = config['device']
    resource_check(device)
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    process = psutil.Process()
    if os.name == 'nt':
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    people = load_people(config['dataset'], config['fold'], config['include_evaluation'])
    assignments = component_assignments(people, config['seed'])
    sources = {str(p.relative_to(root)): sha(p) for p in sorted((root / 'src').rglob('*.py'))}
    sources['scripts/run_key_comparison.py'] = sha(root / 'scripts/run_key_comparison.py')
    source_paths = sorted({f for group in people.values() for person in group for f in person.source_files})
    manifest = dict(config=config, source_sha256=sources, data_sha256={p: sha(p) for p in source_paths},
                    original_model_sha256=sha(SOURCE / 'tools/models/chronos-2/model.safetensors'),
                    protocol_sha256=sha(root / 'configs/daily_protocol.json'),
                    split_people={k: [p.person for p in v] for k, v in people.items()},
                    small_component=assignments, torch=torch.__version__)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf8')) != manifest:
            raise RuntimeError('Run provenance changed; existing experiment is preserved. Use a distinct purpose name or explicitly reconcile it.')
    else:
        save_json(manifest_path, manifest)
        import zipfile
        with zipfile.ZipFile(directory / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for relative in sources:
                archive.write(root / relative, relative)
    status_path = directory / 'status.json'
    save_json(status_path, dict(status='running', pid=os.getpid(), config=config))
    log('start', device=device, threads=config['threads'], split_people=manifest['split_people'])
    timing = {}
    timing_path = directory / 'timing.json'
    if timing_path.exists():
        timing.update(json.loads(timing_path.read_text()))
    try:
        stage_start = time.perf_counter()
        small, small_report = fit_small_components(people, assignments, config, directory / 'small', log)
        timing.setdefault('small_cross_fit_seconds', time.perf_counter() - stage_start)
        save_json(timing_path, timing)
        items = {'train': [], 'validation': [], 'evaluation': []}
        stage_start = time.perf_counter()
        for split, persons in people.items():
            for person in persons:
                component = assignments[person.person]
                if person.person in small_report[component]['fit_people']:
                    raise AssertionError('S cross-fitting leaked the scored person')
                personal_dir = directory / 'evidence' / person.person
                personal_dir.mkdir(parents=True, exist_ok=True)
                cal_path = personal_dir / 'calibration.pt'
                if cal_path.exists():
                    cal = torch.load(cal_path, weights_only=False)
                else:
                    cal = calibrate_person(small[component], person, config['personal_steps'])
                    torch.save(cal, cal_path)
                maximum = config['train_origins_per_person'] if split == 'train' else config['validation_origins_per_person'] if split == 'validation' else 0
                for origin, label in selected_origins(person, maximum):
                    path = personal_dir / f'origin{origin}.pt'
                    if not path.exists():
                        torch.save(origin_evidence(small[component], person, cal, origin), path)
                    items[split].append(dict(person=person.person, origin=origin, label=label,
                                             evidence_path=path, source=person))
                log('personal_evidence_complete', person=person.person, split=split,
                    calibrated_nodes=int(cal['node_available'].sum()), calibrated_edges=int(cal['edge_available'].sum()))
        timing.setdefault('personal_fit_and_evidence_seconds', time.perf_counter() - stage_start)
        save_json(timing_path, timing)
        del small
        resource_check(device)
        stage_start = time.perf_counter()
        # Initialization is independent of whether S components were trained or resumed.
        torch.manual_seed(config['seed'])
        model, head = load_backbone(SOURCE / 'tools/models/chronos-2', device)
        timing['backbone_load_seconds'] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        verification = {}
        cache_dir = directory / 'prefix'; cache_dir.mkdir(exist_ok=True)
        for split, rows in items.items():
            for i, item in enumerate(rows):
                prefix_path = cache_dir / f"{item['person']}_{item['origin']}.pt"
                packed, state, family, keep = item['source'].large_window(item['origin'])
                if not prefix_path.exists():
                    cached = capture_prefix(model, to_device(packed, device))
                    torch.save(cached, prefix_path)
                item.update(prefix_path=prefix_path, state=state, family=family, keep=torch.from_numpy(keep))
                if not verification:
                    from chronos_bridge import encode_packed_segments
                    with torch.no_grad():
                        cached = to_device(torch.load(prefix_path, weights_only=True), device)
                        tail = encoder_tail(model, cached)
                        full = encode_packed_segments(model, **to_device(packed, device))[0]
                        error = float((tail - full).abs().max())
                        if not torch.equal(tail, full):
                            raise AssertionError(f'Frozen-prefix cache changed forward values: {error}')
                        ev = to_device(torch.load(item['evidence_path'], weights_only=True), device)
                        guide = EvidenceBias().to(device)
                        tb, gb, _ = pack_bias(guide, ev, cached, item['keep'])
                        guided = encoder_tail(model, cached, tb, gb)
                        if not torch.equal(tail, guided):
                            raise AssertionError('Zero initialized guidance changed baseline')
                        verification.update(full_pretrained_prefix_exact=True, max_error=error,
                                            zero_guidance_exact=True, original_parameter_count=119477664)
                if (i + 1) % 10 == 0 or i == len(rows) - 1:
                    resource_check(device); log('prefix_cache', split=split, complete=i + 1, total=len(rows))
        timing.setdefault('prefix_cache_seconds', time.perf_counter() - stage_start)
        save_json(timing_path, timing)
        save_json(directory / 'verification.json', verification)
        log('data_ready', counts={k: len(v) for k, v in items.items()})
        counts = {p.person: sum(r['person'] == p.person for r in items['train']) for p in people['train']}
        n_people = len(counts); n_train = len(items['train'])
        def get(item):
            cache = to_device(torch.load(item['prefix_path'], weights_only=True), device)
            ev = to_device(torch.load(item['evidence_path'], weights_only=True), device)
            return cache, ev
        def forward(item, mode, module, need_features=False):
            if mode in ('logit', 'hidden'):
                base = to_device(item['baseline'], device)
                ev = to_device(item['summary'], device)
                return module(base['logit'], base['feature'], ev), None
            if mode in ('full_hidden', 'full_logit', 'evidence'):
                base = to_device(item['baseline'], device)
                ev = to_device(torch.load(item['evidence_path'], weights_only=True), device)
                return module(base['logit'], base['feature'], ev, item['source'].family[0].to(device)), None
            if mode == 'input':
                from chronos_bridge import encode_packed_segments
                ev = to_device(torch.load(item['evidence_path'], weights_only=True), device)
                packed = to_device(item['source'].large_window(item['origin'])[0], device)
                residual = module(ev, item['source'].family[0].to(device), item['keep'])
                hidden = encode_packed_segments(model, **packed, input_residual=residual)[0]
                cache = to_device(torch.load(item['prefix_path'], weights_only=True), device)
                return read_event(head, hidden, cache, item['state'].to(device), item['family'].to(device), config['dataset']), None
            cache, ev = get(item)
            penalty = torch.tensor(0., device=device)
            tb = gb = None
            if mode == 'guidance':
                tb, gb, penalty = pack_bias(module, ev, cache, item['keep'])
            hidden = encoder_tail(model, cache, tb, gb)
            value = read_event(head, hidden, cache, item['state'].to(device), item['family'].to(device),
                               config['dataset'], need_features)
            return value, penalty
        def evaluate(rows, mode, module):
            predictions = []
            with torch.no_grad():
                for item in rows:
                    z, _ = forward(item, mode, module)
                    predictions.append(dict(person=item['person'], origin=item['origin'], label=item['label'],
                                            logit=float(z.item()), probability=float(z.sigmoid().item())))
            return predictions
        def snapshot(mode, module):
            if mode == 'baseline':
                return dict(lora=adapter_state(model), head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
            return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        def restore(mode, module, state):
            if mode == 'baseline':
                load_adapter(model, state['lora']); head.load_state_dict(state['head'])
            else:
                module.load_state_dict(state)
        outputs, reports = {}, {}
        stages = [('baseline', 'M04_C2_LoRA'), ('guidance', 'M10_SAGE_fixed'),
                  ('logit', 'M05_S_logit_residual'), ('hidden', 'M12_S_hidden_late_fusion')]
        if config.get('complete_evidence_controls'):
            stages += [('full_hidden', 'M12_full_evidence_fusion'), ('full_logit', 'M13_full_evidence_logit_fusion'),
                       ('evidence', 'M02_S_evidence_only'), ('input', 'M06_full_evidence_input_residual')]
        for stage_index, (mode, name) in enumerate(stages):
            resource_check(device)
            torch.manual_seed(config['seed'] + stage_index)
            if mode == 'baseline':
                module = head
                groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': .0001},
                          {'params': head.parameters(), 'lr': .001}]
            else:
                model.requires_grad_(False); head.requires_grad_(False)
                if mode == 'guidance':
                    module = EvidenceBias()
                elif mode in ('full_hidden', 'full_logit', 'evidence'):
                    module = FullEvidenceFusion({'full_hidden': 'hidden', 'full_logit': 'logit', 'evidence': 'evidence'}[mode])
                elif mode == 'input':
                    module = InputEvidenceResidual()
                else:
                    module = ResidualFusion(mode)
                module = module.to(device)
                groups = [{'params': module.parameters(), 'lr': .001}]
            optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
            params = [p for group in optimizer.param_groups for p in group['params']]
            path = directory / f'{name}.pt'; resume = directory / f'{name}_training.pt'
            elapsed_prior = 0.; curve = []; best = float('inf'); best_state = None; selected = 0; stale = 0
            trained_best = float('inf'); trained_state = None; trained_epoch = None
            stage_start = time.perf_counter()
            if path.exists():
                done = torch.load(path, map_location='cpu', weights_only=False)
                restore(mode, module, done['state']); report = done['report']
                log('stage_resumed_complete', method=name)
            else:
                first_epoch = 1
                if config.get('include_noop_candidate') and mode not in ('baseline', 'evidence'):
                    initial_metrics = metrics(evaluate(items['validation'], mode, module))
                    best, best_state = initial_metrics['nll'], snapshot(mode, module)
                    curve.append(dict(epoch=0, train_loss=None, validation=initial_metrics,
                                      elapsed_seconds=0., candidate='exact_baseline_noop'))
                if resume.exists():
                    saved = torch.load(resume, map_location='cpu', weights_only=False)
                    restore(mode, module, saved['state']); optimizer.load_state_dict(saved['optimizer'])
                    curve, best, best_state, selected, stale, elapsed_prior = (saved[k] for k in ('curve', 'best', 'best_state', 'selected', 'stale', 'elapsed_seconds'))
                    first_epoch = saved['epoch'] + 1
                    trained_best = saved.get('trained_best', best)
                    trained_state = saved.get('trained_state', best_state)
                    trained_epoch = saved.get('trained_epoch', selected)
                for epoch in range(first_epoch, config['event_epochs'] + 1):
                    resource_check(device)
                    epoch_start = time.perf_counter(); optimizer.zero_grad(); losses = []
                    order = np.random.default_rng(config['seed'] + epoch).permutation(n_train)
                    for step, index in enumerate(order):
                        item = items['train'][index]
                        z, penalty = forward(item, mode, module)
                        target = torch.tensor([item['label']], device=device)
                        loss = nn.functional.binary_cross_entropy_with_logits(z, target)
                        if penalty is not None:
                            loss = loss + .0001 * penalty
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Nonfinite {name} loss')
                        weight = n_train / (n_people * counts[item['person']])
                        # Last partial batch is normalized by its actual batch size.
                        remaining = n_train - (step // config['effective_batch']) * config['effective_batch']
                        denom = min(config['effective_batch'], remaining)
                        (loss * weight / denom).backward()
                        losses.append(float(loss.detach()) * weight)
                        if (step + 1) % config['effective_batch'] == 0 or step + 1 == n_train:
                            torch.nn.utils.clip_grad_norm_(params, 1.); optimizer.step(); optimizer.zero_grad()
                        if (step + 1) % 32 == 0:
                            resource_check(device); log('training_progress', method=name, epoch=epoch, complete=step + 1, total=n_train)
                    val = evaluate(items['validation'], mode, module)
                    m = metrics(val)
                    curve.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation=m,
                                      elapsed_seconds=time.perf_counter() - epoch_start))
                    if epoch >= config['minimum_selection_epoch']:
                        if m['nll'] < trained_best:
                            trained_best, trained_state, trained_epoch = m['nll'], snapshot(mode, module), epoch
                        if m['nll'] < best:
                            best, best_state, selected, stale = m['nll'], snapshot(mode, module), epoch, 0
                        else:
                            stale += 1
                    elapsed = elapsed_prior + time.perf_counter() - stage_start
                    torch.save(dict(state=snapshot(mode, module), optimizer=optimizer.state_dict(), epoch=epoch,
                                    curve=curve, best=best, best_state=best_state, selected=selected,
                                    stale=stale, elapsed_seconds=elapsed, trained_best=trained_best,
                                    trained_state=trained_state, trained_epoch=trained_epoch), resume)
                    log('epoch_complete', method=name, **curve[-1])
                    if stale >= config['event_patience'] and epoch >= config.get('minimum_training_epochs', 1):
                        break
                if best_state is None:
                    raise ValueError('Epoch budget is below minimum selection epoch')
                restore(mode, module, best_state)
                report = dict(selected_epoch=selected, selection_metric='validation_NLL', curve=curve,
                              training_and_validation_seconds=elapsed_prior + time.perf_counter() - stage_start,
                              trainable_parameters=sum(p.numel() for p in params),
                              trained_selected_epoch=trained_epoch,
                              noop_candidate=config.get('include_noop_candidate', False) and mode not in ('baseline', 'evidence'))
                torch.save(dict(state=best_state, report=report, trained_state=trained_state), path)
            outputs[name] = {}
            for split in ('validation', 'evaluation'):
                if items[split]:
                    pred_start = time.perf_counter()
                    outputs[name][split] = evaluate(items[split], mode, module)
                    report[split] = metrics(outputs[name][split])
                    report[f'{split}_inference_seconds'] = time.perf_counter() - pred_start
            if report.get('noop_candidate'):
                saved = torch.load(path, map_location='cpu', weights_only=False)
                restore(mode, module, saved['trained_state'])
                active_name = name + '_always_on'
                outputs[active_name] = {}
                report['always_on'] = {}
                for split in ('validation', 'evaluation'):
                    if items[split]:
                        outputs[active_name][split] = evaluate(items[split], mode, module)
                        report['always_on'][split] = metrics(outputs[active_name][split])
                restore(mode, module, saved['state'])
            reports[name] = report
            save_json(result_dir / f"fold{config['fold']}_seed{config['seed']}_metrics.json", reports)
            log('method_complete', method=name, metrics={k: report[k] for k in ('validation', 'evaluation') if k in report})
            if mode == 'baseline':
                # Freeze the selected baseline once; all other methods reuse it exactly.
                model.requires_grad_(False); head.requires_grad_(False)
                model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
                feature_start = time.perf_counter()
                with torch.no_grad():
                    for rows in items.values():
                        for item in rows:
                            (z, feature), _ = forward(item, 'baseline', None, True)
                            item['baseline'] = dict(logit=z[0].cpu(), feature=feature[0].cpu())
                            item['summary'] = torch.load(item['evidence_path'], weights_only=True)['summary']
                timing['baseline_feature_cache_seconds'] = time.perf_counter() - feature_start
                frozen_baseline = snapshot('baseline', head)
            else:
                check_state = snapshot('baseline', head)
                assert all(torch.equal(v, check_state[k][n]) for k in frozen_baseline for n, v in frozen_baseline[k].items())
                assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in head.parameters())
                verification[f'{name}_baseline_frozen'] = True
        predictions_path = result_dir / f"fold{config['fold']}_seed{config['seed']}_predictions.csv"
        with predictions_path.open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'split', 'person', 'origin', 'label', 'logit', 'probability'])
            writer.writeheader()
            for name, splits in outputs.items():
                for split, rows in splits.items():
                    for row in rows:
                        writer.writerow(dict(method=name, split=split, **row))
        paired = paired_bootstrap(outputs, config['seed'], config['bootstrap_repetitions'])
        save_json(result_dir / f"fold{config['fold']}_seed{config['seed']}_paired.json", paired)
        timing['invocation_wall_seconds'] = time.perf_counter() - start
        timing['process_peak_rss_bytes'] = getattr(process.memory_info(), 'peak_wset', process.memory_info().rss)
        timing['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
        timing['cache_bytes'] = sum(p.stat().st_size for p in cache_dir.glob('*.pt'))
        timing['method_training_seconds'] = {name: report['training_and_validation_seconds'] for name, report in reports.items()}
        save_json(timing_path, timing); save_json(directory / 'verification.json', verification)
        save_json(result_dir / f"fold{config['fold']}_seed{config['seed']}_timing.json", timing)
        save_json(status_path, dict(status='completed', pid=os.getpid(), config=config, timing=timing))
        log('completed', timing=timing)
    except BaseException as exc:
        save_json(status_path, dict(status='failed', pid=os.getpid(), error=repr(exc), config=config,
                                   elapsed_seconds=time.perf_counter() - start))
        log('failed', error=repr(exc)); raise


def paired_bootstrap(outputs, seed, repetitions):
    results = {}
    baseline = outputs['M04_C2_LoRA']
    for split, rows in baseline.items():
        ids = np.asarray([r['person'] for r in rows]); y = np.asarray([r['label'] for r in rows])
        p0 = np.asarray([r['probability'] for r in rows]); people = np.unique(ids)
        for method, splits in outputs.items():
            if method == 'M04_C2_LoRA':
                continue
            target = splits[split]
            if [(r['person'], r['origin']) for r in rows] != [(r['person'], r['origin']) for r in target]:
                raise AssertionError('Paired predictions do not match')
            p1 = np.asarray([r['probability'] for r in target])
            rng = np.random.default_rng(seed); aucs = []; briers = []
            for _ in range(repetitions):
                ix = np.concatenate([np.flatnonzero(ids == p) for p in rng.choice(people, len(people), replace=True)])
                if len(np.unique(y[ix])) == 2:
                    aucs.append(roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix]))
                briers.append(np.mean((y[ix] - p1[ix]) ** 2 - (y[ix] - p0[ix]) ** 2))
            results[f'{split}/{method}'] = dict(
                delta_auroc=float(roc_auc_score(y, p1) - roc_auc_score(y, p0)) if len(np.unique(y)) == 2 else None,
                delta_auroc_ci95=np.quantile(aucs, [.025, .975]).tolist() if aucs else None,
                delta_brier=float(np.mean((y - p1) ** 2 - (y - p0) ** 2)),
                delta_brier_ci95=np.quantile(briers, [.025, .975]).tolist(),
                valid_auroc_repetitions=len(aucs), people=len(people),
                warning='Validation is used for checkpoint selection; these are exploratory, not confirmatory CIs.')
    return results
