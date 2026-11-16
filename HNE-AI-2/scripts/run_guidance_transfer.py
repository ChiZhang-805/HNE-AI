"""Score frozen Bris event models with paired historical-evidence interventions on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import psutil
import torch
from evidence_alignment import alignment_indices, relocate_innovations
from fusion.evidence import FullEvidenceFusion
from pack_guidance import pack_bias
from precursor_guidance import ConditionalPredictor, EvidenceBias
from train_backbone import load_backbone, load_adapter, capture_prefix, encoder_tail, read_event
from train_comparison import save_json, sha, resource_check, metrics
from train_evidence import calibrate_person, origin_evidence
from training_data import SOURCE, load_people, selected_origins, provenance_key

METHODS = ('baseline', 'aligned', 'reversed', 'full_fusion')


def freeze(module, file):
    module.load_state_dict(torch.load(file, map_location='cpu', weights_only=False)['state'])
    return module.eval().requires_grad_(False)


@torch.no_grad()
def forecast(person, origin, small, cal, backbone, head, guide, fusion, cached_prefix=None):
    evidence = origin_evidence(small, person, cal, origin)
    indices = alignment_indices(evidence)
    packed, state, family, keep = person.large_window(origin)
    cache = (torch.load(cached_prefix, weights_only=True) if cached_prefix is not None
             else capture_prefix(backbone, packed))
    keep = torch.from_numpy(keep)
    out = encoder_tail(backbone, cache)
    z, feature = read_event(head, out, cache, state, family, person.dataset, True)
    scores = {'baseline': float(z.sigmoid()), 'full_fusion': float(
        fusion(z[0], feature[0], evidence, person.family[0]).sigmoid())}
    times = {}
    for name in (('aligned', 'reversed') if origin % 2 == 0 else ('reversed', 'aligned')):
        start = time.perf_counter()
        ev = relocate_innovations(evidence, indices, name == 'reversed')
        tb, gb, _ = pack_bias(guide, ev, cache, keep)
        out = encoder_tail(backbone, cache, tb, gb)
        scores[name] = float(read_event(head, out, cache, state, family, person.dataset).sigmoid())
        times[name + '_tail_seconds'] = time.perf_counter() - start
    return {**scores, **times, 'valid_nodes': int(evidence['node_valid'].sum()),
            'valid_edges': int(evidence['edge_valid'].sum())}


def summarize(rows, cfg):
    reports = {}
    for method in METHODS:
        reports[method] = metrics(rows.rename(columns={method: 'probability'}).to_dict('records'))
    # Resample entire people. Repeated origins for that person always travel together.
    # Pool origin-level loss (same estimand as the table); no label-based resampling.
    rng = np.random.default_rng(cfg['seed'])
    contrasts = {}
    for comparator in ('baseline', 'reversed', 'full_fusion'):
        losses = {}
        for model in ('aligned', comparator):
            p = rows[model].to_numpy(dtype=float).clip(np.finfo(float).eps, 1 - np.finfo(float).eps)
            y = rows.label.to_numpy()
            losses[model] = -y * np.log(p) - (1 - y) * np.log1p(-p)
        delta = losses['aligned'] - losses[comparator]
        per_person = pd.DataFrame({'person': rows.person, 'delta': delta}).groupby('person').delta.agg(['sum', 'count'])
        draws = rng.integers(0, len(per_person), size=(cfg['bootstrap_repetitions'], len(per_person)))
        boot = per_person['sum'].to_numpy()[draws].sum(1) / per_person['count'].to_numpy()[draws].sum(1)
        contrasts[comparator] = dict(delta_nll=float(delta.mean()), interval=np.quantile(boot, [.025,.975]).tolist(),
            mean_absolute_probability_change=float((rows.aligned - rows[comparator]).abs().mean()),
            people_improved=int((per_person['sum'] < 0).sum()), people_worsened=int((per_person['sum'] > 0).sum()))
    return dict(metrics=reports, contrasts=contrasts, positive_people=int(rows.groupby('person').label.max().sum()),
                tail_seconds={m: float(rows[m + '_tail_seconds'].median()) for m in ('aligned','reversed')},
                zero_node_origins=int((rows.valid_nodes == 0).sum()), zero_edge_origins=int((rows.valid_edges == 0).sum()))


def main():
    started = time.perf_counter()
    cfg = json.loads((ROOT / 'configs/guidance_transfer.json').read_text(encoding='utf8'))
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu')
    source, target, fold, seed = (cfg[k] for k in ('source_dataset','target_dataset','outer_fold','seed'))
    checkpoint = ROOT / 'artifacts/key_comparison' / cfg['source_profile'] / source / f'fold{fold}' / f'seed{seed}'
    base_file = checkpoint / 'M04_C2_LoRA.pt'
    guide_file = checkpoint / 'M10_SAGE_fixed.pt'
    fusion_file = checkpoint / 'M12_full_evidence_fusion.pt'
    small_files = [checkpoint / 'small' / f'component{i}.pt' for i in range(3)]
    weight_dir = SOURCE / 'tools/models/chronos-2'
    groups = {d: load_people(d, fold, only_split=cfg['scored_split'])[cfg['scored_split']] for d in (target, source)}
    output = ROOT / 'results/guidance_transfer'
    artifacts = ROOT / 'artifacts/guidance_transfer'
    output.mkdir(parents=True, exist_ok=True)
    files = [base_file, guide_file, fusion_file, *small_files, weight_dir / 'model.safetensors',
             weight_dir / 'config.json', ROOT / 'configs/daily_protocol.json', Path(__file__),
             *sorted((ROOT / 'src').rglob('*.py'))]
    original = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf8'))
    if sha(weight_dir / 'model.safetensors') != original['original_model_sha256']:
        raise ValueError('Pretrained weights differ from the source run')
    provenance = dict(config=cfg, torch=str(torch.__version__), files={provenance_key(f, ROOT): sha(f) for f in files},
        data={provenance_key(f, ROOT): sha(f) for group in groups.values() for p in group for f in p.source_files})
    manifest = artifacts / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf8')) != provenance:
        raise RuntimeError('Inputs differ from the preserved transfer run')
    if not manifest.exists():
        save_json(manifest, provenance)
    status = artifacts / 'status.json'
    if status.exists() and json.loads(status.read_text())['state'] == 'complete' and (output / 'summary.json').exists():
        print('Completed transfer already present', flush=True)
        return
    save_json(status, dict(state='running', pid=os.getpid()))
    model, head = load_backbone(weight_dir)
    base = torch.load(base_file, map_location='cpu', weights_only=False)['state']
    load_adapter(model, base['lora'])
    head.load_state_dict(base['head'])
    model.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    guide = freeze(EvidenceBias(), guide_file)
    fusion = freeze(FullEvidenceFusion(), fusion_file)
    small = [freeze(ConditionalPredictor(), f) for f in small_files]
    report = {}
    for dataset, people in groups.items():
        completed = []
        for person in people:
            resource_check('cpu')
            path = artifacts / dataset / (person.person + '.json')
            if path.exists():
                completed.append(json.loads(path.read_text()))
                continue
            mark = time.perf_counter()
            key = f'{dataset}|{person.person}|{seed}'.encode()
            component = int(hashlib.sha256(key).hexdigest(), 16) % len(small)
            if dataset == source and component != original['small_component'][person.person]:
                raise ValueError('Source small-model assignment changed')
            cal = calibrate_person(small[component], person, cfg['personal_steps'])
            cal_seconds = time.perf_counter() - mark
            rows = []
            origins = selected_origins(person, cfg['origins_per_person'])
            for i, (origin, label) in enumerate(origins):
                prefix = checkpoint / 'prefix' / f'{person.person}_{origin}.pt' if dataset == source else None
                values = forecast(person, origin, small[component], cal, model, head, guide, fusion, prefix)
                if not all(np.isfinite(values[m]) for m in METHODS):
                    raise FloatingPointError('Nonfinite probability')
                rows.append(dict(person=person.person, fold=person.fold, origin=origin, label=int(label), **values))
                if (i+1) % 20 == 0:
                    print(f'{dataset} {person.person}: {i+1}/{len(origins)}', flush=True)
            record = dict(component=component, calibration_seconds=cal_seconds,
                          processing_seconds=time.perf_counter()-mark, rows=rows)
            save_json(path, record)
            completed.append(record)
            print(f'{dataset} person {len(completed)}/{len(people)}, {len(rows)} origins', flush=True)
        frame = pd.DataFrame([r for p in completed for r in p['rows']])
        if frame.duplicated(['person','origin']).any():
            raise ValueError('Duplicate prediction')
        if dataset == source:
            old = pd.read_csv(ROOT / 'results/key_comparison' / cfg['source_profile'] / source /
                              f'fold{fold}_seed{seed}_predictions.csv')
            mapped = {'baseline':'M04_C2_LoRA','aligned':'M10_SAGE_fixed','full_fusion':'M12_full_evidence_fusion'}
            errors = {}
            for method, saved in mapped.items():
                pair = frame.merge(old[(old.method == saved) & (old.split == cfg['scored_split'])],
                                   on=['person','origin','label'], validate='one_to_one')
                errors[method] = float((pair[method]-pair.probability).abs().max())
                if len(pair) != len(frame) or errors[method] > 1e-6:
                    raise AssertionError('Source replay disagrees with selected checkpoint')
        frame.to_csv(output / (dataset + '_predictions.csv'), index=False)
        report[dataset] = summarize(frame, cfg)
        report[dataset].update(processing_seconds=sum(p['processing_seconds'] for p in completed),
                               calibration_seconds=sum(p['calibration_seconds'] for p in completed))
        if dataset == source:
            report[dataset]['source_replay_max_abs_error'] = errors
    mem = psutil.Process().memory_info()
    report['runtime'] = dict(invocation_seconds=time.perf_counter()-started, peak_rss_bytes=getattr(mem,'peak_wset',mem.rss),
                             threads=cfg['threads'], device='cpu', pretrained_blocks=12, guided_blocks=4)
    save_json(output / 'summary.json', report)
    save_json(status, dict(state='complete', runtime=report['runtime']))
    print(json.dumps(report['runtime']), flush=True)


if __name__ == '__main__':
    main()
