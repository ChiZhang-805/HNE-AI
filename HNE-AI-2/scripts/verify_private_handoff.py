"""Verify a private handoff and optionally replay one CPU-only real-data origin.

The default command uses only the standard library. --smoke additionally needs
the pinned research dependencies. No training, API, GPU or paper-one access.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from study_inputs import SOURCE


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root, relative):
    if Path(relative).is_absolute() or '\\' in relative:
        raise ValueError('Manifest paths must be relative POSIX paths')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes its declared root')
    return path


def payload_path(relative):
    root = SOURCE if relative.startswith(('data/', 'tools/models/')) else ROOT
    return safe_path(root, relative)


def verify_inventory(manifest):
    seen = set()
    for item in manifest['files']:
        name = item['path']
        if name in seen:
            raise ValueError('Duplicate payload entry')
        seen.add(name)
        path = payload_path(name)
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Missing or changed private payload: {name}')
    protocol = sha(ROOT / 'configs/daily_protocol.json')
    if protocol != manifest['protocol_sha256']:
        raise ValueError('Repository protocol differs from the delivered data')
    for name, expected in manifest['runtime_source_sha256'].items():
        content = safe_path(ROOT, name).read_bytes()
        if manifest.get('runtime_text_hash_normalization') == 'CRLF_to_LF':
            content = content.replace(b'\r\n', b'\n')
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Runtime file differs from the tested handoff commit: {name}')
    index = {x['path']: x['sha256'] for x in manifest['files']}
    for spec in manifest['runs'].values():
        directory = safe_path(ROOT, spec['directory'])
        original = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if original['protocol_sha256'] != protocol:
            raise ValueError('Checkpoint protocol differs from the delivered inputs')
        if original['original_model_sha256'] != index['tools/models/chronos-2/model.safetensors']:
            raise ValueError('Original backbone differs from the checkpoint source')
        with zipfile.ZipFile(directory / 'sources.zip') as archive:
            for name, expected in original['source_sha256'].items():
                # Historical manifests used Windows separators; archive entries
                # are POSIX. Normalize only the lookup, never the saved record.
                if hashlib.sha256(archive.read(name.replace('\\', '/'))).hexdigest() != expected:
                    raise ValueError('Historical training-source snapshot mismatch')
        for name, expected in original['data_sha256'].items():
            normalized = name.replace('\\', '/')
            if '/data/' not in normalized:
                raise ValueError('Unknown historical data path; do not guess a mapping')
            relative = 'data/' + normalized.split('/data/', 1)[1]
            if index.get(relative) != expected:
                raise ValueError('Delivered input differs from the historical checkpoint data')
    return len(seen)


def replay_origin(manifest, profile):
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import numpy as np
    import psutil
    import torch
    from training_data import load_people
    from precursor_guidance import ConditionalPredictor
    from train_evidence import origin_evidence
    from train_backbone import load_backbone, load_adapter, capture_prefix, encoder_tail, read_event

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    data_checks = {}
    for dataset, expected in manifest['datasets'].items():
        hourly = {p.stem.removesuffix('_hourly') for p in (SOURCE / 'data/development' / dataset).glob('*_hourly.npz')}
        daily = sorted((SOURCE / 'data/daily_study' / dataset).glob('*_daily.npz'))
        folds = {str(k): 0 for k in range(5)}
        if hourly != {p.stem.removesuffix('_daily') for p in daily}:
            raise ValueError('Hourly/daily file pairing differs')
        for path in daily:
            with np.load(path, allow_pickle=False) as data:
                folds[str(int(data['fold']))] += 1
                if str(data['protocol_sha256']) != manifest['protocol_sha256']:
                    raise ValueError('A daily record uses another protocol')
        if len(daily) != expected['people'] or folds != expected['fold_counts']:
            raise ValueError('Delivered fixed person folds differ from the receipt')
        data_checks[dataset] = dict(people=len(daily), folds_verified=True)

    spec = manifest['runs'][profile]
    run = safe_path(ROOT, spec['directory'])
    original = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    people = load_people(spec['dataset'], spec['fold'], only_split='validation')['validation']
    with safe_path(ROOT, spec['prediction_file']).open(encoding='utf-8', newline='') as stream:
        predictions = [x for x in csv.DictReader(stream)
                       if x['method'] == 'M04_C2_LoRA' and x['split'] == 'validation']
    available = {p.person: p for p in people}
    candidates = [x for x in predictions if x['person'] in available
                  and (run / 'evidence' / x['person'] / f"origin{int(x['origin'])}.pt").is_file()]
    if not candidates:
        raise ValueError('No matched validation origin for the shared M04/S pair')
    selected = sorted(candidates, key=lambda x: (x['person'], int(x['origin'])))[0]
    person = available[selected['person']]
    origin = int(selected['origin'])
    component = original['small_component'][person.person]
    # These are our own hash-verified historical files; calibration contains
    # NumPy arrays, so it needs the trusted local-pickle load path.
    small_file = torch.load(run / 'small' / f'component{component}.pt', map_location='cpu', weights_only=False)
    if person.person in small_file['report']['fit_people']:
        raise ValueError('S fitting included the selected validation person')
    small = ConditionalPredictor().eval().requires_grad_(False)
    small.load_state_dict(small_file['state'])
    calibration = torch.load(run / 'evidence' / person.person / 'calibration.pt', map_location='cpu', weights_only=False)
    cached = torch.load(run / 'evidence' / person.person / f'origin{origin}.pt', map_location='cpu', weights_only=True)
    actual = origin_evidence(small, person, calibration, origin)
    errors = {}
    for key in ('node', 'node_valid', 'edge', 'edge_valid', 'summary'):
        if actual[key].shape != cached[key].shape:
            raise ValueError(f'Evidence shape differs: {key}')
        errors[key] = float((actual[key].float() - cached[key].float()).abs().max())
        if not torch.allclose(actual[key].float(), cached[key].float(), rtol=1e-5, atol=1e-5):
            raise ValueError(f'Recomputed S evidence differs: {key}')
    backbone, head = load_backbone(SOURCE / 'tools/models/chronos-2', device='cpu')
    base = torch.load(run / 'M04_C2_LoRA.pt', map_location='cpu', weights_only=False)['state']
    load_adapter(backbone, base['lora'])
    head.load_state_dict(base['head'])
    backbone.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    with torch.no_grad():
        packed, state, family, _ = person.large_window(origin)
        prefix = capture_prefix(backbone, packed)
        hidden = encoder_tail(backbone, prefix)
        probability = float(read_event(head, hidden, prefix, state, family, person.dataset).sigmoid())
    error = abs(probability - float(selected['probability']))
    if not np.isfinite(probability) or error > 1e-5:
        raise ValueError(f'Replayed M04 probability differs from its saved output: {error}')
    return dict(profile=profile, data_checks=data_checks, evaluation_fold_loaded=False,
                teacher_exclusion_check='passed', evidence_max_absolute_errors=errors,
                baseline_probability_absolute_error=error,
                cached_prefix_required=False, validation_origins_replayed=1,
                interpretation='Interface and historical replay only; not new performance evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/private_handoff/manifest.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--profile', choices=['cpu_pilot', 'contract_check'], default='cpu_pilot')
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    count = verify_inventory(manifest)
    report = dict(status='passed', verified_payload_files=count, required_code_commit=manifest['code_commit'],
                  input_root=str(SOURCE), training_started=False, gpu_used=False, api_calls=0)
    if args.smoke:
        report['smoke'] = replay_origin(manifest, args.profile)
    report['elapsed_seconds'] = time.perf_counter() - start
    output = ROOT / 'artifacts/handoff_validation' / (args.profile + ('_smoke.json' if args.smoke else '_inventory.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
