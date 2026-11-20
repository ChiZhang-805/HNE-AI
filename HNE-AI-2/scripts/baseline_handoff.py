"""Package or verify historical baselines without training or unpickling models."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
META = 'artifacts/baseline_handoff'
NAME = 'HNE-AI-2-baseline-handoff.zip'
DATASETS = ('BrisT1D-Open', 'RESILIENT')
SEEDS = (20261113, 20261114, 20261115)
METHODS = ('M01_HGB', 'Random Forest', 'XGBoost', 'LightGBM', 'HYDRA')
PROTOCOL = '3d58b3089a3e7d25a442ec8aa78638c509fc03fcf2af901e1cfbe2b5ab17607c'
DIRECTORIES = ('artifacts/local_baselines', 'results/local_baselines',
               'artifacts/local_alarms', 'results/local_alarms',
               'artifacts/local_model_recovery')
SUMMARIES = ('baseline_summary.csv', 'baseline_seed_metrics.csv',
             'baseline_prior_contrasts.csv', 'alarm_summary.csv',
             'alarm_seed_metrics.csv', 'alarm_validation.json',
             'analysis_manifest.json', 'fold_coverage.csv', 'calibration_bins.csv',
             'within_person_metrics.csv', 'within_person_summary.csv', 'within_person_seed_metrics.csv')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name
            and ':' not in name and str(p) == name, 'Unsafe payload path')
    return name


class Reader:
    def __init__(self, root=None, archive=None):
        self.root = Path(root).resolve() if root else None
        self.zip = zipfile.ZipFile(archive) if archive else None
        if self.zip:
            names = self.zip.namelist()
            require(len(names) == len(set(names)), 'Duplicate ZIP entries')
            for name in names:
                safe_name(name)
                require(name.startswith('HNE-AI-2/'), 'Unexpected ZIP prefix')

    def read(self, name):
        safe_name(name)
        if self.zip:
            return self.zip.read('HNE-AI-2/' + name)
        path = (self.root / name).resolve(strict=True)
        require(path.is_relative_to(self.root), 'Payload escapes its root')
        return path.read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def csv(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode('utf-8-sig'))))


def keyed(rows, fields):
    result = {tuple(str(row[x]) for x in fields): row for row in rows}
    require(len(rows) == len(result), 'Duplicate keys in input records')
    return result


def audit(reader, study_root):
    """Check receipt chains, common origin sets, exclusions and saved parity."""
    study = Path(study_root).resolve(strict=True)
    base = reader.json('artifacts/local_baselines/manifest.json')
    alarm = reader.json('artifacts/local_alarms/manifest.json')
    recovery = reader.json('artifacts/local_alarms/recovery_receipt.json')
    require(base['protocol_sha256'] == PROTOCOL, 'Wrong historical protocol')
    require(file_hash(study / 'configs/daily_protocol.json') == PROTOCOL,
            'Input study uses a different protocol')
    identities = ('dataset', 'fold', 'seed', 'method')
    registries = [keyed(reader.csv(f'results/{kind}/run_registry.csv'), identities)
                  for kind in ('local_baselines', 'local_alarms')]
    expected = set(itertools.product(DATASETS, map(str, range(5)), map(str, SEEDS), METHODS))
    require(all(set(r) == expected for r in registries), 'Incomplete or extra 150-unit grid')
    base_hash = digest(reader.read('artifacts/local_baselines/manifest.json'))
    alarm_hash = digest(reader.read('artifacts/local_alarms/manifest.json'))
    recovery_hash = digest(reader.read('artifacts/local_model_recovery/manifest.json'))
    require(alarm['model_recovery_manifest_sha256'] == recovery_hash, 'Recovery chain mismatch')
    require(alarm['recovery_receipt_sha256'] == digest(reader.read('artifacts/local_alarms/recovery_receipt.json')),
            'Alarm recovery receipt mismatch')
    archived_sources = 0
    for kind in ('local_baselines', 'local_alarms', 'local_model_recovery'):
        origin = reader.json(f'artifacts/{kind}/manifest.json')
        with zipfile.ZipFile(io.BytesIO(reader.read(f'artifacts/{kind}/sources.zip'))) as snapshot:
            for name, checksum in origin['source_sha256'].items():
                require(digest(snapshot.read(name.replace('\\', '/'))) == checksum, 'Historical source snapshot mismatch')
                archived_sources += 1
    for archive_name, receipt_key in (('failed_attempt.zip', 'archive_sha256'),
                                      ('native_load_failure.zip', 'native_load_failure_archive_sha256'),
                                      ('cpu_initialization_failure.zip', 'cpu_initialization_failure_archive_sha256')):
        require(digest(reader.read('artifacts/local_alarms/' + archive_name)) == recovery[receipt_key],
                'Failure archive provenance mismatch')
    with zipfile.ZipFile(io.BytesIO(reader.read('artifacts/local_alarms/failed_attempt.zip'))) as initial:
        require(digest(initial.read('artifacts/local_alarms/manifest.json')) == recovery['initial_manifest_sha256'],
                'Original alarm manifest mismatch')
    for kind in ('local_baselines', 'local_alarms'):
        status = reader.json(f'artifacts/{kind}/status.json')
        require(status['status'] == 'completed' and status['completed_units'] == 150,
                'Historical run is not complete')
    folds = {}
    copy_receipt = json.loads((study / 'artifacts/input_handoff/manifest.json').read_text(encoding='utf-8'))
    copied = {i['relative_path']: i['sha256'] for i in copy_receipt['files']}
    for dataset in DATASETS:
        relative = f'data/daily_study/{dataset}/people.csv'
        require(file_hash(study / relative) == copied[relative], 'Fixed person-fold file changed')
        with (study / 'data/daily_study' / dataset / 'people.csv').open(encoding='utf-8-sig', newline='') as stream:
            people = list(csv.DictReader(stream))
        # Historical people tables name the identifier person_id.
        column = 'person_id' if 'person_id' in people[0] else 'person'
        folds[dataset] = {r[column]: int(r['fold']) for r in people}
    inputs = dict(alarm['data_sha256'])
    signatures = {}
    units = []
    max_error = 0.0
    total_predictions = total_issue_days = 0
    for dataset, fold, seed, method in sorted(expected):
        key = (dataset, fold, seed, method)
        suffix = f'{dataset}/fold{fold}/seed{seed}/{method.replace(" ", "_")}'
        b = 'artifacts/local_baselines/' + suffix
        a = 'results/local_alarms/' + suffix
        br = reader.json(b + '/report.json')
        ar = reader.json(a + '/report.json')
        for i, report in enumerate((br, ar)):
            require(report['registry']['status'] == 'completed', 'Incomplete unit')
            for field, value in report['registry'].items():
                actual = registries[i][key][field]
                if value is None:
                    require(actual == '', 'Registry null mismatch')
                elif isinstance(value, (float, int)):
                    require(float(actual) == value, 'Registry number mismatch')
                else:
                    require(actual == value, 'Registry text mismatch')
        require(br['manifest_sha256'] == base_hash, 'Model manifest mismatch')
        model_hash = digest(reader.read(b + '/estimator.joblib'))
        require(model_hash == br['estimator_sha256'] == ar['model_sha256'], 'Model hash mismatch')
        if method == 'HYDRA':
            require(ar['transform_sha256'] == digest(reader.read(b + '/transform.pt')), 'HYDRA transform mismatch')
        if ar['manifest_sha256'] != alarm_hash:
            require(ar['manifest_sha256'] == recovery['initial_manifest_sha256']
                    and recovery['completed_report_sha256'].get(a + '/report.json') == digest(reader.read(a + '/report.json')),
                    'Unexplained historical alarm manifest transition')
        if method == 'XGBoost':
            rec = f'artifacts/local_model_recovery/{dataset}/fold{fold}/seed{seed}'
            rr = reader.json(rec + '/report.json')
            require(rr['original_model_sha256'] == model_hash and rr['manifest_sha256'] == recovery_hash,
                    'XGBoost recovery provenance mismatch')
            require(rr['original_report_sha256'] == digest(reader.read(b + '/report.json'))
                    and rr['model_sha256'] == digest(reader.read(rec + '/estimator.ubj')), 'Recovered XGBoost hash mismatch')
            require(max(rr['parity'].values()) < 1e-6 and rr['round_trip_max_error'] < 1e-6,
                    'Historical XGBoost parity failed')
        for path, checksum in br['data_sha256'].items():
            require(path not in inputs or inputs[path] == checksum, 'Conflicting input hashes')
            inputs[path] = checksum
        training = reader.csv(b + '/training_origins.csv')
        predictions = reader.csv('results/local_baselines/' + suffix + '/predictions.csv')
        days = reader.csv(a + '/issue_days.csv')
        pred = keyed(predictions, ('split', 'person', 'origin'))
        issued = keyed(days, ('split', 'person', 'origin'))
        known = {k: v for k, v in issued.items() if v['label'] not in ('', 'nan', 'NaN')}
        require(set(pred) == set(known), 'Known-day and prediction coverage differ')
        keyed(training, ('person', 'origin'))
        evaluation_fold, validation_fold = int(fold), (int(fold) + 1) % 5
        for row in training:
            require(folds[dataset][row['person']] not in (evaluation_fold, validation_fold), 'Training person leakage')
        for k, row in issued.items():
            require(k[0] in ('validation', 'evaluation'), 'Unexpected prediction split')
            expected_fold = validation_fold if k[0] == 'validation' else evaluation_fold
            require(folds[dataset][k[1]] == expected_fold, 'Prediction fold mismatch')
            probability = float(row['probability'])
            require(math.isfinite(probability) and 0 <= probability <= 1, 'Invalid probability')
            if k in pred:
                require(float(row['label']) == float(pred[k]['label']), 'Label mismatch')
                error = abs(probability - float(pred[k]['probability']))
                require(error < 1e-6, 'Saved forecast parity failed')
                max_error = max(max_error, error)
        for split in ('validation', 'evaluation'):
            require(sum(k[0] == split for k in pred) == br['registry'][split + '_origins'], 'Origin count mismatch')
        require(len(training) == br['registry']['train_origins'], 'Training count mismatch')
        signature = (tuple(sorted((r['person'], r['origin'], r['label']) for r in training)),
                     tuple(sorted((k, v['label']) for k, v in issued.items())))
        group = (dataset, fold)
        require(group not in signatures or signatures[group] == signature, 'Methods/seeds use different origins')
        signatures[group] = signature
        require(ar['validation']['false_per_30_scorable_issue_days'] <= 1 + 1e-10,
                'Validation alarm budget exceeded')
        require(ar['registry']['threshold'] == ar['validation']['threshold'], 'Evaluation threshold changed')
        units.append(dict(dataset=dataset, fold=int(fold), seed=int(seed), method=method,
                          model_receipt=b + '/report.json', alarm_receipt=a + '/report.json',
                          predictions='results/local_baselines/' + suffix + '/predictions.csv',
                          training_origins=b + '/training_origins.csv', issue_days=a + '/issue_days.csv'))
        total_predictions += len(pred)
        total_issue_days += len(issued)
    mapped = {}
    for original, checksum in inputs.items():
        normalized = original.replace('\\', '/')
        require('/data/' in normalized, 'Unknown historical input location')
        relative = safe_name('data/' + normalized.split('/data/', 1)[1])
        path = (study / relative).resolve(strict=True)
        require(path.is_relative_to(study), 'Input escapes second-paper study root')
        require(file_hash(path) == checksum, 'Independent input hash mismatch: ' + relative)
        mapped[relative] = checksum
    analysis = reader.json('results/local_experiments/analysis_manifest.json')
    historical_source = 'scripts/analyze_local_experiments.py'
    with zipfile.ZipFile(io.BytesIO(reader.read('artifacts/local_output_review/sources.zip'))) as snapshot:
        archived_analysis = snapshot.read(historical_source)
    require(digest(archived_analysis) == {k.replace('\\', '/'): v for k, v in analysis['source_sha256'].items()}[historical_source],
            'Historical analysis snapshot mismatch')
    for name in ('results/local_baselines/run_registry.csv', 'results/local_alarms/run_registry.csv'):
        recorded = {k.replace('\\', '/'): v for k, v in analysis['source_sha256'].items()}
        require(recorded[name] == digest(reader.read(name)), 'Summary source registry mismatch')
    return dict(status='passed', model_units=len(units), alarm_units=len(units),
                prediction_rows=total_predictions, issue_day_rows=total_issue_days,
                unique_input_files_verified=len(mapped), fixed_fold_files_verified=2,
                historical_source_files_verified=archived_sources, max_saved_prediction_error=max_error,
                origin_grid_groups=len(signatures), formal_common_model_groups=0,
                scientific_scope='Historical, development-exposed cohorts; not a fresh external evaluation',
                checks='Hashes, receipt chains, fixed folds/person exclusions, common origins, saved prediction parity, recorded validation alarm budget; no model inference or fresh outcome scoring',
                units=units, mapped_input_sha256=mapped)


def verify(archive, study_root):
    reader = Reader(archive=archive)
    try:
        manifest = reader.json(META + '/manifest.json')
        files = manifest['files']
        require(len({item['path'] for item in files}) == len(files), 'Duplicate manifest paths')
        require(set(reader.zip.namelist()) == {'HNE-AI-2/' + i['path'] for i in files} | {'HNE-AI-2/' + META + '/manifest.json'},
                'ZIP inventory differs from manifest')
        for item in files:
            raw = reader.read(item['path'])
            require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'Payload hash mismatch: ' + item['path'])
        result = audit(reader, study_root)
        result.pop('units'); result.pop('mapped_input_sha256')
        return dict(result, archive_sha256=file_hash(Path(archive)), payload_files=len(files),
                    archive_bytes=Path(archive).stat().st_size)
    finally:
        reader.zip.close()


def build(code_root):
    code = Path(code_root).resolve(strict=True)
    changes = subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True)
    require(not changes.strip(), 'Commit reviewed public tooling before packaging')
    commit = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    result = audit(Reader(root=ROOT), ROOT)
    directory = ROOT / META
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    paths = {p for d in DIRECTORIES for p in (ROOT / d).rglob('*') if p.is_file()}
    paths.update(ROOT / 'results/local_experiments' / n for n in SUMMARIES)
    paths.update(ROOT / p for p in ('artifacts/local_output_review/validation.json',
                                   'artifacts/local_output_review/sources.zip',
                                   'scripts/baseline_handoff.py', 'scripts/analyze_local_experiments.py',
                                   'experiments/evaluation/reproduce_historical_baselines.py',
                                   'docs/private_handoff.md'))
    paths.update(directory / p for p in ('audit.json', 'formal_availability.csv', 'delivery_matrix.json', 'B_CODEX_PROMPT.txt'))
    replay = ROOT / 'results/experiments/evaluation/historical_baselines'
    reproduction = json.loads((replay / 'reproduction.json').read_text(encoding='utf-8'))
    require(reproduction['status'] == 'passed', 'Historical statistics reproduction has not passed')
    require(reproduction['runner_sha256'] == file_hash(ROOT / 'experiments/evaluation/reproduce_historical_baselines.py'),
            'Statistics runner changed after reproduction')
    require(reproduction['draw_file_sha256'] == file_hash(replay / 'bootstrap_draws.npz'), 'Bootstrap draws changed')
    paths.update(replay / p for p in ('bootstrap_draws.npz', 'bootstrap_calls.json', 'reproduction.json'))
    files = []
    for path in sorted(paths):
        require(path.resolve(strict=True).is_relative_to(ROOT.resolve()) and not path.is_symlink(), 'Payload escapes paper two')
        files.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=file_hash(path)))
    require(digest((code / 'scripts/baseline_handoff.py').read_bytes().replace(b'\r\n', b'\n')) ==
            digest(Path(__file__).read_bytes().replace(b'\r\n', b'\n')), 'Published verifier differs')
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), code_commit=commit,
                    privacy='Private artifacts; never publish this ZIP or individual records',
                    protocol_sha256=PROTOCOL, files=files, formal_common_model_groups=0,
                    historical_model_units=150, historical_alarm_units=150,
                    historical_bootstrap_reproduced=True,
                    historical_bootstrap_source='scripts/analyze_local_experiments.py',
                    historical_bootstrap_runner='experiments/evaluation/reproduce_historical_baselines.py',
                    recipient_receipt_status='awaiting_confirmation',
                    summary_use='Private independent audit; public aggregates require B review under the fixed statistical contract',
                    data_dependency='Existing HNE-AI-2-private-handoff.zip; configure --study-root to its independent inputs',
                    first_paper_accessed=False)
    manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    output = ROOT / 'packages/private-handoff'
    output.mkdir(parents=True, exist_ok=True)
    partial = output / NAME.replace('.zip', '.partial.zip')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for item in files:
            archive.write(ROOT / item['path'], 'HNE-AI-2/' + item['path'])
        archive.writestr('HNE-AI-2/' + META + '/manifest.json', manifest_bytes)
    validation = verify(partial, ROOT)
    destination = output / NAME
    partial.replace(destination)
    (directory / 'manifest.json').write_bytes(manifest_bytes)
    (directory / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
    (output / 'BASELINE_VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
    (output / 'BASELINE_SHA256SUMS.txt').write_text(f'{validation["archive_sha256"]}  {NAME}\n', encoding='ascii')
    (output / 'B_CODEX_PROMPT.txt').write_bytes((directory / 'B_CODEX_PROMPT.txt').read_bytes())
    original = output / 'HNE-AI-2-private-handoff.zip'
    transfer_path = output / 'TRANSFER_RECEIPT.json'
    previous = json.loads(transfer_path.read_text(encoding='utf-8')) if transfer_path.exists() else {}
    superseded = previous.get('superseded_baseline_archive_sha256', [])
    for entry in previous.get('files', []):
        if entry['name'] == NAME and entry['sha256'] != validation['archive_sha256'] and entry['sha256'] not in superseded:
            superseded.append(entry['sha256'])
    transfer = dict(status='prepared_and_verified_awaiting_recipient_receipt', public_branch='main',
                    public_commit=commit, files=[dict(name=p.name, bytes=p.stat().st_size, sha256=file_hash(p))
                                                for p in (original, destination)],
                    baseline_validation=validation, historical_statistics_reproduction=reproduction,
                    superseded_baseline_archive_sha256=superseded,
                    formal_common_groups_generated=0, formal_common_groups_delivered=0,
                    B_code_received=False, recipient_confirmed_received=False, uploaded_private_files=False)
    transfer_path.write_text(json.dumps(transfer, indent=2) + '\n', encoding='utf-8')
    header = '# Private transfer contents\n\n'
    for entry in transfer['files']:
        header += f'- `{entry["name"]}`: {entry["bytes"]:,} bytes; SHA-256 `{entry["sha256"]}`\n'
    header += ('\nThe baseline supplement uses the same stable filename and supersedes the checksum in the previous transfer. '
               'Keep an already extracted historical directory unchanged; extract this ZIP into a separate private receiving directory. '
               'The original input archive is unchanged. Prepared and verified does not mean received or accepted by B. '
               f'Public code commit: `{commit}`. Formal common groups generated: **0/30**.\n\n')
    (output / 'README.md').write_text(header + (ROOT / 'docs/private_handoff.md').read_text(encoding='utf-8'), encoding='utf-8')
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--code-root', type=Path, required=True)
    v = sub.add_parser('verify'); v.add_argument('--archive', type=Path, required=True)
    v.add_argument('--study-root', type=Path, required=True)
    a = sub.add_parser('audit'); a.add_argument('--study-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.code_root)
    elif args.command == 'verify':
        result = verify(args.archive, args.study_root)
    else:
        result = audit(Reader(root=ROOT), args.study_root)
        result.pop('units'); result.pop('mapped_input_sha256')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
