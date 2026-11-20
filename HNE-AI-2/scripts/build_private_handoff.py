"""Build a private, hash-indexed M04/S handoff; never upload its contents."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True,
                        help='Committed HNE-AI-2 directory that will be published')
    args = parser.parse_args()
    code = args.code_root.resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--', '.'], text=True)
    if changes.strip():
        raise ValueError('Commit the reviewed public code before pinning the private handoff')
    receipt = json.loads((ROOT / 'artifacts/input_handoff/manifest.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'copied_and_verified':
        raise ValueError('Independent prepared inputs have not passed their copy audit')
    paths = {ROOT / item['relative_path'] for item in receipt['files']}
    paths.update(ROOT / 'artifacts/input_handoff' / name for name in ('manifest.json', 'verification.json'))
    runs = {}
    names = ('M04_C2_LoRA.pt', 'manifest.json', 'sources.zip', 'status.json', 'timing.json',
             'verification.json', 'progress.jsonl', 'runtime.json', 'input_integrity.json', 'baseline_features.pt')
    for profile in ('cpu_pilot', 'contract_check'):
        relative = f'artifacts/key_comparison/{profile}/BrisT1D-Open/fold0/seed20261114'
        directory = ROOT / relative
        for name in names:
            path = directory / name
            if path.exists():
                paths.add(path)
        paths.update((directory / 'small').glob('component*.pt'))
        paths.update((directory / 'evidence').rglob('*.pt'))
        prediction = f'results/key_comparison/{profile}/BrisT1D-Open/fold0_seed20261114_predictions.csv'
        paths.add(ROOT / prediction)
        runs[profile] = dict(directory=relative, dataset='BrisT1D-Open', fold=0, seed=20261114,
                             prediction_file=prediction, shared_teacher_components=3,
                             allowed_use='Historical interface replay; not formal full-fold evidence',
                             imported_methods=['M04'], prefix_cache_included=False,
                             optimizer_state_included=False)
    files = []
    for path in sorted(paths):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(ROOT.resolve()) or path.is_symlink():
            raise ValueError('Selected payload escapes paper two')
        relative = path.relative_to(ROOT).as_posix()
        if not relative.startswith(('data/', 'artifacts/', 'tools/models/', 'results/key_comparison/')):
            raise ValueError('Unexpected private payload category')
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path)))
    expected = {x['relative_path']: x['sha256'] for x in receipt['files']}
    if any(item['sha256'] != expected[item['path']] for item in files if item['path'] in expected):
        raise ValueError('Prepared inputs changed since the copy audit')
    datasets = {}
    for dataset in ('BrisT1D-Open', 'RESILIENT'):
        with (ROOT / 'data/daily_study' / dataset / 'people.csv').open(encoding='utf-8', newline='') as stream:
            people = list(csv.DictReader(stream))
        folds = Counter(str(int(row['fold'])) for row in people)
        datasets[dataset] = dict(people=len(people), fold_counts={str(k): folds[str(k)] for k in range(5)})
    runtime = [*sorted((code / 'src').rglob('*.py')), code / 'scripts/verify_private_handoff.py',
               code / 'configs/daily_protocol.json', code / 'requirements.txt']
    manifest = dict(status='private_handoff_prepared', created_at_utc=datetime.now(timezone.utc).isoformat(),
                    repository='https://github.com/ChiZhang-805/HNE-AI', code_commit=commit,
                    protocol_sha256=sha(code / 'configs/daily_protocol.json'), datasets=datasets, runs=runs,
                    runtime_source_sha256={p.relative_to(code).as_posix(): sha(p) for p in runtime},
                    files=files, payload_files=len(files), payload_bytes=sum(x['bytes'] for x in files),
                    privacy='Private research inputs and participant-level artifacts. Do not publish to GitHub.',
                    source_paper_modified=False, formal_multifold_models_included=False,
                    excluded=['First-paper trained models and results', 'Rebuildable prefix caches',
                              'Optimizer states', 'API credentials', 'Alternative unmatched teacher candidates'],
                    historical_paths='Retained as recorded; the verifier maps historical /data/ suffixes to current inputs and checks hashes.')
    directory = ROOT / 'artifacts/private_handoff'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    guide = (code / 'docs/private_handoff.md').read_text(encoding='utf-8')
    guide = f'# 本次交接对应代码\n\n固定提交：`{commit}`。\n\n' + guide
    (directory / 'README.md').write_text(guide, encoding='utf-8')
    output = ROOT / 'packages/private-handoff'
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / 'HNE-AI-2-private-handoff.zip'
    temporary = output / 'HNE-AI-2-private-handoff.partial.zip'
    if not archive_path.resolve().is_relative_to(ROOT.resolve()) or not temporary.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Archive destination escapes paper two')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for item in files:
            archive.write(ROOT / item['path'], 'HNE-AI-2/' + item['path'])
        archive.write(directory / 'manifest.json', 'HNE-AI-2/artifacts/private_handoff/manifest.json')
        archive.write(directory / 'README.md', 'HNE-AI-2/artifacts/private_handoff/README.md')
    # Stable filename, atomic replacement. Sources and previous experiment
    # receipts are never moved or rewritten by packaging.
    temporary.replace(archive_path)
    checksum = sha(archive_path)
    (output / 'SHA256SUMS.txt').write_text(f'{checksum}  {archive_path.name}\n', encoding='ascii')
    (output / 'README.md').write_text(guide, encoding='utf-8')
    report = dict(status='built', archive=str(archive_path), archive_sha256=checksum,
                  archive_bytes=archive_path.stat().st_size, payload_files=len(files),
                  payload_bytes=manifest['payload_bytes'], code_commit=commit, uploaded=False)
    (directory / 'package_receipt.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
