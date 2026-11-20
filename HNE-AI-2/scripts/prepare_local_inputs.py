"""Copy an explicit prepared-input source into this paper; never move originals.

Only hourly observations, daily labels/folds and original public model files are
selected. No paper-specific learned weights or result directories are imported.
The inventory and receipts are private, under artifacts/input_handoff/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('BrisT1D-Open', 'RESILIENT')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve(strict=True)
    destination = ROOT.resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Source and destination must be independent, non-nested directories')
    paths = []
    for dataset in DATASETS:
        hourly = source / 'data/development' / dataset
        daily = source / 'data/daily_study' / dataset
        hf = sorted(hourly.glob('*_hourly.npz'))
        df = sorted(daily.glob('*_daily.npz'))
        if not hf or not df:
            raise FileNotFoundError(f'Missing prepared observations or daily records for {dataset}')
        if {p.name.removesuffix('_hourly.npz') for p in hf} != {p.name.removesuffix('_daily.npz') for p in df}:
            raise ValueError(f'Hourly/daily pairing mismatch for {dataset}')
        paths.extend(hf + df)
        for name in ('people.csv', 'samples.csv', 'origins.csv', 'summary.json'):
            path = daily / name
            if path.exists():
                paths.append(path)
    paths.extend(source / 'tools/models/chronos-2' / name
                 for name in ('config.json', 'model.safetensors'))
    expected_model = 'ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42'
    expected_protocol = digest(ROOT / 'configs/daily_protocol.json')
    inventory = []
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source) or path.is_symlink():
            raise ValueError('Input escapes the selected source or is a symbolic link')
        relative = path.relative_to(source)
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Destination escapes this project')
        value = digest(path)
        if path.name == 'model.safetensors' and value != expected_model:
            raise ValueError('Original public Chronos-2 weight hash differs from the fixed study')
        if path.name == 'summary.json':
            metadata = json.loads(path.read_text(encoding='utf-8'))
            if metadata.get('protocol_sha256') != expected_protocol:
                raise ValueError('Daily protocol differs from this paper')
        if target.exists() and (not target.is_file() or digest(target) != value):
            raise FileExistsError(f'Refusing to overwrite a different existing file: {relative}')
        inventory.append(dict(relative_path=relative.as_posix(), bytes=path.stat().st_size,
                              sha256=value, existed_before=target.exists()))
    record = dict(status='inventoried_before_copy', source_root=str(source),
                  destination_root=str(destination), created_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=expected_protocol, file_count=len(inventory),
                  total_bytes=sum(item['bytes'] for item in inventory), files=inventory,
                  scope='Observations, fixed daily protocol/folds, and original public pretrained weights only',
                  source_files_moved=False, source_files_modified=False,
                  paper_specific_weights_imported=False)
    receipt = ROOT / 'artifacts/input_handoff/manifest.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding='utf-8'))
        expected = [(x['relative_path'], x['bytes'], x['sha256']) for x in previous['files']]
        actual = [(x['relative_path'], x['bytes'], x['sha256']) for x in inventory]
        if expected != actual or previous['source_root'] != str(source):
            raise ValueError('Existing handoff has different provenance; keep it and resolve explicitly')
    save(receipt, record)
    for item in inventory:
        relative = Path(item['relative_path'])
        original, target = source / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(original, target)
        if (target.stat().st_size != item['bytes'] or digest(target) != item['sha256']
                or digest(original) != item['sha256']):
            raise ValueError('Post-copy content verification failed')
    record.update(status='copied_and_verified', verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  verified_destination_files=len(inventory), verified_source_files=len(inventory))
    save(receipt, record)
    print(json.dumps({key: record[key] for key in
                      ('status', 'file_count', 'total_bytes', 'destination_root',
                       'source_files_modified', 'paper_specific_weights_imported')}))


if __name__ == '__main__':
    main()
