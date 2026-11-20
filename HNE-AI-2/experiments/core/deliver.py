"""Export A's private core artifacts and provenance for B's independent scoring.

This exports only completed runs, preserves raw training outputs, and does not
select models, alter outcomes, or compute new paper statistics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from study_inputs import SOURCE


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-key', required=True)
    parser.add_argument('--package', action='store_true')
    args = parser.parse_args()
    directory = (ROOT / 'artifacts/experiments' / args.run_key).resolve()
    output = (ROOT / 'results/experiments' / args.run_key).resolve()
    if not directory.is_relative_to((ROOT / 'artifacts/experiments/core').resolve()):
        raise ValueError('Expected a core run key')
    summary = read(output / 'summary.json'); manifest = read(directory / 'manifest.json')
    assert read(directory / 'status.json')['status'] == summary['status'] == 'completed'
    config = manifest['config']; protocol_hash = manifest['protocol_sha256']
    assert sha(ROOT / 'configs/daily_protocol.json') == protocol_hash
    teacher_root = ROOT / 'artifacts/experiments' / config['teacher_run']
    teacher_manifest = read(teacher_root / 'manifest.json')
    chosen = {r['component']: r for r in manifest['selected_teachers']}
    for record in chosen.values():
        assert sha(ROOT / record['path']) == record['sha256']
    evidence = {(r['person'], r['origin']): r for r in read(directory / 'evidence_receipts.json')}
    for (person, origin), record in evidence.items():
        assert sha(directory / 'evidence' / person / f'origin{origin}.pt') == record['sha256']
        assert sha(directory / 'evidence' / person / 'calibration.pt') == record['calibration_sha256']
    base_path = directory / 'baseline/candidate0.pt'
    assert sha(base_path) == summary['base_checkpoint_sha256']
    assert sha(directory / 'evidence_receipts.json') == summary['evidence_receipts_sha256']
    paths = {'M04': base_path, 'M10': directory / 'M10.pt', 'M12': directory / 'M12.pt'}
    checkpoint_hashes = {key: sha(path) for key, path in paths.items()}
    daily = {}
    clocks = {}
    for person in manifest['split_people']['validation']:
        file = SOURCE / 'data/daily_study' / config['dataset'] / f'{person}_daily.npz'
        assert sha(file) == manifest['data_sha256'][file.relative_to(SOURCE / 'data').as_posix()]
        with np.load(file, allow_pickle=False) as source:
            daily[person] = {key: source[key].copy() for key in ('issue_time', 'dates', 'available_at', 'future_labels', 'fold')}
        hourly = SOURCE / 'data/development' / config['dataset'] / f'{person}_hourly.npz'
        assert sha(hourly) == manifest['data_sha256'][hourly.relative_to(SOURCE / 'data').as_posix()]
        with np.load(hourly, allow_pickle=False) as source:
            # These fixed hourly inputs use UTC-compatible calendar clocks.
            import pandas as pd
            clock = pd.Timestamp(str(source['clock']))
            if clock.tzinfo is not None:
                clock = clock.tz_convert('UTC').tz_localize(None)
            clocks[person] = clock.to_datetime64().astype('datetime64[s]')
    with (output / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
        predictions = list(csv.DictReader(stream))
    standardized = []
    for row in predictions:
        person, origin = row['person'], int(row['origin'])
        method = row['method'].split('_')[0]
        assert row['split'] == 'validation' and int(daily[person]['fold']) == (config['fold'] + 1) % 5
        assert float(row['label']) == float(daily[person]['future_labels'][origin])
        if method == 'M04':
            policy, epoch, candidate, state_key = 'always_off', summary['reports']['M04']['selected_epoch'], 0, 'state'
        else:
            policy = 'always_on' if row['method'].endswith('always_on') else 'selected'
            report = summary['reports'][method]
            candidate = report['always_on_candidate' if policy == 'always_on' else 'selected_candidate']
            epoch = report['candidates'][candidate]['always_on_epoch' if policy == 'always_on' else 'selected_epoch']
            state_key = 'trained_state' if policy == 'always_on' else 'state'
        d = daily[person]
        issue = d['issue_time'][origin]
        target_start = d['dates'][origin + 2].astype('datetime64[s]')
        target_end = (d['dates'][origin + 8] + np.timedelta64(1, 'D')).astype('datetime64[s]')
        mature = d['available_at'][origin + 2:origin + 9].max()
        input_end = clocks[person] + np.timedelta64(origin + 1, 'D')
        assert not np.isnat(issue) and not np.isnat(mature)
        assert input_end <= issue
        assert target_start - issue >= np.timedelta64(1, 'D') and mature >= target_end
        component = teacher_manifest['small_component'][person]
        standardized.append(dict(dataset=config['dataset'], person=person, origin=origin,
            issue_time=str(issue), max_input_time=str(input_end), max_input_time_is_exclusive=True,
            target_start=str(target_start), target_end=str(target_end), target_end_is_exclusive=True,
            label_mature_time=str(mature), evaluation_fold=config['fold'], validation_fold=(config['fold'] + 1) % 5,
            seed=config['seed'], method_id=method, policy=policy, probability=float(row['probability']),
            logit=float(row['logit']), label=float(row['label']), protocol_hash=protocol_hash,
            config_hash=manifest['config_sha256'], code_commit=manifest['code_commit'],
            base_checkpoint_hash=summary['base_checkpoint_sha256'], checkpoint_hash=checkpoint_hashes[method],
            teacher_checkpoint_hash=chosen[component]['sha256'] if method != 'M04' else None,
            evidence_hash=evidence[(person, origin)]['sha256'] if method != 'M04' else None,
            selected_epoch=epoch, selected_candidate=candidate, checkpoint_policy=state_key,
            evaluation_role='source_validation_used_for_selection'))
    keys = [(r['method_id'], r['policy'], r['person'], r['origin']) for r in standardized]
    assert len(keys) == len(set(keys))
    export = output / 'forecast_records.csv'
    with export.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(standardized[0]))
        writer.writeheader(); writer.writerows(standardized)
    receipt = dict(run_key=args.run_key, scope=summary['scope'],
        export_sha256=sha(export), records=len(standardized), protocol_sha256=protocol_hash,
        data_root_configuration='HNE_STUDY_ROOT; independently provided private input root',
        backbone_sha256=manifest['model_sha256'], selected_teachers=manifest['selected_teachers'],
        checkpoints={k: dict(path=p.relative_to(ROOT).as_posix(), sha256=checkpoint_hashes[k]) for k, p in paths.items()},
        evidence_receipts_sha256=summary['evidence_receipts_sha256'],
        raw_predictions_sha256=sha(output / 'predictions.csv'),
        timestamp_semantics='Input cutoff and target end are exclusive; issue and label maturity read from fixed daily data',
        no_independent_test_claim=True,
        warning='This replaces neither the older engineering package nor the pending formal full-fold study. Do not mix anchors or use these validation scores as independent test results.')
    (directory / 'handoff.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    if args.package:
        package = ROOT / 'packages/private-handoff/HNE-AI-2-guidance-study.zip'
        package.parent.mkdir(parents=True, exist_ok=True)
        if package.exists():
            raise FileExistsError('Existing private package preserved; inspect before replacing')
        files = [*paths.values(), directory / 'manifest.json', directory / 'status.json', directory / 'sources.zip', directory / 'handoff.json',
                 directory / 'baseline_features.pt',
                 directory / 'evidence_receipts.json', directory / 'prefix_receipts.json',
                 teacher_root / 'manifest.json', teacher_root / 'selection.json', teacher_root / 'sources.zip',
                 *[ROOT / r['path'] for r in chosen.values()], *sorted((directory / 'evidence').rglob('*.pt')),
                 *sorted(output.glob('*.json')), *sorted(output.glob('*.csv'))]
        inventory = {file.relative_to(ROOT).as_posix(): dict(sha256=sha(file), bytes=file.stat().st_size) for file in files}
        with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file, file.relative_to(ROOT).as_posix())
            archive.writestr('CONTENTS.json', json.dumps(inventory, indent=2))
            archive.writestr('README.txt',
                'PRIVATE: HNE-AI-2 matched guidance development supplement.\n'
                'Do not upload this archive, extracted weights, evidence or predictions to public GitHub.\n'
                'Use the HNE-AI-2 code on exp/teacher-condition and the existing private input handoff.\n'
                'HNE_STUDY_ROOT must point to your own authorized input root containing data/ and tools/models/chronos-2/.\n'
                'Extract the listed project-relative paths under your HNE-AI-2 checkout, never HNE-AI-1.\n'
                'The original pretrained backbone and study data are prerequisites and are not duplicated here.\n'
                'CONTENTS.json records SHA-256 and size for every payload. Verify these before use.\n'
                'Selected teacher checkpoints require src/conditional_teacher.py::teacher_from_checkpoint.\n'
                'M04 is baseline/candidate0.pt; reuse this exact anchor for all dependent comparisons.\n'
                'M10.pt and M12.pt contain state (selected) and trained_state (always-on), with reports identifying candidates/epochs.\n'
                'M12 uses node/edge-only evidence: zero the extra summary via experiments/core/run.py::same_evidence.\n'
                'forecast_records.csv supplies per-row policy, checkpoint/evidence hashes and time boundaries.\n'
                'Source validation was used for model selection; these are development results, not independent tests.\n'
                f'Run key: {args.run_key}\n'
                f'Training source commit: {manifest["code_commit"]}\n'
                f'Optional local export recheck: python experiments/core/deliver.py --run-key {args.run_key}\n')
        with zipfile.ZipFile(package) as archive:
            assert archive.testzip() is None
            for name, record in inventory.items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == record['sha256']
        package.with_suffix('.sha256').write_text(sha(package) + '  ' + package.name + '\n', encoding='utf-8')
        print(json.dumps(dict(package=str(package), files=len(inventory), sha256=sha(package))))
    print(json.dumps(dict(export=str(export), records=len(standardized), inputs_verified=True)))


if __name__ == '__main__':
    main()
