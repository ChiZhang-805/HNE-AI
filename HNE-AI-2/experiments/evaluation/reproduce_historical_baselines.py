"""Replay the archived baseline statistics and export exact bootstrap draws.

Uses saved historical predictions only. Does not fit models, open B's new
evaluation outputs, or run the archived script's unrelated analyses/figures.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/analyze_local_experiments.py'
SOURCE_SHA = '70820a0d95c04787b2e00bcc02ff5c91248a0704ce68fcec6556bacf817bc74e'
SNAPSHOT = 'artifacts/local_output_review/sources.zip'
OUTPUTS = ('baseline_summary.csv', 'baseline_seed_metrics.csv', 'fold_coverage.csv',
           'calibration_bins.csv', 'baseline_prior_contrasts.csv',
           'within_person_metrics.csv', 'within_person_summary.csv',
           'within_person_seed_metrics.csv', 'alarm_seed_metrics.csv',
           'alarm_summary.csv', 'alarm_validation.json')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(artifact_root, output):
    started = time.perf_counter()
    root = artifact_root.resolve(strict=True)
    output = output.resolve()
    if output == root or output.is_relative_to(root / 'artifacts') or output.is_relative_to(root / 'results/local_experiments'):
        raise ValueError('Output must not overwrite historical inputs or receipts')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import numpy as np
    import pandas as pd
    import sklearn
    import psutil
    from threadpoolctl import threadpool_limits
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    with zipfile.ZipFile(root / SNAPSHOT) as archive:
        source = archive.read(SOURCE)
    actual = hashlib.sha256(source).hexdigest()
    manifest = json.loads((root / 'results/local_experiments/analysis_manifest.json').read_text(encoding='utf-8'))
    sources = {k.replace('\\', '/'): v for k, v in manifest['source_sha256'].items()}
    if actual != SOURCE_SHA or actual != sources[SOURCE]:
        raise ValueError('Archived analysis source differs from the historical receipt')
    # Keep the historical code byte-identical. Its main/figure path is not called.
    scope = {'__name__': '_historical_analysis', '__file__': str(PROJECT / SOURCE)}
    exec(compile(source, str(root / SNAPSHOT) + ':' + SOURCE, 'exec'), scope)
    scope.update(ROOT=root, OUT=output)
    original_weights = scope['person_weights']
    draws, calls = {}, []
    phase = ['baseline']

    def record_weights(frame, seed):
        weights = original_weights(frame, seed)
        people, inverse = np.unique(frame.person, return_inverse=True)
        samples = np.random.default_rng(seed).integers(0, len(people), (scope['REPETITIONS'], len(people)))
        counts = np.zeros((len(samples), len(people)), dtype=np.int16)
        for index, row in enumerate(samples):
            counts[index] = np.bincount(row, minlength=len(people))
        if not np.array_equal(counts[:, inverse], weights[1:]):
            raise ValueError('Exported draws do not reconstruct historical weights')
        prefix = f'call{len(calls):02d}'
        draws[prefix + '_people'] = people.astype(str)
        draws[prefix + '_indices'] = samples
        calls.append(dict(prefix=prefix, phase=phase[0], seed=seed,
                          dataset=str(frame.dataset.iloc[0]), fold=int(frame.fold.iloc[0]),
                          person_count=len(people), row_count=len(frame),
                          replicates=len(samples), order='numpy.unique sorted person identifiers'))
        return weights

    scope['person_weights'] = record_weights
    with threadpool_limits(limits=1):
        scope['check_metrics']()
        scope['baseline_analysis']()
        phase[0] = 'within_person'
        scope['within_person_analysis']()
        scope['alarm_analysis']()
    np.savez_compressed(output / 'bootstrap_draws.npz', **draws)
    (output / 'bootstrap_calls.json').write_text(json.dumps(calls, indent=2) + '\n', encoding='utf-8')
    comparisons = []
    all_passed = True
    for name in OUTPUTS:
        reference = root / 'results/local_experiments' / name
        produced = output / name
        error = 0.0
        if name.endswith('.json'):
            equal = json.loads(reference.read_text()) == json.loads(produced.read_text())
        else:
            old, new = pd.read_csv(reference), pd.read_csv(produced)
            equal = old.shape == new.shape and list(old.columns) == list(new.columns)
            if equal:
                for column in old:
                    a, b = old[column], new[column]
                    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                        av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
                        equal = bool(equal and np.allclose(av, bv, atol=1e-12, rtol=0, equal_nan=True))
                        finite = np.isfinite(av) & np.isfinite(bv)
                        if finite.any():
                            error = max(error, float(np.abs(av[finite] - bv[finite]).max()))
                    else:
                        equal = bool(equal and a.fillna('').equals(b.fillna('')))
        comparisons.append(dict(file=name, passed=equal, max_abs_numeric_error=error,
                                original_sha256=sha(reference), reproduced_sha256=sha(produced)))
        all_passed = bool(all_passed and equal)
    report = dict(status='passed' if all_passed else 'failed', seconds=time.perf_counter() - started,
                  analysis_source_sha256=actual, source_archive_sha256=sha(root / SNAPSHOT),
                  runner_sha256=sha(Path(__file__)), output_files=len(OUTPUTS),
                  bootstrap_calls=len(calls), bootstrap_repetitions=scope['REPETITIONS'],
                  draw_file_sha256=sha(output / 'bootstrap_draws.npz'),
                  absolute_numeric_tolerance=1e-12, comparisons=comparisons,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   pandas=pd.__version__, sklearn=sklearn.__version__),
                  scope='Reproduction of historical development-cohort statistics; not new B outer scoring, not refitting uncertainty',
                  raw_draws_private=True)
    (output / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not all_passed:
        raise RuntimeError('Historical output mismatch; inspect reproduction.json, do not overwrite the original results')
    return {k: v for k, v in report.items() if k != 'comparisons'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True,
                        help='Received HNE-AI-2 root containing historical artifacts/ and results/')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/experiments/evaluation/historical_baselines')
    args = parser.parse_args()
    print(json.dumps(run(args.artifact_root, args.output), indent=2))


if __name__ == '__main__':
    main()
