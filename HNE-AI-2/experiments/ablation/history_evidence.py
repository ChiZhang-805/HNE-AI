"""Project matched evidence onto H-only features and audit a source run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import psutil
import torch


ROOT = Path(__file__).resolve().parents[2]
# innovation_features: H/HN/HNE losses, H/HN/HNE signed residuals,
# N/E gains, and observed coverage. Keep H loss, H residual and coverage.
HISTORY_COLUMNS = (0, 3, 8)
REMOVED_COLUMNS = (1, 2, 4, 5, 6, 7)


def history_only(evidence):
    """Keep H values and node masks, with no condition-specific inputs or edges."""
    node = torch.zeros_like(evidence['node'])
    node[..., HISTORY_COLUMNS] = evidence['node'][..., HISTORY_COLUMNS]
    return dict(node=node, node_valid=evidence['node_valid'],
                edge=torch.zeros_like(evidence['edge']),
                edge_valid=torch.zeros_like(evidence['edge_valid']),
                summary=torch.zeros_like(evidence['summary']))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(project, key):
    started = time.perf_counter()
    core = (project / 'artifacts/experiments/core').resolve(strict=True)
    run = (project / 'artifacts/experiments' / key).resolve(strict=True)
    if not run.is_relative_to(core):
        raise ValueError('Expected an existing core run directory')
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['config']['include_evaluation'] or manifest['split_people']['evaluation']:
        raise ValueError('This audit expects a source-only run')
    split_by_person = {person: split for split in ('train', 'validation')
                       for person in manifest['split_people'][split]}
    receipt_path = run / 'evidence_receipts.json'
    receipt_digest = sha(receipt_path)
    rows = json.loads(receipt_path.read_text(encoding='utf-8'))
    features_path = run / 'baseline_features.pt'
    feature_digest = sha(features_path)
    features = torch.load(features_path, weights_only=True, map_location='cpu')
    seen = set()
    split_counts = dict(train=0, validation=0)
    valid_nodes = full_edges = 0
    for row in rows:
        person, origin = row['person'], row['origin']
        feature_key = f'{person}/{origin}'
        if feature_key in seen or feature_key not in features:
            raise ValueError('Missing or duplicate origin in the common M04 cache')
        seen.add(feature_key)
        split_counts[split_by_person[person]] += 1
        source = (run / 'evidence' / person / f'origin{origin}.pt').resolve(strict=True)
        if not source.is_relative_to(run) or sha(source) != row['sha256']:
            raise ValueError('Evidence path or content differs from its receipt')
        full = torch.load(source, weights_only=True, map_location='cpu')
        projected = history_only(full)
        assert full['node'].shape == (6, 21, 9) and full['edge'].shape == (21, 6, 6, 4)
        assert torch.equal(projected['node_valid'], full['node_valid'])
        assert torch.equal(projected['node'][..., HISTORY_COLUMNS], full['node'][..., HISTORY_COLUMNS])
        assert torch.isfinite(projected['node']).all()
        assert not any(projected[k].any() for k in ('edge', 'edge_valid', 'summary'))
        assert not projected['node'][..., REMOVED_COLUMNS].any()
        valid_nodes += int(full['node_valid'].sum())
        full_edges += int(full['edge_valid'].sum())
    if seen != set(features) or not all(split_counts.values()):
        raise ValueError('Evidence and backbone feature origin sets do not match')
    if sha(receipt_path) != receipt_digest or sha(features_path) != feature_digest:
        raise ValueError('The original run updated evidence or features during the audit')
    return dict(status='passed', core_run=key, scope='Matched history-only feature ablation, source development only',
                origin_counts=split_counts, node_masks_preserved=True, retained_columns=list(HISTORY_COLUMNS),
                valid_nodes=valid_nodes, full_evidence_valid_edges=full_edges, history_only_valid_edges=0,
                original_model_or_cache_changed=False, reader_trained=False, outer_outcomes_loaded=False,
                checkpoint_sha256=sha(run / 'baseline/candidate0.pt'), evidence_receipts_sha256=receipt_digest,
                feature_cache_sha256=feature_digest, tool_sha256=sha(Path(__file__)),
                seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--run-key', required=True)
    args = parser.parse_args()
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1)
    project = args.project_root.resolve(strict=True)
    result = audit(project, args.run_key)
    output = project / 'artifacts/verification/history_evidence.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
