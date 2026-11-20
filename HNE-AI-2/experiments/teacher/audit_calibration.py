"""Check whether per-view personal calibration preserves raw H-off identity.

No labels or raw observations are read. This audits existing adapters only and
does not change evidence or selected models in a running core comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import torch
import psutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core-run-key', required=True)
    args = parser.parse_args()
    directory = (ROOT / 'artifacts/experiments' / args.core_run_key).resolve()
    if not directory.is_relative_to((ROOT / 'artifacts/experiments/core').resolve()):
        raise ValueError('Expected a core artifact directory')
    torch.set_num_threads(1)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    rows = []
    for path in sorted((directory / 'evidence').glob('*/calibration.pt')):
        calibration = torch.load(path, weights_only=False)
        valid = torch.as_tensor(calibration['node_available']).bool()
        if not valid.any():
            continue
        bias = calibration['bias'][:, valid]
        scale = calibration['log_scale'][:, valid]
        rows.append(dict(person=path.parent.name, calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         valid_nodes=int(valid.sum()),
                         hn_minus_h_bias_max=float((bias[1] - bias[0]).abs().max()),
                         hne_minus_h_bias_max=float((bias[2] - bias[0]).abs().max()),
                         hn_to_h_scale_ratio_min=float((scale[1] - scale[0]).exp().min()),
                         hn_to_h_scale_ratio_max=float((scale[1] - scale[0]).exp().max()),
                         hne_to_h_scale_ratio_min=float((scale[2] - scale[0]).exp().min()),
                         hne_to_h_scale_ratio_max=float((scale[2] - scale[0]).exp().max()),
                         raw_off_identity_preserved_after_calibration=bool(torch.equal(bias[0], bias[1]) and
                             torch.equal(bias[0], bias[2]) and torch.equal(scale[0], scale[1]) and torch.equal(scale[0], scale[2]))))
    summary = dict(people=len(rows), exact_identity_people=sum(r['raw_off_identity_preserved_after_calibration'] for r in rows),
                   max_hne_bias_difference=max(r['hne_minus_h_bias_max'] for r in rows),
                   hne_scale_ratio_range=[min(r['hne_to_h_scale_ratio_min'] for r in rows), max(r['hne_to_h_scale_ratio_max'] for r in rows)],
                   interpretation='Independent per-view personal adapters can alter H/HN/HNE contrasts even when raw conditional residuals are disabled. Calibrated gains concern the complete prediction pipeline; raw disabled-condition identity does not imply calibrated identity.',
                   core_guidance_off_semantics='Unchanged: disabling guidance bypasses the module and returns the frozen M04.',
                   models_or_evidence_changed=False, event_labels_loaded=False)
    result = dict(summary=summary, rows=rows, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory / 'calibration_identity_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
