"""Verify group isolation, forward/gradient parity and measured CPU batching cost."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import psutil
from train_backbone import load_backbone, encoder_tail
from training_data import load_people, SOURCE
from precursor_guidance import EvidenceBias
from batching import forward_group, combine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-key', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20261114)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    directory = ROOT / 'artifacts/experiments' / args.run_key
    manifest = json.loads((directory / 'manifest.json').read_text())
    people = load_people(manifest['config']['dataset'], manifest['config']['fold'])
    lookup = {p.person: p for group in people.values() for p in group}
    items = []
    # Distinct people with prefixes already committed to disk by the active run.
    for path in sorted((directory / 'prefix').glob('*.pt')):
        person, origin = path.stem.rsplit('_', 1)
        if any(x['person'] == person for x in items):
            continue
        p = lookup[person]; _, state, family, keep = p.large_window(int(origin))
        ev = torch.load(directory / 'evidence' / person / f'origin{origin}.pt', weights_only=True)
        items.append(dict(person=person, prefix_path=path, state=state, family=family, keep=torch.from_numpy(keep), evidence=ev))
        if len(items) == 4:
            break
    assert len(items) == 4
    model, head = load_backbone(SOURCE / 'tools/models/chronos-2')
    model.requires_grad_(False); head.requires_grad_(False)
    guide = EvidenceBias()
    with torch.no_grad():
        for parameter in guide.parameters():
            parameter.uniform_(-.03, .03)
    measurements = {}
    for label, active in [('baseline', None), ('nonzero_guidance', guide)]:
        with torch.no_grad():
            start = time.perf_counter()
            individual = torch.cat([forward_group(model, head, [item], 'BrisT1D-Open', active)[0] for item in items])
            sequential_seconds = time.perf_counter() - start
            start = time.perf_counter(); packed = forward_group(model, head, items, 'BrisT1D-Open', active)[0]
            packed_seconds = time.perf_counter() - start
        torch.testing.assert_close(individual, packed, atol=2e-6, rtol=2e-5)
        measurements[label] = dict(max_logit_error=float((individual - packed).abs().max()),
                                    sequential_seconds=sequential_seconds, packed_seconds=packed_seconds)
    guide.zero_grad(set_to_none=True)
    for item in items[:2]:
        value, penalty = forward_group(model, head, [item], 'BrisT1D-Open', guide)
        (value.square().mean() / 2 + .0001 * penalty.mean() / 2).backward()
    individual_grad = {k: p.grad.clone() for k, p in guide.named_parameters()}
    guide.zero_grad(set_to_none=True)
    value, penalty = forward_group(model, head, items[:2], 'BrisT1D-Open', guide)
    (value.square().mean() + .0001 * penalty.mean()).backward()
    gradient_error = 0.
    for name, parameter in guide.named_parameters():
        torch.testing.assert_close(individual_grad[name], parameter.grad, atol=2e-6, rtol=2e-4)
        gradient_error = max(gradient_error, float((individual_grad[name] - parameter.grad).abs().max()))
    caches = [torch.load(item['prefix_path'], weights_only=True) for item in items[:2]]
    # Include fully masked group rows. A finite foreign mask would fail isolation.
    caches[0]['group_time_mask'].fill_(torch.finfo(torch.float32).min)
    with torch.no_grad():
        merged, _, _, sizes = combine(caches); before = encoder_tail(model, merged)
        caches[1]['hidden'] = caches[1]['hidden'] + 10
        altered, _, _, _ = combine(caches); after = encoder_tail(model, altered)
    assert torch.equal(before[:sizes[0]], after[:sizes[0]]), 'Foreign person changed masked group outputs'
    result = dict(status='passed', measurements=measurements, max_gradient_error=gradient_error,
                  foreign_person_isolation_exact=True, fully_masked_rows_checked=True,
                  threads=1, gpu_used=False, purpose='Numerical engineering check; no models trained or selected')
    out = ROOT / 'artifacts/verification/core_batching.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__':
    main()
