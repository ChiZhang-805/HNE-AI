"""Measure full uncached deployment passes on completed, matched checkpoints."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import psutil
import torch
from training_data import SOURCE, load_people, selected_origins
from train_backbone import load_backbone, load_adapter, capture_prefix, encoder_tail, read_event
from train_evidence import origin_evidence
from train_comparison import ResidualFusion, save_json
from precursor_guidance import ConditionalPredictor, EvidenceBias
from pack_guidance import pack_bias
from fusion.evidence import FullEvidenceFusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='cpu_pilot')
    parser.add_argument('--dataset', default='BrisT1D-Open')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    directory = ROOT / 'artifacts/key_comparison' / args.profile / args.dataset / f'fold{args.fold}' / f'seed{args.seed}'
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'completed':
        raise RuntimeError('Benchmark requires a completed comparison')
    manifest = json.loads((directory / 'manifest.json').read_text())
    people = load_people(args.dataset, args.fold)
    samples = []
    for p in people['validation']:
        for origin, _ in selected_origins(p, 2):
            samples.append((p, origin))
    samples = samples[:args.samples]
    model, head = load_backbone(SOURCE / 'tools/models/chronos-2')
    baseline = torch.load(directory / 'M04_C2_LoRA.pt', map_location='cpu', weights_only=False)
    load_adapter(model, baseline['state']['lora']); head.load_state_dict(baseline['state']['head'])
    guide = EvidenceBias(); guide.load_state_dict(torch.load(directory / 'M10_SAGE_fixed.pt', weights_only=False)['state'])
    logit = ResidualFusion('logit'); logit.load_state_dict(torch.load(directory / 'M05_S_logit_residual.pt', weights_only=False)['state'])
    hidden = ResidualFusion('hidden'); hidden.load_state_dict(torch.load(directory / 'M12_S_hidden_late_fusion.pt', weights_only=False)['state'])
    full = None
    if (directory / 'M12_full_evidence_fusion.pt').exists():
        full = FullEvidenceFusion()
        full.load_state_dict(torch.load(directory / 'M12_full_evidence_fusion.pt', weights_only=False)['state'])
    small = {}
    for i in range(3):
        s = ConditionalPredictor()
        s.load_state_dict(torch.load(directory / 'small' / f'component{i}.pt', weights_only=False)['state'])
        small[i] = s.eval()
    rows = []
    prediction_path = ROOT / 'results/key_comparison' / args.profile / args.dataset / f'fold{args.fold}_seed{args.seed}_predictions.csv'
    predictions = pd.read_csv(prediction_path)
    method_names = dict(baseline='M04_C2_LoRA', guidance='M10_SAGE_fixed',
                        logit='M05_S_logit_residual', hidden='M12_S_hidden_late_fusion',
                        full_evidence='M12_full_evidence_fusion')
    modes = ['baseline', 'guidance', 'logit', 'hidden']
    if full is not None:
        modes.append('full_evidence')
    with torch.no_grad():
        for number, (person, origin) in enumerate(samples):
            cal = torch.load(directory / 'evidence' / person.person / 'calibration.pt', weights_only=False)
            component = manifest['small_component'][person.person]
            # Rotate measurement order to reduce first-pass/cache-order confounding.
            for mode in modes[number % len(modes):] + modes[:number % len(modes)]:
                started = time.perf_counter()
                ev = origin_evidence(small[component], person, cal, origin) if mode != 'baseline' else None
                packed, state, family, keep = person.large_window(origin)
                cache = capture_prefix(model, packed)
                tb = gb = None
                if mode == 'guidance':
                    tb, gb, _ = pack_bias(guide, ev, cache, torch.from_numpy(keep))
                output = encoder_tail(model, cache, tb, gb)
                z, feature = read_event(head, output, cache, state, family, person.dataset, True)
                if mode in ('logit', 'hidden'):
                    z = (logit if mode == 'logit' else hidden)(z[0], feature[0], ev['summary'])
                if mode == 'full_evidence':
                    z = full(z[0], feature[0], ev, person.family[0])
                row = dict(person=person.person, origin=origin, method=mode,
                           seconds=time.perf_counter() - started, probability=float(z.sigmoid()))
                if mode == 'guidance':
                    row.update(time_bias_max=float(tb.abs().max()), group_bias_max=float(gb.abs().max()),
                               valid_node_patches=int(ev['node_valid'].sum()), valid_edges=int(ev['edge_valid'].sum()))
                saved = predictions[(predictions.method == method_names[mode]) & (predictions.person == person.person) & (predictions.origin == origin)]
                if len(saved) == 1:
                    row['cached_probability_error'] = abs(row['probability'] - float(saved.iloc[0].probability))
                    if row['cached_probability_error'] > 1e-6:
                        raise AssertionError('Full deployment disagrees with saved cached prediction')
                rows.append(row); print(json.dumps(row), flush=True)
    report = dict(device='cpu', threads=2, scope='Uncached full 12-block forward; S evidence included for guidance/fusion; model loading and one-time personal calibration excluded',
                  summary={m: dict(median_seconds=float(np.median([r['seconds'] for r in rows if r['method'] == m])),
                                   mean_seconds=float(np.mean([r['seconds'] for r in rows if r['method'] == m]))) for m in modes},
                  maximum_cached_probability_error=max((r.get('cached_probability_error', 0.) for r in rows), default=0.),
                  measurements=rows)
    save_json(ROOT / 'results/key_comparison' / args.profile / args.dataset / f'fold{args.fold}_seed{args.seed}_deployment_timing.json', report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
