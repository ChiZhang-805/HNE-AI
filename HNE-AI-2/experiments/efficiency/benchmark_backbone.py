"""Synthetic-only CPU/GPU capacity test for an exclusive research device.

Uses public pretrained weights and generated tensors; no study data, labels,
private S or trained M04 checkpoint is read or uploaded. Does not rent hardware.
"""
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
from train_backbone import load_backbone, capture_prefix, encoder_tail, read_event
from train_comparison import resource_check, to_device, save_json, sha
from precursor_guidance import EvidenceBias
from pack_guidance import pack_bias
from fusion.evidence import FullEvidenceFusion
from study_inputs import SOURCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--model-path', type=Path, default=SOURCE / 'tools/models/chronos-2')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/verification/backbone_capacity.json')
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError('At least one timed repetition is required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20261114)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check(args.device)  # Refuse an occupied GPU before allocating model weights.
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no CPU result is substituted')
    device = args.device
    model, head = load_backbone(args.model_path, device)
    observed = torch.rand(6, 2112) > .1
    packed = to_device(dict(context=torch.randn(6, 2112), observed=observed,
                           hour_offsets=torch.arange(-2112, 0)[None].expand(6, -1),
                           group_ids=torch.zeros(6, dtype=torch.long)), device)
    state = torch.tensor([[True] * 5 + [False]], device=device)
    family = torch.tensor([[0, 1, 2, 2, 2, 3]], device=device)
    keep = torch.ones(6, dtype=torch.bool, device=device)
    cached = to_device(capture_prefix(model, packed), device)
    evidence = dict(node=torch.randn(6, 21, 9, device=device),
                    node_valid=torch.ones(6, 21, dtype=torch.bool, device=device),
                    edge=torch.randn(21, 6, 6, 4, device=device),
                    edge_valid=torch.ones(21, 6, 6, dtype=torch.bool, device=device),
                    summary=torch.zeros(12, device=device))
    evidence['node_valid'][-1] = False
    evidence['edge_valid'][:, -1] = False
    for channel in range(6):
        evidence['edge_valid'][:, channel, channel] = False

    def synchronize():
        if device == 'cuda':
            torch.cuda.synchronize()

    measurements = {}
    def measure(name, operation):
        operation(); synchronize()  # Separate warm-up from timed repetitions.
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repetitions):
            resource_check(device); synchronize(); start = time.perf_counter()
            operation(); synchronize(); times.append(time.perf_counter() - start)
        measurements[name] = dict(seconds=times, mean_seconds=sum(times) / len(times),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device == 'cuda' else None)

    measure('frozen_prefix', lambda: capture_prefix(model, packed))
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.requires_grad], lr=.0001),
        dict(params=head.parameters(), lr=.001)], weight_decay=.0001)
    target = torch.ones(1, device=device)
    def baseline_step():
        optimizer.zero_grad(set_to_none=True)
        z = read_event(head, encoder_tail(model, cached), cached, state, family, 'BrisT1D-Open')
        torch.nn.functional.binary_cross_entropy_with_logits(z, target).backward()
        torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']], 1.)
        optimizer.step()
    measure('M04_forward_backward_optimizer', baseline_step)
    model.requires_grad_(False); head.requires_grad_(False)
    model.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
    with torch.no_grad():
        base, feature = read_event(head, encoder_tail(model, cached), cached, state, family, 'BrisT1D-Open', True)
    guide = EvidenceBias().to(device)
    guide_optimizer = torch.optim.AdamW(guide.parameters(), lr=.001, weight_decay=.0001)
    def guide_step():
        guide_optimizer.zero_grad(set_to_none=True)
        tb, gb, penalty = pack_bias(guide, evidence, cached, keep)
        z = read_event(head, encoder_tail(model, cached, tb, gb), cached, state, family, 'BrisT1D-Open')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, target) + .0001 * penalty
        loss.backward(); torch.nn.utils.clip_grad_norm_(guide.parameters(), 1.); guide_optimizer.step()
    measure('M10_forward_backward_optimizer', guide_step)
    fusion = FullEvidenceFusion('hidden').to(device)
    fusion_optimizer = torch.optim.AdamW(fusion.parameters(), lr=.001, weight_decay=.0001)
    def fusion_step():
        fusion_optimizer.zero_grad(set_to_none=True)
        z = fusion(base[0], feature[0], evidence, family[0])
        torch.nn.functional.binary_cross_entropy_with_logits(z, target).backward()
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.); fusion_optimizer.step()
    measure('M12_cached_feature_update', fusion_step)
    def inference():
        with torch.no_grad():
            return read_event(head, encoder_tail(model, cached), cached, state, family, 'BrisT1D-Open')
    measure('M04_inference', inference)
    assert all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in head.parameters())
    result = dict(status='completed', device=device, threads=1, synthetic_only=True, private_inputs_loaded=False,
                  gpu_name=torch.cuda.get_device_name() if device == 'cuda' else None, torch=torch.__version__,
                  public_backbone_sha256=sha(args.model_path / 'model.safetensors'),
                  source_sha256=sha(Path(__file__)), measurements=measurements,
                  interpretation='Capacity estimate for one six-channel window; excludes study data loading, complete candidate selection and full-run convergence. No method effectiveness claim.')
    save_json(args.output, result); print(json.dumps(result))


if __name__ == '__main__':
    main()
