"""CPU checks of complete-evidence interfaces with the actual frozen Chronos model."""
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import psutil
import torch
from fusion.evidence import FullEvidenceFusion, InputEvidenceResidual
from training_data import load_people, SOURCE
from train_backbone import load_backbone, load_adapter, capture_prefix, encoder_tail, read_event
from chronos_bridge import encode_packed_segments
from train_comparison import resource_check, save_json, sha


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check('cpu')
    torch.manual_seed(20261114)
    directory = ROOT / 'artifacts/key_comparison/cpu_pilot/BrisT1D-Open/fold0/seed20261114'
    person = load_people('BrisT1D-Open', 0)['validation'][0]
    origin = int(person.origins[0])
    evidence = torch.load(directory / 'evidence' / person.person / f'origin{origin}.pt', weights_only=True)
    model, head = load_backbone(SOURCE / 'tools/models/chronos-2')
    state = torch.load(directory / 'M04_C2_LoRA.pt', weights_only=False)['state']
    load_adapter(model, state['lora']); head.load_state_dict(state['head'])
    model.requires_grad_(False); head.requires_grad_(False)
    packed, state_mask, family, keep = person.large_window(origin)
    keep = torch.from_numpy(keep)
    checks, errors, counts = {}, {}, {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not checks[name]:
            raise AssertionError(name)
    with torch.no_grad():
        cached = capture_prefix(model, packed)
        original = encode_packed_segments(model, **packed)[0]
        tail = encoder_tail(model, cached)
        check('full_size_prefix_tail_exact', torch.equal(original, tail))
        logit, feature = read_event(head, original, cached, state_mask, family, person.dataset, True)
    for mode, expected in [('hidden', 897), ('logit', 905), ('evidence', 897)]:
        module = FullEvidenceFusion(mode)
        counts[mode] = sum(p.numel() for p in module.parameters())
        check(mode + '_parameters', counts[mode] == expected)
        z = module(logit[0], feature[0], evidence, person.family[0])
        check(mode + '_initial_output', torch.equal(z, logit[0].reshape(1)) if mode != 'evidence' else torch.equal(z, torch.zeros_like(z)))
        torch.nn.init.normal_(module.head[-1].weight, std=.1)
        permutation = torch.tensor([2, 4, 0, 5, 1, 3])
        changed = {**evidence, 'node': evidence['node'][permutation], 'node_valid': evidence['node_valid'][permutation],
                   'edge': evidence['edge'][:, permutation][:, :, permutation],
                   'edge_valid': evidence['edge_valid'][:, permutation][:, :, permutation]}
        a = module(logit[0], feature[0], evidence, person.family[0])
        b = module(logit[0], feature[0], changed, person.family[0][permutation])
        check(mode + '_channel_permutation', torch.allclose(a, b, atol=1e-6, rtol=1e-6))
        a.sum().backward()
        check(mode + '_learnable', any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
    residual = InputEvidenceResidual()
    counts['input'] = sum(p.numel() for p in residual.parameters())
    check('input_parameters', counts['input'] == 1785)
    delta = residual(evidence, person.family[0], keep)
    check('input_zero_initialized', torch.count_nonzero(delta) == 0)
    with torch.no_grad():
        z = encode_packed_segments(model, **packed, input_residual=delta)[0]
        check('input_zero_exact_baseline', torch.equal(original, z))
    torch.nn.init.normal_(residual.head[-1].weight, std=.1)
    delta = residual(evidence, person.family[0], keep)
    check('input_reference_unchanged', delta[:, :111].abs().sum() == 0)
    with torch.no_grad():
        off = encode_packed_segments(model, **packed, input_residual=delta, gate=0)[0]
        check('input_explicit_off_exact', torch.equal(original, off))
        missing = {**packed, 'context': packed['context'].masked_fill(~packed['observed'], 1e6)}
        filled = encode_packed_segments(model, **missing)[0]
        check('full_size_missing_fill_independent', torch.equal(original, filled))
        order = torch.arange(original.shape[0] - 1, -1, -1)
        permuted = {k: v[order] for k, v in packed.items()}
        moved = encode_packed_segments(model, **permuted)[0]
        errors['channel_permutation_hidden_max'] = float((original[order] - moved).abs().max())
        check('full_size_channel_permutation', torch.allclose(original[order], moved, atol=1e-5, rtol=1e-5))
        doubled = {k: torch.cat((v, v), 0) for k, v in packed.items()}
        doubled['group_ids'][original.shape[0]:] = 1
        doubled['context'][original.shape[0]:] = doubled['context'][original.shape[0]:] * 2 + 5
        batch = encode_packed_segments(model, **doubled)[0][:original.shape[0]]
        errors['entity_isolation_hidden_max'] = float((original - batch).abs().max())
        check('full_size_group_isolation_and_batch_parity', torch.allclose(original, batch, atol=1e-5, rtol=1e-5))
    modified = encode_packed_segments(model, **packed, input_residual=delta)[0]
    risk = read_event(head, modified, cached, state_mask, family, person.dataset)
    risk.sum().backward()
    check('input_gradient_through_all_frozen_blocks', any(p.grad is not None and p.grad.abs().sum() > 0 for p in residual.parameters()))
    check('backbone_and_event_head_frozen', all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in head.parameters()))
    report = dict(status='passed', checks=checks, errors=errors, parameters=counts,
                  elapsed_seconds=time.perf_counter()-started, device='cpu', threads=2,
                  baseline_sha256=sha(directory/'M04_C2_LoRA.pt'),
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'src/fusion/evidence.py', ROOT/'src/chronos_bridge.py']},
                  scope='Real 119M pretrained model and real historical evidence; engineering checks, not predictive-effect evidence')
    save_json(ROOT/'artifacts/verification/complete_interfaces.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
