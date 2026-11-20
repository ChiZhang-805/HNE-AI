"""Causality, identity and gradient checks for the conditional teacher variants."""
import json
import os
from pathlib import Path
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from conditional_teacher import HistoryPredictor, ResidualConditionalPredictor
from precursor_guidance import ConditionalPredictor


def main():
    torch.set_num_threads(1); torch.manual_seed(43)
    x = torch.randn(2, 6, 16, 16)
    observed = torch.rand_like(x) > .15
    calendar = torch.randn(2, 16, 4)
    state = torch.tensor([[True] * 5 + [False]] * 2)
    family = torch.tensor([[0, 1, 2, 2, 3, 4]] * 2)
    reference = torch.randn(2, 6, 64)
    inputs = dict(values=x, observed=observed, calendar=calendar, state=state, family=family, reference=reference)
    history = HistoryPredictor().eval()
    h = history(**inputs)
    legacy = ConditionalPredictor.forward(history, **inputs)
    assert all(torch.equal(a, b) for a, b in zip(h['views']['H'], legacy['views']['H']))
    assert all(torch.equal(h['views']['H'][0], h['views'][v][0]) for v in ('HN', 'HNE'))
    residual = ResidualConditionalPredictor(history).eval()
    initial = residual(**inputs)
    assert all(torch.equal(h['views']['H'][0], initial['views'][v][0]) for v in ('H', 'HN', 'HNE'))
    for head in (residual.n_head, residual.e_head):
        torch.nn.init.normal_(head[-1].weight, std=.02)
    active = residual(**inputs)
    disabled = residual(**inputs, conditions=False)
    assert all(torch.equal(disabled['views']['H'][j], disabled['views'][v][j]) for v in ('HN', 'HNE') for j in (0, 1))
    changed = x.clone(); changed[:, :, 10:] = 1000 * torch.randn_like(changed[:, :, 10:])
    future = residual(**{**inputs, 'values': changed})
    assert all(torch.equal(active['views'][v][0][:, :, :11], future['views'][v][0][:, :, :11]) for v in ('H', 'HN', 'HNE'))
    filled = residual(**{**inputs, 'values': x.masked_fill(~observed, 10000)})
    assert torch.equal(active['views']['HNE'][0], filled['views']['HNE'][0])
    no_context = observed.clone(); no_context[:, -1] = False
    no_e = residual(**{**inputs, 'observed': no_context})
    assert all(torch.equal(no_e['views']['HN'][j], no_e['views']['HNE'][j]) for j in (0, 1))
    order = torch.tensor([4, 2, 0, 5, 1, 3]); inverse = torch.argsort(order)
    permuted = residual(x[:, order], observed[:, order], calendar, state[:, order], family[:, order], reference[:, order])
    assert torch.allclose(active['views']['HNE'][0], permuted['views']['HNE'][0][:, inverse], atol=1e-5, rtol=1e-5)
    loss = active['views']['HNE'][0].square().mean(); loss.backward()
    assert all(p.grad is None for p in residual.history.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in residual.n_head.parameters())
    lagged = residual(**inputs, condition_lag=7)
    assert torch.isfinite(lagged['views']['HNE'][0]).all()
    checks = dict(history_matches_legacy_H=True, history_has_no_conditional_gain=True,
                  zero_residual_identity=True, disabled_identity_after_nonzero_weights=True,
                  strict_prediction_causality=True, missing_fill_invariance=True,
                  absent_context_has_no_E_effect=True, channel_permutation=True,
                  frozen_history_has_no_gradient=True, residual_receives_gradient=True,
                  causal_older_conditions_finite=True)
    out = ROOT / 'artifacts/verification/conditional_teacher.json'; out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='passed', checks=checks, gpu_used=False, training_started=False,
                  parameters_history=sum(p.numel() for p in history.parameters()),
                  residual_trainable_parameters=sum(p.numel() for p in residual.parameters() if p.requires_grad))
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8'); print(json.dumps(result))


if __name__ == '__main__':
    main()
