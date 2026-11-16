"""Leakage, detached-evidence, sparse packing, and split invariants on real data."""
import copy
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import numpy as np
import pandas as pd
from replay_guidance_alarms import chronological_alerts, source_threshold, score_events
from training_data import load_people, component_assignments
from train_evidence import calibrate_person, origin_evidence, forecast_loss, predict
from precursor_guidance import ConditionalPredictor, EvidenceBias
from pack_guidance import pack_bias
from fusion.evidence import FullEvidenceFusion
from evidence_alignment import alignment_indices, relocate_innovations, patch_indices

torch.set_num_threads(2)
torch.manual_seed(20261114)
checks = {}
start = time.perf_counter()

def check(name, value):
    checks[name] = bool(value)
    if not value:
        raise AssertionError(name)

example = pd.DataFrame(dict(person=['P'] * 4, issue=pd.to_datetime(
    ['2020-01-01', '2020-01-03', '2020-01-08', '2020-01-15']),
    label=[np.nan, 1., 1., 0.], aligned=[.9, .95, .8, .7]))
example['target_start'] = example.issue + pd.Timedelta(days=1)
example['target_end'] = example.issue + pd.Timedelta(days=7)
emitted = chronological_alerts(example, 'aligned', .75, 7)
check('unknown_alarm_silences_next_day_and_exact_seven_days_allowed', list(emitted.index) == [0, 2])
late = score_events(example, 'aligned', .75, 7)[0]
early = score_events(example, 'aligned', .94, 7)[0]
check('overlapping_positive_intervals_merge', late['events'] == 1)
check('late_alarm_excluded_and_unknown_reported', late['timely_hits'] == 0 and late['alerts_unknown'] == 1)
check('early_alarm_leads_first_covered_date', early['timely_hits'] == 1 and early['median_lead_days'] == 1)
check('source_cutoff_obeys_budget', source_threshold(example, 'aligned',
      dict(silence_days=7, false_alarms_per_30_scorable_issue_days=1)) == .8)

people = load_people('BrisT1D-Open', 0)
check('evaluation_outcomes_not_loaded', not people['evaluation'])
check('person_folds_disjoint', not (set(p.person for p in people['train']) & set(p.person for p in people['validation'])))
assign = component_assignments(people, 20261114)
check('all_three_small_components_used', set(assign[p.person] for p in people['train']) == {0, 1, 2})
cohort_contract = {}
for dataset in ('BrisT1D-Open', 'RESILIENT'):
    # Check both datasets' source clocks and window shapes without loading the
    # evaluation fold or evaluating an event model on these outcomes.
    groups = load_people(dataset, 0)
    cohort_contract[dataset] = {}
    for split in ('train', 'validation'):
        count = 0
        for person in groups[split]:
            check(f'{dataset}_{person.person}_midnight_clock', person.clock.hour == 0 and person.clock.minute == 0)
            for index in (0, -1):
                origin_day = int(person.origins[index])
                packed, _, _, _ = person.large_window(origin_day)
                check(f'{dataset}_{person.person}_{index}_bounded_history',
                      packed['context'].shape[1] == 2112 and packed['hour_offsets'].max() == -1)
            count += len(person.origins)
        cohort_contract[dataset][split] = dict(people=len(groups[split]), origins=count)
p = people['train'][0]
model = ConditionalPredictor().eval().requires_grad_(False)
cal = calibrate_person(model, p, steps=3)
origin = int(p.origins[0])
original = origin_evidence(model, p, cal, origin)
future_changed = copy.deepcopy(p)
future_changed.raw[(origin + 1) * 24:] = 1e8
future_changed.labels[:] = 1 - future_changed.labels
after = origin_evidence(model, future_changed, cal, origin)
check('future_values_and_event_labels_cannot_change_S_evidence', all(torch.equal(v, after[k]) for k, v in original.items()))
pack, state, family, keep = p.large_window(origin)
changed_pack = future_changed.large_window(origin)[0]
check('future_values_do_not_enter_L_window', all(torch.equal(torch.nan_to_num(v), torch.nan_to_num(changed_pack[k])) for k, v in pack.items()))
check('history_and_reference_no_overlap', torch.all(pack['hour_offsets'][:, 1344] > pack['hour_offsets'][:, 1343]))
check('last_observed_hour_precedes_issue', torch.all(pack['hour_offsets'][:, -1] == -1))
check('evidence_detached', all(not v.requires_grad for v in original.values()))
valid = torch.tensor([[False, True, False, True, True], [False]*5, [True, False, False, False, False]])
identity, rev = patch_indices(valid)
check('matched_control_reverses_observed_positions_only',
      torch.equal(rev, torch.tensor([[0,4,2,3,1], [0,1,2,3,4], [0,1,2,3,4]])))
indices = alignment_indices(original)
aligned = relocate_innovations(original, indices, False)
reversed_ev = relocate_innovations(original, indices, True)
restored = relocate_innovations(reversed_ev, indices, True)
check('aligned_control_exact_identity', all(torch.equal(v, original[k]) for k,v in aligned.items()))
check('reversing_evidence_twice_recovers_every_value', all(torch.equal(v, original[k]) for k,v in restored.items()))
check('matched_control_preserves_context_and_coverage', torch.equal(original['node'][...,8],reversed_ev['node'][...,8])
      and torch.equal(original['edge'][...,1:],reversed_ev['edge'][...,1:])
      and torch.equal(original['node_valid'],reversed_ev['node_valid'])
      and torch.equal(original['edge_valid'],reversed_ev['edge_valid']))
c = int(keep.sum()); t = 145
cached = dict(hidden=torch.zeros(c, t, 8), attention_mask=torch.zeros(c, 1, 1, t), group_time_mask=torch.zeros(t, 1, c, c))
cached['attention_mask'][0, 0, 0, 120] = torch.finfo(torch.float32).min
cached['group_time_mask'][120, 0, :, 0] = torch.finfo(torch.float32).min
guide = EvidenceBias()
for last in (guide.time[-1], guide.group[-1]):
    torch.nn.init.normal_(last.weight, std=.05)
tb, gb, penalty = pack_bias(guide, original, cached, torch.from_numpy(keep))
check('guidance_finite_and_bounded', tb.isfinite().all() and gb.isfinite().all() and tb.abs().max() <= 2 and gb.abs().max() <= 2)
check('unobserved_keys_not_unmasked', tb[0, :, :, 120].abs().sum() == 0 and gb[120, :, :, 0].abs().sum() == 0)
check('reference_REG_future_queries_unchanged', tb[:, :, :111].abs().sum() == 0 and tb[:, :, 132:].abs().sum() == 0 and gb[:111].abs().sum() == 0 and gb[132:].abs().sum() == 0)
(tb.square().sum() + gb.square().sum()).backward()
check('guide_receives_gradients', sum(float(p.grad.abs().sum()) for p in guide.parameters() if p.grad is not None) > 0)
check('S_receives_no_event_gradient', all(p.grad is None for p in model.parameters()))
model.requires_grad_(True)
loss = forecast_loss(*predict(model, p, 14, 42)); loss.backward()
check('self_supervised_loss_finite_and_differentiable', torch.isfinite(loss) and sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0)
fusion = FullEvidenceFusion()
base_logit = torch.tensor([.2], requires_grad=True)
base_feature = torch.randn(130, requires_grad=True)
check('full_evidence_fusion_parameter_budget', sum(p.numel() for p in fusion.parameters()) == 897)
check('full_evidence_fusion_zero_identity', torch.equal(fusion(base_logit, base_feature, original, p.family[0]), base_logit))
torch.nn.init.normal_(fusion.head[-1].weight, std=.1)
fusion(base_logit, base_feature, original, p.family[0]).sum().backward()
check('full_evidence_fusion_frozen_L', base_logit.grad is None and base_feature.grad is None)
check('full_evidence_fusion_learns', any(v.grad is not None and v.grad.abs().sum() > 0 for v in fusion.parameters()))
permutation = torch.tensor([2, 4, 0, 5, 1, 3])
permuted = {**original, 'node': original['node'][permutation], 'node_valid': original['node_valid'][permutation],
            'edge': original['edge'][:, permutation][:, :, permutation],
            'edge_valid': original['edge_valid'][:, permutation][:, :, permutation]}
check('full_evidence_fusion_channel_permutation', torch.allclose(
    fusion(base_logit, base_feature, original, p.family[0]),
    fusion(base_logit, base_feature, permuted, p.family[0][permutation]), atol=1e-6))
report = dict(status='passed', checks=checks, elapsed_seconds=time.perf_counter() - start,
              cohort_contract=cohort_contract,
              scope='Real hourly data invariants; full pretrained forward equivalence is recorded separately by each run')
target = ROOT / 'artifacts/verification/training_pipeline.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(report, ensure_ascii=False), flush=True)
