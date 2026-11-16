"""Personal-reference evidence and matched hard-path training in Chronos-2."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from chronos.chronos2.model import Chronos2Encoder
from .hne_joint import HNEJoint, EvidenceBasis


class StructuredEvidenceBasis(EvidenceBasis):
    """Keep H/N/E/NE regularization scales while routing three evidence types."""
    def fit(self, x, groups):
        super().fit(x, groups)
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        source_scale = np.sqrt(np.repeat(counts[inverse]*self.repeats, self.repeats))
        self.correction = [np.sqrt(len(index))/source_scale[index] for index in self.indices]
        return self

    def transform(self, x):
        return [np.asarray(b*c, np.float32) for b, c in zip(super().transform(x), self.correction)]


class HNERefined(HNEJoint):
    def __init__(self, config, widths, variant='joint'):
        super().__init__(config, widths, variant)
        d = config['hidden_width']
        # A shared initial state gives the three operators their information in order.
        self.initial_state = nn.Parameter(torch.zeros(d))
        self.anchor_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer('class_log_odds', torch.zeros(()))
        self.dropout = nn.Dropout(config['dropout'])
        if config.get('scale_aware_queries', False):
            self.scale_context = nn.Sequential(nn.Linear(3, d), nn.GELU())
        for p in self.root.parameters():
            p.requires_grad_(False)
        nn.init.zeros_(self.risk.weight); nn.init.zeros_(self.risk.bias)
        if variant in ['fixed', 'no_route_loss']:
            for p in self.router.parameters():
                p.requires_grad_(False)
        if variant == 'coordinates_only':
            for module in [self.tail, self.prompt_up, self.readout]:
                for p in module.parameters():
                    p.requires_grad_(False)
            self.prompt_types.requires_grad_(False)
            if hasattr(self, 'scale_context'):
                for p in self.scale_context.parameters(): p.requires_grad_(False)
        if variant == 'frozen_backbone':
            for p in self.tail.parameters(): p.requires_grad_(False)
        self.omitted_evidence = {'without_H': 0, 'without_N': 1, 'without_E': 2}.get(variant)
        if self.omitted_evidence is not None:
            for module in [self.evidence[self.omitted_evidence], self.gates[self.omitted_evidence]]:
                for p in module.parameters(): p.requires_grad_(False)
            self.evidence_coefficients[self.omitted_evidence].requires_grad_(False)

    def initialize_evidence(self, basis, ids, y, weights):
        x = np.concatenate([b[ids] for b in basis], axis=1)
        if self.variant == 'no_coordinates':
            x = np.zeros_like(x)
        if self.omitted_evidence is not None:
            offsets = np.cumsum([0]+[b.shape[1] for b in basis])
            x[:, offsets[self.omitted_evidence]:offsets[self.omitted_evidence+1]] = 0
        rule = LogisticRegression(C=self.config['anchor_C'], max_iter=2000)
        rule.fit(x, y, sample_weight=weights)
        with torch.no_grad():
            start = 0
            for coefficient in self.evidence_coefficients:
                n = coefficient.numel()
                coefficient.copy_(torch.as_tensor(rule.coef_[0, start:start+n], device=coefficient.device))
                start += n
            self.anchor_bias.copy_(torch.as_tensor(rule.intercept_[0], device=self.anchor_bias.device))

    def encode(self, x, mask, reference, reference_mask, basis, stats=None):
        numeric = torch.stack([encoder(b) for encoder, b in zip(self.evidence, basis)], 1)
        root = self.initial_state[None].expand(len(numeric), -1)
        if self.variant == 'coordinates_only':
            return root, self.dropout(numeric)
        b, c, t, _ = x.shape
        state_count = c-self.config.get('context_channels', 1)
        prompts = self.prompt_up(numeric) + self.prompt_types
        channel_prompts = prompts[:, None].expand(-1, c, -1, -1)
        if self.config.get('scale_aware_queries', False):
            channel_prompts = channel_prompts+self.prompt_up(self.scale_context(stats))[:, :, None]
        full = torch.cat([x, channel_prompts], 2)
        length = t+3
        observed = torch.cat([mask, torch.ones((b, c, 3), dtype=torch.bool, device=x.device)], 2)
        flat = full.reshape(b*c, length, 768)
        fm = observed.reshape(b*c, length).to(flat.dtype)
        time_mask = Chronos2Encoder._expand_and_invert_time_attention_mask(fm, flat.dtype)
        time_mask = time_mask.expand(-1, 1, length, -1).clone()
        # Sequence queries cannot read prompts. Each evidence query can read its own
        # prompt and the observed sequence, but cannot read the other evidence prompts.
        forbidden = torch.zeros((length, length), dtype=torch.bool, device=x.device)
        forbidden[:t, t:] = True
        forbidden[t:, t:] = ~torch.eye(3, dtype=torch.bool, device=x.device)
        time_mask.masked_fill_(forbidden[None, None], torch.finfo(flat.dtype).min)
        groups = torch.arange(b, device=x.device).repeat_interleave(c)
        group_mask = Chronos2Encoder._construct_and_invert_group_time_mask(groups, fm, flat.dtype)
        tail = self.tail(flat, position_ids=torch.arange(length, device=x.device)[None],
                         attention_mask=time_mask, group_time_mask=group_mask).hidden_states
        tail = tail.reshape(b, c, length, 768)
        embedded = self.readout(tail[:, :, -3:])
        rm = reference_mask[:, :state_count, :, None]
        remembered = (self.readout(reference[:, :state_count])*rm).sum((1, 2))/rm.sum((1, 2))
        h = embedded[:, :state_count, 0].mean(1)-remembered
        n = embedded[:, :state_count, 1].mean(1)
        e = .5*(embedded[:, :state_count, 2].mean(1)+embedded[:, state_count:, 2].mean(1))
        return root, self.dropout(numeric+torch.stack([h, n, e], 1))

    def forward(self, x, mask, reference, reference_mask, stats, basis):
        if self.variant == 'no_coordinates':
            basis = [torch.zeros_like(v) for v in basis]
        if self.omitted_evidence is not None:
            basis = [torch.zeros_like(v) if i == self.omitted_evidence else v for i, v in enumerate(basis)]
        root, evidence = self.encode(x, mask, reference, reference_mask, basis, stats)
        if self.omitted_evidence is not None:
            active = evidence.new_tensor([i != self.omitted_evidence for i in range(3)])
            evidence = evidence*active[None, :, None]
        with torch.autocast(x.device.type, enabled=False):
            evidence = evidence.float(); state = root.float()[:, None].expand(-1, 6, -1)
            b = len(state); log_policy = torch.zeros((b, 6), device=x.device)
            accumulated = torch.zeros_like(log_policy)
            used = torch.zeros((6, 3), dtype=torch.bool, device=x.device)
            context = evidence.mean(1)[:, None].expand(-1, 6, -1)
            for step in range(3):
                route = self.router(torch.cat([state, context], -1)).masked_fill(used[None], -1e4)
                action = self.paths[:, step]
                log_policy = log_policy+route.log_softmax(-1).gather(-1, action[None, :, None].expand(b, -1, 1)).squeeze(-1)
                for a in range(3):
                    gate = 1+self.config['gate_amplitude']*torch.tanh(self.gates[a](state))
                    increment = (basis[a].float()[:, None]*gate.repeat_interleave(3, -1)*self.evidence_coefficients[a]).sum(-1)
                    accumulated = accumulated+increment*(action == a)[None]
                updated = self.recurrence(evidence[:, action].reshape(-1, evidence.shape[-1]),
                                          state.reshape(-1, state.shape[-1])).reshape(b, 6, -1)
                state = updated if self.omitted_evidence is None else torch.where(
                    (action != self.omitted_evidence)[None, :, None], updated, state)
                used = used.clone(); used.scatter_(1, action[:, None], True)
            residual = self.config['neural_logit_scale']*torch.tanh(self.risk(state).squeeze(-1))
            candidates = self.anchor_bias+accumulated+residual
            if not self.training:
                candidates = candidates-self.class_log_odds
            selected = log_policy.argmax(-1)
            if self.variant == 'fixed':
                selected = torch.zeros_like(selected)
            return dict(logits=candidates[torch.arange(b, device=x.device), selected],
                        all_logits=candidates, policy=log_policy.exp(), log_policy=log_policy,
                        selected=selected, routes=self.paths[selected],
                        future=self.future(state).reshape(b, 6, 5, 2))


def refined_objective(model, out, y, weights, person, future, config, epoch, training_n):
    loss = F.binary_cross_entropy_with_logits(out['all_logits'], y[:, None].expand(-1, 6), reduction='none')
    hard = F.one_hot(out['selected'], 6).float()
    if model.variant == 'fixed':
        path_weight = hard; route = loss.sum()*0
    elif epoch <= config['route_warmup_epochs']:
        path_weight = torch.full_like(hard, 1/6); route = loss.sum()*0
    else:
        path_weight = (1-config['path_exploration_mass'])*hard+config['path_exploration_mass']/6
        # The router learns relative event loss only. Future targets and rank loss
        # cannot teach it a path that optimizes a different prediction task.
        advantage = loss.detach()-loss.detach().mean(1, keepdim=True)
        route = ((out['policy']*advantage).sum(1)*weights).mean()
        if model.variant == 'no_route_loss': route = route*0
    event = ((path_weight*loss).sum(1)*weights).mean()
    risk = (path_weight*out['all_logits']).sum(1)
    per_person = []
    for who in torch.unique(person):
        positive = risk[(person == who)&(y == 1)]
        negative = risk[(person == who)&(y == 0)]
        if len(positive) and len(negative):
            per_person.append(F.softplus(-(positive[:, None]-negative[None])).mean())
    rank = torch.stack(per_person).mean() if per_person else risk.sum()*0
    observed = torch.isfinite(future)
    target = torch.nan_to_num(future)[:, None].expand(-1, 6, -1, -1)
    future_loss = (F.smooth_l1_loss(out['future'], target, reduction='none')*observed[:, None]).sum((2, 3))
    future_loss = future_loss/observed.sum((1, 2)).clamp_min(1)[:, None]
    prediction = ((path_weight*future_loss).sum(1)*weights).mean()
    penalty = sum((v*v).sum() for v in model.evidence_coefficients)/(2*config['anchor_C']*training_n)
    total = event+config['path_supervision_weight']*route+config['within_person_rank_weight']*rank+config['future_loss_weight']*prediction+penalty
    components = dict(event_bce=event, route_advantage=route, within_person_rank=rank,
                      future_huber=prediction, evidence_penalty=penalty, objective=total)
    return total, components


def compact_state(model):
    """Store adapters and task parameters; the fixed public backbone is reloaded."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith('tail.') or k.endswith(('.a', '.b'))}
