"""CPU-verifiable building blocks for the SAGE-TS design, not a trained system.

Axes: values [batch, channel, patch, within_patch]; edges [batch, patch, target, source].
No event labels enter the small predictor. Personal fitting/calibration is external.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    allowed = allowed.expand_as(scores)
    scores = scores.masked_fill(~allowed, -torch.inf)
    any_key = allowed.any(-1, keepdim=True)
    scores = torch.where(any_key, scores, torch.zeros_like(scores))
    return torch.softmax(scores, -1).masked_fill(~allowed, 0.0)


class CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=dilation) for _ in range(2)])
        # Normalize feature coordinates only. GroupNorm over time would leak future inputs.
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])

    def forward(self, x):
        residual = x
        for conv, norm in zip(self.convs, self.norms):
            x = conv(F.pad(x, (2 * self.dilation, 0)))
            x = F.gelu(norm(x.transpose(1, 2))).transpose(1, 2)
        return residual + x


class ConditionalPredictor(nn.Module):
    """Shared causal TCN; H, HN, HNE and single-source views use the same head.

    `reference` is a fixed prefix representation [B,C,W] available before all query
    patches. The caller must not derive it from the recent/future window.
    `state` indicates target-state channels (False means observed context).
    No fixed channel-name embeddings are used.
    """
    def __init__(self, patch_size=16, width=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.patch_size, self.width = patch_size, width
        self.input = nn.Linear(2 * patch_size + 5, width)
        self.blocks = nn.Sequential(*[CausalBlock(width, d) for d in dilations])
        self.reference_projection = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.message = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, 2 * patch_size)

    def encode(self, values, observed, calendar, state):
        b, c, p, k = values.shape
        if k != self.patch_size or observed.shape != values.shape:
            raise ValueError('Incorrect patch/mask shape')
        if calendar.shape != (b, p, 4) or state.shape != (b, c):
            raise ValueError('Calendar must be [B,P,4], state [B,C]')
        clean = torch.where(observed, values, torch.zeros_like(values))
        if not torch.isfinite(clean).all():
            raise ValueError('Observed values must be finite')
        role = state[:, :, None, None].expand(b, c, p, 1).to(values.dtype)
        clock = calendar[:, None].expand(b, c, p, 4)
        features = torch.cat((clean, observed.to(values.dtype), clock, role), -1)
        x = self.input(features).reshape(b * c, p, self.width).transpose(1, 2)
        return self.blocks(x).transpose(1, 2).reshape(b, c, p, self.width)

    def reference_summary(self, values, observed, calendar, state):
        h = self.encode(values, observed, calendar, state)
        w = observed.float().mean(-1, keepdim=True)
        return (h * w).sum(2) / w.sum(2).clamp_min(1.0)

    def distribution(self, h):
        mu, raw_scale = self.head(h).chunk(2, dim=-1)
        return mu, F.softplus(raw_scale) + 1e-3

    def forward(self, values, observed, calendar, state, family, reference=None, single_source=True):
        b, c, p, _ = values.shape
        encoded = self.encode(values, observed, calendar, state)
        # Prediction for patch p uses representations ending strictly before p.
        prior = F.pad(encoded[:, :, :-1], (0, 0, 1, 0))
        if reference is not None:
            if reference.shape != (b, c, self.width):
                raise ValueError('Reference must have shape [B,C,width]')
            prior = prior + self.reference_projection(reference)[:, :, None]
        h = prior.permute(0, 2, 1, 3)  # B,P,C,W
        score = torch.einsum('bpjw,bpkw->bpjk', self.query(h), self.key(h)) / math.sqrt(self.width)
        # State targets need half a patch; sparse observed context events need one
        # real observation to be usable. Missing context is still unknown, not zero.
        minimum_source = torch.where(state, self.patch_size // 2, 1)[:, :, None]
        prior_valid = F.pad(observed.sum(-1)[:, :, :-1] >= minimum_source, (1, 0))
        source_ok = prior_valid.permute(0, 2, 1)[:, :, None, :]
        not_self = ~torch.eye(c, dtype=torch.bool, device=values.device)[None, None]
        # Equalize repeated fields within each measurement family, separately by role.
        same = (family[:, :, None] == family[:, None, :]) & (state[:, :, None] == state[:, None, :])
        available = prior_valid.permute(0, 2, 1)
        count = (same[:, None] & available[:, :, None, :]).sum(-1)
        # Target self-edges are excluded, so remove the target from the eligible
        # source-family cardinality for this target row as well.
        count = count[:, :, None, :] - (same[:, None] & available[:, :, :, None]).to(count.dtype)
        score = score - count.clamp_min(1).log()
        a_n = masked_softmax(score, source_ok & not_self & state[:, None, None, :])
        a_e = masked_softmax(score, source_ok & not_self & ~state[:, None, None, :])
        message = self.message(h)
        hn = h + torch.einsum('bpjk,bpkw->bpjw', a_n, message)
        hne = hn + torch.einsum('bpjk,bpkw->bpjw', a_e, message)
        views = {}
        for name, value in [('H', h), ('HN', hn), ('HNE', hne)]:
            mu, scale = self.distribution(value)
            views[name] = (mu.permute(0, 2, 1, 3), scale.permute(0, 2, 1, 3))
        if single_source:
            # Same head evaluated with exactly one neighbour. This view must also
            # be included in the self-supervised training objective, not only inference.
            pair = h[:, :, :, None, :] + message[:, :, None, :, :]
            views['pair'] = self.distribution(pair)  # B,P,target,source,K
        return {'views': views, 'a_n': a_n, 'a_e': a_e,
                'pair_allowed': (source_ok & not_self).expand(b, p, c, c)}


def patch_nll(target, observed, mu, scale):
    clean = torch.where(observed, target, torch.zeros_like(target))
    log_loss = -torch.distributions.StudentT(5.0, mu, scale).log_prob(clean)
    count = observed.sum(-1)
    loss = torch.where(observed, log_loss, torch.zeros_like(log_loss)).sum(-1)
    return loss / count.clamp_min(1), count >= max(1, target.shape[-1] // 2)


def innovation_features(prediction, values, observed):
    """Returns completed-patch evidence. Later event loss cannot update S through it.

    Calibration into reference-relative units remains an explicit caller step.
    node features: NLL(H,HN,HNE), signed residual(H,HN,HNE), gain_N, gain_E, coverage.
    """
    losses, residuals = [], []
    for view in ('H', 'HN', 'HNE'):
        mu, sigma = prediction['views'][view]
        loss, valid = patch_nll(values, observed, mu, sigma)
        z = torch.where(observed, (torch.nan_to_num(values) - mu) / sigma, 0.0)
        losses.append(loss)
        residuals.append(z.sum(-1) / observed.sum(-1).clamp_min(1))
    node = torch.stack([*losses, *residuals, losses[0] - losses[1],
                        losses[1] - losses[2], observed.float().mean(-1)], -1)
    edge, edge_valid = None, None
    if 'pair' in prediction['views']:
        mu, sigma = prediction['views']['pair']
        y = values.permute(0, 2, 1, 3)[:, :, :, None, :].expand_as(mu)
        mask = observed.permute(0, 2, 1, 3)[:, :, :, None, :].expand_as(mu)
        loss, pair_valid = patch_nll(y, mask, mu, sigma)
        edge = losses[0].permute(0, 2, 1)[:, :, :, None] - loss
        edge_valid = prediction['pair_allowed'] & pair_valid
        edge = edge.masked_fill(~edge_valid, 0.0).detach()
    return node.detach(), valid, edge, edge_valid


def bounded_centered_bias(raw, allowed, bound=2.0):
    """Bound survives row-centering. A fully masked row returns zero.

    `allowed` must contain only valid original attention keys. It cannot create edges.
    """
    allowed = allowed.expand_as(raw)
    z = torch.tanh(raw).masked_fill(~allowed, 0.0)
    mean = z.sum(-1, keepdim=True) / allowed.sum(-1, keepdim=True).clamp_min(1)
    return (0.5 * bound * (z - mean)).masked_fill(~allowed, 0.0)


class EvidenceBias(nn.Module):
    """Shared across layers/heads; caller packs time and group tensor axes explicitly."""
    def __init__(self, node_dim=9, edge_dim=4, hidden=32, bound=2.0):
        super().__init__()
        self.bound = bound
        self.time = nn.Sequential(nn.Linear(2 * node_dim + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.group = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        for head in (self.time[-1], self.group[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, time_pair_features, group_edge_features, time_allowed, group_allowed,
                time_active=None, group_active=None):
        t = self.time(time_pair_features.detach()).squeeze(-1)
        g = self.group(group_edge_features.detach()).squeeze(-1)
        # Inactive queries (reference, REG, future) receive exactly zero guidance.
        # Center over all native valid keys, including inactive edges: the resulting
        # row-constant shift leaves their relative softmax weights unchanged.
        if time_active is not None:
            t = t.masked_fill(~time_active, 0.0)
        if group_active is not None:
            g = g.masked_fill(~group_active, 0.0)
        return (bounded_centered_bias(t, time_allowed, self.bound),
                bounded_centered_bias(g, group_allowed, self.bound))


def observed_utility(off_logits, on_logits, labels):
    """Training-only target from an entirely held-out pipeline; positive means helpful."""
    return (F.binary_cross_entropy_with_logits(off_logits, labels, reduction='none')
            - F.binary_cross_entropy_with_logits(on_logits, labels, reduction='none')).detach()


class UtilityGate(nn.Module):
    """Two-action gate. Inputs contain S/calibration/missingness summaries only."""
    def __init__(self, input_dim=12):
        super().__init__()
        self.utility = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.utility(features.detach()).squeeze(-1)

    def choose(self, features, threshold=0.0):
        return (self(features) > threshold).to(features.dtype)
