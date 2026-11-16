"""Name-independent, equal-family event readout shared by all C2 comparisons."""
import torch
from torch import nn


class FamilyEventHead(nn.Module):
    def __init__(self, backbone_width=768, width=64):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(backbone_width, width), nn.GELU())
        self.channel = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.risk = nn.Sequential(nn.Linear(2 * width + 2, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, hidden, token_observed, state, family, segment_weights,
                expected_state_family_count=None, return_features=False):
        # Three segments: reference, recent history, future placeholders overlapping
        # d+2:d+8. Future masks indicate valid placeholder tokens, not observations.
        if segment_weights.shape != (3, hidden.shape[2]):
            raise ValueError('Expected three segment weight vectors')
        x = self.project(hidden)
        w = token_observed[..., None] * segment_weights.T[None, None]
        pooled = torch.einsum('bctw,bcts->bcsw', x, w) / w.sum(2)[..., None].clamp_min(1.0)
        channel = self.channel(pooled.flatten(-2))
        valid = state & (w[..., 1].sum(-1) > 0)
        result, features = [], []
        for batch in range(hidden.shape[0]):
            groups = []
            family_coverage = []
            for fam in torch.unique(family[batch][valid[batch]]):
                use = valid[batch] & (family[batch] == fam)
                groups.append(channel[batch, use].mean(0))
                family_coverage.append((w[batch, use, :, 1].sum(-1)
                    / segment_weights[1].sum().clamp_min(1)).mean())
            if not groups:
                raise ValueError('No observed state family for event readout')
            groups = torch.stack(groups)
            total_families = (torch.unique(family[batch][state[batch]]).numel()
                              if expected_state_family_count is None
                              else int(expected_state_family_count[batch]))
            if total_families < len(groups):
                raise ValueError('Expected family count cannot exclude observed families')
            availability = hidden.new_tensor([len(groups) / max(1, total_families)])
            coverage = torch.stack(family_coverage).mean().reshape(1)
            feature = torch.cat((groups.mean(0), groups.max(0).values, availability, coverage))
            features.append(feature)
            result.append(self.risk(feature).squeeze(-1))
        logits = torch.stack(result)
        return (logits, torch.stack(features)) if return_features else logits
