"""Learned late fusion with the complete detached node/edge evidence arrays."""
import torch
from torch import nn


def family_pool(values, valid, family, width=16):
    """Per-channel patch mean, per-family channel mean, then family mean/max."""
    families = []
    for f in torch.unique(family):
        channels = []
        for c in torch.where(family == f)[0]:
            if valid[c].any():
                channels.append(values[c][valid[c]].mean(0))
        if channels:
            families.append(torch.stack(channels).mean(0))
    if not families:
        return values.new_zeros(2 * width)
    stacked = torch.stack(families)
    return torch.cat((stacked.mean(0), stacked.max(0).values))


class FullEvidenceFusion(nn.Module):
    """897 parameters; receives all evidence used by the internal guidance.

    No fixed event-summary bottleneck before learning: shared node/edge encoders
    process every valid patch/edge. Age is supplied to preserve time information.
    The same 12 quality summaries and frozen 130-dimensional L feature are also
    available. Final zero initialization starts from the exact frozen baseline.
    """
    def __init__(self, mode='hidden'):
        super().__init__()
        if mode not in ('hidden', 'logit', 'evidence'):
            raise ValueError(mode)
        self.mode = mode
        self.node = nn.Sequential(nn.Linear(10, 16), nn.GELU())
        self.edge = nn.Sequential(nn.Linear(5, 16), nn.GELU())
        inputs, width = {'hidden': (206, 3), 'logit': (77, 8), 'evidence': (76, 8)}[mode]
        self.head = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, logit, feature, evidence, family):
        node, edge = evidence['node'].detach(), evidence['edge'].detach()
        nv, ev = evidence['node_valid'], evidence['edge_valid']
        channels, patches, _ = node.shape
        age = torch.linspace(-1, 0, patches, device=node.device)
        encoded_node = self.node(torch.cat((node, age[None, :, None].expand(channels, -1, -1)), -1))
        encoded_edge = self.edge(torch.cat((edge, age[:, None, None, None].expand(-1, channels, channels, -1)), -1))
        # Average sources within a measurement family before combining families.
        source_means, source_valid = [], []
        for f in torch.unique(family):
            use = family == f
            mask = ev[:, :, use]
            x = (encoded_edge[:, :, use] * mask[..., None]).sum(2) / mask.sum(2)[..., None].clamp_min(1)
            source_means.append(x); source_valid.append(mask.any(2))
        values = torch.stack(source_means, 2)
        valid = torch.stack(source_valid, 2)
        by_target = (values * valid[..., None]).sum(2) / valid.sum(2)[..., None].clamp_min(1)
        node_pool = family_pool(encoded_node, nv, family)
        edge_pool = family_pool(by_target.transpose(0, 1), valid.any(2).T, family)
        shared = (evidence['summary'].detach(), node_pool, edge_pool)
        if self.mode == 'evidence':
            return self.head(torch.cat(shared)).reshape(1)
        base = feature.detach() if self.mode == 'hidden' else logit.detach().reshape(1)
        return logit.detach().reshape(1) + self.head(torch.cat((base, *shared))).reshape(1)


class InputEvidenceResidual(nn.Module):
    """Complete detached nodes/edges into historical patch embeddings (1785 parameters).

    Uses the original input path, so an unmodified frozen-prefix cache is invalid.
    Only recent observed patches receive residuals; native normalization is intact.
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(10, 16), nn.GELU())
        self.edge = nn.Sequential(nn.Linear(5, 8), nn.GELU())
        self.head = nn.Sequential(nn.Linear(24, 1), nn.GELU(), nn.Linear(1, hidden_size))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, evidence, family, keep, context_patches=132):
        node, edge = evidence['node'].detach(), evidence['edge'].detach()
        nv, ev = evidence['node_valid'], evidence['edge_valid']
        channels, patches, _ = node.shape
        age = torch.linspace(-1, 0, patches, device=node.device)
        n = self.node(torch.cat((node, age[None, :, None].expand(channels, -1, -1)), -1))
        e = self.edge(torch.cat((edge, age[:, None, None, None].expand(-1, channels, channels, -1)), -1))
        source_values, source_valid = [], []
        for f in torch.unique(family):
            mask = ev[:, :, family == f]
            source_values.append((e[:, :, family == f] * mask[..., None]).sum(2) / mask.sum(2)[..., None].clamp_min(1))
            source_valid.append(mask.any(2))
        values, valid = torch.stack(source_values, 2), torch.stack(source_valid, 2)
        pooled = ((values * valid[..., None]).sum(2) / valid.sum(2)[..., None].clamp_min(1)).transpose(0, 1)
        valid_patch = nv | ev.any(-1).T
        n = n * nv[..., None]
        delta = self.head(torch.cat((n, pooled), -1)) * valid_patch[..., None]
        delta = delta[keep.to(delta.device)]
        output = delta.new_zeros((delta.shape[0], context_patches, delta.shape[-1]))
        output[:, context_patches - patches:] = delta
        return output
