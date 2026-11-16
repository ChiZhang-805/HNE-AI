"""Sparse evidence scores scattered onto native attention token axes."""
import torch
from precursor_guidance import bounded_centered_bias


def pack_bias(guide, evidence, cached, keep):
    # Compute only the active 21 x 21 patch scores. Center on all native valid keys.
    node = evidence['node'][keep].detach()
    valid = evidence['node_valid'][keep]
    channels, patches, _ = node.shape
    times = torch.arange(patches, device=node.device) * 16
    lag = torch.asinh((times[:, None] - times[None, :]) / 168.)
    pairs = torch.cat((node[:, :, None].expand(-1, -1, patches, -1),
                       node[:, None].expand(-1, patches, -1, -1),
                       lag[None, :, :, None].expand(channels, -1, -1, -1)), -1)
    raw_time = guide.time(pairs).squeeze(-1).masked_fill(~(valid[:, :, None] & valid[:, None, :]), 0)
    edge = evidence['edge'][:, keep][:, :, keep].detach()
    ev = evidence['edge_valid'][:, keep][:, :, keep]
    raw_group = guide.group(edge).squeeze(-1).masked_fill(~ev, 0)
    tokens = cached['hidden'].shape[1]
    t = raw_time.new_zeros(channels, tokens, tokens)
    g = raw_group.new_zeros(tokens, channels, channels)
    t[:, 111:132, 111:132] = raw_time
    g[111:132] = raw_group
    allowed_t = cached['attention_mask'][:, 0] == 0
    allowed_g = cached['group_time_mask'][:, 0] == 0
    tb = bounded_centered_bias(t, allowed_t, guide.bound)
    gb = bounded_centered_bias(g, allowed_g, guide.bound)
    penalty = tb.square().mean() + gb.square().mean()
    return tb[:, None], gb[:, None], penalty
