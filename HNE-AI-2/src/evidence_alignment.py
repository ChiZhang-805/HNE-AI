"""Relocate innovations only among valid completed patches at a single issue time."""
import torch


def patch_indices(valid):
    """Return aligned and reversed indices with identical missing positions."""
    positions = torch.arange(valid.shape[-1], device=valid.device).expand_as(valid)
    rank = valid.long().cumsum(-1)
    ordered = positions.masked_fill(~valid, valid.shape[-1]).sort(-1).values
    reversed_valid = ordered.gather(-1, (valid.sum(-1, keepdim=True) - rank).clamp_min(0))
    return positions, torch.where(valid, reversed_valid, positions)


def alignment_indices(evidence):
    return (patch_indices(evidence['node_valid']),
            patch_indices(evidence['edge_valid'].permute(1, 2, 0)))


def relocate_innovations(evidence, indices, reverse):
    node_index, edge_index = (pair[int(reverse)] for pair in indices)
    node = evidence['node']
    edge = evidence['edge'].permute(1, 2, 0, 3)
    # Node 8 and edge 1:4 are coverage or reference/source context. Keep them at their time.
    node_innovation = node[..., :8].gather(1, node_index[..., None].expand_as(node[..., :8]))
    edge_innovation = edge[..., :1].gather(2, edge_index[..., None])
    return {**evidence, 'node': torch.cat((node_innovation, node[..., 8:]), -1),
            'edge': torch.cat((edge_innovation, edge[..., 1:]), -1).permute(2, 0, 1, 3)}
