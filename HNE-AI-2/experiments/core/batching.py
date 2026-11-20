"""Block-isolated batching of independent Chronos series groups.

Foreign groups receive negative infinity, not the finite native mask minimum:
otherwise a fully masked native row could average other people's values.
"""
import torch
from train_backbone import encoder_tail, read_event
from pack_guidance import pack_bias


def combine(caches, time_biases=None, group_biases=None):
    expected = {'hidden', 'position_ids', 'attention_mask', 'group_time_mask'}
    if any(set(cache) != expected for cache in caches):
        raise ValueError('Unknown cached encoder fields; do not silently drop masks')
    sizes = [cache['hidden'].shape[0] for cache in caches]
    total = sum(sizes); tokens = caches[0]['hidden'].shape[1]
    group = caches[0]['group_time_mask'].new_full((tokens, 1, total, total), float('-inf'))
    group_bias = torch.zeros_like(group) if group_biases is not None else None
    offset = 0
    for index, (cache, size) in enumerate(zip(caches, sizes)):
        assert cache['hidden'].shape[1] == tokens
        group[:, :, offset:offset + size, offset:offset + size] = cache['group_time_mask']
        if group_bias is not None:
            group_bias[:, :, offset:offset + size, offset:offset + size] = group_biases[index]
        offset += size
    merged = {key: torch.cat([cache[key] for cache in caches], dim=0)
              for key in ('hidden', 'position_ids', 'attention_mask')}
    merged['group_time_mask'] = group
    time_bias = torch.cat(time_biases, dim=0) if time_biases is not None else None
    return merged, time_bias, group_bias, sizes


def forward_group(model, head, items, dataset, guide=None, return_features=False):
    caches = [torch.load(item['prefix_path'], weights_only=True) for item in items]
    time_biases, group_biases, penalties = [], [], []
    if guide is not None:
        for item, cache in zip(items, caches):
            tb, gb, penalty = pack_bias(guide, item['evidence'], cache, item['keep'])
            time_biases.append(tb); group_biases.append(gb); penalties.append(penalty)
    merged, tb, gb, sizes = combine(caches, time_biases if guide is not None else None,
                                  group_biases if guide is not None else None)
    hidden = encoder_tail(model, merged, tb, gb)
    output, features, offset = [], [], 0
    for item, cache, size in zip(items, caches, sizes):
        value = read_event(head, hidden[offset:offset + size], cache, item['state'], item['family'], dataset, return_features)
        if return_features:
            logit, feature = value; output.append(logit); features.append(feature)
        else:
            output.append(value)
        offset += size
    penalty = torch.stack(penalties) if penalties else hidden.new_zeros(len(items))
    if return_features:
        return torch.cat(output), torch.cat(features), penalty
    return torch.cat(output), penalty
