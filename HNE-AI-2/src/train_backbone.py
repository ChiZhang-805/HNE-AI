"""LoRA, exact frozen-prefix caching, and the shared event readout."""
import math
import torch
from torch import nn
from chronos_bridge import encode_packed_segments, attention_guidance
from event_readout import FamilyEventHead
from training_data import segment_weights, FAMILIES


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.multiplier = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.multiplier


def load_backbone(path, device='cpu'):
    from chronos.chronos2.model import Chronos2Model
    model = Chronos2Model.from_pretrained(str(path), local_files_only=True,
                                         torch_dtype=torch.float32, attn_implementation='sdpa')
    model.requires_grad_(False)
    for index in (8, 9, 10, 11):
        for layer in (0, 1):
            attention = model.encoder.block[index].layer[layer].self_attention
            for projection in ('q', 'v'):
                setattr(attention, projection, LoRALinear(getattr(attention, projection)))
    # Backbone dropout disabled identically in every comparison. Gradients stay enabled.
    return model.eval().to(device), FamilyEventHead().to(device)


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if n.endswith('.a') or n.endswith('.b')}


def load_adapter(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


class _PrefixReady(Exception):
    pass


@torch.no_grad()
def capture_prefix(model, packed):
    """Stop at block 8 without running the remaining blocks or output projection."""
    saved = {}
    def hook(module, args, kwargs):
        hidden = args[0] if args else kwargs['hidden_states']
        saved['hidden'] = hidden.detach().cpu().clone()
        saved.update({k: v.detach().cpu().clone() for k, v in kwargs.items()
                      if isinstance(v, torch.Tensor) and k != 'hidden_states'})
        raise _PrefixReady()
    handle = model.encoder.block[8].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        encode_packed_segments(model, **packed)
    except _PrefixReady:
        pass
    finally:
        handle.remove()
    if not saved:
        raise RuntimeError('Frozen-prefix hook did not execute')
    return saved


def encoder_tail(model, cached, time_bias=None, group_bias=None):
    hidden = cached['hidden']
    kwargs = {k: v for k, v in cached.items() if k != 'hidden'}
    with attention_guidance(model, time_bias, group_bias):
        for block in model.encoder.block[8:]:
            hidden = block(hidden, **kwargs, output_attentions=False)[0]
    return model.encoder.dropout(model.encoder.final_layer_norm(hidden))


def token_mask(cached):
    return cached['attention_mask'][:, 0, 0, :] == 0


def read_event(head, hidden, cached, state, family, dataset, return_features=False):
    expected = len(set(FAMILIES[dataset][:5]))
    return head(hidden[None], token_mask(cached)[None], state, family,
                segment_weights(hidden.shape[1]).to(hidden.device),
                torch.tensor([expected], device=hidden.device), return_features)
