"""Vectorize frozen checkpoints while sharing the pretrained core weights."""
import torch
from torch.func import functional_call, vmap


def stack_states(states, device):
    return {name: torch.stack([state[name] for state in states]).to(device) for name in states[0]}


def predict_members(model, states, inputs, basis, chunk_size=5):
    def apply(parameters, coordinates):
        return functional_call(model, parameters, (*inputs, coordinates), strict=False)['logits'].float().sigmoid()
    return vmap(apply, in_dims=(0, 0), chunk_size=chunk_size)(states, basis)
