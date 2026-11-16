"""Continuous hidden-state feedback through a shared pretrained Transformer core."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from chronos.chronos2.model import Chronos2EncoderBlock, Chronos2Encoder
from chronos.chronos2.layers import Chronos2LayerNorm, Chronos2RotaryEmbedding
from chronos.chronos2.config import Chronos2CoreConfig
from .hne_routed import LoRALinear

ROOT = Path(__file__).resolve().parents[1]


class HNELatent(nn.Module):
    def __init__(self, config, widths, variant='recurrent'):
        super().__init__()
        self.config = config; self.variant = variant
        payload = torch.load(ROOT/'tools/models/chronos-2/latent_core.pt', map_location='cpu', weights_only=False)
        core = Chronos2CoreConfig(**payload['config']); core._attn_implementation = 'sdpa'
        self.core = nn.ModuleList([Chronos2EncoderBlock(core) for _ in payload['blocks']])
        self.output_norm = Chronos2LayerNorm(core.d_model, eps=core.layer_norm_epsilon)
        self.output_norm.load_state_dict(payload['norm'])
        for block, weights in zip(self.core, payload['blocks']):
            block.load_state_dict(weights)
        for module in [self.core, self.output_norm]:
            for p in module.parameters(): p.requires_grad_(False)
        for block in self.core:
            for layer in block.layer[:2]:
                for key in ['q', 'v']:
                    adapter = LoRALinear(getattr(layer.self_attention, key), config['lora_rank'])
                    if variant == 'frozen_core':
                        for parameter in adapter.parameters(): parameter.requires_grad_(False)
                    setattr(layer.self_attention, key, adapter)
        d = config['hidden_width']; width = core.d_model
        self.evidence = nn.ModuleList([nn.Sequential(nn.Linear(n, d), nn.GELU(), nn.LayerNorm(d),
                                                    nn.Linear(d, width)) for n in widths])
        self.channel_stats = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, width))
        self.token_types = nn.Parameter(torch.randn(5, width)*.01)
        self.initial_latent = nn.Parameter(torch.randn(width)*.01)
        self.feedback_gain = nn.Parameter(torch.ones(()))
        self.dropout = nn.Dropout(config['dropout'])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d), nn.GELU())
        self.risk = nn.Linear(2*d, 1)
        self.future = nn.Linear(d, 2)
        self.register_buffer('class_log_odds', torch.zeros(()))
        self.register_buffer('future_mean', torch.zeros(2))
        self.register_buffer('future_scale', torch.ones(2))

    def memory(self, x, mask, reference, reference_mask, stats, basis):
        b, c, _, _ = x.shape
        numeric = [encoder(values) for encoder, values in zip(self.evidence, basis)]
        omitted = {'no_history_coordinates': 0, 'no_network_coordinates': 1, 'no_environment_coordinates': 2}.get(self.variant)
        if omitted is not None: numeric[omitted] = torch.zeros_like(numeric[omitted])
        evidence = torch.stack(numeric, 1)[:, None].expand(-1, c, -1, -1)
        ref_mask = reference_mask[..., None]
        remembered = (reference*ref_mask).sum(2)/ref_mask.sum(2).clamp_min(1)
        if self.variant == 'no_personal_reference': remembered = torch.zeros_like(remembered)
        scale = self.channel_stats(stats.float().clamp(-8, 8))
        extra = torch.cat([remembered[:, :, None], scale[:, :, None], evidence], 2)
        extra = self.dropout(extra+self.token_types[None, None])
        memory = torch.cat([x, extra], 2)
        valid = torch.cat([mask, torch.ones((b, c, 5), device=x.device, dtype=torch.bool)], 2)
        return memory, valid

    def core_step(self, hidden, positions, time_mask, group_mask, cache):
        updated = []
        for index, block in enumerate(self.core):
            time_layer = block.layer[0]; attention = time_layer.self_attention
            normed = time_layer.layer_norm(hidden)
            shape = lambda value: value.reshape(len(value), -1, attention.n_heads, attention.kv_proj_dim).transpose(1, 2)
            q, k, v = [shape(layer(normed)) for layer in [attention.q, attention.k, attention.v]]
            cos, sin = attention.rope_embed(v, positions)
            q, k = Chronos2RotaryEmbedding.apply_rotary_pos_emb(q, k, cos, sin)
            if cache is not None:
                k = torch.cat([cache[index][0], k], 2)
                v = torch.cat([cache[index][1], v], 2)
            # Keep the graph: future losses must reach earlier continuous states and evidence.
            updated.append((k, v))
            attended, _ = attention._sdpa_attention(q, k, v, time_mask)
            attended = attended.transpose(1, 2).reshape(len(hidden), hidden.shape[1], attention.inner_dim)
            hidden = hidden+time_layer.dropout(attention.o(attended))
            hidden = block.layer[1](hidden, attention_mask=group_mask).hidden_states
            hidden = block.layer[2](hidden)
        return hidden, updated

    def forward(self, x, mask, reference, reference_mask, stats, basis, steps=None,
                intervention=None, return_states=False, use_cache=None):
        memory, valid = self.memory(x, mask, reference, reference_mask, stats, basis)
        b, c, m, width = memory.shape
        steps = steps or (1 if self.variant == 'single_pass' else self.config['latent_steps'])
        seed = self.initial_latent[None, None].expand(b, c, -1)
        latent_inputs = [seed]; states = []; logits = []; futures = []; cache = None
        use_cache = self.config.get('use_kv_cache', False) if use_cache is None else use_cache
        groups = torch.arange(b, device=x.device).repeat_interleave(c)
        n_state = c-self.config.get('context_channels', 1)
        for step in range(steps):
            tokens = torch.cat([memory, torch.stack(latent_inputs, 2)], 2)
            length = tokens.shape[2]
            full_valid = torch.cat([valid, torch.ones((b, c, step+1), device=x.device, dtype=torch.bool)], 2)
            flat_valid = full_valid.reshape(b*c, length)
            # Observations cannot read latent tokens; each latent position reads only earlier states.
            allowed = torch.ones((length, length), device=x.device, dtype=torch.bool)
            allowed[:m, m:] = False
            allowed[m:, m:] = torch.ones((step+1, step+1), device=x.device, dtype=torch.bool).tril()
            allowed = allowed[None, None] & flat_valid[:, None, None, :]
            time_mask = torch.zeros(allowed.shape, device=x.device, dtype=tokens.dtype).masked_fill(~allowed, -1e4)
            start = length-1 if use_cache and cache is not None else 0
            group_mask = Chronos2Encoder._construct_and_invert_group_time_mask(groups, flat_valid[:, start:].to(tokens.dtype), tokens.dtype)
            hidden = tokens[:, :, start:].reshape(b*c, length-start, width)
            positions = torch.arange(start, length, device=x.device)[None]
            if use_cache:
                hidden, cache = self.core_step(hidden, positions, time_mask[:, :, start:], group_mask, cache)
            else:
                for block in self.core:
                    hidden = block(hidden, position_ids=positions, attention_mask=time_mask,
                                   group_time_mask=group_mask).hidden_states
            state = self.output_norm(hidden[:, -1]).reshape(b, c, width)
            representation = self.readout(state[:, :n_state])
            pooled = torch.cat([representation.mean(1), representation.amax(1)], -1)
            logits.append(self.risk(pooled).squeeze(-1)-self.class_log_odds)
            futures.append(self.future(representation))
            states.append(state)
            feedback = self.feedback_gain*state
            if self.variant == 'detached_feedback': feedback = feedback.detach()
            if self.variant == 'reset_feedback' or intervention == 'reset': feedback = seed
            if intervention == 'shuffle': feedback = feedback.roll(1, 0)
            latent_inputs.append(feedback)
        output = dict(logits=logits[-1], step_logits=torch.stack(logits, 1), future=futures[-1])
        if return_states:
            output['states'] = torch.stack(states, 1)
            output['state_list'] = states
        return output


def latent_objective(model, out, y, weight, target, config):
    logits = out['step_logits'].float()+model.class_log_odds
    losses = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits), reduction='none')
    event = (losses[:, -1]*weight).mean()
    intermediate = (losses[:, :-1].mean(1)*weight).mean() if logits.shape[1] > 1 else event*0
    observed = torch.isfinite(target)
    normalized = (torch.nan_to_num(target)-model.future_mean)/model.future_scale
    future = F.smooth_l1_loss(out['future'].float(), normalized.clamp(-8, 8), reduction='none')
    # Preserve equal participant groups when execution micro-batches are combined.
    group_size = config['batch_size']//config['people_per_batch']
    future = ((future*observed).reshape(-1, group_size, *target.shape[1:]).sum((1, 2, 3)) /
              observed.reshape(-1, group_size, *target.shape[1:]).sum((1, 2, 3)).clamp_min(1)).mean()
    intermediate_weight = config['intermediate_loss_weight'] if logits.shape[1] > 1 else 0.
    total = (event+intermediate_weight*intermediate)/(1+intermediate_weight)+config['future_loss_weight']*future
    return total, dict(objective=total, event_bce=event, intermediate_bce=intermediate, future_loss=future)


def compact_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}


class LatentData:
    def __init__(self, dataset):
        folder = ROOT/'data/hne_routed'/dataset
        self.arrays = dict(np.load(folder/'arrays.npz'))
        prefix = ROOT/'data/hne_latent'/dataset
        self.history = torch.tensor(np.load(prefix/'history_prefix.npy'), device='cuda')
        self.reference = torch.tensor(np.load(prefix/'reference_prefix.npy'), device='cuda')
        self.mask = torch.tensor(np.load(folder/'history_mask.npy'), device='cuda', dtype=torch.bool)
        self.rmask = torch.tensor(np.load(folder/'reference_mask.npy'), device='cuda', dtype=torch.bool)
        self.stats = torch.tensor(self.arrays['stats'], device='cuda')
        self.people = torch.tensor(self.arrays['person_index'], device='cuda', dtype=torch.long)

    def batch(self, ids):
        ids = torch.as_tensor(ids, device='cuda'); people = self.people[ids]
        return [self.history[ids].float(), self.mask[ids], self.reference[people].float(),
                self.rmask[people], self.stats[ids]]
