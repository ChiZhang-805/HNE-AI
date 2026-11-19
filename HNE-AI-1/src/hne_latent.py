"""HNE: evidence memory -> conditioned core -> warning -> continuous feedback.

Algorithm references and author-code links: literature/continuous_reasoning_methods.md.
The forward loop follows Coconut's hidden-state feedback. The optional core
modulation adapts LoopFormer-style conditioning to pretrained Chronos-2 blocks.
"""
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
    """Shared core for both cohorts; H/N/E are evidence, not ordered reasoning steps."""

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
        self.core_modulation = config.get('core_modulation', False)
        if self.core_modulation:
            self.iteration_embedding = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, width))
            self.core_modulators = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d), nn.SiLU(), nn.Linear(d, 4*width))
                for _ in self.core
            ])
            # Begin at the pretrained computation; learn bounded deviations inside its blocks.
            for modulator in self.core_modulators:
                nn.init.zeros_(modulator[-1].weight)
                nn.init.zeros_(modulator[-1].bias)
        self.register_buffer('class_log_odds', torch.zeros(()))
        self.register_buffer('future_mean', torch.zeros(2))
        self.register_buffer('future_scale', torch.ones(2))

    def memory(self, x, mask, reference, reference_mask, stats, basis):
        """Append reference, scale and H/N/E tokens to each channel's observed history.

        x: [batch, channels, history_tokens, width]. The returned memory has
        five extra tokens per channel; valid marks the observable key positions.
        """
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

    def core_step(self, hidden, positions, time_mask, group_mask, cache, condition=None, latent_mask=None):
        """Update states inside shared temporal/channel attention and feedforward layers.

        Modulation is restricted to latent positions. Zero modulation preserves
        the pretrained residual branches through (1 + gate). Cached K/V retain
        their computation graph so later losses can train earlier computations.
        """
        updated = []
        for index, block in enumerate(self.core):
            time_layer = block.layer[0]; attention = time_layer.self_attention
            normed = time_layer.layer_norm(hidden)
            scale = gate_time = gate_group = gate_ff = 0.
            if condition is not None and self.variant != 'no_modulation':
                modulation = .25*torch.tanh(self.core_modulators[index](condition))
                modulation = modulation*latent_mask
                scale, gate_time, gate_group, gate_ff = modulation.chunk(4, -1)
                normed = normed*(1+scale)
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
            hidden = hidden+(1+gate_time)*time_layer.dropout(attention.o(attended))
            grouped = block.layer[1](hidden, attention_mask=group_mask).hidden_states
            if condition is None or self.variant == 'no_modulation':
                hidden = block.layer[2](grouped)
            else:
                hidden = hidden+(1+gate_group)*(grouped-hidden)
                transformed = block.layer[2](hidden)
                hidden = hidden+(1+gate_ff)*(transformed-hidden)
        return hidden, updated

    def step_condition(self, memory, previous_input, seed, step, steps):
        """Combine the current latent input, H/N/E evidence and internal step time."""
        clock = memory.new_tensor([step/steps, 1/steps])
        condition_input = previous_input+memory[:, :, -3:].mean(2)
        if self.variant == 'clock_only_modulation':
            condition_input = seed
        return condition_input+self.iteration_embedding(clock)

    def prepare_step(self, memory, valid, latent_inputs, conditions, cached, reset):
        """Build the visible prefix and masks for one internal computation step.

        Observations never read latent positions. Each latent position reads
        observations, itself and earlier latent positions. With a K/V cache,
        only the newest query is recomputed; all valid prefix keys remain.
        """
        b, c, m, width = memory.shape
        count = len(latent_inputs)
        tokens = torch.cat([memory, torch.stack(latent_inputs, 2)], 2)
        length = tokens.shape[2]
        full_valid = torch.cat([valid, torch.ones((b, c, count), device=memory.device, dtype=torch.bool)], 2)
        flat_valid = full_valid.reshape(b*c, length)
        allowed = torch.ones((length, length), device=memory.device, dtype=torch.bool)
        allowed[:m, m:] = False
        allowed[m:, m:] = torch.ones((count, count), device=memory.device, dtype=torch.bool).tril()
        if self.core_modulation and reset:
            # This control removes earlier latent K/V reads as well as explicit feedback.
            allowed[m:, m:] = torch.eye(count, device=memory.device, dtype=torch.bool)
        allowed = allowed[None, None] & flat_valid[:, None, None, :]
        time_mask = torch.zeros(allowed.shape, device=memory.device, dtype=tokens.dtype).masked_fill(~allowed, -1e4)
        start = length-1 if cached else 0
        groups = torch.arange(b, device=memory.device).repeat_interleave(c)
        group_mask = Chronos2Encoder._construct_and_invert_group_time_mask(
            groups, flat_valid[:, start:].to(tokens.dtype), tokens.dtype)
        hidden = tokens[:, :, start:].reshape(b*c, length-start, width)
        positions = torch.arange(start, length, device=memory.device)[None]
        condition = latent_mask = None
        if self.core_modulation:
            condition = torch.cat([torch.zeros_like(memory), torch.stack(conditions, 2)], 2)
            condition = condition[:, :, start:].reshape(b*c, length-start, width)
            latent_mask = (positions >= m).to(hidden.dtype)[..., None]
        return hidden, positions, time_mask[:, :, start:], group_mask, condition, latent_mask

    def read_state(self, hidden, batch_size, channels):
        """Read risk and future summaries from state channels at the newest position."""
        state = self.output_norm(hidden[:, -1]).reshape(batch_size, channels, -1)
        n_state = channels-self.config.get('context_channels', 1)
        representation = self.readout(state[:, :n_state])
        pooled = torch.cat([representation.mean(1), representation.amax(1)], -1)
        logit = self.risk(pooled).squeeze(-1)-self.class_log_odds
        return state, logit, self.future(representation)

    def feedback_input(self, state, seed, intervention):
        """Write the continuous state into the next position, without text decoding."""
        feedback = self.feedback_gain*state
        if self.variant == 'detached_feedback':
            # Detach this edge only; the separate K/V graph is still retained.
            feedback = feedback.detach()
        if self.variant == 'reset_feedback' or intervention == 'reset':
            feedback = seed
        if intervention == 'shuffle':
            feedback = feedback.roll(1, 0)
        return feedback

    def forward(self, x, mask, reference, reference_mask, stats, basis, steps=None,
                intervention=None, return_states=False, use_cache=None):
        """Refine one prediction over internal steps; no new observations arrive.

        return_states exposes [batch, steps, channels, width] and the original
        state tensors for trajectory analysis and cross-step gradient probes.
        """
        memory, valid = self.memory(x, mask, reference, reference_mask, stats, basis)
        b, c = memory.shape[:2]
        steps = steps or (1 if self.variant == 'single_pass' else self.config['latent_steps'])
        seed = self.initial_latent[None, None].expand(b, c, -1)
        latent_inputs, conditions = [seed], []
        states, logits, futures = [], [], []
        cache = None
        use_cache = self.config.get('use_kv_cache', False) if use_cache is None else use_cache
        reset = self.variant == 'reset_feedback' or intervention == 'reset'
        for step in range(steps):
            # 1. Let evidence and the previous state condition this internal step.
            if self.core_modulation:
                conditions.append(self.step_condition(memory, latent_inputs[-1], seed, step, steps))
            hidden, positions, time_mask, group_mask, condition, latent_mask = self.prepare_step(
                memory, valid, latent_inputs, conditions, use_cache and cache is not None, reset)

            # 2. Reuse the same pretrained blocks, including their trainable LoRA.
            if use_cache or self.core_modulation:
                hidden, new_cache = self.core_step(hidden, positions, time_mask, group_mask,
                    cache if use_cache else None, condition, latent_mask)
                cache = new_cache if use_cache else None
            else:
                for block in self.core:
                    hidden = block(hidden, position_ids=positions, attention_mask=time_mask,
                                   group_time_mask=group_mask).hidden_states

            # 3. Read the current prediction and feed its hidden state forward.
            state, logit, future = self.read_state(hidden, b, c)
            logits.append(logit)
            futures.append(future)
            states.append(state)
            latent_inputs.append(self.feedback_input(state, seed, intervention))
        output = dict(logits=logits[-1], step_logits=torch.stack(logits, 1), future=futures[-1],
                      step_future=torch.stack(futures, 1))
        if return_states:
            output['states'] = torch.stack(states, 1)
            output['state_list'] = states
        return output


def latent_objective(model, out, y, weight, target, config):
    """Train event prediction and future-summary decoding through the latent steps.

    Every step predicts the same future window. These observed training targets
    ground the states; they do not prescribe a different semantic task per step.
    """
    logits = out['step_logits'].float()+model.class_log_odds
    losses = F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits), reduction='none')
    event = (losses[:, -1]*weight).mean()
    intermediate = (losses[:, :-1].mean(1)*weight).mean() if logits.shape[1] > 1 else event*0
    observed = torch.isfinite(target)
    normalized = (torch.nan_to_num(target)-model.future_mean)/model.future_scale
    prediction = out['future'][:, None]
    if config.get('dense_future_supervision', False) and model.variant != 'final_future_only':
        prediction = out['step_future']
    future = F.smooth_l1_loss(prediction.float(), normalized[:, None].clamp(-8, 8).expand_as(prediction), reduction='none').mean(1)
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
