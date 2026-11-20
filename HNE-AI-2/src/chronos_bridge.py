"""Explicit, reversible Chronos-2 attention interface for the design.

Uses the installed package without editing it. Model weights are supplied by the
caller. No downloading, GPU allocation, optimizer, or training loop is performed.
"""
from contextlib import contextmanager
import torch


def preserve_mask_add_bias(native_mask, bias):
    bias = bias.to(device=native_mask.device, dtype=native_mask.dtype)
    original, delta = torch.broadcast_tensors(native_mask, bias)
    # Native masks use 0 and finfo.min. A guidance signal can never unmask padding.
    return torch.where(original < 0, original, original + delta)


@contextmanager
def attention_guidance(model, time_bias=None, group_bias=None, blocks=(8, 9, 10, 11), gate=1.0):
    """blocks are zero-based; not intended for simultaneous calls on one model.

    Time bias: [flattened_channel,1,token,token].
    Group bias: [token,1,flattened_channel,flattened_channel].
    Scalar gate is used for one-origin inference. Batched per-origin gates must be
    packed onto each bias tensor by the caller before entering this context.
    """
    handles = []

    def make_hook(bias):
        def hook(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs['mask'] = preserve_mask_add_bias(kwargs['mask'], bias * gate)
            return args, kwargs
        return hook

    try:
        for index in blocks:
            block = model.encoder.block[index]
            if time_bias is not None:
                handles.append(block.layer[0].self_attention.register_forward_pre_hook(make_hook(time_bias), with_kwargs=True))
            if group_bias is not None:
                handles.append(block.layer[1].self_attention.register_forward_pre_hook(make_hook(group_bias), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def encode_packed_segments(model, context, observed, hour_offsets, group_ids,
                           num_output_patches=12, time_bias=None, group_bias=None,
                           blocks=(8, 9, 10, 11), gate=1.0, input_residual=None):
    """Pack prefix/recent observations without pretending the gap is continuous.

    context/mask/hour_offsets: [flattened_channel, hours]. Missing values are NaN
    BEFORE native normalization. Offsets are real hours relative to issue time,
    ending at -1. Every patch is internally contiguous; gaps occur between patches.
    Only masked future tokens are constructed. No future observations are accepted.
    """
    if context.shape != observed.shape or context.shape != hour_offsets.shape:
        raise ValueError('Context, observed mask and offsets must have equal shapes')
    patch = model.chronos_config.input_patch_size
    if context.shape[1] % patch:
        raise ValueError('Segments must be independently aligned to full patches')
    if context.shape[1] > model.chronos_config.context_length:
        raise ValueError('Packed context would be truncated by the backbone')
    if (hour_offsets >= 0).any():
        raise ValueError('Only completed historical observations are accepted')
    if not torch.all(hour_offsets[:, 1:] > hour_offsets[:, :-1]):
        raise ValueError('Hour offsets must be strictly chronological')
    offsets = hour_offsets.reshape(context.shape[0], -1, patch)
    if not torch.all(offsets[:, :, 1:] - offsets[:, :, :-1] == 1):
        raise ValueError('A real time gap cannot occur inside a patch')
    if not torch.all(hour_offsets[:, -1] == -1):
        raise ValueError('Packed window must end immediately before the issue time')
    observed = observed.bool() & torch.isfinite(context)
    if (~observed.any(-1)).any():
        raise ValueError('Remove entirely unobserved channels before packing')
    clean = context.masked_fill(~observed, torch.nan)
    # The native encoder subtracts its mask from 1.0; it expects a floating mask.
    patched, attention_mask, loc_scale = model._prepare_patched_context(clean, observed.to(context.dtype))
    n_context = attention_mask.shape[1]
    patched = patched.clone()
    patched[..., :patch] = offsets.to(patched.dtype) / model.chronos_config.time_encoding_scale
    embeddings = model.input_patch_embedding(patched)
    if input_residual is not None:
        if input_residual.shape != embeddings.shape:
            raise ValueError('Input guidance must match historical patch embeddings')
        embeddings = embeddings + input_residual.to(embeddings) * attention_mask[..., None] * gate
    # Shift all RoPE positions by n_context to match the official contiguous path
    # exactly in the no-gap case. Nonintegral offsets are permitted by RoPE.
    positions = offsets[..., 0].to(patched.dtype) / patch + n_context
    if model.chronos_config.use_reg_token:
        ids = torch.full((context.shape[0], 1), model.config.reg_token_id,
                         dtype=torch.long, device=context.device)
        embeddings = torch.cat((embeddings, model.shared(ids)), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(ids, dtype=torch.bool)), dim=1)
        positions = torch.cat((positions, torch.full_like(positions[:, :1], n_context)), dim=1)
    future, _ = model._prepare_patched_future(
        future_covariates=None, future_covariates_mask=None, loc_scale=loc_scale,
        num_output_patches=num_output_patches, batch_size=context.shape[0])
    future_embedding = model.input_patch_embedding(future)
    future_start = n_context + int(model.chronos_config.use_reg_token)
    future_positions = torch.arange(future_start, future_start + num_output_patches,
                                    dtype=positions.dtype, device=context.device)
    positions = torch.cat((positions, future_positions[None].expand(context.shape[0], -1)), dim=1)
    embeddings = torch.cat((embeddings, future_embedding), dim=1)
    attention_mask = torch.cat((attention_mask, torch.ones((context.shape[0], num_output_patches),
                                      dtype=torch.bool, device=context.device)), dim=1)
    with attention_guidance(model, time_bias, group_bias, blocks, gate):
        attention_mask = attention_mask.to(embeddings.dtype)
        outputs = model.encoder(inputs_embeds=embeddings, attention_mask=attention_mask,
                                group_ids=group_ids, position_ids=positions)
    return outputs.last_hidden_state, attention_mask, positions, loc_scale
