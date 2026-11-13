"""History/network/environment conditional distributions of future trajectories."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hne_temporal import HNETemporal, WIDTH
from .hne_set import _masked_pool


class TrajectoryHNE(HNETemporal):
    def __init__(self, groups, history_steps=32, rank=4):
        super().__init__(groups, history_steps)
        self.rank = rank
        del self.risk
        del self.relation
        self.decoder = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.GELU(),
                                     nn.Linear(WIDTH, 7*(3+rank)))

    def fit_input_scalers(self, windows, nodes, pooled, weights):
        super().fit_scalers(np.arcsinh(windows), np.arcsinh(nodes), pooled, weights)
        for prefix in ['state', 'context']:
            getattr(self, prefix+'_mean').zero_()
            getattr(self, prefix+'_scale').fill_(1.)

    def forward(self, windows, nodes, pooled):
        windows, nodes = torch.asinh(windows), torch.asinh(nodes)
        p = nodes.shape[1]
        temporal, state_observed = self._temporal(windows[:, :, :p], 'state')
        environment, context_observed = self._temporal(windows[:, :, p:], 'context')
        reference, node_observed = self._scaled(nodes, 'node')
        reference = self.node_reference(torch.cat((reference, node_observed.to(reference.dtype)), -1))
        state = self.node_fusion(torch.cat((temporal, reference), -1))
        present = state_observed.any(1) | node_observed[..., :48].any(-1)
        context = self.context_fusion(_masked_pool(environment, context_observed.any(1)))
        gain, shift = self.modulate(context).chunk(2, -1)
        state = state * (2*torch.sigmoid(gain[:, None])) + torch.tanh(shift[:, None])
        scores = self.query(state) @ self.key(state).transpose(1, 2)/math.sqrt(WIDTH)
        scores = scores.masked_fill(~present[:, None], -torch.inf)
        scores = torch.where(present.any(1)[:, None, None], scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, -1)*present[:, None].to(scores.dtype)
        state = self.graph_norm(state + attention @ self.value(state))
        output = self.decoder(state).reshape(len(state), p, 7, 3+self.rank)
        mean = output[..., 0]
        sd = .05+F.softplus(output[..., 1])
        observation_logit = output[..., 2]
        factors = output[..., 3:]/math.sqrt(max(1, self.rank))
        return mean, sd, factors, observation_logit


def joint_loss(parameters, target):
    """Exact observed-submatrix Gaussian likelihood via the Woodbury identity."""
    mean, sd, factors, observation_logit = parameters
    target = target.to(mean.dtype)
    observed = torch.isfinite(target)
    actual = torch.where(observed, target, torch.zeros_like(target))
    transformed = torch.asinh(actual)
    shape = mean.shape
    residual = (transformed-mean).reshape(shape[0], -1)
    inverse = (observed/sd.square()).reshape(shape[0], -1)
    logdet = (2*observed*torch.log(sd)).sum((1, 2))
    quadratic = (inverse*residual.square()).sum(1)
    rank = factors.shape[-1]
    if rank:
        factors = factors.reshape(shape[0], -1, rank)
        weighted = inverse[..., None]*factors
        middle = factors.transpose(1, 2) @ weighted + torch.eye(rank, device=mean.device)
        cholesky = torch.linalg.cholesky(middle)
        projection = factors.transpose(1, 2) @ (inverse*residual)[..., None]
        quadratic -= (projection*torch.cholesky_solve(projection, cholesky)).sum((1, 2))
        logdet += 2*torch.log(cholesky.diagonal(dim1=-2, dim2=-1)).sum(1)
    count = observed.sum((1, 2)).to(mean.dtype)
    jacobian = (.5*torch.log1p(actual.square())*observed).sum((1, 2))
    values = (.5*(quadratic+logdet+count*math.log(2*math.pi))+jacobian)/count.clamp_min(1)
    availability = F.binary_cross_entropy_with_logits(observation_logit, observed.float(), reduction='none').mean((1, 2))
    return values+availability, values, availability


def simulated_event_probability(parameters, reference, families, protocol, draws, generator):
    """Apply the original three-valued event operator to simulated future records."""
    mean, sd, factors, observation_logit = parameters
    batch, channels, steps = mean.shape
    noise = torch.randn(batch, draws, channels, steps, device=mean.device, generator=generator)
    latent = mean[:, None]+sd[:, None]*noise
    if factors.shape[-1]:
        shared = torch.randn(batch, draws, factors.shape[-1], device=mean.device, generator=generator)
        latent += torch.einsum('bdk,bptk->bdpt', shared, factors)
    # Float64 preserves inverse-asinh values beyond float32's finite range.
    values = torch.sinh(latent.double())
    present = torch.rand(batch, draws, channels, steps, device=mean.device,
                         generator=generator) < torch.sigmoid(observation_logit)[:, None]
    eligible = reference['eligible'][:, None]
    count = present.sum(-1)
    valid = eligible & (count >= protocol['minimum_block_observations'])
    ordered = values.masked_fill(~present, torch.inf).sort(-1).values

    def quantile(q):
        position = (count.clamp_min(1)-1)*q
        low, high = position.floor().long(), position.ceil().long()
        lower = ordered.gather(-1, low[..., None]).squeeze(-1)
        upper = ordered.gather(-1, high[..., None]).squeeze(-1)
        return lower+(upper-lower)*(position-low)
    amplitude = quantile(protocol['amplitude_quantiles'][1])-quantile(protocol['amplitude_quantiles'][0])
    location = torch.where(present, values, 0.).sum(-1)/count.clamp_min(1)
    extreme = ((values.abs() > reference['absolute_point_threshold'][:, None, :, None]) & present).sum(-1)
    upper = ((values > protocol['bias_same_sign_z']) & present).sum(-1)
    lower = ((values < -protocol['bias_same_sign_z']) & present).sum(-1)
    abnormal = ((amplitude > reference['amplitude_threshold'][:, None]) &
                (extreme >= protocol['amplitude_minimum_extreme_days'])) | (
                (location.abs() > reference['bias_threshold'][:, None]) &
                (torch.maximum(upper, lower) >= protocol['bias_minimum_same_sign_days']))
    variable = torch.where(valid, abnormal.long(), -1)
    positive, unknown = torch.zeros((batch, draws), device=mean.device), torch.zeros((batch, draws), device=mean.device)
    for family in range(int(families.max().item())+1):
        member = (families == family) & reference['eligible']
        active = member.any(1)[:, None]
        yes = ((variable == 1) & member[:, None]).any(-1)
        no = ((variable == 0) | ~member[:, None]).all(-1)
        positive += (yes & active)
        unknown += (~yes & ~no & active)
    event = positive >= protocol['minimum_abnormal_families']
    resolved = event | ((positive+unknown) < protocol['minimum_abnormal_families'])
    return ((event.sum(1)+.5)/(resolved.sum(1)+1.)).cpu().numpy(), resolved.float().mean(1).cpu().numpy()
