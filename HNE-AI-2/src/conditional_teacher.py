"""History-only and switchable conditional teachers; legacy teacher stays intact."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from precursor_guidance import ConditionalPredictor, masked_softmax


def history_inputs(model, values, observed, calendar, state, reference):
    encoded = model.encode(values, observed, calendar, state)
    prior = F.pad(encoded[:, :, :-1], (0, 0, 1, 0))
    if reference is not None:
        prior = prior + model.reference_projection(reference)[:, :, None]
    minimum = torch.where(state, model.patch_size // 2, 1)[:, :, None]
    valid = F.pad(observed.sum(-1)[:, :, :-1] >= minimum, (1, 0))
    return prior.permute(0, 2, 1, 3), valid.permute(0, 2, 1)


def source_mask(valid):
    channels = valid.shape[-1]
    return valid[:, :, None, :] & ~torch.eye(channels, dtype=torch.bool, device=valid.device)[None, None]


class HistoryPredictor(ConditionalPredictor):
    def forward(self, values, observed, calendar, state, family, reference=None, single_source=True,
                condition_lag=0, conditions=True):
        h, valid = history_inputs(self, values, observed, calendar, state, reference)
        mu, sigma = self.distribution(h)
        view = (mu.permute(0, 2, 1, 3), sigma.permute(0, 2, 1, 3))
        views = {name: view for name in ('H', 'HN', 'HNE')}
        if single_source:
            views['pair'] = (mu[:, :, :, None].expand(-1, -1, -1, h.shape[2], -1),
                             sigma[:, :, :, None].expand(-1, -1, -1, h.shape[2], -1))
        allowed = source_mask(valid).expand(-1, -1, h.shape[2], -1)
        return dict(views=views, pair_allowed=allowed,
                    a_n=torch.zeros_like(allowed, dtype=values.dtype),
                    a_e=torch.zeros_like(allowed, dtype=values.dtype))


class ResidualConditionalPredictor(nn.Module):
    """Frozen H plus N/E residuals that vanish with disabled or absent messages.

    Subtracting the zero-message head rules out a generic extra H-only bias.
    Conditions use observed prior patches; a positive lag only makes them older.
    """
    def __init__(self, history=None):
        super().__init__()
        self.history = history if history is not None else HistoryPredictor()
        self.history.requires_grad_(False)
        width = self.history.width
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.message = nn.Linear(width, width, bias=False)
        self.n_head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 32))
        self.e_head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 32))
        for head in (self.n_head, self.e_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def reference_summary(self, *args, **kwargs):
        return self.history.reference_summary(*args, **kwargs)

    def correction(self, head, target, message, allowed):
        target = target.expand_as(message)
        raw = head(torch.cat((target, message), -1)) - head(torch.cat((target, torch.zeros_like(message)), -1))
        return raw.masked_fill(~allowed[..., None], 0).chunk(2, -1)

    def forward(self, values, observed, calendar, state, family, reference=None, single_source=True,
                condition_lag=0, conditions=True):
        h, valid = history_inputs(self.history, values, observed, calendar, state, reference)
        source = h
        if condition_lag:
            if condition_lag < 0:
                raise ValueError('Condition controls cannot use future patches')
            source = F.pad(h[:, :-condition_lag], (0, 0, 0, 0, condition_lag, 0))
            valid = F.pad(valid[:, :-condition_lag], (0, 0, condition_lag, 0))
        allowed = source_mask(valid).expand(-1, -1, h.shape[2], -1)
        if not conditions:
            allowed = torch.zeros_like(allowed)
        scores = torch.einsum('bpjw,bpkw->bpjk', self.query(h), self.key(source)) / math.sqrt(h.shape[-1])
        same = (family[:, :, None] == family[:, None, :]) & (state[:, :, None] == state[:, None, :])
        count = (same[:, None] & valid[:, :, None, :]).sum(-1)
        count = count[:, :, None, :] - (same[:, None] & valid[:, :, :, None]).to(count.dtype)
        scores = scores - count.clamp_min(1).log()
        n_allowed = allowed & state[:, None, None, :]
        e_allowed = allowed & ~state[:, None, None, :]
        a_n, a_e = masked_softmax(scores, n_allowed), masked_softmax(scores, e_allowed)
        message = self.message(source)
        n = torch.einsum('bpjk,bpkw->bpjw', a_n, message)
        e = torch.einsum('bpjk,bpkw->bpjw', a_e, message)
        dn, sn = self.correction(self.n_head, h, n, n_allowed.any(-1))
        de, se = self.correction(self.e_head, h, e, e_allowed.any(-1))
        mu, sigma = self.history.distribution(h)
        mu_n = mu + 2 * sigma * dn.tanh()
        sigma_n = sigma * (math.log(2) * sn.tanh()).exp()
        mu_ne = mu_n + 2 * sigma * de.tanh()
        sigma_ne = sigma_n * (math.log(2) * se.tanh()).exp()
        views = {name: (m.permute(0, 2, 1, 3), s.permute(0, 2, 1, 3))
                 for name, m, s in [('H', mu, sigma), ('HN', mu_n, sigma_n), ('HNE', mu_ne, sigma_ne)]}
        if single_source:
            shape = (*h.shape[:3], h.shape[2], h.shape[-1])
            target = h[:, :, :, None].expand(shape)
            single = message[:, :, None].expand(shape)
            nd, ns = self.correction(self.n_head, target, single, n_allowed)
            ed, es = self.correction(self.e_head, target, single, e_allowed)
            views['pair'] = (mu[:, :, :, None] + 2 * sigma[:, :, :, None] * (nd + ed).tanh(),
                             sigma[:, :, :, None] * (math.log(2) * (ns + es).tanh()).exp())
        return dict(views=views, pair_allowed=allowed, a_n=a_n, a_e=a_e)


def teacher_from_checkpoint(checkpoint):
    kind = checkpoint.get('architecture', 'shared')
    model = {'shared': ConditionalPredictor, 'history': HistoryPredictor,
             'residual': ResidualConditionalPredictor}[kind]()
    model.load_state_dict(checkpoint['state'])
    return model.eval().requires_grad_(False)
