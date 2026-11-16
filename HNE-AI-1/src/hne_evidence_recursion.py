"""A shared recurrent block reads H/N/E evidence inside a frozen Chronos-2 encoder extension."""
import torch
from torch import nn


class EvidenceRecursion(nn.Module):
    def __init__(self, widths, prefix_width=768, width=48, depth=4, residual_scale=None):
        super().__init__()
        self.depth = depth
        self.widths = widths
        self.residual_scale = residual_scale
        self.coordinates = nn.ModuleList([
            nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.Linear(width, width))
            for size in widths
        ])
        self.prefix = nn.Sequential(nn.LayerNorm(prefix_width), nn.Linear(prefix_width, width))
        self.initial = nn.Parameter(torch.zeros(width))
        self.step_embedding = nn.Parameter(torch.randn(depth, width) * 0.02)
        self.key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.update = nn.GRUCell(width, width)
        self.readout = nn.Linear(width, 1)
        if residual_scale is not None:
            self.register_buffer('anchor_weights',torch.zeros(sum(widths)))
            self.register_buffer('anchor_bias',torch.zeros(()))
            nn.init.zeros_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    def anchor(self, coordinates, omit=None):
        terms = [x @ w for j,(x,w) in enumerate(zip(coordinates,self.anchor_weights.split(self.widths))) if j!=omit]
        return self.anchor_bias+sum(terms)

    def forward(self, coordinates, prefix, variant='feedback', omit=None):
        # Prefix entries: historical state difference, recent context difference.
        h, n, e = [encoder(x) for encoder, x in zip(self.coordinates, coordinates)]
        memory = torch.stack([h+self.prefix(prefix[:, 0]), n, e+self.prefix(prefix[:, 1])], dim=1)
        keys = self.key(self.norm(memory))
        seed = self.initial.expand(len(h), -1)
        state = seed
        anchor = self.anchor(coordinates,omit) if self.residual_scale is not None else None
        scores = []
        for step in range(1 if variant=='one_step' else self.depth):
            previous = seed if variant=='reset' else state
            query_state = seed if variant=='fixed_query' else previous
            position = self.step_embedding[step]
            query = self.query(self.norm(query_state + position))
            attention = (keys * query[:, None]).sum(-1) / query.shape[-1]**0.5
            if omit is not None:
                attention = attention.masked_fill(torch.arange(3, device=h.device)==omit, -torch.inf)
            evidence = (attention.softmax(-1).unsqueeze(-1)*memory).sum(1)
            state = self.update(evidence + position, previous)
            score = self.readout(self.norm(state)).squeeze(-1)
            if anchor is not None:
                score = anchor + self.residual_scale*score.tanh()
            scores.append(score)
        return torch.stack(scores, dim=1)
