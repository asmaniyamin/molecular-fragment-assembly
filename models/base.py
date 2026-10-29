# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap

import torch
import torch.nn as nn
from env.features import STATE_DIM, ACTION_FEAT_DIM


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_feat_dim: int = ACTION_FEAT_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_enc = _mlp(state_dim, hidden_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + action_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action_feats: torch.Tensor,
    ) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        s_enc = self.state_enc(state)
        n = action_feats.shape[0]
        s_exp = s_enc.expand(n, -1)
        inp = torch.cat([s_exp, action_feats], dim=-1)
        return self.scorer(inp).squeeze(-1)

    def action_dist(
        self,
        state: torch.Tensor,
        action_feats: torch.Tensor,
    ) -> torch.distributions.Categorical:
        logits = self.forward(state, action_feats)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
