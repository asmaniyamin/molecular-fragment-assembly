# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap

from typing import Optional, Tuple

import torch
import torch.nn as nn

from core.structs import MolGraph
from env.features import ACTION_FEAT_DIM
from env.data import GOAL_DIM

from .gnn import MolGNN, mol_graph_to_tensors, _mlp


class GNNSuccessor(nn.Module):

    def __init__(
        self,
        gnn_dim:         int = 128,
        goal_dim:        int = GOAL_DIM,
        action_feat_dim: int = ACTION_FEAT_DIM,
        hidden_dim:      int = 256,
        site_actions:    bool = False,
        frag_site_emb:   bool = False,
        num_frag_sites:  int = 0,
        frag_site_dim:   int = 64,
    ):
        super().__init__()
        self.gnn          = MolGNN(hidden_dim=gnn_dim)
        self.goal_dim     = goal_dim
        self.gnn_dim      = gnn_dim
        self.site_actions = site_actions
        self.frag_site_emb = bool(frag_site_emb and site_actions)
        self.num_frag_sites = num_frag_sites

        self.state_enc = _mlp(gnn_dim + goal_dim, hidden_dim, hidden_dim)
        frag_dim = frag_site_dim if self.frag_site_emb else 0
        scorer_in_dim = hidden_dim + action_feat_dim + (gnn_dim if site_actions else 0) + frag_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )
        if self.frag_site_emb:
            self.frag_proj = nn.Linear(gnn_dim, frag_site_dim)
            self.register_buffer("frag_site_cache",
                                 torch.zeros(num_frag_sites + 1, gnn_dim), persistent=False)


    @torch.no_grad()
    def refresh_frag_cache(self, env, device: torch.device) -> None:
        if not self.frag_site_emb:
            return
        cache = torch.zeros(self.num_frag_sites + 1, self.gnn_dim, device=device)
        by_frag: dict = {}
        for (fi, dummy_idx), sid in env._frag_site_id.items():
            by_frag.setdefault(fi, []).append((dummy_idx, sid))
        for fi, sites in by_frag.items():
            mg = env._frag_graph(fi)
            if mg is None:
                continue
            nf, ei, ea = mol_graph_to_tensors(mg, device)
            node_emb = self.gnn.encode_nodes(nf, ei, ea)
            n_nodes = node_emb.shape[0]
            for dummy_idx, sid in sites:
                if 0 <= dummy_idx < n_nodes:
                    cache[sid] = node_emb[dummy_idx]
        self.frag_site_cache = cache


    def encode(
        self,
        mg:     MolGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nf, ei, ea = mol_graph_to_tensors(mg, device)
        node_emb = self.gnn.encode_nodes(nf, ei, ea)
        g_emb = self.gnn.pool(node_emb)
        return g_emb, node_emb

    def _action_inputs(
        self,
        s_enc:          torch.Tensor,
        node_emb:       torch.Tensor,
        action_feats:   torch.Tensor,
        action_site_ix: Optional[torch.Tensor],
        device:         torch.device,
    ) -> torch.Tensor:
        n = action_feats.shape[0]
        parts = [s_enc.unsqueeze(0).expand(n, -1), action_feats]
        if self.site_actions:
            if action_site_ix is None:
                mol_ix = torch.full((n,), -1, dtype=torch.long, device=device)
                frag_ix = torch.zeros((n,), dtype=torch.long, device=device)
            elif action_site_ix.dim() == 2:
                mol_ix, frag_ix = action_site_ix[:, 0], action_site_ix[:, 1]
            else:
                mol_ix, frag_ix = action_site_ix, torch.zeros_like(action_site_ix)
            site_emb = torch.zeros(n, self.gnn_dim, dtype=node_emb.dtype, device=device)
            valid = mol_ix >= 0
            if valid.any():
                site_emb[valid] = node_emb[mol_ix[valid]]
            parts.append(site_emb)
            if self.frag_site_emb:
                frag_ix = frag_ix.clamp(min=0, max=self.num_frag_sites)
                parts.append(self.frag_proj(self.frag_site_cache[frag_ix]))
        return torch.cat(parts, dim=-1)

    def psi(
        self,
        mg:             MolGraph,
        goal_norm:      torch.Tensor,
        action_feats:   torch.Tensor,
        device:         torch.device,
        action_site_ix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g_emb, node_emb = self.encode(mg, device)
        s_enc = self.state_enc(torch.cat([g_emb, goal_norm], dim=-1))
        inp = self._action_inputs(s_enc, node_emb, action_feats, action_site_ix, device)
        return self.scorer(inp)


    @staticmethod
    def q_from_psi(psi: torch.Tensor, reward_goal: torch.Tensor) -> torch.Tensor:
        return -torch.linalg.norm(psi - reward_goal.unsqueeze(0), dim=-1)

    def action_dist(
        self,
        mg:             MolGraph,
        goal_norm:      torch.Tensor,
        action_feats:   torch.Tensor,
        device:         torch.device,
        action_site_ix: Optional[torch.Tensor] = None,
        temperature:    float = 0.2,
    ) -> torch.distributions.Categorical:
        psi = self.psi(mg, goal_norm, action_feats, device, action_site_ix)
        q = self.q_from_psi(psi, goal_norm)
        return torch.distributions.Categorical(logits=q / max(temperature, 1e-6))


    @torch.no_grad()
    def gpi_scores(
        self,
        mg:             MolGraph,
        reward_goal:    torch.Tensor,
        base_goals:     torch.Tensor,
        action_feats:   torch.Tensor,
        device:         torch.device,
        action_site_ix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_emb, node_emb = self.encode(mg, device)
        K = base_goals.shape[0]
        n = action_feats.shape[0]
        q_all = torch.empty(K, n, device=device)
        for k in range(K):
            s_enc = self.state_enc(torch.cat([g_emb, base_goals[k]], dim=-1))
            inp = self._action_inputs(s_enc, node_emb, action_feats, action_site_ix, device)
            psi_k = self.scorer(inp)
            q_all[k] = self.q_from_psi(psi_k, reward_goal)
        best_q, best_idx = q_all.max(dim=0)
        return best_q, best_idx
