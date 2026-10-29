# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.structs import MolGraph
from env.features import ACTION_FEAT_DIM
from env.data import GOAL_DIM


ELEMENTS = [0, 6, 7, 8, 9, 15, 16, 17, 35, 53]
_ELEM2IDX = {e: i for i, e in enumerate(ELEMENTS)}
N_ELEM    = len(ELEMENTS)
EDGE_DIM  = 4
NODE_DIM  = N_ELEM + 1 + 1 + 1 + 17


def mol_graph_to_tensors(
    mg: MolGraph,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arr = mg.arrays
    N   = int(arr.atomic_num.shape[0])

    node_rows = np.zeros((N, NODE_DIM), dtype=np.float32)
    for i in range(N):
        z = int(arr.atomic_num[i])

        node_rows[i, _ELEM2IDX.get(z, 0)] = 1.0

        node_rows[i, N_ELEM]     = float(arr.formal_charge[i]) / 2.0
        node_rows[i, N_ELEM + 1] = float(arr.is_aromatic[i]) if arr.is_aromatic is not None else 0.0
        node_rows[i, N_ELEM + 2] = 1.0 if z == 0 else 0.0

        if z == 0 and arr.attachment_label is not None:
            lbl = arr.attachment_label[i]
            if lbl is not None:
                try:
                    li = int(lbl)
                    if 0 <= li < 17:
                        node_rows[i, N_ELEM + 3 + li] = 1.0
                except (ValueError, TypeError):
                    pass

    node_feat = torch.tensor(node_rows, dtype=torch.float32, device=device)

    M = int(arr.bonds.shape[0])
    if M > 0:
        b   = arr.bonds.astype(np.int64)
        src = np.concatenate([b[:, 0], b[:, 1]])
        dst = np.concatenate([b[:, 1], b[:, 0]])
        btype = np.concatenate([arr.bond_type, arr.bond_type]).astype(np.int64)
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
        bt_oh = np.zeros((2 * M, EDGE_DIM), dtype=np.float32)
        for j, bt in enumerate(btype):
            if 0 <= bt < EDGE_DIM:
                bt_oh[j, bt] = 1.0
        edge_attr = torch.tensor(bt_oh, dtype=torch.float32, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr  = torch.zeros((0, EDGE_DIM), dtype=torch.float32, device=device)

    return node_feat, edge_index, edge_attr



class MPNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.msg_fn    = nn.Linear(node_dim + edge_dim, out_dim)
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        h:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
    ) -> torch.Tensor:
        N = h.shape[0]
        if edge_index.shape[1] == 0:
            agg = torch.zeros(N, self.msg_fn.out_features, device=h.device)
        else:
            src, dst = edge_index[0], edge_index[1]
            msgs = F.relu(self.msg_fn(torch.cat([h[src], edge_attr], dim=-1)))
            agg  = torch.zeros(N, msgs.shape[-1], device=h.device)
            deg  = torch.zeros(N, 1, device=h.device)
            agg.index_add_(0, dst, msgs)
            deg.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device))
            agg = agg / (deg + 1e-6)

        return self.update_fn(torch.cat([h, agg], dim=-1))



class MolGNN(nn.Module):
    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM,
                 hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def encode_nodes(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.node_embed(node_feat))
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.out_proj(h.mean(dim=0)))

    def forward(
        self,
        node_feat:  torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode_nodes(node_feat, edge_index, edge_attr)
        return self.pool(h)



def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    layers: list = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GNNActor(nn.Module):
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
        self.gnn       = MolGNN(hidden_dim=gnn_dim)
        self.site_actions = site_actions
        self.frag_site_emb = bool(frag_site_emb and site_actions)
        self.gnn_dim = gnn_dim
        self.num_frag_sites = num_frag_sites
        self.state_enc = _mlp(gnn_dim + goal_dim, hidden_dim, hidden_dim)
        frag_dim = frag_site_dim if self.frag_site_emb else 0
        scorer_in_dim = hidden_dim + action_feat_dim + (gnn_dim if site_actions else 0) + frag_dim
        self.scorer    = nn.Sequential(
            nn.Linear(scorer_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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

    def encode_graph(
        self,
        mg:        MolGraph,
        goal_norm: torch.Tensor,
        device:    torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nf, ei, ea = mol_graph_to_tensors(mg, device)
        node_emb = self.gnn.encode_nodes(nf, ei, ea)
        g_emb = self.gnn.pool(node_emb)
        return self.state_enc(torch.cat([g_emb, goal_norm], dim=-1)), node_emb

    def forward(
        self,
        mg:           MolGraph,
        goal_norm:    torch.Tensor,
        action_feats: torch.Tensor,
        device:       torch.device,
        action_site_ix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s_enc, node_emb = self.encode_graph(mg, goal_norm, device)
        n     = action_feats.shape[0]
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
        inp = torch.cat(parts, dim=-1)
        return self.scorer(inp).squeeze(-1)

    def action_dist(
        self,
        mg:           MolGraph,
        goal_norm:    torch.Tensor,
        action_feats: torch.Tensor,
        device:       torch.device,
        action_site_ix: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(
            logits=self.forward(mg, goal_norm, action_feats, device, action_site_ix)
        )


class GNNCritic(nn.Module):
    def __init__(
        self,
        gnn_dim:    int = 128,
        goal_dim:   int = GOAL_DIM,
        hidden_dim: int = 256,
        site_actions: bool = False,
    ):
        super().__init__()
        self.site_actions = site_actions
        self.gnn = MolGNN(hidden_dim=gnn_dim)
        self.net = _mlp(gnn_dim + goal_dim, hidden_dim, 1)

    def forward(
        self,
        mg:        MolGraph,
        goal_norm: torch.Tensor,
        device:    torch.device,
    ) -> torch.Tensor:
        nf, ei, ea = mol_graph_to_tensors(mg, device)
        g_emb = self.gnn(nf, ei, ea)
        return self.net(torch.cat([g_emb, goal_norm], dim=-1)).squeeze(-1)
