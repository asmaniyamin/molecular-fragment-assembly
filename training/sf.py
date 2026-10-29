# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import FRAGMENTS_PATH, PARENTS_PATH, ARTIFACTS_DIR

import argparse
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from env import load_fragment_library, load_target_distribution, sample_target
from env import MolEnv, TERMINATE, StepResult, DEFAULT_PROPERTY_NAMES, parse_property_names
from env import Episode, Transition
from models import GNNSuccessor


DEFAULT_CFG: Dict[str, Any] = dict(
    fragments_parquet=str(FRAGMENTS_PATH),
    parents_parquet=str(PARENTS_PATH),
    n_frags=200,
    min_frag_count=5_000,
    n_targets=300,
    max_steps=6,
    goal_properties=",".join(DEFAULT_PROPERTY_NAMES),
    site_actions=False,
    frag_site_emb=False,
    counterfactual_actions=False,
    fast_props=False,
    refresh_every=50,
    n_episodes=10_000,
    batch_size=128,
    hidden_dim=256,
    gnn_dim=128,
    lr=3e-4,
    her_k=4,
    grad_clip=1.0,
    sf_temp=1.0,
    sf_temp_final=0.2,
    n_base_goals=16,
    log_every=50,
    checkpoint_every=500,
    checkpoint_dir=str(ARTIFACTS_DIR / "checkpoints_sf"),
    resume="",
    exp_name="sf_run",
    seed=1,
    use_wandb=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
)



class SFBuffer:

    def __init__(self, max_episodes: int = 2_000, her_k: int = 4):
        self.max_episodes = max_episodes
        self.her_k = her_k
        self._episodes: List[Tuple[Episode, np.ndarray]] = []

    def push(self, ep: Episode) -> None:
        achieved = ep.achieved_goal
        if achieved is None:
            return
        self._episodes.append((ep, np.asarray(achieved, dtype=np.float32)))
        if len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)

    def __len__(self) -> int:
        return len(self._episodes)

    def sample(self, n: int) -> List[Tuple[Transition, np.ndarray, np.ndarray]]:
        pool: List[Tuple[Transition, np.ndarray, np.ndarray]] = []
        for ep, achieved in self._episodes:
            for t in ep.transitions:
                pool.append((t, achieved, t.goal))
                for _ in range(self.her_k):
                    pool.append((t, achieved, achieved))
        if not pool:
            return []
        k = min(n, len(pool))
        idx = np.random.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]



def rollout_sf(
    env: MolEnv,
    model: GNNSuccessor,
    device: torch.device,
    temperature: float,
) -> Episode:
    state, goal, valid_actions = env.reset()
    ep = Episode()

    for _ in range(env.max_steps):
        mol_graph = deepcopy(env._mol)
        af_np = env.get_action_features(valid_actions)
        site_ix_np = env.get_action_site_indices(valid_actions)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=device)
        af_t = torch.tensor(af_np, dtype=torch.float32, device=device)
        site_ix_t = torch.tensor(site_ix_np, dtype=torch.long, device=device)

        with torch.no_grad():
            dist = model.action_dist(mol_graph, goal_t, af_t, device, site_ix_t,
                                     temperature=temperature)
            action_idx = int(dist.sample().item())

        result: StepResult = env.step(valid_actions[action_idx])
        next_valid = result.info.get("valid_actions", [TERMINATE])

        ep.add(Transition(
            state=state,
            action_feats=af_np,
            action_idx=action_idx,
            reward=result.reward,
            next_state=result.state,
            next_action_feats=af_np,
            done=result.done,
            goal=goal,
            info={
                "mol_graph": mol_graph,
                "action_site_indices": site_ix_np.copy(),
                "terminal_distance": result.info.get("terminal_distance"),
            },
        ))

        if result.done:
            ep.achieved_goal = result.info.get("achieved_goal")
            break

        state = result.state
        valid_actions = next_valid

    return ep



def update_sf(
    model: GNNSuccessor,
    opt: optim.Optimizer,
    batch: List[Tuple[Transition, np.ndarray, np.ndarray]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    losses: List[torch.Tensor] = []
    for t, target_phi, cond_goal in batch:
        mg = t.info.get("mol_graph")
        site_ix_np = t.info.get("action_site_indices")
        cond_t = torch.tensor(cond_goal, dtype=torch.float32, device=device)
        af_t = torch.tensor(t.action_feats, dtype=torch.float32, device=device)
        site_ix_t = (
            torch.tensor(site_ix_np, dtype=torch.long, device=device)
            if site_ix_np is not None else None
        )
        target_t = torch.tensor(target_phi, dtype=torch.float32, device=device)

        psi = model.psi(mg, cond_t, af_t, device, site_ix_t)
        psi_a = psi[t.action_idx]
        losses.append((psi_a - target_t).pow(2).sum())

    loss = torch.stack(losses).mean()

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    opt.step()

    return {"sf_loss": float(loss.item())}



def _temperature(ep_idx: int, cfg: Dict[str, Any]) -> float:
    frac = min(1.0, ep_idx / max(1, cfg["n_episodes"]))
    return cfg["sf_temp"] + frac * (cfg["sf_temp_final"] - cfg["sf_temp"])


def _latest_checkpoint(ckpt_dir: str) -> str:
    import glob, re
    paths = glob.glob(os.path.join(ckpt_dir, "ckpt_ep*.pt"))
    if not paths:
        return ""
    def _epnum(p: str) -> int:
        m = re.search(r"ckpt_ep(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(paths, key=_epnum)


def _save_checkpoint(path, ep_idx, model, opt, cfg, base_goals):
    torch.save({
        "episode": ep_idx,
        "successor": model.state_dict(),
        "opt": opt.state_dict(),
        "config": cfg,
        "base_goals": base_goals,
        "algo": "sf",
    }, path)



def train(cfg: Optional[Dict[str, Any]] = None) -> GNNSuccessor:
    if cfg is None:
        cfg = dict(DEFAULT_CFG)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if cfg.get("frag_site_emb") and not cfg.get("site_actions"):
        raise ValueError("--frag_site_emb requires --site_actions")

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.init(
            project="cs224r-fragment-assembly",
            name=cfg["exp_name"],
            config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            tags=["sf", "usfa", "gnn"],
        )

    device = torch.device(cfg["device"])

    print("Loading fragment library ...")
    frags = load_fragment_library(
        cfg["fragments_parquet"], n=cfg["n_frags"], min_count=cfg["min_frag_count"],
    )
    print(f"  {len(frags)} fragments loaded")

    print("Loading target distribution ...")
    property_names = parse_property_names(cfg["goal_properties"])
    targets = load_target_distribution(
        cfg["parents_parquet"], n=cfg["n_targets"], property_names=property_names,
    )
    print(
        f"  {len(targets)} target molecules "
        + " ".join(
            f"| {name}∈[{targets[:, i].min():.2f},{targets[:, i].max():.2f}]"
            for i, name in enumerate(property_names)
        )
    )

    env = MolEnv(
        frags, targets,
        max_steps=cfg["max_steps"],
        property_names=property_names,
        reward_config={"mode": "sparse"},
        site_actions=cfg.get("site_actions", False),
        counterfactual_actions=cfg.get("counterfactual_actions", False),
        fast_props=cfg.get("fast_props", False),
    )

    model = GNNSuccessor(
        gnn_dim=cfg["gnn_dim"],
        goal_dim=env.goal_dim,
        action_feat_dim=env.action_feat_dim,
        hidden_dim=cfg["hidden_dim"],
        site_actions=cfg.get("site_actions", False),
        frag_site_emb=cfg.get("frag_site_emb", False),
        num_frag_sites=getattr(env, "num_frag_sites", 0),
    ).to(device)
    if cfg.get("frag_site_emb"):
        model.refresh_frag_cache(env, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GNNSuccessor params: {n_params:,}  (head outputs {env.goal_dim}-dim phi)")

    opt = optim.Adam(model.parameters(), lr=cfg["lr"])
    buffer = SFBuffer(max_episodes=2_000, her_k=cfg["her_k"])
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    start_ep = 0
    resume_path = cfg.get("resume", "") or ""
    if resume_path in ("auto", "latest"):
        resume_path = _latest_checkpoint(cfg["checkpoint_dir"])
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["successor"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_ep = int(ckpt.get("episode", 0))
        print(f"  Resumed from {resume_path} at episode {start_ep}")
    if cfg.get("frag_site_emb"):
        model.refresh_frag_cache(env, device)

    ep_dists: List[float] = []
    ep_valid: List[float] = []

    print(f"\nTraining (USFA) from episode {start_ep} to {cfg['n_episodes']} ...\n")

    for ep_idx in range(start_ep, cfg["n_episodes"]):
        model.train()
        temp = _temperature(ep_idx, cfg)
        ep = rollout_sf(env, model, device, temp)
        buffer.push(ep)

        achieved = ep.achieved_goal
        ep_valid.append(1.0 if achieved is not None else 0.0)
        td = ep.transitions[-1].info.get("terminal_distance") if ep.transitions else None
        if td is not None:
            ep_dists.append(float(td))

        if len(buffer) >= 5:
            batch = buffer.sample(cfg["batch_size"])
            metrics = update_sf(model, opt, batch, cfg, device)
        else:
            metrics = {}

        if cfg.get("frag_site_emb") and (ep_idx + 1) % cfg.get("refresh_every", 50) == 0:
            model.refresh_frag_cache(env, device)

        if (ep_idx + 1) % cfg["log_every"] == 0:
            w = cfg["log_every"]
            v_pct = float(np.mean(ep_valid[-w:])) * 100.0
            d_mean = float(np.mean(ep_dists[-w:])) if ep_dists else float("nan")
            loss_s = f"  sf_loss={metrics['sf_loss']:.4f}" if metrics else ""
            print(
                f"[{ep_idx+1:5d}] dist={d_mean:.3f}  valid={v_pct:.1f}%  "
                f"temp={temp:.2f}{loss_s}"
            )
            if cfg.get("use_wandb") and _wandb is not None:
                log = {"train/mean_dist": d_mean, "train/valid_pct": v_pct,
                       "train/temperature": temp}
                if metrics:
                    log["train/sf_loss"] = metrics["sf_loss"]
                _wandb.log(log, step=ep_idx + 1)

        if (ep_idx + 1) % cfg["checkpoint_every"] == 0:
            base_goals = np.stack(
                [sample_target(targets) for _ in range(cfg["n_base_goals"])]
            ).astype(np.float32)
            path = os.path.join(cfg["checkpoint_dir"], f"ckpt_ep{ep_idx+1}.pt")
            _save_checkpoint(path, ep_idx + 1, model, opt, cfg, base_goals)
            print(f"  -> checkpoint saved: {path}")

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.finish()

    return model


def _parse_args(default_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_cfg = dict(DEFAULT_CFG if default_cfg is None else default_cfg)
    p = argparse.ArgumentParser(description="Train USFA goal-conditioned fragment assembly agent.")
    p.add_argument("--n_episodes", type=int, default=base_cfg["n_episodes"])
    p.add_argument("--n_frags", type=int, default=base_cfg["n_frags"])
    p.add_argument("--min_frag_count", type=int, default=base_cfg["min_frag_count"])
    p.add_argument("--n_targets", type=int, default=base_cfg["n_targets"])
    p.add_argument("--max_steps", type=int, default=base_cfg["max_steps"])
    p.add_argument("--goal_properties", type=str, default=base_cfg["goal_properties"])
    p.add_argument("--site_actions", action="store_true", default=base_cfg["site_actions"])
    p.add_argument("--frag_site_emb", action="store_true", default=base_cfg["frag_site_emb"])
    p.add_argument("--counterfactual_actions", action="store_true", default=base_cfg["counterfactual_actions"])
    p.add_argument("--fast_props", action="store_true", default=base_cfg["fast_props"])
    p.add_argument("--refresh_every", type=int, default=base_cfg["refresh_every"])
    p.add_argument("--n_base_goals", type=int, default=base_cfg["n_base_goals"])
    p.add_argument("--hidden_dim", type=int, default=base_cfg["hidden_dim"])
    p.add_argument("--gnn_dim", type=int, default=base_cfg["gnn_dim"])
    p.add_argument("--lr", type=float, default=base_cfg["lr"])
    p.add_argument("--batch_size", type=int, default=base_cfg["batch_size"])
    p.add_argument("--her_k", type=int, default=base_cfg["her_k"])
    p.add_argument("--grad_clip", type=float, default=base_cfg["grad_clip"])
    p.add_argument("--sf_temp", type=float, default=base_cfg["sf_temp"])
    p.add_argument("--sf_temp_final", type=float, default=base_cfg["sf_temp_final"])
    p.add_argument("--device", type=str, default=base_cfg["device"])
    p.add_argument("--checkpoint_dir", type=str, default=base_cfg["checkpoint_dir"])
    p.add_argument("--resume", type=str, default=base_cfg["resume"],
                   help="path to a ckpt_ep*.pt, or 'auto' to pick the latest in checkpoint_dir")
    p.add_argument("--log_every", type=int, default=base_cfg["log_every"])
    p.add_argument("--checkpoint_every", type=int, default=base_cfg["checkpoint_every"])
    p.add_argument("--exp_name", type=str, default=base_cfg["exp_name"])
    p.add_argument("--seed", type=int, default=base_cfg["seed"])
    p.add_argument("--use_wandb", action="store_true", default=base_cfg["use_wandb"])
    p.add_argument("--fragments_parquet", default=base_cfg["fragments_parquet"])
    p.add_argument("--parents_parquet", default=base_cfg["parents_parquet"])
    args = p.parse_args()
    cfg = dict(base_cfg)
    cfg.update(vars(args))
    return cfg


if __name__ == "__main__":
    train(_parse_args())
