# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import FRAGMENTS_PATH, PARENTS_PATH, ARTIFACTS_DIR

import argparse
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from env import (
    load_fragment_library, load_target_distribution, parse_property_names,
    MolEnv, TERMINATE,
)
from env.properties import denormalize_props
from models import GNNSuccessor



def run_episode_to_goal(
    env: MolEnv,
    model: GNNSuccessor,
    g_new: np.ndarray,
    device: torch.device,
    mode: str,
    base_goals_t: Optional[torch.Tensor] = None,
    greedy: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    _state, _goal, valid = env.reset()
    env._goal = np.asarray(g_new, dtype=np.float32)
    g_t = torch.tensor(env._goal, dtype=torch.float32, device=device)

    for _ in range(env.max_steps):
        mg = deepcopy(env._mol)
        af = env.get_action_features(valid)
        six = env.get_action_site_indices(valid)
        af_t = torch.tensor(af, dtype=torch.float32, device=device)
        six_t = torch.tensor(six, dtype=torch.long, device=device)

        with torch.no_grad():
            if mode == "random":
                q = torch.zeros(len(valid), device=device)
            elif mode == "gpi":
                q, _ = model.gpi_scores(mg, g_t, base_goals_t, af_t, device, six_t)
            else:
                psi = model.psi(mg, g_t, af_t, device, six_t)
                q = model.q_from_psi(psi, g_t)
            if mode == "random":
                a = int(np.random.randint(len(valid)))
            elif greedy:
                a = int(torch.argmax(q).item())
            else:
                a = int(torch.distributions.Categorical(logits=q / 0.2).sample().item())

        res = env.step(valid[a])
        valid = res.info.get("valid_actions", [TERMINATE])
        if res.done:
            return res.info.get("achieved_goal"), res.info.get("terminal_distance")
    return None, None


def mean_distance_to_goal(
    env, model, g_new, device, mode, base_goals_t, n_episodes,
) -> Tuple[float, float]:
    dists, n_valid = [], 0
    for _ in range(n_episodes):
        achieved, dist = run_episode_to_goal(env, model, g_new, device, mode, base_goals_t)
        if achieved is not None and dist is not None:
            dists.append(float(dist))
            n_valid += 1
    mean_d = float(np.mean(dists)) if dists else float("nan")
    return mean_d, n_valid / max(1, n_episodes)



def target_grid(
    env, model, targets, device, base_goals_t,
    modes: List[str],
    axes: Tuple[int, int] = (0, 1),
    grid: int = 7,
    episodes: int = 8,
) -> Dict[str, np.ndarray]:
    i, j = axes
    median = np.median(targets, axis=0).astype(np.float32)
    xs = np.linspace(0.0, 1.0, grid)
    ys = np.linspace(0.0, 1.0, grid)

    out = {m: np.full((grid, grid), np.nan, dtype=np.float32) for m in modes}
    for gy, yv in enumerate(ys):
        for gx, xv in enumerate(xs):
            g = median.copy()
            g[i] = xv
            g[j] = yv
            for m in modes:
                d, _ = mean_distance_to_goal(env, model, g, device, m, base_goals_t, episodes)
                out[m][gy, gx] = d
        print(f"  grid row {gy+1}/{grid} done")

    out["_extent"] = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    out["_train_box"] = np.array([
        targets[:, i].min(), targets[:, i].max(),
        targets[:, j].min(), targets[:, j].max(),
    ], dtype=np.float32)
    out["_median"] = median
    out["_axes"] = np.array([i, j])
    return out



def make_figure(grids: Dict[str, np.ndarray], property_names, axes, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    i, j = axes
    modes = [m for m in ("usfa", "gpi") if m in grids]
    extent = grids["_extent"]
    box = grids["_train_box"]

    vals = np.concatenate([grids[m][~np.isnan(grids[m])].ravel() for m in modes]) \
        if modes else np.array([0.0, 1.0])
    vmin, vmax = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)

    fig, axarr = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.6), squeeze=False)
    titles = {"usfa": "USFA (condition on new target)",
              "gpi": "USFA + GPI (zero-shot transfer)"}
    for ax, m in zip(axarr[0], modes):
        im = ax.imshow(grids[m], origin="lower", extent=extent, aspect="auto",
                       cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                               fill=False, edgecolor="white", lw=2, ls="--"))
        ax.set_title(titles.get(m, m))
        ax.set_xlabel(f"target {property_names[i]} (normalised)")
        ax.set_ylabel(f"target {property_names[j]} (normalised)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="achieved distance ↓")
    fig.suptitle("Zero-shot multi-property targeting  (dashed box = training-goal region)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"\nFigure saved: {out_path}")



def load_from_checkpoint(ckpt_path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    property_names = parse_property_names(cfg.get("goal_properties", args.goal_properties))

    frags = load_fragment_library(
        cfg.get("fragments_parquet", args.fragments_parquet),
        n=cfg.get("n_frags", args.n_frags),
        min_count=cfg.get("min_frag_count", args.min_frag_count),
    )
    targets = load_target_distribution(
        cfg.get("parents_parquet", args.parents_parquet),
        n=cfg.get("n_targets", args.n_targets),
        property_names=property_names,
    )
    env = MolEnv(
        frags, targets,
        max_steps=cfg.get("max_steps", 6),
        property_names=property_names,
        reward_config={"mode": "sparse"},
        site_actions=cfg.get("site_actions", False),
        counterfactual_actions=cfg.get("counterfactual_actions", False),
    )
    model = GNNSuccessor(
        gnn_dim=cfg.get("gnn_dim", 128),
        goal_dim=env.goal_dim,
        action_feat_dim=env.action_feat_dim,
        hidden_dim=cfg.get("hidden_dim", 256),
        site_actions=cfg.get("site_actions", False),
        frag_site_emb=cfg.get("frag_site_emb", False),
        num_frag_sites=getattr(env, "num_frag_sites", 0),
    ).to(device)
    model.load_state_dict(ckpt["successor"])
    model.eval()
    if cfg.get("frag_site_emb"):
        model.refresh_frag_cache(env, device)

    base_goals = ckpt.get("base_goals")
    if base_goals is None:
        base_goals = np.stack([targets[np.random.randint(len(targets))]
                               for _ in range(16)]).astype(np.float32)
    base_goals_t = torch.tensor(np.asarray(base_goals, dtype=np.float32), device=device)
    return env, model, targets, property_names, base_goals_t



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--axes", type=str, default="0,1", help="two property indices to sweep")
    default_out = ARTIFACTS_DIR / "figures" / "zero_shot.png"
    p.add_argument("--out", type=str, default=str(default_out))
    p.add_argument("--goal_properties", type=str, default="sLogP,QED,TPSA")
    p.add_argument("--fragments_parquet", type=str, default=str(FRAGMENTS_PATH))
    p.add_argument("--parents_parquet", type=str, default=str(PARENTS_PATH))
    p.add_argument("--n_frags", type=int, default=200)
    p.add_argument("--min_frag_count", type=int, default=5_000)
    p.add_argument("--n_targets", type=int, default=300)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    axes = tuple(int(x) for x in args.axes.split(","))[:2]

    env, model, targets, property_names, base_goals_t = load_from_checkpoint(
        args.checkpoint, args, device)
    print(f"Loaded USFA model | properties={list(property_names)} | "
          f"base-goal library K={base_goals_t.shape[0]}")

    modes = ["random", "usfa", "gpi"]
    grids = target_grid(env, model, targets, device, base_goals_t,
                        modes=["usfa", "gpi"], axes=axes,
                        grid=args.grid, episodes=args.episodes)

    i, j = axes
    box = grids["_train_box"]
    xs = np.linspace(0.0, 1.0, args.grid)
    ys = np.linspace(0.0, 1.0, args.grid)
    XX, YY = np.meshgrid(xs, ys)
    in_box = (XX >= box[0]) & (XX <= box[1]) & (YY >= box[2]) & (YY <= box[3])

    print(f"\n{'='*60}\nZero-shot distance (lower is better)\n{'='*60}")
    print(f"{'policy':>8} | {'in-distribution':>16} | {'out-of-distribution':>20}")
    print("-" * 60)
    for m in ("usfa", "gpi"):
        g = grids[m]
        id_mean = float(np.nanmean(g[in_box])) if in_box.any() else float("nan")
        ood_mean = float(np.nanmean(g[~in_box])) if (~in_box).any() else float("nan")
        print(f"{m:>8} | {id_mean:>16.4f} | {ood_mean:>20.4f}")
    print("=" * 60)

    make_figure(grids, property_names, axes, args.out)


if __name__ == "__main__":
    main()
