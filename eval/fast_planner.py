# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import FRAGMENTS_PATH, PARENTS_PATH, ARTIFACTS_DIR

import argparse
import json
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import (
    load_fragment_library, load_target_distribution, parse_property_names,
    MolEnv, TERMINATE,
)
from env.fast_properties import AdditivePropertyModel, _bondpair_index
from env.properties import normalize_props



def _surrogate_scores(model, base_acc, actions, goal):
    props = model.action_props_norm(base_acc, actions)
    return np.linalg.norm(props - goal[None, :], axis=1)


def _rdkit_scores(env, actions, goal):
    out = np.full(len(actions), np.inf)
    cur = env._current_surrogate_props()
    for i, a in enumerate(actions):
        if a.is_terminate:
            props = cur
        else:
            _, props, _, ok, _ = env._simulate_action(a)
            if not ok or props is None:
                continue
        out[i] = float(np.linalg.norm(props - goal))
    return out



def run_planner_episode(env, model, mode: str, goal: Optional[np.ndarray] = None,
                        topk: int = 8):
    _s, g, valid = env.reset()
    if goal is not None:
        env._goal = np.asarray(goal, dtype=np.float32)
        g = env._goal
    base_acc = model.frag_accum(env._seed_frag_idx).copy() if mode != "random" else None

    n_steps = 0
    while True:
        n_steps += 1
        if mode == "random":
            idx = int(np.random.randint(len(valid)))
        elif mode == "rdkit":
            idx = int(np.argmin(_rdkit_scores(env, valid, g)))
        elif mode == "hybrid":
            ss = _surrogate_scores(model, base_acc, valid, g)
            cand = np.argsort(ss)[:min(topk, len(valid))]
            rs = _rdkit_scores(env, [valid[c] for c in cand], g)
            idx = int(cand[int(np.argmin(rs))])
        else:
            idx = int(np.argmin(_surrogate_scores(model, base_acc, valid, g)))

        chosen = valid[idx]
        res = env.step(chosen)
        if mode in ("surrogate", "hybrid") and not chosen.is_terminate and not res.info.get("error"):
            base_acc = base_acc + model.frag_accum(chosen.frag_idx) \
                + model.bond_correction(chosen.mol_label, chosen.frag_label)
        if res.done:
            return res.info.get("terminal_distance"), res.info.get("valid", False), n_steps
        valid = res.info.get("valid_actions", [TERMINATE])


def benchmark(env, model, mode: str, n_episodes: int, topk: int = 8) -> Dict[str, float]:
    dists, valids = [], []
    t0 = time.time()
    for _ in range(n_episodes):
        d, v, _ = run_planner_episode(env, model, mode, topk=topk)
        valids.append(1.0 if v else 0.0)
        if d is not None:
            dists.append(float(d))
    dt = time.time() - t0
    return {
        "mean_distance": float(np.mean(dists)) if dists else float("nan"),
        "success_rate": float(np.mean([d < 0.1 for d in dists])) if dists else 0.0,
        "validity_rate": float(np.mean(valids)),
        "episodes_per_min": n_episodes / dt * 60.0,
        "n_episodes": n_episodes,
    }



def make_figure(results: Dict[str, Dict[str, float]], out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [m for m in ("random", "surrogate", "hybrid", "rdkit") if m in results]
    labels = {"random": "random", "surrogate": "CAFS\nsurrogate",
              "hybrid": "hybrid\n+ RDKit", "rdkit": "RDKit\ngreedy"}
    colors = {"random": "#9a9a9a", "surrogate": "#0072B2",
              "hybrid": "#009E73", "rdkit": "#D55E00"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    x = np.arange(len(order))

    d = [results[m]["mean_distance"] for m in order]
    ax1.bar(x, d, color=[colors[m] for m in order])
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[m] for m in order], fontsize=10)
    ax1.set_ylabel("mean distance to goal ↓ (RDKit-scored)")
    ax1.set_title("Accuracy")
    for xi, di in zip(x, d):
        ax1.text(xi, di, f"{di:.3f}", ha="center", va="bottom", fontsize=9)

    sp = [results[m]["episodes_per_min"] for m in order]
    ax2.bar(x, sp, color=[colors[m] for m in order])
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[m] for m in order], fontsize=10)
    ax2.set_ylabel("throughput ↑ (episodes / min)")
    ax2.set_title("Speed")
    ax2.set_yscale("log")
    for xi, si in zip(x, sp):
        ax2.text(xi, si, f"{si:.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Analytic greedy planner "
                 "(additive property oracle)", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95), w_pad=1.4)
    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"\nFigure saved: {out_path}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_frags", type=int, default=200)
    p.add_argument("--min_frag_count", type=int, default=5_000)
    p.add_argument("--n_targets", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--goal_properties", type=str, default="sLogP,QED,TPSA")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--topk", type=int, default=8, help="hybrid: # surrogate-shortlisted actions to RDKit-score")
    p.add_argument("--skip_rdkit", action="store_true", help="RDKit-free: only random + analytic-greedy (drops hybrid + exact)")
    p.add_argument("--skip_exact", action="store_true", help="skip only the full exact RDKit-greedy (keep hybrid); use for large libraries")
    p.add_argument("--fragments_parquet", type=str, default=str(FRAGMENTS_PATH))
    p.add_argument("--parents_parquet", type=str, default=str(PARENTS_PATH))
    default_out = ARTIFACTS_DIR / "figures" / "fast_planner.png"
    p.add_argument("--out", type=str, default=str(default_out))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=str, default="", help="optional path to dump the results dict")
    args = p.parse_args()

    np.random.seed(args.seed)
    names = parse_property_names(args.goal_properties)
    frags = load_fragment_library(args.fragments_parquet, n=args.n_frags, min_count=args.min_frag_count)
    targets = load_target_distribution(args.parents_parquet, n=args.n_targets, property_names=names)
    env = MolEnv(frags, targets, max_steps=args.max_steps, property_names=names)

    print("Calibrating additive property oracle ...")
    t = time.time()
    model = AdditivePropertyModel(names)
    model.fit_or_load(env)
    print(f"  ready in {time.time()-t:.1f}s")

    modes = ["random", "surrogate"]
    if not args.skip_rdkit:
        modes.append("hybrid")
        if not args.skip_exact:
            modes.append("rdkit")
    results = {}
    for m in modes:
        print(f"\nBenchmarking '{m}' ...")
        results[m] = benchmark(env, model, m, args.n_episodes, topk=args.topk)
        r = results[m]
        print(f"  dist={r['mean_distance']:.4f}  success={r['success_rate']*100:.1f}%  "
              f"valid={r['validity_rate']*100:.1f}%  {r['episodes_per_min']:.0f} eps/min")

    print(f"\n{'='*64}")
    print(f"{'policy':>18} | {'distance↓':>10} | {'success':>8} | {'eps/min↑':>10}")
    print("-" * 64)
    for m in modes:
        r = results[m]
        print(f"{m:>18} | {r['mean_distance']:>10.4f} | {r['success_rate']*100:>7.1f}% | "
              f"{r['episodes_per_min']:>10.0f}")
    print("=" * 64)

    if args.json:
        _Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        payload = {"seed": args.seed, "n_frags": args.n_frags,
                   "n_episodes": args.n_episodes, "topk": args.topk,
                   "goal_properties": list(names), "results": results}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results JSON saved: {args.json}")

    make_figure(results, args.out)


if __name__ == "__main__":
    main()
