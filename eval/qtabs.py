# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import FRAGMENTS_PATH, PARENTS_PATH, ARTIFACTS_DIR

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from env import (
    Action,
    MolEnv,
    TERMINATE,
    load_fragment_library,
    load_target_distribution,
    parse_property_names,
)
from env.env import BRICS_COMPAT
from env.fast_properties import AdditivePropertyModel
from eval.fast_planner import benchmark as benchmark_fast


@dataclass(frozen=True)
class PlannerCandidate:
    score: float
    seq: Tuple[int, ...]


@dataclass
class BeamNode:
    acc: torch.Tensor
    open_counts: torch.Tensor
    seq: Tuple[int, ...]
    score: float


class ActionBank:

    def __init__(self, env: MolEnv, model: AdditivePropertyModel, device: torch.device):
        self.device = device
        self.model = model
        self.goal_dim = env.goal_dim
        self.acc_dim = int(model.frag_desc.shape[1])
        self.templates: List[Action] = []

        frag_counts = np.zeros((len(env.library), 17), dtype=np.int16)
        for fi, frag in enumerate(env.library):
            for lbl in frag.labels:
                try:
                    li = int(lbl)
                except (TypeError, ValueError):
                    continue
                if 0 <= li < 17:
                    frag_counts[fi, li] += 1
        self.frag_label_counts = frag_counts

        frag_idx: List[int] = []
        mol_label_idx: List[int] = []
        frag_label_idx: List[int] = []
        delta_open: List[np.ndarray] = []
        delta_acc: List[np.ndarray] = []

        compatible_by_frag = {
            frag_label: tuple(sorted(ml for ml, fset in BRICS_COMPAT.items() if frag_label in fset))
            for frag_label in range(17)
        }

        for fi, frag in enumerate(env.library):
            if model.frag_ok is not None and not bool(model.frag_ok[fi]):
                continue
            counts = frag_counts[fi]
            for raw_flab in frag.labels:
                try:
                    flab = int(raw_flab)
                except (TypeError, ValueError):
                    continue
                if not (0 <= flab < 17):
                    continue
                frag_delta = counts.astype(np.int16, copy=True)
                frag_delta[flab] -= 1
                for mlab in compatible_by_frag.get(flab, ()):
                    full_delta = frag_delta.copy()
                    full_delta[mlab] -= 1
                    self.templates.append(
                        Action(
                            is_terminate=False,
                            frag_idx=fi,
                            mol_label=str(mlab),
                            frag_label=str(flab),
                        )
                    )
                    frag_idx.append(fi)
                    mol_label_idx.append(mlab)
                    frag_label_idx.append(flab)
                    delta_open.append(full_delta.astype(np.int16, copy=False))
                    delta_acc.append(
                        model.frag_accum(fi) + model.bond_correction(str(mlab), str(flab))
                    )

        self.n_templates = len(self.templates)
        self.frag_idx = torch.as_tensor(frag_idx, dtype=torch.long, device=device)
        self.mol_label_idx = torch.as_tensor(mol_label_idx, dtype=torch.long, device=device)
        self.frag_label_idx = torch.as_tensor(frag_label_idx, dtype=torch.long, device=device)
        self.delta_open = torch.as_tensor(np.stack(delta_open, axis=0), dtype=torch.int16, device=device)
        self.delta_acc = torch.as_tensor(np.stack(delta_acc, axis=0), dtype=torch.float32, device=device)

    def initial_open_counts(self, env: MolEnv) -> torch.Tensor:
        counts = np.zeros(17, dtype=np.int16)
        for _, lbl in env._open_sites():
            try:
                li = int(lbl)
            except (TypeError, ValueError):
                continue
            if 0 <= li < 17:
                counts[li] += 1
        return torch.as_tensor(counts, dtype=torch.int16, device=self.device)

    def replay_action(self, action_idx: int) -> Action:
        return self.templates[int(action_idx)]


class QTabsPlanner:

    def __init__(
        self,
        bank: ActionBank,
        beam_width: int = 16,
        candidate_multiplier: int = 4,
        final_keep: int = 64,
        action_chunk_size: int = 4096,
        open_penalty: float = 0.05,
        quantize_scale: float = 1000.0,
    ):
        self.bank = bank
        self.beam_width = beam_width
        self.candidate_keep = max(beam_width, beam_width * candidate_multiplier)
        self.final_keep = max(final_keep, beam_width)
        self.action_chunk_size = max(1, action_chunk_size)
        self.open_penalty = open_penalty
        self.quantize_scale = quantize_scale

    def search(
        self,
        base_acc: np.ndarray,
        open_counts: torch.Tensor,
        goal: np.ndarray,
        max_steps: int,
    ) -> List[PlannerCandidate]:
        device = self.bank.device
        goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
        start = BeamNode(
            acc=torch.as_tensor(base_acc, dtype=torch.float32, device=device),
            open_counts=open_counts.to(device=device, dtype=torch.int16),
            seq=(),
            score=float("-inf"),
        )
        beam: List[BeamNode] = [start]
        completed: List[PlannerCandidate] = []

        for depth in range(max_steps + 1):
            completed.extend(self._terminal_candidates(beam, goal_t))
            completed = self._dedupe_candidates(completed)[:self.final_keep]
            if depth == max_steps:
                break
            beam = self._expand_beam(beam, goal_t, depth + 1)
            if not beam:
                break

        completed = sorted(completed, key=lambda c: c.score, reverse=True)
        return completed[:self.final_keep]

    def _terminal_candidates(
        self,
        beam: List[BeamNode],
        goal_t: torch.Tensor,
    ) -> List[PlannerCandidate]:
        if not beam:
            return []
        accs = torch.stack([node.acc for node in beam], dim=0)
        opens = torch.stack([node.open_counts for node in beam], dim=0).float()
        props = self.bank.model.properties_norm_torch(accs)
        dists = torch.linalg.norm(props - goal_t.unsqueeze(0), dim=-1)
        penalties = self.open_penalty * opens.sum(dim=-1)
        scores = -(dists + penalties)
        return [
            PlannerCandidate(score=float(scores[i].item()), seq=node.seq)
            for i, node in enumerate(beam)
        ]

    def _expand_beam(
        self,
        beam: List[BeamNode],
        goal_t: torch.Tensor,
        depth: int,
    ) -> List[BeamNode]:
        if not beam:
            return []

        device = self.bank.device
        beam_accs = torch.stack([node.acc for node in beam], dim=0).to(device)
        beam_opens = torch.stack([node.open_counts for node in beam], dim=0).to(device)
        B = beam_accs.shape[0]
        candidates: List[BeamNode] = []

        for start in range(0, self.bank.n_templates, self.action_chunk_size):
            end = min(start + self.action_chunk_size, self.bank.n_templates)
            mol_req = self.bank.mol_label_idx[start:end]
            compat = beam_opens[:, mol_req] > 0
            n_valid = int(compat.sum().item())
            if n_valid == 0:
                continue

            next_acc = beam_accs[:, None, :] + self.bank.delta_acc[start:end].unsqueeze(0)
            next_open = beam_opens[:, None, :] + self.bank.delta_open[start:end].unsqueeze(0)
            next_props = self.bank.model.properties_norm_torch(
                next_acc.reshape(-1, self.bank.acc_dim)
            ).reshape(B, end - start, self.bank.goal_dim)

            dists = torch.linalg.norm(next_props - goal_t.view(1, 1, -1), dim=-1)
            penalties = self.open_penalty * next_open.sum(dim=-1).float()
            scores = -(dists + penalties)
            scores = scores.masked_fill(~compat, float("-inf"))

            keep = min(self.candidate_keep, n_valid)
            vals, flat_idx = torch.topk(scores.reshape(-1), k=keep)
            chunk_w = end - start
            for rank in range(keep):
                score = float(vals[rank].item())
                if not math.isfinite(score):
                    break
                flat = int(flat_idx[rank].item())
                beam_idx = flat // chunk_w
                local_action_idx = flat % chunk_w
                global_action_idx = start + local_action_idx
                candidates.append(
                    BeamNode(
                        acc=next_acc[beam_idx, local_action_idx].detach().clone(),
                        open_counts=next_open[beam_idx, local_action_idx].detach().clone(),
                        seq=beam[beam_idx].seq + (global_action_idx,),
                        score=score,
                    )
                )

        if not candidates:
            return []

        candidates.sort(key=lambda node: node.score, reverse=True)
        merged: Dict[Tuple[int, ...], BeamNode] = {}
        for cand in candidates:
            key = self._state_key(cand, depth)
            if key in merged:
                continue
            merged[key] = cand
            if len(merged) >= self.beam_width:
                break
        return list(merged.values())

    def _state_key(self, node: BeamNode, depth: int) -> Tuple[int, ...]:
        acc_q = tuple(torch.round(node.acc * self.quantize_scale).to(torch.int32).tolist())
        open_q = tuple(node.open_counts.to(torch.int16).tolist())
        return (depth, *open_q, *acc_q)

    def _dedupe_candidates(self, candidates: List[PlannerCandidate]) -> List[PlannerCandidate]:
        seen: Dict[Tuple[int, ...], PlannerCandidate] = {}
        ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
        for cand in ordered:
            if cand.seq in seen:
                continue
            seen[cand.seq] = cand
            if len(seen) >= self.final_keep:
                break
        return list(seen.values())


def _make_exact_env(start_env: MolEnv) -> MolEnv:
    exact_env = deepcopy(start_env)
    exact_env.fast_props = False
    exact_env._prop_model = None
    exact_env._base_acc = None
    return exact_env


def replay_sequence(
    start_env: MolEnv,
    bank: ActionBank,
    seq: Sequence[int],
    exact: bool = True,
) -> Tuple[Optional[float], bool, int]:
    env = _make_exact_env(start_env) if exact else deepcopy(start_env)
    steps = 0
    for action_idx in seq:
        steps += 1
        res = env.step(bank.replay_action(action_idx))
        if res.info.get("error"):
            return None, False, steps
        if res.done:
            return res.info.get("terminal_distance"), bool(res.info.get("valid", False)), steps
    res = env.step(TERMINATE)
    return res.info.get("terminal_distance"), bool(res.info.get("valid", False)), steps


def run_qtabs_episode(
    env: MolEnv,
    planner: QTabsPlanner,
    bank: ActionBank,
    mode: str,
    exact_rerank_topk: int,
) -> Tuple[Optional[float], bool, int]:
    _state, goal, _valid = env.reset()
    base_acc = env._base_acc.copy() if env._base_acc is not None else env._prop_model.frag_accum(env._seed_frag_idx).copy()
    open_counts = bank.initial_open_counts(env)
    start_env = deepcopy(env)

    candidates = planner.search(
        base_acc=base_acc,
        open_counts=open_counts,
        goal=goal,
        max_steps=max(0, env.max_steps - 1),
    )
    if not candidates:
        return replay_sequence(start_env, bank, (), exact=True)

    if mode == "qtabs_hybrid":
        best_dist, best_valid, best_steps = None, False, 0
        for cand in candidates[:max(1, exact_rerank_topk)]:
            dist, valid, n_steps = replay_sequence(start_env, bank, cand.seq, exact=True)
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist, best_valid, best_steps = dist, valid, n_steps
        return best_dist, best_valid, best_steps

    best = candidates[0]
    return replay_sequence(start_env, bank, best.seq, exact=True)


def benchmark_qtabs(
    env: MolEnv,
    planner: QTabsPlanner,
    bank: ActionBank,
    mode: str,
    n_episodes: int,
    exact_rerank_topk: int = 8,
) -> Dict[str, float]:
    dists, valids, steps = [], [], []
    if planner.bank.device.type == "cuda":
        torch.cuda.synchronize(planner.bank.device)
    t0 = time.time()
    for _ in range(n_episodes):
        dist, valid, n_steps = run_qtabs_episode(env, planner, bank, mode, exact_rerank_topk)
        valids.append(1.0 if valid else 0.0)
        steps.append(float(n_steps))
        if dist is not None:
            dists.append(float(dist))
    if planner.bank.device.type == "cuda":
        torch.cuda.synchronize(planner.bank.device)
    dt = max(time.time() - t0, 1e-9)
    return {
        "mean_distance": float(np.mean(dists)) if dists else float("nan"),
        "success_rate": float(np.mean([d < 0.1 for d in dists])) if dists else 0.0,
        "validity_rate": float(np.mean(valids)),
        "episodes_per_min": n_episodes / dt * 60.0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "n_episodes": n_episodes,
    }


def make_figure(results: Dict[str, Dict[str, float]], out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [m for m in ("random", "surrogate", "qtabs", "qtabs_hybrid") if m in results]
    labels = {
        "random": "random",
        "surrogate": "greedy\n(CAFS)",
        "qtabs": "Q-TABS\nbeam",
        "qtabs_hybrid": "Q-TABS\nhybrid",
    }
    colors = {
        "random": "#9a9a9a",
        "surrogate": "#0072B2",
        "qtabs": "#E69F00",
        "qtabs_hybrid": "#009E73",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    x = np.arange(len(order))

    d = [results[m]["mean_distance"] for m in order]
    ax1.bar(x, d, color=[colors[m] for m in order])
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[m] for m in order])
    ax1.set_ylabel("mean distance to goal ↓")
    ax1.set_title("Accuracy")
    for xi, di in zip(x, d):
        ax1.text(xi, di, f"{di:.3f}", ha="center", va="bottom", fontsize=9)

    sp = [results[m]["episodes_per_min"] for m in order]
    ax2.bar(x, sp, color=[colors[m] for m in order])
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[m] for m in order])
    ax2.set_ylabel("throughput ↑ (episodes / min)")
    ax2.set_title("Speed")
    ax2.set_yscale("log")
    for xi, si in zip(x, sp):
        ax2.text(xi, si, f"{si:.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Q-TABS-Lite: quotient-state tensor beam search over additive chemistry", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    p.add_argument("--beam_width", type=int, default=16)
    p.add_argument("--candidate_multiplier", type=int, default=4)
    p.add_argument("--final_keep", type=int, default=64)
    p.add_argument("--action_chunk_size", type=int, default=4096)
    p.add_argument("--exact_rerank_topk", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fragments_parquet", type=str, default=str(FRAGMENTS_PATH))
    p.add_argument("--parents_parquet", type=str, default=str(PARENTS_PATH))
    default_out = ARTIFACTS_DIR / "figures" / "qtabs.png"
    p.add_argument("--out", type=str, default=str(default_out))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", type=str, default="")
    p.add_argument("--skip_baselines", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    property_names = parse_property_names(args.goal_properties)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    frags = load_fragment_library(args.fragments_parquet, n=args.n_frags, min_count=args.min_frag_count)
    targets = load_target_distribution(args.parents_parquet, n=args.n_targets, property_names=property_names)
    env = MolEnv(
        frags,
        targets,
        max_steps=args.max_steps,
        property_names=property_names,
        reward_config={"mode": "sparse"},
        fast_props=True,
    )
    env_exact = MolEnv(
        frags,
        targets,
        max_steps=args.max_steps,
        property_names=property_names,
        reward_config={"mode": "sparse"},
        fast_props=False,
    )

    print("Preparing additive property oracle ...")
    model = env._prop_model if env._prop_model is not None else AdditivePropertyModel(property_names).fit_or_load(env)
    bank = ActionBank(env, model, device=device)
    planner = QTabsPlanner(
        bank=bank,
        beam_width=args.beam_width,
        candidate_multiplier=args.candidate_multiplier,
        final_keep=args.final_keep,
        action_chunk_size=args.action_chunk_size,
    )
    print(
        f"  action bank: {bank.n_templates} templates  | beam={args.beam_width}  "
        f"| chunk={args.action_chunk_size}  | device={device}"
    )

    results: Dict[str, Dict[str, float]] = {}
    if not args.skip_baselines:
        print("\nBenchmarking greedy baselines ...")
        results["random"] = benchmark_fast(env_exact, model, "random", args.n_episodes)
        results["surrogate"] = benchmark_fast(env_exact, model, "surrogate", args.n_episodes)

    print("\nBenchmarking Q-TABS ...")
    results["qtabs"] = benchmark_qtabs(
        env=env,
        planner=planner,
        bank=bank,
        mode="qtabs",
        n_episodes=args.n_episodes,
        exact_rerank_topk=1,
    )
    print(
        f"  dist={results['qtabs']['mean_distance']:.4f} "
        f"success={results['qtabs']['success_rate']*100:.1f}% "
        f"valid={results['qtabs']['validity_rate']*100:.1f}% "
        f"{results['qtabs']['episodes_per_min']:.0f} eps/min"
    )

    print("\nBenchmarking Q-TABS hybrid rerank ...")
    results["qtabs_hybrid"] = benchmark_qtabs(
        env=env,
        planner=planner,
        bank=bank,
        mode="qtabs_hybrid",
        n_episodes=args.n_episodes,
        exact_rerank_topk=args.exact_rerank_topk,
    )
    print(
        f"  dist={results['qtabs_hybrid']['mean_distance']:.4f} "
        f"success={results['qtabs_hybrid']['success_rate']*100:.1f}% "
        f"valid={results['qtabs_hybrid']['validity_rate']*100:.1f}% "
        f"{results['qtabs_hybrid']['episodes_per_min']:.0f} eps/min"
    )

    print(f"\n{'='*72}")
    print(f"{'policy':>14} | {'distance↓':>10} | {'success':>8} | {'eps/min↑':>10}")
    print("-" * 72)
    for mode, row in results.items():
        print(
            f"{mode:>14} | {row['mean_distance']:>10.4f} | "
            f"{row['success_rate']*100:>7.1f}% | {row['episodes_per_min']:>10.0f}"
        )
    print("=" * 72)

    if args.json:
        payload = {
            "seed": args.seed,
            "n_frags": args.n_frags,
            "n_episodes": args.n_episodes,
            "beam_width": args.beam_width,
            "candidate_multiplier": args.candidate_multiplier,
            "final_keep": args.final_keep,
            "action_chunk_size": args.action_chunk_size,
            "exact_rerank_topk": args.exact_rerank_topk,
            "device": str(device),
            "goal_properties": list(property_names),
            "results": results,
        }
        _Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results JSON saved: {args.json}")

    make_figure(results, args.out)


if __name__ == "__main__":
    main()
