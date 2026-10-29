# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import ARTIFACTS_DIR

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cs224r-trajectory")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from chem.build.molgraph_to_mol import molgraph_to_mol
from core.structs import MolGraph
from env import (
    MolEnv,
    TERMINATE,
    compute_norm_properties,
    load_fragment_library,
    load_target_distribution,
    parse_property_names,
)
from env.properties import denormalize_props
from models import Actor, GNNActor, GNNSuccessor
from training.a2c import DEFAULT_CFG as A2C_DEFAULT_CFG


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ARTIFACTS_DIR
CHECKPOINT_ROOTS = [RESULTS_ROOT / "modal_ckpts", RESULTS_ROOT]
EPSILON = 0.1

plt.style.use("seaborn-v0_8-paper")


def _default_eval_json() -> Path:
    candidates = [
        ROOT / "results" / "report_eval_results.json",
        ROOT / "results" / "eval_results.json",
        RESULTS_ROOT / "report_eval_results.json",
        RESULTS_ROOT / "eval_results.json",
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)
    return candidates[0]


def _latest_checkpoint(path_or_dir: Path) -> Optional[Path]:
    if path_or_dir.is_file():
        return path_or_dir
    if not path_or_dir.exists():
        return None
    def _epnum(path: Path) -> int:
        match = re.search(r"ckpt_ep(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    direct = sorted(path_or_dir.glob("ckpt_ep*.pt"), key=_epnum)
    if direct:
        return direct[-1]
    ckpts = sorted(path_or_dir.rglob("ckpt_ep*.pt"), key=lambda p: (_epnum(p), -len(p.parts)))
    return ckpts[-1] if ckpts else None


def _cap_dummies(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        rw = Chem.RWMol(Chem.Mol(mol))
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomicNum(1)
                atom.SetNoImplicit(False)
                atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(rw, catchErrors=True)
        return rw.GetMol()
    except Exception:
        return None


def _clean_for_display(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetIsotope(0)
    return rw.GetMol()


def mol_to_pil(mol: Optional[Chem.Mol], size: Tuple[int, int] = (300, 260)) -> Image.Image:
    blank = Image.new("RGB", size, (255, 255, 255))
    if mol is None:
        return blank
    try:
        mol = _clean_for_display(mol)
        Chem.SanitizeMol(mol, catchErrors=True)
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().padding = 0.15
        drawer.drawOptions().bondLineWidth = 1.5
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        return blank


def _format_prop_name(name: str) -> str:
    return {"sLogP": "LogP"}.get(name, name)


def _format_prop_value(name: str, value: float) -> str:
    label = _format_prop_name(name)
    if name == "TPSA":
        return rf"{label} $= {value:.1f}$ \AA$^2$"
    if name in {"sLogP", "LogP"}:
        return rf"{label} $= {value:+.2f}$"
    return rf"{label} $= {value:.2f}$"


def _best_checkpoint_from_eval(eval_json: Path) -> Path:
    payload = json.loads(eval_json.read_text())
    runs = payload.get("runs", {})
    ranked = []
    for name, metrics in runs.items():
        if metrics.get("goal_properties") != "sLogP,QED,TPSA":
            continue
        sr = metrics.get("success_rate")
        dist = metrics.get("mean_distance")
        if sr is None or dist is None:
            continue
        ranked.append((float(sr), -float(dist), int(metrics.get("episode", 0)), name))
    ranked.sort(reverse=True)
    for _sr, _nd, _ep, name in ranked:
        for root in CHECKPOINT_ROOTS:
            direct = root / f"checkpoints_{name}"
            ckpt = _latest_checkpoint(direct)
            if ckpt is not None:
                return ckpt
    raise FileNotFoundError(
        f"Could not resolve any checkpoint for the top evaluated runs in {eval_json}."
    )


def _resolve_checkpoint(checkpoint_arg: str, eval_json: Path) -> Path:
    if not checkpoint_arg or checkpoint_arg == "auto-best":
        return _best_checkpoint_from_eval(eval_json)
    ckpt = _latest_checkpoint(Path(checkpoint_arg))
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_arg}")
    return ckpt


def _build_env_and_policy(
    ckpt_path: Path,
    device: torch.device,
    eval_offset: int,
    n_eval_targets: int,
    n_frags_override: int = 0,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    algo = str(ckpt.get("algo", cfg.get("algo", "a2c")))
    property_names = tuple(parse_property_names(cfg.get("goal_properties", A2C_DEFAULT_CFG["goal_properties"])))
    n_total = eval_offset + n_eval_targets
    n_frags = int(n_frags_override or cfg.get("n_frags", A2C_DEFAULT_CFG["n_frags"]))
    min_frag_count = int(cfg.get("min_frag_count", A2C_DEFAULT_CFG["min_frag_count"]))

    frags = load_fragment_library(
        A2C_DEFAULT_CFG["fragments_parquet"],
        n=n_frags,
        min_count=min_frag_count,
    )
    targets = load_target_distribution(
        A2C_DEFAULT_CFG["parents_parquet"],
        n=n_total,
        property_names=property_names,
    )[eval_offset:]
    env = MolEnv(
        frags,
        targets,
        max_steps=int(cfg.get("max_steps", A2C_DEFAULT_CFG["max_steps"])),
        property_names=property_names,
        reward_config={"mode": "sparse"},
        site_actions=bool(cfg.get("site_actions", False)),
        counterfactual_actions=bool(cfg.get("counterfactual_actions", False)),
        fast_props=bool(cfg.get("fast_props", False)),
    )

    if "successor" in ckpt or algo == "sf":
        policy = GNNSuccessor(
            gnn_dim=int(cfg.get("gnn_dim", 128)),
            goal_dim=env.goal_dim,
            action_feat_dim=env.action_feat_dim,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            site_actions=bool(cfg.get("site_actions", False)),
            frag_site_emb=bool(cfg.get("frag_site_emb", False)),
            num_frag_sites=getattr(env, "num_frag_sites", 0),
        ).to(device)
        policy.load_state_dict(ckpt["successor"])
        kind = "sf"
    elif cfg.get("encoder") == "gnn":
        policy = GNNActor(
            gnn_dim=int(cfg.get("gnn_dim", 128)),
            goal_dim=env.goal_dim,
            action_feat_dim=env.action_feat_dim,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            site_actions=bool(cfg.get("site_actions", False)),
            frag_site_emb=bool(cfg.get("frag_site_emb", False)),
            num_frag_sites=getattr(env, "num_frag_sites", 0),
        ).to(device)
        policy.load_state_dict(ckpt["actor"])
        kind = "gnn"
    else:
        policy = Actor(
            state_dim=env.state_dim,
            action_feat_dim=env.action_feat_dim,
            hidden_dim=int(cfg.get("hidden_dim", 256)),
        ).to(device)
        policy.load_state_dict(ckpt["actor"])
        kind = "fp"

    if bool(cfg.get("frag_site_emb", False)) and hasattr(policy, "refresh_frag_cache"):
        policy.refresh_frag_cache(env, device)
    policy.eval()
    return ckpt, env, policy, kind, property_names


def _pick_action(
    policy,
    kind: str,
    env: MolEnv,
    state: np.ndarray,
    goal_norm: np.ndarray,
    valid_actions,
    device: torch.device,
    greedy: bool,
) -> int:
    af_np = env.get_action_features(valid_actions)
    if kind == "fp":
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        af_t = torch.tensor(af_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = policy.action_dist(state_t, af_t)
            return int(dist.probs.argmax().item()) if greedy else int(dist.sample().item())

    mg = deepcopy(env._mol)
    site_ix_np = env.get_action_site_indices(valid_actions)
    goal_t = torch.tensor(goal_norm, dtype=torch.float32, device=device)
    af_t = torch.tensor(af_np, dtype=torch.float32, device=device)
    site_ix_t = torch.tensor(site_ix_np, dtype=torch.long, device=device)
    with torch.no_grad():
        if kind == "sf":
            psi = policy.psi(mg, goal_t, af_t, device, site_ix_t)
            q = policy.q_from_psi(psi, goal_t)
            return int(q.argmax().item()) if greedy else int(torch.distributions.Categorical(logits=q).sample().item())
        dist = policy.action_dist(mg, goal_t, af_t, device, site_ix_t)
        return int(dist.probs.argmax().item()) if greedy else int(dist.sample().item())


def _episode_summary(
    goal_norm: np.ndarray,
    frames: List[Dict[str, object]],
    property_names: Sequence[str],
) -> Dict[str, object]:
    final_props = frames[-1].get("achieved")
    if final_props is None:
        mol = molgraph_to_mol(frames[-1]["mol"], sanitize=False, remove_hs=False)
        capped = _cap_dummies(mol) if mol is not None else None
        final_props = compute_norm_properties(capped, property_names) if capped is not None else None
    final_dist = float(np.linalg.norm(np.asarray(final_props) - goal_norm)) if final_props is not None else float("inf")
    success = bool(final_props is not None and np.all(np.abs(np.asarray(final_props) - goal_norm) < EPSILON))
    return {
        "goal_norm": goal_norm,
        "frames": frames,
        "final_props": final_props,
        "final_dist": final_dist,
        "success": success,
        "num_steps": max(0, len(frames) - 1),
    }


def run_episode(
    env: MolEnv,
    seed: int,
    policy,
    kind: str,
    property_names: Sequence[str],
    device: torch.device,
    greedy: bool = True,
) -> Dict[str, object]:
    np.random.seed(seed)
    state, goal_norm, valid_actions = env.reset()
    frames: List[Dict[str, object]] = [{"mol": deepcopy(env._mol), "step": 0}]

    done = False
    while not done:
        idx = _pick_action(policy, kind, env, state, goal_norm, valid_actions, device, greedy)
        result = env.step(valid_actions[idx])
        state = result.state
        done = result.done
        frames.append({
            "mol": deepcopy(env._mol),
            "step": len(frames),
            "achieved": result.info.get("achieved_goal"),
            "done": result.done,
        })
        if not done:
            valid_actions = result.info.get("valid_actions", [TERMINATE])

    summary = _episode_summary(goal_norm, frames, property_names)
    summary["rollout_seed"] = seed
    return summary


def _rollout_rank(summary: Dict[str, object]) -> Tuple[int, float, int]:
    final_dist = float(summary["final_dist"])
    return (
        1 if bool(summary["success"]) else 0,
        -final_dist if np.isfinite(final_dist) else -1e9,
        int(summary["num_steps"]),
    )


def _compact_rollout_result(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "rollout_seed": int(summary["rollout_seed"]),
        "success": bool(summary["success"]),
        "final_dist": float(summary["final_dist"]),
        "num_steps": int(summary["num_steps"]),
    }


def _search_chunk(
    ckpt_path_str: str,
    eval_offset: int,
    n_eval_targets: int,
    n_frags_override: int,
    seed_start: int,
    seed_stop: int,
    greedy: bool,
) -> Dict[str, object]:
    ckpt_path = Path(ckpt_path_str)
    device = torch.device("cpu")
    _ckpt, env, policy, kind, property_names = _build_env_and_policy(
        ckpt_path,
        device=device,
        eval_offset=eval_offset,
        n_eval_targets=n_eval_targets,
        n_frags_override=n_frags_override,
    )
    best: Optional[Dict[str, object]] = None
    for seed in range(seed_start, seed_stop):
        summary = run_episode(env, seed, policy, kind, property_names, device, greedy=greedy)
        if best is None or _rollout_rank(summary) > _rollout_rank(best):
            best = summary
    assert best is not None
    return _compact_rollout_result(best)


def select_best_rollout(
    env: MolEnv,
    policy,
    kind: str,
    property_names: Sequence[str],
    device: torch.device,
    start_seed: int,
    search_rollouts: int,
    greedy: bool = True,
    workers: int = 1,
    ckpt_path: Optional[Path] = None,
    eval_offset: int = 300,
    n_eval_targets: int = 300,
    n_frags_override: int = 0,
) -> Dict[str, object]:
    if workers > 1 and search_rollouts > 1:
        if ckpt_path is None:
            raise ValueError("Parallel search requires the resolved checkpoint path.")
        total = max(search_rollouts, 1)
        chunk = max(1, math.ceil(total / workers))
        futures = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for offset in range(0, total, chunk):
                seed_lo = start_seed + offset
                seed_hi = min(start_seed + total, seed_lo + chunk)
                futures.append(pool.submit(
                    _search_chunk,
                    str(ckpt_path),
                    eval_offset,
                    n_eval_targets,
                    n_frags_override,
                    seed_lo,
                    seed_hi,
                    greedy,
                ))
        best_small = max((future.result() for future in futures), key=_rollout_rank)
        return run_episode(
            env,
            int(best_small["rollout_seed"]),
            policy,
            kind,
            property_names,
            device,
            greedy=greedy,
        )

    best: Optional[Dict[str, object]] = None
    for seed in range(start_seed, start_seed + max(search_rollouts, 1)):
        summary = run_episode(env, seed, policy, kind, property_names, device, greedy=greedy)
        if best is None or _rollout_rank(summary) > _rollout_rank(best):
            best = summary
    assert best is not None
    return best


def make_figure(
    goal_norm: np.ndarray,
    frames: List[Dict[str, object]],
    property_names: Sequence[str],
    out: str,
) -> None:
    n = len(frames)
    mol_size = (300, 240)
    goal_raw = denormalize_props(goal_norm, property_names)

    props_list = []
    l2s = []
    for frame in frames:
        mol = molgraph_to_mol(frame["mol"], sanitize=False, remove_hs=False)
        capped = _cap_dummies(mol) if mol is not None else None
        props = compute_norm_properties(capped, property_names) if capped is not None else None
        if frame.get("achieved") is not None:
            props = frame["achieved"]
        props_list.append(props)
        l2s.append(float(np.linalg.norm(np.asarray(props) - goal_norm)) if props is not None else float("nan"))

    pils = []
    for frame in frames:
        mol = molgraph_to_mol(frame["mol"], sanitize=False, remove_hs=False)
        capped = _cap_dummies(mol) if mol is not None else None
        pils.append(mol_to_pil(capped, mol_size))

    fig_w = max(6.6, 0.94 * n)
    fig, axes_row = plt.subplots(1, n, figsize=(fig_w, 2.2), gridspec_kw=dict(wspace=0.02))
    axes = np.atleast_1d(axes_row)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.30)

    fig.canvas.draw()
    ax0_top = axes[0].get_position().y1
    title_text = "Target:  " + ",  ".join(
        _format_prop_value(name, float(value))
        for name, value in zip(property_names, goal_raw)
    )
    fig.text(0.5, ax0_top + 0.07, title_text, ha="center", va="bottom", fontsize=7)

    for i, (ax, pil, l2, props) in enumerate(zip(axes, pils, l2s, props_list)):
        ax.imshow(np.array(pil))
        ax.axis("off")
        ax.set_title("Seed" if i == 0 else rf"$t = {i}$", fontsize=7, pad=2)

        if props is not None:
            raw = denormalize_props(np.asarray(props), property_names)
            lines = [rf"$\ell_2 = {l2:.3f}$"] + [
                _format_prop_value(name, float(value))
                for name, value in zip(property_names, raw)
            ]
        else:
            lines = [""]
        ax.text(
            0.5,
            -0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.5,
            linespacing=1.5,
        )

        if i < n - 1:
            ax.annotate(
                "",
                xy=(1.04, 0.52),
                xytext=(0.88, 0.52),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7),
            )

    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="auto-best")
    p.add_argument("--eval-json", type=str, default=str(_default_eval_json()))
    p.add_argument("--seed", type=int, default=42,
                   help="Starting rollout seed. If --search-rollouts > 1, seeds are searched consecutively from here.")
    p.add_argument("--search-rollouts", type=int, default=1,
                   help="Number of greedy rollouts to search before selecting the best single trajectory.")
    p.add_argument("--n-frags", type=int, default=0,
                   help="Optional fragment-count override. Default uses the checkpoint config.")
    p.add_argument("--eval-offset", type=int, default=300)
    p.add_argument("--n-eval-targets", type=int, default=300)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--greedy", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    eval_json = Path(args.eval_json)
    ckpt_path = _resolve_checkpoint(args.checkpoint, eval_json)
    ckpt, env, policy, kind, property_names = _build_env_and_policy(
        ckpt_path,
        device=device,
        eval_offset=args.eval_offset,
        n_eval_targets=args.n_eval_targets,
        n_frags_override=args.n_frags,
    )
    best = select_best_rollout(
        env,
        policy,
        kind,
        property_names,
        device=device,
        start_seed=args.seed,
        search_rollouts=args.search_rollouts,
        greedy=args.greedy,
        workers=args.workers,
        ckpt_path=ckpt_path,
        eval_offset=args.eval_offset,
        n_eval_targets=args.n_eval_targets,
        n_frags_override=args.n_frags,
    )

    out = args.out or f"fig_trajectory_seed{best['rollout_seed']}.png"
    make_figure(
        goal_norm=np.asarray(best["goal_norm"]),
        frames=best["frames"],
        property_names=property_names,
        out=out,
    )
    print(
        "Selected rollout: "
        f"checkpoint={ckpt_path} "
        f"algo={ckpt.get('algo', ckpt.get('config', {}).get('algo', 'a2c'))} "
        f"seed={best['rollout_seed']} "
        f"success={bool(best['success'])} "
        f"steps={best['num_steps']} "
        f"final_l2={float(best['final_dist']):.3f}"
    )


if __name__ == "__main__":
    main()
