# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import FRAGMENTS_PATH, PARENTS_PATH, ARTIFACTS_DIR

import argparse
import io
from copy import deepcopy
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

import torch

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from env import load_fragment_library, load_target_distribution, denormalize_props
from env import MolEnv, TERMINATE
from env import compute_norm_properties
from models import Actor
from chem.build.molgraph_to_mol import molgraph_to_mol
from core.structs import MolGraph



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


def partial_properties(mg: MolGraph) -> Optional[np.ndarray]:
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return None
    capped = _cap_dummies(mol)
    if capped is None:
        return None
    return compute_norm_properties(capped)



def mol_to_pil(mol: Optional[Chem.Mol], size: Tuple[int, int] = (300, 240)) -> Image.Image:
    if mol is None:
        img = Image.new("RGB", size, (240, 240, 240))
        return img
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        Chem.SanitizeMol(mol, catchErrors=True)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        img = Image.new("RGB", size, (240, 240, 240))
        return img


def molgraph_to_pil(mg: MolGraph, size: Tuple[int, int] = (300, 240)) -> Image.Image:
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return Image.new("RGB", size, (240, 240, 240))
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        pass
    return mol_to_pil(mol, size)



from dataclasses import dataclass

@dataclass
class StepSnapshot:
    step:        int
    action_desc: str
    mol_graph:   MolGraph
    props_norm:  Optional[np.ndarray]
    reward:      float
    done:        bool
    n_open_sites: int


def run_episode(
    env: MolEnv,
    seed: int,
    actor: Optional[Actor] = None,
    device: torch.device = torch.device("cpu"),
    greedy: bool = True,
) -> Tuple[np.ndarray, List[StepSnapshot]]:
    np.random.seed(seed)
    state, goal_norm, valid_actions = env.reset()

    snapshots: List[StepSnapshot] = [StepSnapshot(
        step=0,
        action_desc="seed",
        mol_graph=deepcopy(env._mol),
        props_norm=partial_properties(env._mol),
        reward=0.0,
        done=False,
        n_open_sites=len(env._open_labels()),
    )]

    done = False
    step = 0
    while not done:
        if actor is not None:
            af_np   = env.get_action_features(valid_actions)
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            af_t    = torch.tensor(af_np,  dtype=torch.float32, device=device)
            with torch.no_grad():
                dist_t = actor.action_dist(state_t, af_t)
                idx = int(dist_t.probs.argmax().item()) if greedy else int(dist_t.sample().item())
        else:
            idx = np.random.randint(len(valid_actions))

        action = valid_actions[idx]
        result = env.step(action)
        step  += 1

        if action.is_terminate:
            desc = "terminate"
        else:
            desc = (f"attach frag#{action.frag_idx}\n"
                    f"mol←{action.mol_label!r} frag←{action.frag_label!r}")

        props = partial_properties(env._mol)
        if result.done and result.info.get("achieved_goal") is not None:
            props = result.info["achieved_goal"]

        display_mg = deepcopy(env._mol)
        if result.done:
            raw_mol = molgraph_to_mol(display_mg, sanitize=False, remove_hs=False)
            capped  = _cap_dummies(raw_mol) if raw_mol is not None else None
            if capped is not None:
                from chem.build.create_molgraph import smiles_to_molgraph as _s2mg
                from rdkit.Chem import MolToSmiles as _smi
                try:
                    display_mg = _s2mg(_smi(capped)) or display_mg
                except Exception:
                    pass

        snapshots.append(StepSnapshot(
            step=step,
            action_desc=desc,
            mol_graph=display_mg,
            props_norm=props,
            reward=result.reward,
            done=result.done,
            n_open_sites=len(env._open_labels()),
        ))

        state = result.state
        done  = result.done
        if not done:
            valid_actions = result.info.get("valid_actions", [TERMINATE])

    return goal_norm, snapshots



PROP_LABELS = ["LogP", "QED", "TPSA(norm)"]
PROP_UNITS  = ["", "", ""]

def _delta_colour(delta: float) -> str:
    a = abs(delta)
    if a < 0.10: return "#2ecc71"
    if a < 0.25: return "#f39c12"
    return "#e74c3c"


def _prop_block(
    ax: plt.Axes,
    props_norm: Optional[np.ndarray],
    goal_norm:  np.ndarray,
    label:      str,
    is_target:  bool = False,
) -> None:
    ax.axis("off")
    lines = [label, ""]

    if is_target:
        raw = denormalize_props(goal_norm)
        raw_names = ["LogP", "QED", "TPSA"]
        raw_units = ["", "", " Å²"]
        for nm, u, vn, vr in zip(raw_names, raw_units, goal_norm, raw):
            lines.append(f"{nm}: {vr:+.3f}{u}  [{vn:.3f}]")
        ax.text(0.5, 0.5, "\n".join(lines),
                transform=ax.transAxes, va="center", ha="center",
                fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#dce8f5", ec="#3498db", lw=1.5))
        return

    if props_norm is None:
        lines.append("(no props)")
        ax.text(0.5, 0.5, "\n".join(lines),
                transform=ax.transAxes, va="center", ha="center",
                fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#aaa"))
        return

    raw_props = denormalize_props(props_norm)
    raw_goal  = denormalize_props(goal_norm)
    raw_names = ["LogP", "QED", "TPSA"]
    raw_units = ["", "", " Å²"]

    dist = float(np.linalg.norm(props_norm - goal_norm))
    lines.append(f"L2 dist: {dist:.3f}")
    lines.append("")
    for nm, u, pn, pr, gn in zip(
        raw_names, raw_units,
        props_norm, raw_props,
        goal_norm,
    ):
        delta = pn - gn
        sign  = "+" if delta >= 0 else ""
        lines.append(f"{nm}: {pr:+.3f}{u} [Δ={sign}{delta:.3f}]")

    dist_colour = _delta_colour(dist)
    ax.text(0.5, 0.5, "\n".join(lines),
            transform=ax.transAxes, va="center", ha="center",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9",
                      ec=dist_colour, lw=2.0))


def render_trajectory(
    goal_norm:  np.ndarray,
    snapshots:  List[StepSnapshot],
    seed:       int,
    out_path:   str,
    mol_size:   Tuple[int, int] = (300, 240),
):
    n_rows = 1 + len(snapshots)
    n_cols = 3

    fig_w = 12.0
    row_h  = 3.2
    fig_h  = row_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("#fafafa")

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        width_ratios=[0.9, 1.6, 1.2],
        hspace=0.12,
        wspace=0.08,
        left=0.02, right=0.98,
        top=0.98, bottom=0.01,
    )

    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    goal_raw = denormalize_props(goal_norm)
    head_txt = (
        f"Trajectory  (seed={seed})\n"
        f"TARGET:  LogP={goal_raw[0]:+.3f}    QED={goal_raw[1]:.3f}    "
        f"TPSA={goal_raw[2]:.1f} Å²    "
        f"(norm [{', '.join(f'{v:.3f}' for v in goal_norm)}])"
    )
    ax_head.text(
        0.5, 0.5, head_txt,
        transform=ax_head.transAxes,
        va="center", ha="center",
        fontsize=10, fontweight="bold", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#dce8f5", ec="#2980b9", lw=2),
    )

    for row_idx, snap in enumerate(snapshots):
        r = row_idx + 1

        ax_info = fig.add_subplot(gs[r, 0])
        ax_info.axis("off")

        if snap.step == 0:
            top_line   = "t = 0  (seed)"
            action_txt = ""
            bg_colour  = "#eaf7ea"
            border     = "#27ae60"
        elif snap.done:
            top_line   = f"t = {snap.step}   DONE"
            action_txt = f"reward = {snap.reward:.4f}\nopen sites = {snap.n_open_sites}"
            bg_colour  = "#fdecea"
            border     = "#c0392b"
        else:
            top_line   = f"t = {snap.step}   open = {snap.n_open_sites}"
            action_txt = snap.action_desc
            bg_colour  = "#fef9e7"
            border     = "#f39c12"

        info_txt = top_line
        if action_txt:
            info_txt += "\n\n" + action_txt

        ax_info.text(
            0.5, 0.5, info_txt,
            transform=ax_info.transAxes,
            va="center", ha="center",
            fontsize=8, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=bg_colour, ec=border, lw=1.5),
        )

        ax_mol = fig.add_subplot(gs[r, 1])
        ax_mol.axis("off")
        pil_img = molgraph_to_pil(snap.mol_graph, size=mol_size)
        ax_mol.imshow(np.array(pil_img))

        ax_prop = fig.add_subplot(gs[r, 2])
        block_label = "partial  (dummies → H)" if not snap.done else "FINAL  (exact)"
        _prop_block(ax_prop, snap.props_norm, goal_norm, block_label)

    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")



def main(
    n_frags: int = 200,
    max_steps: int = 6,
    seeds: List[int] = None,
    checkpoint: Optional[str] = None,
    greedy: bool = True,
    device_str: str = "cpu",
    fragments_parquet: str = str(FRAGMENTS_PATH),
    parents_parquet: str = str(PARENTS_PATH),
    min_frag_count: int = 1,
    output_dir: str = str(ARTIFACTS_DIR / "trajectories"),
):
    output = _Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = [7, 42, 99]

    device = torch.device(device_str)

    print("Loading data ...")
    frags   = load_fragment_library(
        fragments_parquet,
        n=n_frags, min_count=min_frag_count,
    )
    targets = load_target_distribution(
        parents_parquet,
        n=300,
    )
    counterfactual = False
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        counterfactual = ckpt.get("config", {}).get("counterfactual_actions", False)
    env = MolEnv(frags, targets, max_steps=max_steps, counterfactual_actions=counterfactual)

    actor = None
    if checkpoint is not None:
        print(f"Loading actor from {checkpoint} ...")
        ckpt  = torch.load(checkpoint, map_location=device)
        actor = Actor(
            state_dim=env.state_dim,
            action_feat_dim=env.action_feat_dim,
            hidden_dim=ckpt["config"].get("hidden_dim", 256),
        ).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        policy_label = "trained"
    else:
        policy_label = "random"

    for seed in seeds:
        print(f"\nSeed {seed}  [{policy_label} policy]")
        goal_norm, snapshots = run_episode(env, seed, actor=actor, device=device, greedy=greedy)

        goal_raw = denormalize_props(goal_norm)
        print(f"  Goal: LogP={goal_raw[0]:+.3f}  QED={goal_raw[1]:.3f}  TPSA={goal_raw[2]:.1f}")

        for snap in snapshots:
            dist_s = f"L2={np.linalg.norm(snap.props_norm - goal_norm):.3f}" if snap.props_norm is not None else "L2=N/A"
            tag = "DONE" if snap.done else "    "
            print(f"    t={snap.step} {tag}  {dist_s}  open={snap.n_open_sites}  r={snap.reward:.3f}")

        out = str(output / f"trajectory_{policy_label}_seed{seed}.png")
        render_trajectory(goal_norm, snapshots, seed=seed, out_path=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_frags",    type=int,   default=200)
    p.add_argument("--max_steps",  type=int,   default=6)
    p.add_argument("--seeds",      type=int,   nargs="+", default=[7, 42, 99])
    p.add_argument("--checkpoint", type=str,   default=None)
    p.add_argument("--greedy",     action="store_true", default=True)
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--fragments_parquet", default=str(FRAGMENTS_PATH))
    p.add_argument("--parents_parquet", default=str(PARENTS_PATH))
    p.add_argument("--min_frag_count", type=int, default=1)
    p.add_argument("--output_dir", default=str(ARTIFACTS_DIR / "trajectories"))
    args = p.parse_args()
    main(
        n_frags=args.n_frags,
        max_steps=args.max_steps,
        seeds=args.seeds,
        checkpoint=args.checkpoint,
        greedy=args.greedy,
        device_str=args.device,
        fragments_parquet=args.fragments_parquet, parents_parquet=args.parents_parquet,
        min_frag_count=args.min_frag_count, output_dir=args.output_dir,
    )
