# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import FRAGMENTS_PATH, PARENTS_PATH, ARTIFACTS_DIR

import argparse
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.QED import qed as _qed
from rdkit.Chem.Draw import rdMolDraw2D

from env import (
    load_fragment_library, load_target_distribution,
    denormalize_props, PROP_NAMES,
    MolEnv, TERMINATE, _normalise_brics_smiles,
)
from chem.build.molgraph_to_mol import molgraph_to_mol



def _draw_mol(mol: Chem.Mol, path: str, label: str = "", size: tuple = (400, 300)) -> None:
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addAtomIndices = False
    if label:
        drawer.drawOptions().legendFontSize = 14
    try:
        Chem.SanitizeMol(mol)
        drawer.DrawMolecule(mol, legend=label)
    except Exception:
        drawer.DrawMolecule(mol, legend=label)
    drawer.FinishDrawing()
    with open(path, "wb") as fh:
        fh.write(drawer.GetDrawingText())
    print(f"  Saved: {path}")


def _prop_table(raw: np.ndarray) -> str:
    names = ["LogP", "QED ", "TPSA"]
    units = ["    ", "    ", " Å² "]
    lines = []
    for nm, u, v in zip(names, units, raw):
        lines.append(f"    {nm} {u} {v:+.3f}")
    return "\n".join(lines)



def run_demo(n_frags: int = 50, max_steps: int = 4, seed: int = 42,
             fragments_parquet: str = str(FRAGMENTS_PATH),
             parents_parquet: str = str(PARENTS_PATH), min_frag_count: int = 1,
             output_dir: str = str(ARTIFACTS_DIR / "demo")):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading data …")
    frags   = load_fragment_library(
        fragments_parquet,
        n=n_frags, min_count=min_frag_count,
    )
    targets = load_target_distribution(
        parents_parquet,
        n=200,
    )

    env = MolEnv(frags, targets, max_steps=max_steps)

    print("\n─── Episode start ───────────────────────────────────────")
    state, goal_norm, valid_actions = env.reset()

    seed_mg  = env._mol
    seed_mol = molgraph_to_mol(seed_mg, sanitize=False, remove_hs=False)
    seed_smi = Chem.MolToSmiles(seed_mol) if seed_mol else "(unknown)"

    print(f"\nSeed fragment SMILES : {seed_smi}")
    print(f"Open attachment sites: {env._open_labels()}")

    goal_raw = denormalize_props(goal_norm)
    print(f"\nTarget goal (normalised): [{', '.join(f'{v:.3f}' for v in goal_norm)}]")
    print(f"Target goal (raw):\n{_prop_table(goal_raw)}")

    if seed_mol:
        _draw_mol(seed_mol, str(output / "seed.png"), label="Seed fragment")

    print("\n─── Rollout ─────────────────────────────────────────────")
    done = False
    step = 0
    while not done:
        idx    = np.random.randint(len(valid_actions))
        action = valid_actions[idx]
        result = env.step(action)

        if action.is_terminate:
            tag = "terminate"
        else:
            tag = (f"attach frag#{action.frag_idx} "
                   f"mol_lbl={action.mol_label!r} frag_lbl={action.frag_label!r}")

        print(f"  step {step}: {tag}  reward={result.reward:.4f}  done={result.done}")
        step  += 1
        done   = result.done
        if not done:
            valid_actions = result.info.get("valid_actions", [TERMINATE])

    print("\n─── Result ──────────────────────────────────────────────")
    achieved = result.info.get("achieved_goal")
    is_valid = result.info.get("valid", False)

    out_mg  = env._mol
    out_mol = molgraph_to_mol(out_mg, sanitize=True, remove_hs=True)
    if out_mol is None:
        out_mol = molgraph_to_mol(out_mg, sanitize=False, remove_hs=False)
    out_smi = Chem.MolToSmiles(out_mol) if out_mol else "(could not convert)"

    print(f"Output SMILES  : {out_smi}")
    print(f"Valid molecule : {is_valid}")

    if achieved is not None:
        raw = denormalize_props(achieved)
        print(f"Achieved (normalised): [{', '.join(f'{v:.3f}' for v in achieved)}]")
        print(f"Achieved (raw):\n{_prop_table(raw)}")
        dist = float(np.linalg.norm(achieved - goal_norm))
        print(f"\nL2 distance to goal  : {dist:.4f}")
        print(f"Per-property delta   :")
        for nm, g, a in zip(["LogP", "QED ", "TPSA"], goal_raw, raw):
            print(f"    {nm}  goal={g:+.3f}  achieved={a:+.3f}  Δ={a-g:+.3f}")
    else:
        print("(molecule did not produce valid properties)")

    if out_mol:
        label = f"Output | reward={result.reward:.3f}"
        _draw_mol(out_mol, str(output / "output.png"), label=label)

    print(f"\nDone. Images written to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_frags",  type=int, default=50)
    p.add_argument("--steps",    type=int, default=4)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--fragments_parquet", default=str(FRAGMENTS_PATH))
    p.add_argument("--parents_parquet", default=str(PARENTS_PATH))
    p.add_argument("--min_frag_count", type=int, default=1)
    p.add_argument("--output_dir", default=str(ARTIFACTS_DIR / "demo"))
    args = p.parse_args()
    run_demo(n_frags=args.n_frags, max_steps=args.steps, seed=args.seed,
             fragments_parquet=args.fragments_parquet, parents_parquet=args.parents_parquet,
             min_frag_count=args.min_frag_count, output_dir=args.output_dir)
