# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import ARTIFACTS_DIR

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from env import load_fragment_library, load_target_distribution, parse_property_names
from env import MolEnv, TERMINATE
from models import Actor
from training.a2c import DEFAULT_CFG

_ROOT    = _Path(__file__).resolve().parent.parent
CKPT_DIR = ARTIFACTS_DIR / "checkpoints_a2c"
OUT_FILE = ARTIFACTS_DIR / "val_results.json"
VAL_SEED  = 99991
VAL_N     = 200


def eval_actor(actor, env, n=VAL_N, device=torch.device("cpu")):
    np.random.seed(VAL_SEED)
    actor.eval()
    dists = []
    for _ in range(n):
        state, goal, valid_actions = env.reset()
        done = False
        while not done:
            af   = env.get_action_features(valid_actions)
            st   = torch.tensor(state, dtype=torch.float32, device=device)
            af_t = torch.tensor(af,    dtype=torch.float32, device=device)
            with torch.no_grad():
                idx = int(actor.action_dist(st, af_t).probs.argmax().item())
            result = env.step(valid_actions[idx])
            state  = result.state
            done   = result.done
            if not done:
                valid_actions = result.info.get("valid_actions", [TERMINATE])
        props = result.info.get("achieved_goal")
        if props is not None:
            dists.append(float(np.linalg.norm(props - goal)))
    return float(np.mean(dists)) if dists else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_frags",   type=int, default=DEFAULT_CFG["n_frags"])
    p.add_argument("--n_targets", type=int, default=DEFAULT_CFG["n_targets"])
    p.add_argument("--min_frag_count", type=int, default=DEFAULT_CFG["min_frag_count"])
    p.add_argument("--checkpoint_dir", type=Path, default=CKPT_DIR)
    p.add_argument("--out", type=Path, default=OUT_FILE)
    p.add_argument("--val_n",     type=int, default=VAL_N)
    p.add_argument("--fragments_parquet", default=DEFAULT_CFG["fragments_parquet"])
    p.add_argument("--parents_parquet", default=DEFAULT_CFG["parents_parquet"])
    args = p.parse_args()
    DEFAULT_CFG.update(fragments_parquet=args.fragments_parquet, parents_parquet=args.parents_parquet)

    device = torch.device("cpu")

    print("Loading data ...")
    frags = load_fragment_library(
        DEFAULT_CFG["fragments_parquet"],
        n=args.n_frags,
        min_count=args.min_frag_count,
    )

    ckpts = sorted(args.checkpoint_dir.glob("ckpt_ep*.pt"),
                   key=lambda p: int(p.stem.split("ep")[1]))

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
    results = {}
    for ckpt_path in ckpts:
        ep = int(ckpt_path.stem.split("ep")[1])
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        property_names = parse_property_names(cfg.get("goal_properties", DEFAULT_CFG["goal_properties"]))
        targets = load_target_distribution(
            args.parents_parquet,
            n=args.n_targets,
            property_names=property_names,
        )
        env = MolEnv(
            frags,
            targets,
            max_steps=cfg.get("max_steps", DEFAULT_CFG["max_steps"]),
            property_names=property_names,
            counterfactual_actions=cfg.get("counterfactual_actions", False),
        )
        hidden = cfg.get("hidden_dim", 256)
        actor  = Actor(
            state_dim=env.state_dim,
            action_feat_dim=env.action_feat_dim,
            hidden_dim=hidden,
        ).to(device)
        actor.load_state_dict(ckpt["actor"])

        mean_dist = eval_actor(actor, env, n=args.val_n, device=device)
        results[ep] = mean_dist
        print(f"  ep {ep:6d}  val_dist = {mean_dist:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
