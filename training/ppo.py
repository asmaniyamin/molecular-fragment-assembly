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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from env import (
    load_fragment_library, load_target_distribution,
    MolEnv, Action, TERMINATE, StepResult,
    DEFAULT_PROPERTY_NAMES, DEFAULT_PROPERTY_SURROGATE_CFG,
    build_env_reward_config, parse_reward_vector, parse_property_names,
    reward_from_context,
)
from models import Actor, Critic, GNNActor, GNNCritic
from env import Episode, Transition



DEFAULT_CFG: Dict = dict(
    fragments_parquet=str(FRAGMENTS_PATH),
    parents_parquet=str(PARENTS_PATH),
    n_frags=200,
    min_frag_count=5_000,
    n_targets=300,
    max_steps=6,
    goal_properties=",".join(DEFAULT_PROPERTY_NAMES),
    reward_properties="",
    reward_mode="baseline",
    distance_scale=1.0,
    shaping_gamma=0.99,
    normalize=False,
    use_gae=False,
    gae_lambda=0.95,
    encoder="fp",
    site_actions=False,
    frag_site_emb=False,
    counterfactual_actions=False,
    fast_props=False,
    refresh_every=50,
    gnn_dim=128,
    n_episodes=10_000,
    rollout_eps=16,
    ppo_epochs=4,
    clip_eps=0.2,
    batch_size=64,
    hidden_dim=256,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    her_k=4,
    entropy_coef=0.005,
    critic_coef=0.5,
    grad_clip=1.0,
    use_property_surrogate_reward=False,
    property_surrogate_scale=DEFAULT_PROPERTY_SURROGATE_CFG["scale"],
    property_surrogate_dummy_bonus=DEFAULT_PROPERTY_SURROGATE_CFG["dummy_bonus"],
    property_surrogate_step_penalty=DEFAULT_PROPERTY_SURROGATE_CFG["step_penalty"],
    property_surrogate_weights="",
    property_surrogate_temperatures="",
    property_surrogate_invalid_score=DEFAULT_PROPERTY_SURROGATE_CFG["invalid_score"],
    log_every=50,
    checkpoint_every=500,
    checkpoint_dir=str(ARTIFACTS_DIR / "checkpoints_ppo"),
    resume="",
    exp_name="ppo_run",
    seed=1,
    use_wandb=False,
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
)



@dataclass
class PPOTransition:
    state:        np.ndarray
    action_feats: np.ndarray
    action_idx:   int
    old_log_prob: float
    reward:       float
    done:         bool
    goal:         np.ndarray
    ret:          float = 0.0
    mol_graph:    Any   = None
    action_site_indices: Optional[np.ndarray] = None
    value:        float = 0.0
    adv:          Optional[float] = None



def rollout_ppo(env: MolEnv, actor: Actor, critic: Critic, device: torch.device,
                cfg: Optional[Dict] = None) -> Tuple[Episode, List[PPOTransition]]:
    cfg = cfg or DEFAULT_CFG
    state, goal, valid_actions = env.reset()
    ep = Episode()
    ppo_ts: List[PPOTransition] = []

    for _ in range(env.max_steps):
        af_np   = env.get_action_features(valid_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=device)
        af_t    = torch.tensor(af_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            dist       = actor.action_dist(state_t, af_t)
            action_idx = int(dist.sample().item())
            old_lp     = float(dist.log_prob(torch.tensor(action_idx, device=device)).item())
            value      = float(critic(state_t).reshape(()).item())

        result: StepResult = env.step(valid_actions[action_idx])
        next_valid = result.info.get("valid_actions", [TERMINATE])
        next_af_np = env.get_action_features(next_valid)

        ep.add(Transition(
            state=state,
            action_feats=af_np,
            action_idx=action_idx,
            reward=result.reward,
            next_state=result.state,
            next_action_feats=next_af_np,
            done=result.done,
            goal=goal,
            info=result.info,
        ))
        ppo_ts.append(PPOTransition(
            state=state,
            action_feats=af_np,
            action_idx=action_idx,
            old_log_prob=old_lp,
            reward=result.reward,
            done=result.done,
            goal=goal,
            value=value,
        ))

        if result.done:
            ep.achieved_goal = result.info.get("achieved_goal")
            break
        state         = result.state
        valid_actions = next_valid

    if cfg.get("use_gae"):
        _gae(ppo_ts, cfg["gamma"], cfg["gae_lambda"])
    else:
        G = 0.0
        for t in reversed(ppo_ts):
            G = t.reward + cfg["gamma"] * G * (0.0 if t.done else 1.0)
            t.ret = G

    return ep, ppo_ts


def her_transitions(
    ep: Episode,
    her_k: int,
    gamma: float,
    reward_cfg: Optional[Dict] = None,
) -> List[PPOTransition]:
    achieved = ep.achieved_goal
    if achieved is None:
        return []
    reward_cfg = reward_cfg or {"mode": "sparse"}
    out: List[PPOTransition] = []
    for _ in range(her_k):
        relabelled: List[Tuple[np.ndarray, np.ndarray, int, float, float, bool]] = []
        goal_dim = len(achieved)
        for t in ep.transitions:
            new_s = t.state.copy()
            new_s[-goal_dim:] = achieved
            reward_ctx = t.info.get("reward_ctx")
            new_r = (
                reward_from_context(reward_ctx, achieved, reward_cfg)
                if reward_ctx is not None else
                (0.0 if t.done else t.reward)
            )
            relabelled.append((new_s, t.action_feats, t.action_idx, new_r, t.done, t.goal))

        G = 0.0
        rets: List[float] = []
        for _, _, _, r, done, _ in reversed(relabelled):
            G = r + gamma * G * (0.0 if done else 1.0)
            rets.insert(0, G)

        for (new_s, af, aidx, r, done, _), ret in zip(relabelled, rets):
            out.append(PPOTransition(
                state=new_s,
                action_feats=af,
                action_idx=aidx,
                old_log_prob=0.0,
                reward=r,
                done=done,
                goal=achieved,
                ret=ret,
            ))
    return out



def update_ppo(
    actor:     Actor,
    critic:    Critic,
    opt_actor: optim.Optimizer,
    opt_critic: optim.Optimizer,
    batch:     List[PPOTransition],
    cfg:       Dict,
    device:    torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    with torch.no_grad():
        states_all = torch.tensor(
            np.stack([t.state for t in batch]), dtype=torch.float32, device=device
        )
        Vs_all  = critic(states_all)
        Gs_all  = torch.tensor([t.ret for t in batch], dtype=torch.float32, device=device)
        targets_all = _norm_targets(Gs_all, cfg)
        gae_adv = torch.tensor([(t.adv if t.adv is not None else float("nan")) for t in batch],
                               dtype=torch.float32, device=device)
        adv_raw = torch.where(torch.isnan(gae_adv), targets_all - Vs_all, gae_adv)
        adv_all = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    actor_losses_log:  List[float] = []
    critic_losses_log: List[float] = []
    entropies_log:     List[float] = []

    for _ in range(cfg["ppo_epochs"]):
        idx_perm = torch.randperm(len(batch))
        for start in range(0, len(batch), cfg["batch_size"]):
            mb_idx = idx_perm[start: start + cfg["batch_size"]]

            actor_losses:  List[torch.Tensor] = []
            critic_losses: List[torch.Tensor] = []

            for i in mb_idx.tolist():
                t   = batch[i]
                adv = adv_all[i]

                state_t = torch.tensor(t.state,        dtype=torch.float32, device=device)
                af_t    = torch.tensor(t.action_feats, dtype=torch.float32, device=device)
                G_t     = targets_all[i]

                dist     = actor.action_dist(state_t, af_t)
                new_lp   = dist.log_prob(torch.tensor(t.action_idx, device=device))
                entropy  = dist.entropy()

                ratio    = torch.exp(new_lp - t.old_log_prob)
                surr1    = ratio * adv
                surr2    = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * adv
                a_loss   = -torch.min(surr1, surr2) - cfg["entropy_coef"] * entropy

                V_s      = critic(state_t)
                c_loss   = cfg["critic_coef"] * (G_t - V_s).pow(2)

                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                entropies_log.append(float(entropy.item()))

            total_actor  = torch.stack(actor_losses).mean()
            total_critic = torch.stack(critic_losses).mean()

            opt_actor.zero_grad()
            total_actor.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg["grad_clip"])
            opt_actor.step()

            opt_critic.zero_grad()
            total_critic.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg["grad_clip"])
            opt_critic.step()

            actor_losses_log.append(float(total_actor.item()))
            critic_losses_log.append(float(total_critic.item()))

    return {
        "actor_loss":  float(np.mean(actor_losses_log)),
        "critic_loss": float(np.mean(critic_losses_log)),
        "entropy":     float(np.mean(entropies_log)),
    }



class RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        bc = x.numel()
        if bc == 0:
            return
        bm = float(x.mean().item())
        bv = float(x.var(unbiased=False).item())
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc
                    + delta * delta * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return max(self.var, 1e-8) ** 0.5


def _norm_targets(returns_t: torch.Tensor, cfg: Dict) -> torch.Tensor:
    rms = cfg.get("_ret_rms")
    if rms is None:
        return returns_t
    rms.update(returns_t)
    return (returns_t - rms.mean) / (rms.std + 1e-8)


def _gae(ppo_ts: List[PPOTransition], gamma: float, lam: float) -> None:
    adv = 0.0
    n = len(ppo_ts)
    for i in reversed(range(n)):
        t = ppo_ts[i]
        v_next = 0.0 if (i == n - 1 or t.done) else ppo_ts[i + 1].value
        mask = 0.0 if t.done else 1.0
        delta = t.reward + gamma * v_next * mask - t.value
        adv = delta + gamma * lam * adv * mask
        t.adv = adv
        t.ret = adv + t.value


def _latest_checkpoint(ckpt_dir: str) -> str:
    import glob, re
    paths = glob.glob(os.path.join(ckpt_dir, "ckpt_ep*.pt"))
    if not paths:
        return ""
    def _epnum(p: str) -> int:
        m = re.search(r"ckpt_ep(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(paths, key=_epnum)



def rollout_ppo_gnn(env: MolEnv, actor: GNNActor, critic: GNNCritic, device: torch.device,
                    cfg: Optional[Dict] = None) -> Tuple[Episode, List[PPOTransition]]:
    cfg = cfg or DEFAULT_CFG
    state, goal, valid_actions = env.reset()
    ep = Episode()
    ppo_ts: List[PPOTransition] = []

    for _ in range(env.max_steps):
        mol_graph = deepcopy(env._mol)
        af_np  = env.get_action_features(valid_actions)
        action_site_ix_np = env.get_action_site_indices(valid_actions)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=device)
        af_t   = torch.tensor(af_np, dtype=torch.float32, device=device)
        action_site_ix_t = torch.tensor(action_site_ix_np, dtype=torch.long, device=device)

        with torch.no_grad():
            dist       = actor.action_dist(mol_graph, goal_t, af_t, device, action_site_ix_t)
            action_idx = int(dist.sample().item())
            old_lp     = float(dist.log_prob(torch.tensor(action_idx, device=device)).item())
            value      = float(critic(mol_graph, goal_t, device).reshape(()).item())

        result: StepResult = env.step(valid_actions[action_idx])
        next_valid = result.info.get("valid_actions", [TERMINATE])
        next_af_np = env.get_action_features(next_valid)

        ep.add(Transition(
            state=state, action_feats=af_np, action_idx=action_idx,
            reward=result.reward, next_state=result.state,
            next_action_feats=next_af_np, done=result.done, goal=goal,
            info={
                **result.info,
                "mol_graph": mol_graph,
                "action_site_indices": action_site_ix_np.copy(),
            },
        ))
        ppo_ts.append(PPOTransition(
            state=state, action_feats=af_np, action_idx=action_idx,
            old_log_prob=old_lp, reward=result.reward, done=result.done,
            goal=goal, mol_graph=mol_graph, action_site_indices=action_site_ix_np.copy(), value=value,
        ))

        if result.done:
            ep.achieved_goal = result.info.get("achieved_goal")
            break
        state         = result.state
        valid_actions = next_valid

    if cfg.get("use_gae"):
        _gae(ppo_ts, cfg["gamma"], cfg["gae_lambda"])
    else:
        G = 0.0
        for t in reversed(ppo_ts):
            G = t.reward + cfg["gamma"] * G * (0.0 if t.done else 1.0)
            t.ret = G
    return ep, ppo_ts


def her_transitions_gnn(
    ep: Episode, her_k: int, gamma: float, reward_cfg: Optional[Dict] = None,
) -> List[PPOTransition]:
    achieved = ep.achieved_goal
    if achieved is None:
        return []
    reward_cfg = reward_cfg or {"mode": "sparse"}
    out: List[PPOTransition] = []
    for _ in range(her_k):
        relabelled = []
        for t in ep.transitions:
            mg         = t.info.get("mol_graph")
            action_site_ix = t.info.get("action_site_indices")
            reward_ctx = t.info.get("reward_ctx")
            new_r = (
                reward_from_context(reward_ctx, achieved, reward_cfg)
                if reward_ctx is not None else
                (0.0 if t.done else t.reward)
            )
            relabelled.append((mg, t.action_feats, t.action_idx, action_site_ix, new_r, t.done))

        G = 0.0
        rets: List[float] = []
        for _, _, _, _, r, done in reversed(relabelled):
            G = r + gamma * G * (0.0 if done else 1.0)
            rets.insert(0, G)

        for (mg, af, aidx, action_site_ix, r, done), ret in zip(relabelled, rets):
            out.append(PPOTransition(
                state=np.empty(0), action_feats=af, action_idx=aidx,
                old_log_prob=0.0,
                reward=r, done=done, goal=achieved, ret=ret, mol_graph=mg,
                action_site_indices=None if action_site_ix is None else action_site_ix.copy(),
            ))
    return out


def update_ppo_gnn(
    actor: GNNActor, critic: GNNCritic,
    opt_actor: optim.Optimizer, opt_critic: optim.Optimizer,
    batch: List[PPOTransition], cfg: Dict, device: torch.device,
) -> Dict[str, float]:
    if not batch:
        return {}

    with torch.no_grad():
        Vs = []
        for t in batch:
            goal_t = torch.tensor(t.goal, dtype=torch.float32, device=device)
            Vs.append(critic(t.mol_graph, goal_t, device).reshape(()))
        Vs_all  = torch.stack(Vs)
        Gs_all  = torch.tensor([t.ret for t in batch], dtype=torch.float32, device=device)
        targets_all = _norm_targets(Gs_all, cfg)
        gae_adv = torch.tensor([(t.adv if t.adv is not None else float("nan")) for t in batch],
                               dtype=torch.float32, device=device)
        adv_raw = torch.where(torch.isnan(gae_adv), targets_all - Vs_all, gae_adv)
        adv_all = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    actor_losses_log:  List[float] = []
    critic_losses_log: List[float] = []
    entropies_log:     List[float] = []

    for _ in range(cfg["ppo_epochs"]):
        idx_perm = torch.randperm(len(batch))
        for start in range(0, len(batch), cfg["batch_size"]):
            mb_idx = idx_perm[start: start + cfg["batch_size"]]
            actor_losses:  List[torch.Tensor] = []
            critic_losses: List[torch.Tensor] = []

            for i in mb_idx.tolist():
                t      = batch[i]
                adv    = adv_all[i]
                goal_t = torch.tensor(t.goal, dtype=torch.float32, device=device)
                af_t   = torch.tensor(t.action_feats, dtype=torch.float32, device=device)
                action_site_ix_t = (
                    torch.tensor(t.action_site_indices, dtype=torch.long, device=device)
                    if t.action_site_indices is not None else
                    None
                )
                G_t    = targets_all[i]

                dist    = actor.action_dist(t.mol_graph, goal_t, af_t, device, action_site_ix_t)
                new_lp  = dist.log_prob(torch.tensor(t.action_idx, device=device))
                entropy = dist.entropy()

                ratio  = torch.exp(new_lp - t.old_log_prob)
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * adv
                a_loss = -torch.min(surr1, surr2) - cfg["entropy_coef"] * entropy

                V_s    = critic(t.mol_graph, goal_t, device).reshape(())
                c_loss = cfg["critic_coef"] * (G_t - V_s).pow(2)

                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                entropies_log.append(float(entropy.item()))

            total_actor  = torch.stack(actor_losses).mean()
            total_critic = torch.stack(critic_losses).mean()

            opt_actor.zero_grad()
            total_actor.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg["grad_clip"])
            opt_actor.step()

            opt_critic.zero_grad()
            total_critic.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg["grad_clip"])
            opt_critic.step()

            actor_losses_log.append(float(total_actor.item()))
            critic_losses_log.append(float(total_critic.item()))

    return {
        "actor_loss":  float(np.mean(actor_losses_log)),
        "critic_loss": float(np.mean(critic_losses_log)),
        "entropy":     float(np.mean(entropies_log)),
    }



def train(cfg: Optional[Dict] = None) -> Tuple[Actor, Critic]:
    if cfg is None:
        cfg = DEFAULT_CFG

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg.get("site_actions") and cfg["encoder"] != "gnn":
        raise ValueError("--site_actions currently requires --encoder gnn")
    if cfg.get("frag_site_emb") and not cfg.get("site_actions"):
        raise ValueError("--frag_site_emb requires --site_actions")

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.init(
            project="cs224r-fragment-assembly",
            name=cfg["exp_name"],
            config={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))},
            tags=["ppo"],
        )

    device = torch.device(cfg["device"])

    print("Loading fragment library ...")
    frags = load_fragment_library(
        cfg["fragments_parquet"], n=cfg["n_frags"], min_count=cfg["min_frag_count"]
    )
    print(f"  {len(frags)} fragments")

    print("Loading target distribution ...")
    property_names = parse_property_names(cfg["goal_properties"])
    targets = load_target_distribution(
        cfg["parents_parquet"],
        n=cfg["n_targets"],
        property_names=property_names,
    )
    print(f"  {len(targets)} targets")

    reward_cfg = build_env_reward_config(cfg)
    env    = MolEnv(
        frags,
        targets,
        max_steps=cfg["max_steps"],
        property_names=property_names,
        reward_config=reward_cfg,
        site_actions=cfg.get("site_actions", False),
        counterfactual_actions=cfg.get("counterfactual_actions", False),
        fast_props=cfg.get("fast_props", False),
    )
    if cfg["encoder"] == "gnn":
        actor  = GNNActor(gnn_dim=cfg["gnn_dim"], goal_dim=env.goal_dim,
                          action_feat_dim=env.action_feat_dim,
                          hidden_dim=cfg["hidden_dim"],
                          site_actions=cfg.get("site_actions", False),
                          frag_site_emb=cfg.get("frag_site_emb", False),
                          num_frag_sites=getattr(env, "num_frag_sites", 0)).to(device)
        critic = GNNCritic(gnn_dim=cfg["gnn_dim"], goal_dim=env.goal_dim,
                           hidden_dim=cfg["hidden_dim"],
                           site_actions=cfg.get("site_actions", False)).to(device)
        if cfg.get("frag_site_emb"):
            actor.refresh_frag_cache(env, device)
        _rollout, _her, _update = rollout_ppo_gnn, her_transitions_gnn, update_ppo_gnn
    else:
        actor  = Actor(state_dim=env.state_dim, action_feat_dim=env.action_feat_dim,
                       hidden_dim=cfg["hidden_dim"]).to(device)
        critic = Critic(state_dim=env.state_dim, hidden_dim=cfg["hidden_dim"]).to(device)
        _rollout, _her, _update = rollout_ppo, her_transitions, update_ppo

    opt_actor  = optim.Adam(actor.parameters(),  lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    if cfg.get("normalize"):
        cfg["_ret_rms"] = RunningMeanStd()

    start_ep = 0
    resume_path = cfg.get("resume", "") or ""
    if resume_path in ("auto", "latest"):
        resume_path = _latest_checkpoint(cfg["checkpoint_dir"])
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        if "opt_actor" in ckpt:
            opt_actor.load_state_dict(ckpt["opt_actor"])
        if "opt_critic" in ckpt:
            opt_critic.load_state_dict(ckpt["opt_critic"])
        start_ep = int(ckpt.get("episode", 0))
        print(f"  Resumed from {resume_path} at episode {start_ep}")

    ep_rewards: List[float] = []
    ep_dists:   List[float] = []
    ep_valid:   List[float] = []
    metrics:    Dict = {}

    print(f"\nTraining from episode {start_ep} to {cfg['n_episodes']} ...\n")

    ep_idx = start_ep
    while ep_idx < cfg["n_episodes"]:
        actor.train(); critic.train()

        batch: List[PPOTransition] = []
        for _ in range(cfg["rollout_eps"]):
            if ep_idx >= cfg["n_episodes"]:
                break

            ep, ppo_ts = _rollout(env, actor, critic, device, cfg)
            batch.extend(ppo_ts)
            batch.extend(_her(ep, cfg["her_k"], cfg["gamma"], reward_cfg))

            total_r = sum(t.reward for t in ep.transitions)
            ep_rewards.append(total_r)
            achieved = ep.achieved_goal
            ep_valid.append(1.0 if achieved is not None else 0.0)
            terminal_dist = ep.transitions[-1].info.get("terminal_distance") if ep.transitions else None
            if terminal_dist is not None:
                ep_dists.append(float(terminal_dist))

            ep_idx += 1

            if ep_idx % cfg["log_every"] == 0:
                w = cfg["log_every"]
                r_mean = float(np.mean(ep_rewards[-w:]))
                v_pct  = float(np.mean(ep_valid[-w:])) * 100.0
                d_mean = float(np.mean(ep_dists[-w:])) if ep_dists else float("nan")
                loss_s = ""
                if metrics:
                    loss_s = (f"  actor={metrics['actor_loss']:.4f}"
                              f"  critic={metrics['critic_loss']:.4f}"
                              f"  H={metrics['entropy']:.3f}")
                print(f"[{ep_idx:5d}] reward={r_mean:+.3f}  "
                      f"dist={d_mean:.3f}  valid={v_pct:.1f}%{loss_s}")
                if cfg.get("use_wandb") and _wandb is not None:
                    log = {
                        "train/mean_reward": r_mean,
                        "train/mean_dist":   d_mean,
                        "train/valid_pct":   v_pct,
                    }
                    if metrics:
                        log["train/actor_loss"]  = metrics["actor_loss"]
                        log["train/critic_loss"] = metrics["critic_loss"]
                        log["train/entropy"]     = metrics["entropy"]
                    _wandb.log(log, step=ep_idx)

            if ep_idx % cfg["checkpoint_every"] == 0:
                path = os.path.join(cfg["checkpoint_dir"], f"ckpt_ep{ep_idx}.pt")
                torch.save({
                    "episode": ep_idx,
                    "actor":   actor.state_dict(),
                    "critic":  critic.state_dict(),
                    "opt_actor":  opt_actor.state_dict(),
                    "opt_critic": opt_critic.state_dict(),
                    "config":  cfg,
                }, path)
                print(f"  -> checkpoint: {path}")

        metrics = _update(actor, critic, opt_actor, opt_critic, batch, cfg, device)

        if cfg.get("frag_site_emb"):
            actor.refresh_frag_cache(env, device)

    if cfg.get("use_wandb") and _wandb is not None:
        _wandb.finish()

    return actor, critic



def _parse_args() -> Dict:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes",      type=int,   default=DEFAULT_CFG["n_episodes"])
    p.add_argument("--rollout_eps",     type=int,   default=DEFAULT_CFG["rollout_eps"])
    p.add_argument("--ppo_epochs",      type=int,   default=DEFAULT_CFG["ppo_epochs"])
    p.add_argument("--clip_eps",        type=float, default=DEFAULT_CFG["clip_eps"])
    p.add_argument("--batch_size",      type=int,   default=DEFAULT_CFG["batch_size"])
    p.add_argument("--n_frags",         type=int,   default=DEFAULT_CFG["n_frags"])
    p.add_argument("--n_targets",       type=int,   default=DEFAULT_CFG["n_targets"])
    p.add_argument("--goal_properties", type=str,   default=DEFAULT_CFG["goal_properties"])
    p.add_argument("--reward_properties", type=str, default=DEFAULT_CFG["reward_properties"])
    p.add_argument("--reward_mode", choices=["baseline", "distance_delta", "potential", "potential_based"], default=DEFAULT_CFG["reward_mode"])
    p.add_argument("--distance_scale", type=float, default=DEFAULT_CFG["distance_scale"])
    p.add_argument("--shaping_gamma", type=float, default=DEFAULT_CFG["shaping_gamma"])
    p.add_argument("--normalize", action="store_true", default=DEFAULT_CFG["normalize"])
    p.add_argument("--use_gae", action="store_true", default=DEFAULT_CFG["use_gae"])
    p.add_argument("--gae_lambda", type=float, default=DEFAULT_CFG["gae_lambda"])
    p.add_argument("--gamma", type=float, default=DEFAULT_CFG["gamma"])
    p.add_argument("--encoder", choices=["fp", "gnn"], default=DEFAULT_CFG["encoder"])
    p.add_argument("--site_actions", action="store_true", default=DEFAULT_CFG.get("site_actions", False))
    p.add_argument("--frag_site_emb", action="store_true", default=DEFAULT_CFG.get("frag_site_emb", False))
    p.add_argument("--counterfactual_actions", action="store_true", default=DEFAULT_CFG.get("counterfactual_actions", False))
    p.add_argument("--fast_props", action="store_true", default=DEFAULT_CFG.get("fast_props", False))
    p.add_argument("--refresh_every", type=int, default=DEFAULT_CFG.get("refresh_every", 50))
    p.add_argument("--gnn_dim", type=int, default=DEFAULT_CFG["gnn_dim"])
    p.add_argument("--hidden_dim",      type=int,   default=DEFAULT_CFG["hidden_dim"])
    p.add_argument("--her_k",           type=int,   default=DEFAULT_CFG["her_k"])
    p.add_argument("--entropy_coef",    type=float, default=DEFAULT_CFG["entropy_coef"])
    p.add_argument("--use_property_surrogate_reward", action="store_true", default=False)
    p.add_argument("--property_surrogate_scale", type=float, default=DEFAULT_CFG["property_surrogate_scale"])
    p.add_argument("--property_surrogate_dummy_bonus", type=float, default=DEFAULT_CFG["property_surrogate_dummy_bonus"])
    p.add_argument("--property_surrogate_step_penalty", type=float, default=DEFAULT_CFG["property_surrogate_step_penalty"])
    p.add_argument(
        "--property_surrogate_weights",
        type=str,
        default=DEFAULT_CFG["property_surrogate_weights"],
    )
    p.add_argument(
        "--property_surrogate_temperatures",
        type=str,
        default=DEFAULT_CFG["property_surrogate_temperatures"],
    )
    p.add_argument("--property_surrogate_invalid_score", type=float, default=DEFAULT_CFG["property_surrogate_invalid_score"])
    p.add_argument("--device",          type=str,   default=DEFAULT_CFG["device"])
    p.add_argument("--checkpoint_dir",  type=str,   default=DEFAULT_CFG["checkpoint_dir"])
    p.add_argument("--resume", type=str, default=DEFAULT_CFG["resume"],
                   help="path to a ckpt_ep*.pt, or 'auto' to pick the latest in checkpoint_dir")
    p.add_argument("--log_every",       type=int,   default=DEFAULT_CFG["log_every"])
    p.add_argument("--checkpoint_every",type=int,   default=DEFAULT_CFG["checkpoint_every"])
    p.add_argument("--exp_name",   type=str,            default=DEFAULT_CFG["exp_name"])
    p.add_argument("--seed",       type=int,            default=DEFAULT_CFG["seed"])
    p.add_argument("--use_wandb",  action="store_true", default=False)
    p.add_argument("--fragments_parquet", default=DEFAULT_CFG["fragments_parquet"])
    p.add_argument("--parents_parquet", default=DEFAULT_CFG["parents_parquet"])
    p.add_argument("--min_frag_count", type=int, default=DEFAULT_CFG["min_frag_count"])
    p.add_argument("--max_steps", type=int, default=DEFAULT_CFG["max_steps"])
    args = p.parse_args()
    cfg  = dict(DEFAULT_CFG)
    cfg.update(vars(args))
    if cfg["property_surrogate_weights"]:
        reward_names = parse_property_names(
            cfg["reward_properties"],
            default=parse_property_names(cfg["goal_properties"]),
        )
        cfg["property_surrogate_weights"] = parse_reward_vector(
            cfg["property_surrogate_weights"], expected_dim=len(reward_names)
        )
    if cfg["property_surrogate_temperatures"]:
        reward_names = parse_property_names(
            cfg["reward_properties"],
            default=parse_property_names(cfg["goal_properties"]),
        )
        cfg["property_surrogate_temperatures"] = parse_reward_vector(
            cfg["property_surrogate_temperatures"], expected_dim=len(reward_names)
        )
    return cfg


if __name__ == "__main__":
    train(_parse_args())
