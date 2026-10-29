# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem

from core.structs import MolGraph, BondType
from chem.build.create_molgraph import smiles_to_molgraph
from chem.build.molgraph_to_mol import molgraph_to_mol
from chem.merge.merge import merge_by_labels
from chem.dummy.query import dummy_indices

from .data import FragInfo, sample_target
from .features import (
    state_features, action_features, compute_norm_properties,
    smiles_to_fp, molgraph_to_fp, state_dim as feature_state_dim, ACTION_FEAT_DIM, FP_BITS,
)
from .properties import DEFAULT_PROPERTY_NAMES, parse_property_names
from .rewards import (
    OPEN_DUMMY_PENALTY,
    reward_from_context,
    terminal_reward,
)



_BRICS_RE = re.compile(r"\[(\d+)\*\]")


def _normalise_brics_smiles(smiles: str) -> str:
    return _BRICS_RE.sub(r"[*:\1]", smiles)



BRICS_COMPAT: Dict[int, Set[int]] = {
    1:  {3, 5, 10},
    3:  {1, 4, 13, 14, 15, 16},
    4:  {3, 5, 11},
    5:  {1, 4, 12, 13, 14, 15, 16},
    6:  {13, 14, 15, 16},
    7:  {7},
    8:  {9, 10, 13, 14, 15, 16},
    9:  {8, 13, 14, 15, 16},
    10: {1, 8, 13, 14, 15, 16},
    11: {4, 13, 14, 15, 16},
    12: {5},
    13: {3, 5, 6, 8, 9, 10, 11, 14, 15, 16},
    14: {3, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16},
    15: {3, 5, 6, 8, 9, 10, 11, 13, 14, 16},
    16: {3, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16},
}

def _brics_compatible(mol_label: str, frag_label: str) -> bool:
    try:
        mt = int(mol_label)
        ft = int(frag_label)
    except (ValueError, TypeError):
        return False
    return ft in BRICS_COMPAT.get(mt, set())



def _cap_dummies_with_h(mol: Chem.Mol) -> Optional[Chem.Mol]:
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



@dataclass(frozen=True)
class Action:
    is_terminate: bool
    frag_idx: int = -1
    mol_label: str = ""
    frag_label: str = ""
    mol_dummy_idx: int = -1
    frag_dummy_idx: int = -1


TERMINATE = Action(is_terminate=True)



@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)



class MolEnv:
    def __init__(
        self,
        fragment_library: List[FragInfo],
        target_distribution: np.ndarray,
        max_steps: int = 8,
        property_names: Optional[Tuple[str, ...]] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        site_actions: bool = False,
        counterfactual_actions: bool = False,
        fast_props: bool = False,
    ):
        self.library = fragment_library
        self.targets = target_distribution
        self.max_steps = max_steps
        self.property_names = parse_property_names(property_names, default=DEFAULT_PROPERTY_NAMES)
        self.goal_dim = len(self.property_names)
        if target_distribution.shape[1] != self.goal_dim:
            raise ValueError(
                f"Target distribution has dim {target_distribution.shape[1]}, "
                f"but property_names imply goal_dim {self.goal_dim}."
            )
        self.reward_config = reward_config or {"mode": "sparse"}
        self.site_actions = site_actions
        self.counterfactual_actions = counterfactual_actions

        self.frag_fps: List[np.ndarray] = [
            smiles_to_fp(_normalise_brics_smiles(f.smiles)) for f in fragment_library
        ]
        self._frag_graphs: Dict[int, Optional[MolGraph]] = {}
        self._frag_sites_cache: Dict[int, List[Tuple[int, str]]] = {}

        self._mol:  Optional[MolGraph]    = None
        self._goal: Optional[np.ndarray] = None
        self._step: int                   = 0

        self._seed_frag_idx: int = -1
        self._used_frags: List[int] = []
        self._bond_pairs: List[Tuple[str, str]] = []
        self.fast_props = fast_props
        self._prop_model = None
        self._base_acc: Optional[np.ndarray] = None
        if fast_props:
            from .fast_properties import AdditivePropertyModel
            self._prop_model = AdditivePropertyModel(self.property_names).fit_or_load(self)

        self.num_frag_sites = 0
        self._frag_site_id: Dict[Tuple[int, int], int] = {}
        if site_actions:
            next_id = 1
            for fi in range(len(self.library)):
                for frag_dummy_idx, _lab in self._frag_sites(fi):
                    self._frag_site_id[(fi, int(frag_dummy_idx))] = next_id
                    next_id += 1
            self.num_frag_sites = next_id - 1


    def reset(self) -> Tuple[np.ndarray, np.ndarray, List[Action]]:
        for _ in range(len(self.library)):
            idx = np.random.randint(len(self.library))
            mg  = self._frag_graph(idx)
            if mg is not None and dummy_indices(mg).size > 0:
                break
        else:
            raise RuntimeError("No valid seed fragment found in library.")

        self._mol  = mg
        self._goal = sample_target(self.targets)
        self._step = 0

        self._seed_frag_idx = int(idx)
        self._used_frags = [int(idx)]
        self._bond_pairs = []
        if self.fast_props:
            self._base_acc = self._prop_model.frag_accum(int(idx)).copy()

        state = state_features(self._mol, self._goal)
        valid = self._valid_actions()
        return state, self._goal.copy(), valid

    def step(self, action: Action) -> StepResult:
        self._step += 1
        prev_open = len(self._open_labels())
        prev_props = self._current_surrogate_props()

        if action.is_terminate or self._step >= self.max_steps:
            return self._finalise(prev_props=prev_props, prev_open=prev_open)

        merged, next_props, next_open, ok, err = self._simulate_action(action)
        if not ok or merged is None:
            return self._soft_fail(err or "action_simulation_failed")

        self._mol = merged
        self._used_frags.append(int(action.frag_idx))
        self._bond_pairs.append((action.mol_label, action.frag_label))
        if self.fast_props:
            self._base_acc = self._base_acc \
                + self._prop_model.frag_accum(int(action.frag_idx)) \
                + self._prop_model.bond_correction(action.mol_label, action.frag_label)
            next_props = self._prop_model.properties_norm(self._base_acc)

        if next_open == 0:
            return self._finalise(prev_props=prev_props, prev_open=prev_open)

        state = state_features(self._mol, self._goal)
        valid = self._valid_actions()
        reward_ctx = {
            "kind": "intermediate",
            "base_reward": 0.0,
            "prev_props": None if prev_props is None else prev_props.copy(),
            "next_props": None if next_props is None else next_props.copy(),
            "prev_open": prev_open,
            "next_open": next_open,
        }
        reward = reward_from_context(reward_ctx, self._goal, self.reward_config)
        return StepResult(
            state=state,
            reward=reward,
            done=False,
            info={
                "valid_actions": valid,
                "surrogate_props": None if next_props is None else next_props.copy(),
                "reward_ctx": reward_ctx,
            },
        )


    def _open_labels(self) -> List[str]:
        return [label for _, label in self._open_sites()]

    def _open_sites(self) -> List[Tuple[int, str]]:
        return self._open_sites_for_graph(self._mol)

    def _open_sites_for_graph(self, graph: Optional[MolGraph]) -> List[Tuple[int, str]]:
        g = graph
        if g is None:
            return []
        d_idx = dummy_indices(g)
        if d_idx.size == 0:
            return []
        lab_arr = g.arrays.attachment_label
        sites: List[Tuple[int, str]] = []
        for i in d_idx:
            ii = int(i)
            lbl = lab_arr[ii] if lab_arr is not None else None
            sites.append((ii, str(lbl) if lbl is not None else "1"))
        return sites

    def _frag_sites(self, frag_idx: int) -> List[Tuple[int, str]]:
        if frag_idx in self._frag_sites_cache:
            return self._frag_sites_cache[frag_idx]

        mg = self._frag_graph(frag_idx)
        if mg is None:
            self._frag_sites_cache[frag_idx] = []
            return []

        d_idx = dummy_indices(mg)
        lab_arr = mg.arrays.attachment_label
        sites: List[Tuple[int, str]] = []
        for i in d_idx:
            ii = int(i)
            lbl = lab_arr[ii] if lab_arr is not None else None
            sites.append((ii, str(lbl) if lbl is not None else "1"))
        self._frag_sites_cache[frag_idx] = sites
        return sites

    def _valid_actions(self) -> List[Action]:
        actions: List[Action] = [TERMINATE]
        mol_sites = self._open_sites()
        if not mol_sites:
            return actions

        if self.site_actions:
            for fi, _frag in enumerate(self.library):
                for frag_dummy_idx, flab in self._frag_sites(fi):
                    for mol_dummy_idx, mlab in mol_sites:
                        if not _brics_compatible(mlab, flab):
                            continue
                        actions.append(Action(
                            is_terminate=False,
                            frag_idx=fi,
                            mol_label=mlab,
                            frag_label=flab,
                            mol_dummy_idx=mol_dummy_idx,
                            frag_dummy_idx=frag_dummy_idx,
                        ))
            return actions

        seen: set = set()
        mol_labels = [label for _, label in mol_sites]
        for fi, frag in enumerate(self.library):
            for flab in frag.labels:
                for mlab in mol_labels:
                    if not _brics_compatible(mlab, flab):
                        continue
                    key = (fi, flab, mlab)
                    if key not in seen:
                        seen.add(key)
                        actions.append(Action(
                            is_terminate=False,
                            frag_idx=fi,
                            mol_label=mlab,
                            frag_label=flab,
                        ))
        return actions

    def get_action_features(self, actions: List[Action]) -> np.ndarray:
        rows: List[np.ndarray] = []
        current_props = self._current_surrogate_props() if self.counterfactual_actions else None
        current_open = len(self._open_labels()) if self.counterfactual_actions else 0
        current_fp = molgraph_to_fp(self._mol) if self.counterfactual_actions else None
        for a in actions:
            if a.is_terminate:
                base = np.zeros(ACTION_FEAT_DIM, dtype=np.float32)
            else:
                base = action_features(
                    self.frag_fps[a.frag_idx], a.frag_label, a.mol_label
                )
            if not self.counterfactual_actions:
                rows.append(base)
                continue

            next_graph, next_props, next_open, ok, _ = self._simulate_action(a)
            next_fp = np.zeros(FP_BITS, dtype=np.float32)
            props_vec = np.zeros(self.goal_dim, dtype=np.float32)
            delta_vec = np.zeros(self.goal_dim, dtype=np.float32)
            valid_flag = 0.0
            done_flag = 1.0 if a.is_terminate else 0.0

            if a.is_terminate:
                if current_fp is not None:
                    next_fp = current_fp.astype(np.float32, copy=False)
                if current_props is not None:
                    props_vec = current_props.astype(np.float32, copy=False)
                    valid_flag = 1.0
            elif ok and next_graph is not None:
                next_fp = molgraph_to_fp(next_graph)
                if next_props is not None:
                    props_vec = next_props.astype(np.float32, copy=False)
                    valid_flag = 1.0
                    if current_props is not None:
                        delta_vec = (props_vec - current_props).astype(np.float32, copy=False)
                done_flag = 1.0 if next_open == 0 else 0.0

            scalars = np.array([
                float(next_open) / max(1.0, float(self.max_steps)),
                float(next_open - current_open) / max(1.0, float(self.max_steps)),
                valid_flag,
                done_flag,
            ], dtype=np.float32)
            rows.append(np.concatenate([base, next_fp, props_vec, delta_vec, scalars]))
        return np.stack(rows, axis=0)

    def get_action_site_indices(self, actions: List[Action]) -> np.ndarray:
        rows = []
        for a in actions:
            if a.is_terminate:
                rows.append((-1, 0))
            else:
                fid = self._frag_site_id.get((int(a.frag_idx), int(a.frag_dummy_idx)), 0)
                rows.append((int(a.mol_dummy_idx), fid))
        return np.array(rows, dtype=np.int64).reshape(-1, 2)

    def _relabel_selected_dummy(
        self,
        graph: MolGraph,
        dummy_idx: int,
        unique_label: str,
    ) -> MolGraph:
        relabelled = deepcopy(graph)
        labels = relabelled.arrays.attachment_label
        if labels is None:
            raise ValueError("selected-site actions require attachment labels")
        labels_obj = labels.astype(object, copy=True)
        labels_obj[int(dummy_idx)] = unique_label
        relabelled.arrays.attachment_label = labels_obj
        return relabelled


    def _finalise(
        self,
        prev_props: Optional[np.ndarray] = None,
        prev_open: Optional[int] = None,
    ) -> StepResult:
        n_open = len(self._open_labels())

        if self.fast_props and self._base_acc is not None:
            props_norm = self._prop_model.properties_norm(self._base_acc)
            state = state_features(self._mol, self._goal)
            reward_ctx = {
                "kind": "terminal",
                "prev_props": None if prev_props is None else prev_props.copy(),
                "final_props": props_norm.copy(),
                "prev_open": n_open if prev_open is None else prev_open,
                "final_open": n_open,
                "open_dummy_penalty": OPEN_DUMMY_PENALTY,
            }
            reward = reward_from_context(reward_ctx, self._goal, self.reward_config)
            _, dist = terminal_reward(props_norm, self._goal, n_open, OPEN_DUMMY_PENALTY)
            return StepResult(
                state=state, reward=reward, done=True,
                info={
                    "valid": True,
                    "achieved_goal": props_norm.copy(),
                    "n_open_dummies": n_open,
                    "valid_actions": [TERMINATE],
                    "surrogate_props": props_norm.copy(),
                    "terminal_distance": dist,
                    "reward_ctx": reward_ctx,
                },
            )

        mol = molgraph_to_mol(self._mol, sanitize=False, remove_hs=False)

        closed = _cap_dummies_with_h(mol) if mol is not None else None

        if closed is None and mol is not None:
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
                closed = mol
            except Exception:
                pass

        props_norm = (
            compute_norm_properties(closed, self.property_names)
            if closed is not None else None
        )
        state = state_features(self._mol, self._goal)

        reward_ctx = {
            "kind": "terminal",
            "prev_props": None if prev_props is None else prev_props.copy(),
            "final_props": None if props_norm is None else props_norm.copy(),
            "prev_open": n_open if prev_open is None else prev_open,
            "final_open": n_open,
            "open_dummy_penalty": OPEN_DUMMY_PENALTY,
        }
        reward = reward_from_context(reward_ctx, self._goal, self.reward_config)
        _, dist = terminal_reward(props_norm, self._goal, n_open, OPEN_DUMMY_PENALTY)

        return StepResult(
            state=state,
            reward=reward,
            done=True,
            info={
                "valid": props_norm is not None,
                "achieved_goal": None if props_norm is None else props_norm.copy(),
                "n_open_dummies": n_open,
                "valid_actions": [TERMINATE],
                "surrogate_props": None if props_norm is None else props_norm.copy(),
                "terminal_distance": dist,
                "reward_ctx": reward_ctx,
            },
        )

    def _soft_fail(self, reason: str) -> StepResult:
        state = state_features(self._mol, self._goal)
        valid = self._valid_actions()
        reward_ctx = {"kind": "constant", "value": -0.1}
        return StepResult(
            state=state,
            reward=-0.1,
            done=False,
            info={"error": reason, "valid_actions": valid, "reward_ctx": reward_ctx},
        )

    def _current_surrogate_props(self) -> Optional[np.ndarray]:
        if self.fast_props and self._base_acc is not None:
            return self._prop_model.properties_norm(self._base_acc)
        return self._surrogate_props_for_graph(self._mol)

    def _surrogate_props_for_graph(self, graph: Optional[MolGraph]) -> Optional[np.ndarray]:
        mol = molgraph_to_mol(graph, sanitize=False, remove_hs=False)
        if mol is None:
            return None

        closed = _cap_dummies_with_h(mol)
        if closed is None:
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
                closed = mol
            except Exception:
                return None
        return compute_norm_properties(closed, self.property_names)

    def _simulate_action(
        self,
        action: Action,
    ) -> Tuple[Optional[MolGraph], Optional[np.ndarray], int, bool, Optional[str]]:
        if action.is_terminate:
            props = self._current_surrogate_props()
            return self._mol, props, len(self._open_labels()), True, None

        frag_mg = self._frag_graph(action.frag_idx)
        if frag_mg is None:
            return None, None, len(self._open_labels()), False, "fragment_load_failed"

        try:
            if self.site_actions and action.mol_dummy_idx >= 0 and action.frag_dummy_idx >= 0:
                mol_graph = self._relabel_selected_dummy(
                    self._mol,
                    action.mol_dummy_idx,
                    "__selected_mol_site__",
                )
                frag_graph = self._relabel_selected_dummy(
                    frag_mg,
                    action.frag_dummy_idx,
                    "__selected_frag_site__",
                )
                merged = merge_by_labels(
                    mol_graph,
                    frag_graph,
                    label_a="__selected_mol_site__",
                    label_b="__selected_frag_site__",
                    bond_type_code=BondType.SINGLE,
                    validate_with_rdkit=False,
                )
            else:
                merged = merge_by_labels(
                    self._mol,
                    frag_mg,
                    label_a=action.mol_label,
                    label_b=action.frag_label,
                    bond_type_code=BondType.SINGLE,
                    validate_with_rdkit=False,
                )
        except Exception as exc:
            return None, None, len(self._open_labels()), False, str(exc)

        next_open = len(self._open_sites_for_graph(merged))
        if self.fast_props and self._base_acc is not None:
            acc = self._base_acc \
                + self._prop_model.frag_accum(int(action.frag_idx)) \
                + self._prop_model.bond_correction(action.mol_label, action.frag_label)
            next_props = self._prop_model.properties_norm(acc)
        else:
            next_props = self._surrogate_props_for_graph(merged)
        return merged, next_props, next_open, True, None


    def _frag_graph(self, idx: int) -> Optional[MolGraph]:
        if idx not in self._frag_graphs:
            norm_smi = _normalise_brics_smiles(self.library[idx].smiles)
            self._frag_graphs[idx] = smiles_to_molgraph(norm_smi)
        return self._frag_graphs[idx]

    @property
    def state_dim(self) -> int:
        return feature_state_dim(self.goal_dim)

    @property
    def action_feat_dim(self) -> int:
        if not self.counterfactual_actions:
            return ACTION_FEAT_DIM
        return ACTION_FEAT_DIM + FP_BITS + 2 * self.goal_dim + 4
