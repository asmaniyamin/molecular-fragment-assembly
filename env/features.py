# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap

from typing import Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from core.structs import MolGraph
from chem.build.molgraph_to_mol import molgraph_to_mol
from .data import GOAL_DIM
from .properties import (
    DEFAULT_PROPERTY_NAMES,
    compute_norm_properties as compute_named_norm_properties,
    compute_raw_properties as compute_named_raw_properties,
)

FP_BITS = 512
BRICS_DIM = 17
ACTION_FEAT_DIM = FP_BITS + 2 * BRICS_DIM
STATE_DIM = FP_BITS + GOAL_DIM


def state_dim(goal_dim: int) -> int:
    return FP_BITS + goal_dim



def _mol_to_fp(mol: Chem.Mol) -> np.ndarray:
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(FP_BITS, dtype=np.float32)


def molgraph_to_fp(mg: MolGraph) -> np.ndarray:
    if mg is None or mg.arrays is None or mg.arrays.atomic_num.shape[0] == 0:
        return np.zeros(FP_BITS, dtype=np.float32)
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return _fallback_count_fp(mg)
    return _mol_to_fp(mol)


def _fallback_count_fp(mg: MolGraph) -> np.ndarray:
    feat = np.zeros(FP_BITS, dtype=np.float32)
    z = mg.arrays.atomic_num
    for i, elem in enumerate([6, 7, 8, 9, 15, 16, 17, 35, 53]):
        feat[i] = float(np.sum(z == elem))
    feat[9] = float(z.shape[0])
    return feat


def smiles_to_fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return np.zeros(FP_BITS, dtype=np.float32)
    return _mol_to_fp(mol)



def brics_onehot(label: str) -> np.ndarray:
    v = np.zeros(BRICS_DIM, dtype=np.float32)
    try:
        idx = int(label)
        if 0 <= idx < BRICS_DIM:
            v[idx] = 1.0
    except (ValueError, TypeError):
        pass
    return v



def state_features(mg: MolGraph, goal_norm: np.ndarray) -> np.ndarray:
    fp = molgraph_to_fp(mg)
    return np.concatenate([fp, goal_norm.astype(np.float32)])


def action_features(frag_fp: np.ndarray, frag_label: str, mol_label: str) -> np.ndarray:
    return np.concatenate([
        frag_fp,
        brics_onehot(frag_label),
        brics_onehot(mol_label),
    ])



def compute_raw_properties(
    mol: Chem.Mol,
    property_names: Sequence[str] = DEFAULT_PROPERTY_NAMES,
) -> Optional[np.ndarray]:
    return compute_named_raw_properties(mol, property_names)


def compute_norm_properties(
    mol: Chem.Mol,
    property_names: Sequence[str] = DEFAULT_PROPERTY_NAMES,
) -> Optional[np.ndarray]:
    return compute_named_norm_properties(mol, property_names)
