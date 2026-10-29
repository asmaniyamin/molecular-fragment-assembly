# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.QED import qed as _rdkit_qed


@dataclass(frozen=True)
class PropertySpec:
    name: str
    evaluator: Callable[[Chem.Mol], float]
    min_value: float
    max_value: float


def _formal_charge(mol: Chem.Mol) -> float:
    return float(Chem.GetFormalCharge(mol))


PROPERTY_SPECS = {
    "sLogP": PropertySpec("sLogP", Descriptors.MolLogP, -5.0, 10.0),
    "QED": PropertySpec("QED", _rdkit_qed, 0.0, 1.0),
    "TPSA": PropertySpec("TPSA", Descriptors.TPSA, 0.0, 200.0),
    "MW": PropertySpec("MW", Descriptors.MolWt, 0.0, 1000.0),
    "HBD": PropertySpec("HBD", Lipinski.NumHDonors, 0.0, 12.0),
    "HBA": PropertySpec("HBA", Lipinski.NumHAcceptors, 0.0, 20.0),
    "RotB": PropertySpec("RotB", Lipinski.NumRotatableBonds, 0.0, 20.0),
    "RingCount": PropertySpec("RingCount", Lipinski.RingCount, 0.0, 12.0),
    "HeavyAtomCount": PropertySpec("HeavyAtomCount", Lipinski.HeavyAtomCount, 0.0, 80.0),
    "FractionCSP3": PropertySpec("FractionCSP3", Lipinski.FractionCSP3, 0.0, 1.0),
    "FormalCharge": PropertySpec("FormalCharge", _formal_charge, -5.0, 5.0),
}

_PROPERTY_ALIASES = {name.lower(): name for name in PROPERTY_SPECS}

DEFAULT_PROPERTY_NAMES: Tuple[str, ...] = ("sLogP", "QED", "TPSA")


def parse_property_names(
    value: Optional[str | Sequence[str]],
    default: Sequence[str] = DEFAULT_PROPERTY_NAMES,
) -> Tuple[str, ...]:
    if value is None:
        names = tuple(default)
    elif isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
        if not names:
            names = tuple(default)
    else:
        names = tuple(str(part).strip() for part in value if str(part).strip())
        if not names:
            names = tuple(default)

    canonical: list[str] = []
    for name in names:
        resolved = _PROPERTY_ALIASES.get(name.lower())
        if resolved is None:
            choices = ", ".join(sorted(PROPERTY_SPECS))
            raise ValueError(f"Unknown property {name!r}. Available: {choices}")
        canonical.append(resolved)
    return tuple(canonical)


def property_bounds(property_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    names = parse_property_names(property_names)
    mins = np.array([PROPERTY_SPECS[name].min_value for name in names], dtype=np.float32)
    maxs = np.array([PROPERTY_SPECS[name].max_value for name in names], dtype=np.float32)
    return mins, maxs


def normalize_props(raw: np.ndarray, property_names: Sequence[str] = DEFAULT_PROPERTY_NAMES) -> np.ndarray:
    mins, maxs = property_bounds(property_names)
    return (np.clip(raw, mins, maxs) - mins) / np.maximum(maxs - mins, 1e-6)


def denormalize_props(normed: np.ndarray, property_names: Sequence[str] = DEFAULT_PROPERTY_NAMES) -> np.ndarray:
    mins, maxs = property_bounds(property_names)
    return normed * (maxs - mins) + mins


def compute_raw_properties(
    mol: Chem.Mol,
    property_names: Sequence[str] = DEFAULT_PROPERTY_NAMES,
) -> Optional[np.ndarray]:
    names = parse_property_names(property_names)
    try:
        values = [float(PROPERTY_SPECS[name].evaluator(mol)) for name in names]
        return np.asarray(values, dtype=np.float32)
    except Exception:
        return None


def compute_norm_properties(
    mol: Chem.Mol,
    property_names: Sequence[str] = DEFAULT_PROPERTY_NAMES,
) -> Optional[np.ndarray]:
    raw = compute_raw_properties(mol, property_names)
    return None if raw is None else normalize_props(raw, property_names)


def property_indices(
    full_property_names: Sequence[str],
    selected_property_names: Sequence[str],
) -> Tuple[int, ...]:
    full = parse_property_names(full_property_names)
    selected = parse_property_names(selected_property_names, default=full)
    index = {name: i for i, name in enumerate(full)}
    missing = [name for name in selected if name not in index]
    if missing:
        raise ValueError(
            f"reward_properties must be a subset of goal_properties. Missing: {missing}"
        )
    return tuple(index[name] for name in selected)


def vector_for_indices(values: Optional[Sequence[float]], indices: Iterable[int]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return arr[list(indices)].astype(np.float32, copy=False)
