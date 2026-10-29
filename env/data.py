# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap

import heapq
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import psutil
import pyarrow.parquet as pq
from rdkit import Chem

from .properties import (
    DEFAULT_PROPERTY_NAMES,
    compute_raw_properties,
    denormalize_props,
    normalize_props,
    parse_property_names,
    property_bounds,
)

PROP_NAMES = DEFAULT_PROPERTY_NAMES
PROP_MIN, PROP_MAX = property_bounds(PROP_NAMES)
PROP_RANGE = PROP_MAX - PROP_MIN
GOAL_DIM = len(PROP_NAMES)



def _available_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _check_headroom(max_mem_gb: float, label: str) -> None:
    avail = _available_gb()
    if avail < 2.0:
        raise MemoryError(
            f"{label}: only {avail:.1f} GB RAM available; refusing to load more data."
        )


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except Exception:
        return False



@dataclass
class FragInfo:
    smiles: str
    labels: List[str]
    n_attach: int
    count: int


def load_fragment_library(
    parquet_path: str,
    n: int = 500,
    min_count: int = 1_000,
    max_mem_gb: float = 32.0,
) -> List[FragInfo]:
    COLS = ["frag_smiles", "count", "attach_labels", "n_attach"]

    if not Path(parquet_path).is_file():
        raise FileNotFoundError(f"Missing fragment data: {parquet_path}. Provide your fragment dataset with --fragments_parquet or MFA_DATA_DIR.")
    pf          = pq.ParquetFile(parquet_path)
    n_groups    = pf.metadata.num_row_groups
    used_start  = psutil.virtual_memory().used / 1e9

    heap:  list = []
    entry_idx   = 0

    for g in range(n_groups):
        used_now = psutil.virtual_memory().used / 1e9
        delta    = used_now - used_start
        if delta >= max_mem_gb:
            print(f"  [data] memory cap hit ({delta:.1f}/{max_mem_gb:.0f} GB used by "
                  f"this load); stopping after {g}/{n_groups} row groups.")
            break

        _check_headroom(max_mem_gb, "load_fragment_library")

        batch = pf.read_row_group(g, columns=COLS).to_pandas()

        batch = batch[(batch["n_attach"] >= 1) & (batch["count"] >= min_count)]

        for _, row in batch.iterrows():
            cnt = int(row["count"])
            if len(heap) < n:
                heapq.heappush(heap, (cnt, entry_idx, row.to_dict()))
                entry_idx += 1
            elif cnt > heap[0][0]:
                heapq.heapreplace(heap, (cnt, entry_idx, row.to_dict()))
                entry_idx += 1

    top = sorted(heap, key=lambda x: -x[0])

    frags: List[FragInfo] = []
    for _, _, row in top:
        raw_labels = str(row["attach_labels"])
        labels = [lbl.strip() for lbl in raw_labels.split(",") if lbl.strip()]
        frags.append(FragInfo(
            smiles=str(row["frag_smiles"]),
            labels=labels,
            n_attach=int(row["n_attach"]),
            count=int(row["count"]),
        ))

    if not frags:
        raise ValueError("No usable fragments remain. Check --min_frag_count and the fragment dataset.")
    print(f"  [data] fragment library: {len(frags)} fragments "
          f"(scanned {min(g+1, n_groups)}/{n_groups} row groups)")
    return frags



def load_target_distribution(
    parquet_path: str,
    n: int = 1_000,
    property_names: Optional[Sequence[str]] = None,
    max_mem_gb: float = 32.0,
) -> np.ndarray:
    property_names = parse_property_names(property_names)

    if not Path(parquet_path).is_file():
        raise FileNotFoundError(f"Missing target data: {parquet_path}. Provide your target dataset with --parents_parquet or MFA_DATA_DIR.")
    pf         = pq.ParquetFile(parquet_path)
    n_groups   = pf.metadata.num_row_groups
    used_start = psutil.virtual_memory().used / 1e9
    schema_names = set(pf.schema.names)

    available_props = [name for name in property_names if name in schema_names]
    missing_props = [name for name in property_names if name not in schema_names]
    smiles_col = next(
        (name for name in ("smiles", "canonical_smiles", "parent_smiles", "SMILES",
                           "parent_smiles_canon", "parent_smiles_mapped", "frag_smiles")
         if name in schema_names),
        None,
    )
    if missing_props and smiles_col is None:
        raise RuntimeError(
            f"Target load requested properties {missing_props}, but parquet has no matching columns "
            f"and no SMILES column for recomputation."
        )

    cols_to_read = list(available_props)
    if smiles_col is not None and missing_props:
        cols_to_read.append(smiles_col)

    rows: list = []
    for g in range(n_groups):
        if len(rows) >= n:
            break

        used_now = psutil.virtual_memory().used / 1e9
        delta    = used_now - used_start
        if delta >= max_mem_gb:
            print(f"  [data] memory cap hit ({delta:.1f}/{max_mem_gb:.0f} GB); "
                  f"stopping target load after {g}/{n_groups} row groups.")
            break

        _check_headroom(max_mem_gb, "load_target_distribution")

        batch = pf.read_row_group(g, columns=cols_to_read).to_pandas()

        if missing_props:
            prop_rows = []
            for _, row in batch.iterrows():
                values = {name: row[name] for name in available_props}
                smiles = row.get(smiles_col)
                if any(name not in values or _is_missing_scalar(values[name]) for name in available_props):
                    continue
                if missing_props:
                    if not isinstance(smiles, str) or not smiles:
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    raw_missing = compute_raw_properties(mol, missing_props)
                    if raw_missing is None:
                        continue
                    for name, value in zip(missing_props, raw_missing):
                        values[name] = value
                prop_rows.append([values[name] for name in property_names])
                if len(prop_rows) >= (n - sum(len(chunk) for chunk in rows)):
                    break
            if prop_rows:
                rows.append(np.asarray(prop_rows, dtype=np.float32))
        else:
            batch = batch[list(property_names)].dropna()
            need = n - sum(len(chunk) for chunk in rows)
            rows.append(batch.head(need).values.astype(np.float32))

    if not rows:
        raise RuntimeError("No target rows loaded — check parquet path and columns.")

    raw = np.concatenate(rows, axis=0)[:n]

    print(f"  [data] target distribution: {len(raw)} molecules "
          f"(scanned {min(g+1, n_groups)}/{n_groups} row groups)")
    return normalize_props(raw, property_names)


def sample_target(targets: np.ndarray) -> np.ndarray:
    return targets[np.random.randint(len(targets))].copy()
