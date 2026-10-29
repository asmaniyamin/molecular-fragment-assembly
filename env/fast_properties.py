# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from scripts import _path_bootstrap
from paths import ARTIFACTS_DIR

import hashlib
import os
from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED

from chem.build.molgraph_to_mol import molgraph_to_mol
from .properties import normalize_props, parse_property_names

_ACC = ["MW", "CRIPPEN", "TPSA", "HBD", "HBA", "ROTB", "RING", "HAC",
        "NCSP3", "NC", "FCHG", "QHBA", "QHBD", "QAROM", "QALERTS"]
_AIX = {n: i for i, n in enumerate(_ACC)}
_NACC = len(_ACC)

_NB = 17
_NPAIR = _NB * _NB

_QEDProps = namedtuple("QEDproperties", "MW ALOGP HBA HBD PSA ROTB AROM ALERTS")
_QEDWeights = np.asarray((0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95), dtype=np.float32)
_QEDAds = {
    "MW": (2.817065973, 392.5754953, 290.7489764, 2.419764353, 49.22325677, 65.37051707, 104.9805561),
    "ALOGP": (3.172690585, 137.8624751, 2.534937431, 4.581497897, 0.822739154, 0.576295591, 131.3186604),
    "HBA": (2.948620388, 160.4605972, 3.615294657, 4.435986202, 0.290141953, 1.300669958, 148.7763046),
    "HBD": (1.618662227, 1010.051101, 0.985094388, 0.000000001, 0.713820843, 0.920922555, 258.1632616),
    "PSA": (1.876861559, 125.2232657, 62.90773554, 87.83366614, 12.01999824, 28.51324732, 104.5686167),
    "ROTB": (0.010000000, 272.4121427, 2.558379970, 1.565547684, 1.271567166, 2.758063707, 105.4420403),
    "AROM": (3.217788970, 957.7374108, 2.274627939, 0.000000001, 1.317690384, 0.375760881, 312.3372610),
    "ALERTS": (0.010000000, 1199.094025, -0.09002883, 0.000000001, 0.185904477, 0.875193782, 417.7253140),
}

_CACHE_DIR = str(ARTIFACTS_DIR / "cafs_cache")


def _capped(mg) -> Optional[Chem.Mol]:
    from .env import _cap_dummies_with_h
    mol = molgraph_to_mol(mg, sanitize=False, remove_hs=False)
    if mol is None:
        return None
    return _cap_dummies_with_h(mol)


def _descriptor_vec(c: Chem.Mol) -> Optional[np.ndarray]:
    try:
        qp = QED.properties(c)
        nc = sum(1 for a in c.GetAtoms() if a.GetAtomicNum() == 6)
        ncsp3 = sum(1 for a in c.GetAtoms()
                    if a.GetAtomicNum() == 6
                    and a.GetHybridization() == Chem.HybridizationType.SP3)
        v = np.zeros(_NACC, dtype=np.float64)
        v[_AIX["MW"]] = Descriptors.MolWt(c)
        v[_AIX["CRIPPEN"]] = Descriptors.MolLogP(c)
        v[_AIX["TPSA"]] = Descriptors.TPSA(c)
        v[_AIX["HBD"]] = Lipinski.NumHDonors(c)
        v[_AIX["HBA"]] = Lipinski.NumHAcceptors(c)
        v[_AIX["ROTB"]] = Lipinski.NumRotatableBonds(c)
        v[_AIX["RING"]] = Lipinski.RingCount(c)
        v[_AIX["HAC"]] = Lipinski.HeavyAtomCount(c)
        v[_AIX["NCSP3"]] = ncsp3
        v[_AIX["NC"]] = nc
        v[_AIX["FCHG"]] = Chem.GetFormalCharge(c)
        v[_AIX["QHBA"]] = qp.HBA
        v[_AIX["QHBD"]] = qp.HBD
        v[_AIX["QAROM"]] = qp.AROM
        v[_AIX["QALERTS"]] = qp.ALERTS
        return v
    except Exception:
        return None


def _bondpair_index(mol_label: str, frag_label: str) -> Optional[int]:
    try:
        return int(mol_label) * _NB + int(frag_label)
    except (ValueError, TypeError):
        return None


def _qed_ads_np(x: np.ndarray, params) -> np.ndarray:
    A, B, C, D, E, F, DMAX = params
    exp1 = 1.0 + np.exp(-(x - C + D / 2.0) / E)
    exp2 = 1.0 + np.exp(-(x - C - D / 2.0) / F)
    dx = A + B / exp1 * (1.0 - 1.0 / exp2)
    return dx / DMAX


def _qed_from_acc_batch_np(accs: np.ndarray) -> np.ndarray:
    mw = np.maximum(0.0, accs[:, _AIX["MW"]])
    alogp = accs[:, _AIX["CRIPPEN"]]
    hba = np.round(np.maximum(0.0, accs[:, _AIX["QHBA"]]))
    hbd = np.round(np.maximum(0.0, accs[:, _AIX["QHBD"]]))
    psa = np.maximum(0.0, accs[:, _AIX["TPSA"]])
    rotb = np.round(np.maximum(0.0, accs[:, _AIX["ROTB"]]))
    arom = np.round(np.maximum(0.0, accs[:, _AIX["QAROM"]]))
    alerts = np.round(np.maximum(0.0, accs[:, _AIX["QALERTS"]]))
    cols = (
        _qed_ads_np(mw, _QEDAds["MW"]),
        _qed_ads_np(alogp, _QEDAds["ALOGP"]),
        _qed_ads_np(hba, _QEDAds["HBA"]),
        _qed_ads_np(hbd, _QEDAds["HBD"]),
        _qed_ads_np(psa, _QEDAds["PSA"]),
        _qed_ads_np(rotb, _QEDAds["ROTB"]),
        _qed_ads_np(arom, _QEDAds["AROM"]),
        _qed_ads_np(alerts, _QEDAds["ALERTS"]),
    )
    stacked = np.stack([np.clip(col, 1e-12, None) for col in cols], axis=1)
    return np.exp((np.log(stacked) * _QEDWeights[None, :]).sum(axis=1) / float(_QEDWeights.sum()))


class AdditivePropertyModel:

    def __init__(self, property_names: Sequence[str]):
        self.property_names = parse_property_names(property_names)
        self.frag_desc: Optional[np.ndarray] = None
        self.frag_ok: Optional[np.ndarray] = None
        self.correction = np.zeros((_NPAIR, _NACC), dtype=np.float64)
        self.calibrated = False


    def precompute_fragments(self, env) -> None:
        n = len(env.library)
        desc = np.zeros((n, _NACC), dtype=np.float64)
        ok = np.zeros(n, dtype=bool)
        for i in range(n):
            mg = env._frag_graph(i)
            if mg is None:
                continue
            c = _capped(mg)
            if c is None:
                continue
            v = _descriptor_vec(c)
            if v is not None:
                desc[i] = v
                ok[i] = True
        self.frag_desc = desc
        self.frag_ok = ok

    def _lib_signature(self, env) -> str:
        h = hashlib.md5()
        h.update(",".join(f.smiles for f in env.library).encode())
        h.update(",".join(self.property_names).encode())
        return h.hexdigest()[:16]

    def fit_or_load(self, env, n_samples: int = 800, seed: int = 0,
                    use_cache: bool = True) -> "AdditivePropertyModel":
        if self.frag_desc is None:
            self.precompute_fragments(env)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache = os.path.join(_CACHE_DIR, f"corr_{self._lib_signature(env)}.npy")
        if use_cache and os.path.exists(cache):
            self.correction = np.load(cache)
            self.calibrated = True
            return self
        self._calibrate(env, n_samples=n_samples, seed=seed)
        if use_cache:
            np.save(cache, self.correction)
        return self

    def _random_assembly(self, env, rng) -> Optional[Tuple[List[int], List[Tuple[str, str]], Chem.Mol]]:
        from copy import deepcopy
        n = len(env.library)
        si = int(rng.integers(n))
        mg = env._frag_graph(si)
        if mg is None or not self.frag_ok[si]:
            return None
        env._mol = deepcopy(mg)
        env._step = 0
        used, bonds = [si], []
        for _ in range(env.max_steps - 1):
            valid = env._valid_actions()
            attach = [a for a in valid if not a.is_terminate]
            if not attach:
                break
            a = attach[int(rng.integers(len(attach)))]
            merged, _np, nopen, ok, _ = env._simulate_action(a)
            if not ok or merged is None or not self.frag_ok[a.frag_idx]:
                break
            env._mol = merged
            used.append(a.frag_idx)
            bonds.append((a.mol_label, a.frag_label))
            if nopen == 0:
                break
        c = _capped(env._mol)
        if c is None:
            return None
        return used, bonds, c

    def _calibrate(self, env, n_samples: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        X, Y = [], []
        tries = 0
        while len(X) < n_samples and tries < n_samples * 4:
            tries += 1
            r = self._random_assembly(env, rng)
            if r is None:
                continue
            used, bonds, c = r
            true = _descriptor_vec(c)
            if true is None:
                continue
            add = self.frag_desc[used].sum(axis=0)
            bp = np.zeros(_NPAIR, dtype=np.float64)
            for ml, fl in bonds:
                k = _bondpair_index(ml, fl)
                if k is not None:
                    bp[k] += 1.0
            X.append(bp)
            Y.append(true - add)
        if not X:
            self.calibrated = True
            return
        X = np.asarray(X)
        Y = np.asarray(Y)
        lam = 1.0
        A = X.T @ X + lam * np.eye(_NPAIR)
        self.correction = np.linalg.solve(A, X.T @ Y)
        self.calibrated = True


    def frag_accum(self, frag_idx: int) -> np.ndarray:
        return self.frag_desc[frag_idx]

    def bond_correction(self, mol_label: str, frag_label: str) -> np.ndarray:
        k = _bondpair_index(mol_label, frag_label)
        return self.correction[k] if k is not None else np.zeros(_NACC)


    def _raw_from_accum(self, acc: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.property_names), dtype=np.float32)
        for j, name in enumerate(self.property_names):
            out[j] = self._raw_one(acc, name)
        return out

    def _raw_one(self, acc: np.ndarray, name: str) -> float:
        if name == "sLogP":
            return float(acc[_AIX["CRIPPEN"]])
        if name == "TPSA":
            return float(max(0.0, acc[_AIX["TPSA"]]))
        if name == "MW":
            return float(max(0.0, acc[_AIX["MW"]]))
        if name == "HBD":
            return float(max(0.0, acc[_AIX["HBD"]]))
        if name == "HBA":
            return float(max(0.0, acc[_AIX["HBA"]]))
        if name == "RotB":
            return float(max(0.0, acc[_AIX["ROTB"]]))
        if name == "RingCount":
            return float(max(0.0, acc[_AIX["RING"]]))
        if name == "HeavyAtomCount":
            return float(max(0.0, acc[_AIX["HAC"]]))
        if name == "FormalCharge":
            return float(acc[_AIX["FCHG"]])
        if name == "FractionCSP3":
            nc = acc[_AIX["NC"]]
            return float(acc[_AIX["NCSP3"]] / nc) if nc > 0 else 0.0
        if name == "QED":
            qp = _QEDProps(
                MW=float(max(0.0, acc[_AIX["MW"]])),
                ALOGP=float(acc[_AIX["CRIPPEN"]]),
                HBA=int(round(max(0.0, acc[_AIX["QHBA"]]))),
                HBD=int(round(max(0.0, acc[_AIX["QHBD"]]))),
                PSA=float(max(0.0, acc[_AIX["TPSA"]])),
                ROTB=int(round(max(0.0, acc[_AIX["ROTB"]]))),
                AROM=int(round(max(0.0, acc[_AIX["QAROM"]]))),
                ALERTS=int(round(max(0.0, acc[_AIX["QALERTS"]]))),
            )
            try:
                return float(QED.qed(None, qedProperties=qp))
            except Exception:
                return 0.0
        raise ValueError(f"AdditivePropertyModel cannot derive {name!r}")

    def properties_norm(self, acc: np.ndarray) -> np.ndarray:
        return normalize_props(self._raw_from_accum(acc), self.property_names)

    def properties_norm_batch(self, accs: np.ndarray) -> np.ndarray:
        accs = np.asarray(accs, dtype=np.float32)
        if accs.ndim == 1:
            accs = accs[None, :]
        raw = np.zeros((accs.shape[0], len(self.property_names)), dtype=np.float32)
        for j, name in enumerate(self.property_names):
            if name == "sLogP":
                raw[:, j] = accs[:, _AIX["CRIPPEN"]]
            elif name == "TPSA":
                raw[:, j] = np.maximum(0.0, accs[:, _AIX["TPSA"]])
            elif name == "MW":
                raw[:, j] = np.maximum(0.0, accs[:, _AIX["MW"]])
            elif name == "HBD":
                raw[:, j] = np.maximum(0.0, accs[:, _AIX["HBD"]])
            elif name == "HBA":
                raw[:, j] = np.maximum(0.0, accs[:, _AIX["HBA"]])
            elif name == "RotB":
                raw[:, j] = np.maximum(0.0, accs[:, _AIX["ROTB"]])
            elif name == "RingCount":
                raw[:, j] = np.maximum(0.0, accs[:, _AIX["RING"]])
            elif name == "HeavyAtomCount":
                raw[:, j] = np.maximum(0.0, accs[:, _AIX["HAC"]])
            elif name == "FormalCharge":
                raw[:, j] = accs[:, _AIX["FCHG"]]
            elif name == "FractionCSP3":
                nc = np.maximum(accs[:, _AIX["NC"]], 1e-6)
                raw[:, j] = np.where(nc > 0.0, accs[:, _AIX["NCSP3"]] / nc, 0.0)
            elif name == "QED":
                raw[:, j] = _qed_from_acc_batch_np(accs).astype(np.float32)
            else:
                raise ValueError(f"AdditivePropertyModel cannot batch-derive {name!r}")
        return normalize_props_batch(raw, self.property_names)

    def properties_norm_torch(self, accs):
        import torch
        from .properties import property_bounds

        if accs.dim() == 1:
            accs = accs.unsqueeze(0)
        accs = accs.float()
        raw_cols = []
        for name in self.property_names:
            if name == "sLogP":
                raw_cols.append(accs[:, _AIX["CRIPPEN"]])
            elif name == "TPSA":
                raw_cols.append(torch.clamp(accs[:, _AIX["TPSA"]], min=0.0))
            elif name == "MW":
                raw_cols.append(torch.clamp(accs[:, _AIX["MW"]], min=0.0))
            elif name == "HBD":
                raw_cols.append(torch.clamp(accs[:, _AIX["HBD"]], min=0.0))
            elif name == "HBA":
                raw_cols.append(torch.clamp(accs[:, _AIX["HBA"]], min=0.0))
            elif name == "RotB":
                raw_cols.append(torch.clamp(accs[:, _AIX["ROTB"]], min=0.0))
            elif name == "RingCount":
                raw_cols.append(torch.clamp(accs[:, _AIX["RING"]], min=0.0))
            elif name == "HeavyAtomCount":
                raw_cols.append(torch.clamp(accs[:, _AIX["HAC"]], min=0.0))
            elif name == "FormalCharge":
                raw_cols.append(accs[:, _AIX["FCHG"]])
            elif name == "FractionCSP3":
                nc = torch.clamp(accs[:, _AIX["NC"]], min=1e-6)
                raw_cols.append(torch.where(nc > 0.0, accs[:, _AIX["NCSP3"]] / nc, torch.zeros_like(nc)))
            elif name == "QED":
                raw_cols.append(self._qed_from_acc_torch(accs))
            else:
                raise ValueError(f"AdditivePropertyModel cannot torch-derive {name!r}")
        raw = torch.stack(raw_cols, dim=1)
        mins_np, maxs_np = property_bounds(self.property_names)
        mins = torch.as_tensor(mins_np, dtype=raw.dtype, device=raw.device)
        maxs = torch.as_tensor(maxs_np, dtype=raw.dtype, device=raw.device)
        return (torch.clamp(raw, mins, maxs) - mins) / torch.clamp(maxs - mins, min=1e-6)

    def _qed_from_acc_torch(self, accs):
        import torch

        def ads(x, key):
            A, B, C, D, E, F, DMAX = _QEDAds[key]
            exp1 = 1.0 + torch.exp(-(x - C + D / 2.0) / E)
            exp2 = 1.0 + torch.exp(-(x - C - D / 2.0) / F)
            dx = A + B / exp1 * (1.0 - 1.0 / exp2)
            return dx / DMAX

        mw = torch.clamp(accs[:, _AIX["MW"]], min=0.0)
        alogp = accs[:, _AIX["CRIPPEN"]]
        hba = torch.round(torch.clamp(accs[:, _AIX["QHBA"]], min=0.0))
        hbd = torch.round(torch.clamp(accs[:, _AIX["QHBD"]], min=0.0))
        psa = torch.clamp(accs[:, _AIX["TPSA"]], min=0.0)
        rotb = torch.round(torch.clamp(accs[:, _AIX["ROTB"]], min=0.0))
        arom = torch.round(torch.clamp(accs[:, _AIX["QAROM"]], min=0.0))
        alerts = torch.round(torch.clamp(accs[:, _AIX["QALERTS"]], min=0.0))
        cols = torch.stack([
            ads(mw, "MW"),
            ads(alogp, "ALOGP"),
            ads(hba, "HBA"),
            ads(hbd, "HBD"),
            ads(psa, "PSA"),
            ads(rotb, "ROTB"),
            ads(arom, "AROM"),
            ads(alerts, "ALERTS"),
        ], dim=1)
        weights = torch.as_tensor(_QEDWeights, dtype=cols.dtype, device=cols.device)
        cols = torch.clamp(cols, min=1e-12)
        return torch.exp((torch.log(cols) * weights.unsqueeze(0)).sum(dim=1) / float(weights.sum().item()))


    def action_props_norm(self, base_acc: np.ndarray, actions) -> np.ndarray:
        n = len(actions)
        accs = np.tile(base_acc.astype(np.float64), (n, 1))
        fidx = np.fromiter(
            (-1 if getattr(a, "is_terminate", False) else int(a.frag_idx) for a in actions),
            dtype=np.int64, count=n)
        bp = np.fromiter(
            (-1 if getattr(a, "is_terminate", False)
             else (_bondpair_index(a.mol_label, a.frag_label) or -1) for a in actions),
            dtype=np.int64, count=n)
        m = fidx >= 0
        if m.any():
            accs[m] += self.frag_desc[fidx[m]]
        mb = bp >= 0
        if mb.any():
            accs[mb] += self.correction[bp[mb]]
        return self.properties_norm_batch(accs)


def normalize_props_batch(raw: np.ndarray, property_names) -> np.ndarray:
    from .properties import property_bounds
    mins, maxs = property_bounds(property_names)
    return ((np.clip(raw, mins, maxs) - mins) / np.maximum(maxs - mins, 1e-6)).astype(np.float32)
