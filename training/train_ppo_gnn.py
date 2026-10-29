# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import ARTIFACTS_DIR

import training.ppo as _ppo


_ppo.DEFAULT_CFG.update({
    "encoder": "gnn",
    "checkpoint_dir": str(ARTIFACTS_DIR / "checkpoints_ppo_gnn"),
    "exp_name": "ppo_gnn_run",
})


if __name__ == "__main__":
    _ppo.train(_ppo._parse_args())
