# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from scripts import _path_bootstrap
from paths import ARTIFACTS_DIR

from typing import Dict

from training.a2c import DEFAULT_CFG as A2C_DEFAULT_CFG
from training.a2c import _parse_args as _parse_a2c_args
from training.a2c import train


DEFAULT_CFG: Dict = dict(A2C_DEFAULT_CFG)
DEFAULT_CFG.update({
    "encoder": "gnn",
    "checkpoint_dir": str(ARTIFACTS_DIR / "checkpoints_gnn"),
    "exp_name": "gnn_run",
})


def _parse_args() -> Dict:
    return _parse_a2c_args(DEFAULT_CFG)


if __name__ == "__main__":
    train(_parse_args())
