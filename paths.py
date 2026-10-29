# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

"""Defaults shared by local commands.

Set MFA_DATA_DIR for a full dataset and MFA_RESULTS_DIR for generated outputs.
Command-line data options take precedence over these defaults.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MFA_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
ARTIFACTS_DIR = Path(os.environ.get("MFA_RESULTS_DIR", PROJECT_ROOT / "artifacts")).expanduser().resolve()
FRAGMENTS_PATH = DATA_DIR / "fragments.parquet"
PARENTS_PATH = DATA_DIR / "parents.parquet"
