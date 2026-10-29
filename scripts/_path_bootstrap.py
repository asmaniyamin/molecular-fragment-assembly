# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# The chemistry toolkit is a separate dependency.
import os

_candidates = []
if os.environ.get("MFA_TOOLKIT_DIR"):
    _candidates.append(Path(os.environ["MFA_TOOLKIT_DIR"]).expanduser())
_candidates.extend(parent / "small-molecule-mod-toolkit"
                   for parent in [_THIS_DIR.parent, *_THIS_DIR.parent.parents])
for _toolkit in _candidates:
    if (_toolkit / "chem").is_dir() and (_toolkit / "core").is_dir():
        if str(_toolkit) not in sys.path:
            sys.path.insert(0, str(_toolkit))
        break


for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
