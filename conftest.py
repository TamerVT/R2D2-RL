"""Pytest discovery shim.

Adds ``r2d2_rl/`` to ``sys.path`` so existing imports like
``from estimation.pixel_to_table import ...`` keep resolving after the
2026-05-14 restructure that moved project modules under ``r2d2_rl/``
to match the layout used in Felix's branch.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
R2D2_RL = REPO_ROOT / "r2d2_rl"

for path in (R2D2_RL, REPO_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
