"""Pytest discovery shim.

Adds ``OUR_stuff/`` to ``sys.path`` so existing imports like
``from estimation.pixel_to_table import ...`` keep resolving after the
2026-05-14 restructure that moved project modules under ``OUR_stuff/``
to match the layout used in Felix's branch.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUR_STUFF = REPO_ROOT / "OUR_stuff"

for path in (OUR_STUFF, REPO_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
