"""Project 3 hybrid-control/RL package.

Most project modules still support the legacy flat imports used by scripts
(``from estimation...``). Importing ``r2d2_rl`` adds this directory to
``sys.path`` so package-style imports also work from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

