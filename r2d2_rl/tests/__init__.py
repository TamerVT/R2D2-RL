"""Package init for r2d2_rl/tests/.

Adds both the repository root and ``r2d2_rl/`` to ``sys.path`` so tests can
use package-style imports while project scripts can keep their legacy flat
imports (``from estimation.block_belief import ...``).

Also exposes :data:`BASE_CONFIG_PATH` so tests can load
``configs/hybrid_control_rl/base.yaml`` without depending on the process
current working directory.
"""

import sys
from pathlib import Path

R2D2_RL = Path(__file__).resolve().parents[1]
REPO_ROOT = R2D2_RL.parent
BASE_CONFIG_PATH = R2D2_RL / "configs" / "hybrid_control_rl" / "base.yaml"

for path in (REPO_ROOT, R2D2_RL):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
