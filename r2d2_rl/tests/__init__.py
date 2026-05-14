"""Package init for r2d2_rl/tests/.

Adds ``r2d2_rl/`` to ``sys.path`` so test modules can use the same flat
imports (``from estimation.block_belief import ...``) they used before the
restructure that moved project code under ``r2d2_rl/``.

Also exposes :data:`BASE_CONFIG_PATH` so tests can load
``configs/hybrid_control_rl/base.yaml`` without depending on the process
current working directory.
"""

import sys
from pathlib import Path

R2D2_RL = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = R2D2_RL / "configs" / "hybrid_control_rl" / "base.yaml"

if str(R2D2_RL) not in sys.path:
    sys.path.insert(0, str(R2D2_RL))
