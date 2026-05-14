"""Package init for OUR_stuff/tests/.

Adds ``OUR_stuff/`` to ``sys.path`` so test modules can use the same flat
imports (``from estimation.block_belief import ...``) they used before the
restructure that moved project code under ``OUR_stuff/``.

Also exposes :data:`BASE_CONFIG_PATH` so tests can load
``configs/hybrid_control_rl/base.yaml`` without depending on the process
current working directory.
"""

import sys
from pathlib import Path

OUR_STUFF = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = OUR_STUFF / "configs" / "hybrid_control_rl" / "base.yaml"

if str(OUR_STUFF) not in sys.path:
    sys.path.insert(0, str(OUR_STUFF))
