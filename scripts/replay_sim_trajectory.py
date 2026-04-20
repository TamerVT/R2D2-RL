from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rcs.sim_state_replay import app

if __name__ == "__main__":
    app()
