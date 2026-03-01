from __future__ import annotations

import sys
from pathlib import Path

_SIM_ROOT = Path(__file__).resolve().parent
_SIM_SRC = _SIM_ROOT / "src"
if str(_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SIM_SRC))

from browsecraft_sim.rl.hud_env import env


if __name__ == "__main__":
    env.run(transport="stdio")
