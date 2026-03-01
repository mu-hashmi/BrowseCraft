from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_SCENARIOS = {
    "t1_absolute",
    "t2_relative_single_ref",
    "t3_primitives",
    "t4_structure_relative",
    "t5_modification",
    "t6_composition",
}

REQUIRED_TOOLS = {
    "player_position",
    "player_inventory",
    "inspect_area",
    "place_blocks",
    "fill_region",
    "undo_last",
    "get_active_overlay",
    "modify_overlay",
    "get_blueprints",
    "save_blueprint",
    "load_blueprint",
}


def _check_env_registry() -> None:
    sim_dir = Path(__file__).resolve().parents[1]
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    from env import env

    scenario_names = set(env._scenarios.keys())
    missing_scenarios = REQUIRED_SCENARIOS - scenario_names
    if missing_scenarios:
        missing = ", ".join(sorted(missing_scenarios))
        raise RuntimeError(f"missing HUD scenarios: {missing}")

    tools = asyncio.run(env.list_tools())
    tool_names = {tool.name for tool in tools}
    missing_tools = REQUIRED_TOOLS - tool_names
    if missing_tools:
        missing = ", ".join(sorted(missing_tools))
        raise RuntimeError(f"missing HUD tools: {missing}")


def _run_hud_debug(sim_dir: Path) -> None:
    config_payload = {
        "browsecraft": {
            "command": "uv",
            "args": ["run", "python", "env.py"],
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(config_payload, handle)
        config_path = Path(handle.name)
    try:
        command = ["hud", "debug", "--config", str(config_path), "--max-phase", "5"]
        result = subprocess.run(command, cwd=sim_dir, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"hud debug failed with exit code {result.returncode}")
    finally:
        config_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HUD environment smoke checks.")
    parser.add_argument("--skip-debug", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sim_dir = Path(__file__).resolve().parents[1]
    _check_env_registry()
    if not args.skip_debug:
        _run_hud_debug(sim_dir)
    print("HUD smoke checks passed")


if __name__ == "__main__":
    main()
