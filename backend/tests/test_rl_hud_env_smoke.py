from __future__ import annotations

import asyncio

import pytest


def test_hud_env_registers_expected_tools_and_scenarios() -> None:
    pytest.importorskip("hud")
    from browsecraft_sim.rl.hud_env import env

    scenario_names = set(env._scenarios.keys())
    assert scenario_names >= {
        "t1_absolute",
        "t2_relative_single_ref",
        "t3_primitives",
        "t4_structure_relative",
        "t5_modification",
        "t6_composition",
    }

    tool_names = {tool.name for tool in asyncio.run(env.list_tools())}
    assert tool_names >= {
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
