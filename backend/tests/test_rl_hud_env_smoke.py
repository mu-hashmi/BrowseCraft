from __future__ import annotations

import asyncio

import pytest

from browsecraft_sim.rl.agent_config import compose_remote_user_prompt
from browsecraft_sim.rl.task_generator import generate_task


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


def test_hud_scenario_generator_protocol_round_trip() -> None:
    pytest.importorskip("hud")
    from browsecraft_sim.rl.hud_env import env

    async def _run() -> None:
        prompt = await env.run_scenario_setup("t1_absolute", {"seed": 17, "index": 0, "reward_config": {}})
        expected_task = generate_task(tier="t1_absolute", seed=17, index=0)
        assert prompt == compose_remote_user_prompt(expected_task.prompt)
        await env.submit("t1_absolute", "")
        result = await env.run_scenario_evaluate("t1_absolute")
        assert 0.0 <= result.reward <= 1.0

    asyncio.run(_run())
