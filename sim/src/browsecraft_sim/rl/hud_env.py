from __future__ import annotations

import contextvars
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from hud import Environment

from browsecraft_sim.tool_dispatch import dispatch_tool

from .config import RewardConfig
from .grader import grade_task
from .task_generator import generate_task
from .types import EpisodeTrace, TaskSpec, Tier, ToolCallRecord
from .world_setup import build_world, diff_to_blocks, serialize_snapshot


ENV_NAME = "browsecraft-spatial-rl"
_CURRENT_SESSION: contextvars.ContextVar["_ScenarioSession"] = contextvars.ContextVar("browsecraft_rl_session")
env = Environment(name=ENV_NAME)


@dataclass(slots=True)
class _ScenarioSession:
    task: TaskSpec
    world: Any
    before_snapshot: dict[tuple[int, int, int], str]
    trace: EpisodeTrace
    reward_config: RewardConfig


def _load_task(
    *,
    tier: Tier,
    task_spec: dict[str, Any] | None,
    seed: int,
    index: int,
) -> TaskSpec:
    if task_spec is not None:
        task = TaskSpec.model_validate(task_spec)
        if task.tier != tier:
            raise ValueError(f"task_spec tier mismatch: expected={tier} actual={task.tier}")
        return task
    return generate_task(tier=tier, seed=seed, index=index)


def _start_session(task: TaskSpec, reward_config: RewardConfig) -> tuple[_ScenarioSession, contextvars.Token[_ScenarioSession]]:
    world = build_world(task)
    before_snapshot = world.snapshot()
    trace = EpisodeTrace(
        task_id=task.task_id,
        tier=task.tier,
        seed=task.seed,
        model="",
        initial_world=serialize_snapshot(before_snapshot),
        started_at=datetime.now(UTC),
    )
    session = _ScenarioSession(
        task=task,
        world=world,
        before_snapshot=before_snapshot,
        trace=trace,
        reward_config=reward_config,
    )
    token = _CURRENT_SESSION.set(session)
    return session, token


def _active_session() -> _ScenarioSession:
    try:
        return _CURRENT_SESSION.get()
    except LookupError as exc:
        raise RuntimeError("tool call outside active scenario session") from exc


def _dispatch(name: str, params: dict[str, Any]) -> dict[str, Any]:
    session = _active_session()
    try:
        result = dispatch_tool(session.world, name, params)
        session.trace.tool_calls.append(ToolCallRecord(name=name, args=params, success=True))
        return result
    except Exception as exc:
        session.trace.format_valid = False
        session.trace.tool_calls.append(ToolCallRecord(name=name, args=params, success=False, error=str(exc)))
        raise


def _finish(session: _ScenarioSession) -> float:
    session.trace.ended_at = datetime.now(UTC)
    session.trace.final_world_diff = diff_to_blocks(session.world.diff(session.before_snapshot))
    breakdown = grade_task(task=session.task, world=session.world, trace=session.trace, config=session.reward_config)
    return breakdown.reward_normalized


@env.tool()
async def player_position() -> dict[str, Any]:
    return _dispatch("player_position", {})


@env.tool()
async def player_inventory() -> dict[str, Any]:
    return _dispatch("player_inventory", {})


@env.tool()
async def inspect_area(
    center: dict[str, int],
    radius: int,
    detailed: bool = False,
    filter_terrain: bool = True,
) -> dict[str, Any]:
    return _dispatch(
        "inspect_area",
        {
            "center": center,
            "radius": radius,
            "detailed": detailed,
            "filter_terrain": filter_terrain,
        },
    )


@env.tool()
async def place_blocks(placements: list[dict[str, Any]]) -> dict[str, Any]:
    return _dispatch("place_blocks", {"placements": placements})


@env.tool()
async def fill_region(from_corner: dict[str, int], to_corner: dict[str, int], block_id: str) -> dict[str, Any]:
    return _dispatch(
        "fill_region",
        {
            "from_corner": from_corner,
            "to_corner": to_corner,
            "block_id": block_id,
        },
    )


@env.tool()
async def undo_last() -> dict[str, Any]:
    return _dispatch("undo_last", {})


@env.tool()
async def get_active_overlay() -> dict[str, Any]:
    return _dispatch("get_active_overlay", {})


@env.tool()
async def modify_overlay(
    op: str,
    quarters: int | None = None,
    dy: int | None = None,
    x: int | None = None,
    y: int | None = None,
    z: int | None = None,
    from_block: str | None = None,
    to_block: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"op": op}
    if quarters is not None:
        params["quarters"] = quarters
    if dy is not None:
        params["dy"] = dy
    if x is not None:
        params["x"] = x
    if y is not None:
        params["y"] = y
    if z is not None:
        params["z"] = z
    if from_block is not None:
        params["from"] = from_block
    if to_block is not None:
        params["to"] = to_block
    return _dispatch("modify_overlay", params)


@env.tool()
async def get_blueprints() -> dict[str, Any]:
    return _dispatch("get_blueprints", {})


@env.tool()
async def save_blueprint(name: str) -> dict[str, Any]:
    return _dispatch("save_blueprint", {"name": name})


@env.tool()
async def load_blueprint(name: str) -> dict[str, Any]:
    return _dispatch("load_blueprint", {"name": name})


async def _run_scenario(
    *,
    tier: Tier,
    task_spec: dict[str, Any] | None,
    seed: int,
    index: int,
    reward_config: dict[str, Any] | None,
) -> Any:
    task = _load_task(tier=tier, task_spec=task_spec, seed=seed, index=index)
    config = RewardConfig.model_validate(reward_config or {})
    session, token = _start_session(task=task, reward_config=config)
    try:
        _ = yield task.prompt
        yield _finish(session)
    finally:
        _CURRENT_SESSION.reset(token)


@env.scenario("t1_absolute")
async def t1_absolute(
    task_spec: dict[str, Any] | None = None,
    seed: int = 7,
    index: int = 0,
    reward_config: dict[str, Any] | None = None,
) -> Any:
    async for item in _run_scenario(
        tier="t1_absolute",
        task_spec=task_spec,
        seed=seed,
        index=index,
        reward_config=reward_config,
    ):
        yield item


@env.scenario("t2_relative_single_ref")
async def t2_relative_single_ref(
    task_spec: dict[str, Any] | None = None,
    seed: int = 7,
    index: int = 0,
    reward_config: dict[str, Any] | None = None,
) -> Any:
    async for item in _run_scenario(
        tier="t2_relative_single_ref",
        task_spec=task_spec,
        seed=seed,
        index=index,
        reward_config=reward_config,
    ):
        yield item


@env.scenario("t3_primitives")
async def t3_primitives(
    task_spec: dict[str, Any] | None = None,
    seed: int = 7,
    index: int = 0,
    reward_config: dict[str, Any] | None = None,
) -> Any:
    async for item in _run_scenario(
        tier="t3_primitives",
        task_spec=task_spec,
        seed=seed,
        index=index,
        reward_config=reward_config,
    ):
        yield item


@env.scenario("t4_structure_relative")
async def t4_structure_relative(
    task_spec: dict[str, Any] | None = None,
    seed: int = 7,
    index: int = 0,
    reward_config: dict[str, Any] | None = None,
) -> Any:
    async for item in _run_scenario(
        tier="t4_structure_relative",
        task_spec=task_spec,
        seed=seed,
        index=index,
        reward_config=reward_config,
    ):
        yield item


@env.scenario("t5_modification")
async def t5_modification(
    task_spec: dict[str, Any] | None = None,
    seed: int = 7,
    index: int = 0,
    reward_config: dict[str, Any] | None = None,
) -> Any:
    async for item in _run_scenario(
        tier="t5_modification",
        task_spec=task_spec,
        seed=seed,
        index=index,
        reward_config=reward_config,
    ):
        yield item


@env.scenario("t6_composition")
async def t6_composition(
    task_spec: dict[str, Any] | None = None,
    seed: int = 7,
    index: int = 0,
    reward_config: dict[str, Any] | None = None,
) -> Any:
    async for item in _run_scenario(
        tier="t6_composition",
        task_spec=task_spec,
        seed=seed,
        index=index,
        reward_config=reward_config,
    ):
        yield item
