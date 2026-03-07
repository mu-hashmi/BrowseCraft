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
_SCENARIO_SESSION: contextvars.ContextVar["_ScenarioSession | None"] = contextvars.ContextVar(
    "browsecraft_rl_session",
    default=None,
)
_TOOL_SESSION: "_ScenarioSession | None" = None
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


def _start_session(
    task: TaskSpec,
    reward_config: RewardConfig,
) -> _ScenarioSession:
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
    return session


def _active_session() -> _ScenarioSession:
    session = _SCENARIO_SESSION.get()
    if session is not None:
        return session
    session = _TOOL_SESSION
    if session is None:
        raise RuntimeError("tool call outside active scenario session")
    return session


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


def _score_session(session: _ScenarioSession) -> Any:
    session.trace.ended_at = datetime.now(UTC)
    session.trace.final_world_diff = diff_to_blocks(session.world.diff(session.before_snapshot))
    return grade_task(task=session.task, world=session.world, trace=session.trace, config=session.reward_config)


def _finish(session: _ScenarioSession) -> float:
    return _score_session(session).reward_normalized


def _clear_tool_session() -> None:
    global _TOOL_SESSION
    _TOOL_SESSION = None


@env.tool()
async def player_position() -> dict[str, Any]:
    """Return the current simulated player position and facing."""
    return _dispatch("player_position", {})


@env.tool()
async def player_inventory() -> dict[str, Any]:
    """Return the simulated player inventory summary."""
    return _dispatch("player_inventory", {})


@env.tool()
async def inspect_area(
    center: dict[str, int],
    radius: int,
    detailed: bool = False,
    filter_terrain: bool = True,
) -> dict[str, Any]:
    """Inspect blocks around a center point within a cubic radius."""
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
    """Place explicit block placements at the given coordinates."""
    return _dispatch("place_blocks", {"placements": placements})


@env.tool()
async def fill_region(from_corner: dict[str, int], to_corner: dict[str, int], block_id: str) -> dict[str, Any]:
    """Fill an axis-aligned region, inclusive of both corners, with one block type."""
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
    """Undo the most recent world-modifying action."""
    return _dispatch("undo_last", {})


@env.tool()
async def get_active_overlay() -> dict[str, Any]:
    """Return the currently loaded overlay blueprint, if one exists."""
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
    """Transform the active overlay with rotation, translation, pivot, or material swap operations."""
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
    """List saved blueprint names available in the simulator."""
    return _dispatch("get_blueprints", {})


@env.tool()
async def save_blueprint(name: str) -> dict[str, Any]:
    """Save the active overlay under a blueprint name."""
    return _dispatch("save_blueprint", {"name": name})


@env.tool()
async def load_blueprint(name: str) -> dict[str, Any]:
    """Load a saved blueprint into the active overlay."""
    return _dispatch("load_blueprint", {"name": name})


@env.tool()
async def rl_setup_task(
    task_spec: dict[str, Any],
    reward_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialize a legacy HUD eval episode from an explicit RL task spec."""
    global _TOOL_SESSION
    if _TOOL_SESSION is not None:
        raise RuntimeError("task session already active")

    task = TaskSpec.model_validate(task_spec)
    config = RewardConfig.model_validate(reward_config or {})
    session = _start_session(task=task, reward_config=config)
    _TOOL_SESSION = session
    return {
        "task_id": task.task_id,
        "tier": task.tier,
        "family": task.family,
        "prompt": task.prompt,
        "expected_tool_calls": task.expected_tool_calls,
    }


@env.tool()
async def rl_grade_task() -> dict[str, Any]:
    """Score the active legacy HUD eval episode and return reward details."""
    session = _active_session()
    try:
        breakdown = _score_session(session)
        config = session.reward_config
        return {
            "reward": breakdown.reward_normalized,
            "score": breakdown.reward_normalized,
            "task_id": session.task.task_id,
            "tier": session.task.tier,
            "subscores": {
                "format": breakdown.format_score,
                "correctness": breakdown.correctness_score,
                "efficiency": breakdown.efficiency_score,
                "structural": breakdown.structural_score,
            },
            "weights": {
                "format": config.weight_format,
                "correctness": config.weight_correctness,
                "efficiency": config.weight_efficiency,
                "structural": config.weight_structural,
            },
            "details": breakdown.details,
        }
    finally:
        _clear_tool_session()


async def _run_scenario(
    *,
    tier: Tier,
    task_spec: dict[str, Any] | None,
    seed: int,
    index: int,
    reward_config: dict[str, Any] | None,
) -> Any:
    global _TOOL_SESSION
    task = _load_task(tier=tier, task_spec=task_spec, seed=seed, index=index)
    config = RewardConfig.model_validate(reward_config or {})
    session = _start_session(task=task, reward_config=config)
    if _TOOL_SESSION is not None:
        raise RuntimeError("scenario session already active")
    token = _SCENARIO_SESSION.set(session)
    _TOOL_SESSION = session
    try:
        yield task.prompt
        yield _finish(session)
    finally:
        _clear_tool_session()
        _SCENARIO_SESSION.reset(token)


def _register_scenario(tier: Tier, scenario_name: str) -> Any:
    @env.scenario(scenario_name)
    async def _scenario(
        task_spec: dict[str, Any] | None = None,
        seed: int = 7,
        index: int = 0,
        reward_config: dict[str, Any] | None = None,
    ) -> Any:
        async for item in _run_scenario(
            tier=tier,
            task_spec=task_spec,
            seed=seed,
            index=index,
            reward_config=reward_config,
        ):
            yield item

    return _scenario


t1_absolute = _register_scenario("t1_absolute", "t1_absolute")
t2_relative_single_ref = _register_scenario("t2_relative_single_ref", "t2_relative_single_ref")
t3_primitives = _register_scenario("t3_primitives", "t3_primitives")
t4_structure_relative = _register_scenario("t4_structure_relative", "t4_structure_relative")
t5_modification = _register_scenario("t5_modification", "t5_modification")
t6_composition = _register_scenario("t6_composition", "t6_composition")
