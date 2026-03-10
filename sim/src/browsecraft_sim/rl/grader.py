from __future__ import annotations

from browsecraft_sim.main import HeadlessVoxelWorld

from .config import RewardConfig
from .metrics import (
    changed_map,
    exact_match,
    grounding_ratio,
    iou_score,
    is_connected,
    preservation_score,
    span_length,
)
from .reward import binary_reward, compose_reward, effective_efficiency_score
from .types import BlockPlacement, EpisodeTrace, RewardBreakdown, TaskSpec
from .world_setup import deserialize_snapshot


def grade_task(
    task: TaskSpec,
    world: HeadlessVoxelWorld,
    trace: EpisodeTrace,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    reward_config = config or RewardConfig()
    expected_changed = changed_map(task.target_blocks)
    actual_changed = changed_map(finalize_trace_diff(world=world, trace=trace))

    format_valid = trace.format_valid and all(call.success for call in trace.tool_calls)
    expected_calls = reward_config.expected_tool_calls(task)
    actual_calls = max(trace.tool_call_count, 1)
    efficiency_score = min(1.0, expected_calls / actual_calls)

    tier = task.tier
    details: dict[str, float | int | str | bool] = {
        "expected_tool_calls": expected_calls,
        "actual_tool_calls": trace.tool_call_count,
    }

    if tier in {"t1_absolute", "t2_relative_single_ref"}:
        correctness_score = exact_match(actual_changed, expected_changed)
    elif tier == "t5_modification":
        changed_iou = iou_score(actual_changed, expected_changed)
        preserve = preservation_score(world, task.preserved_blocks)
        correctness_score = 0.7 * changed_iou + 0.3 * preserve
        details["changed_iou"] = round(changed_iou, 6)
        details["preservation"] = round(preserve, 6)
    else:
        correctness_score = iou_score(actual_changed, expected_changed)

    structural_score = _structural_score(task, world)
    effective_efficiency = effective_efficiency_score(
        correctness_score=correctness_score,
        efficiency_score=efficiency_score,
        config=reward_config,
    )
    format_score, reward_raw, reward_normalized = compose_reward(
        format_valid=format_valid,
        correctness_score=correctness_score,
        efficiency_score=efficiency_score,
        structural_score=structural_score,
        config=reward_config,
    )
    reward_binary = binary_reward(normalized_reward=reward_normalized, config=reward_config)

    details["correctness"] = round(correctness_score, 6)
    details["efficiency_base"] = round(efficiency_score, 6)
    details["efficiency_effective"] = round(effective_efficiency, 6)
    details["efficiency_min_correctness"] = round(reward_config.efficiency_min_correctness, 6)
    details["binary_reward_threshold"] = round(reward_config.binary_reward_threshold, 6)
    details["structural"] = round(structural_score, 6)

    return RewardBreakdown(
        task_id=task.task_id,
        tier=task.tier,
        task_mode=trace.task_mode,
        format_valid=format_valid,
        format_score=format_score,
        correctness_score=correctness_score,
        efficiency_score=effective_efficiency,
        structural_score=structural_score,
        reward_raw=reward_raw,
        reward_normalized=reward_normalized,
        reward_binary=reward_binary,
        details=details,
    )


def finalize_trace_diff(
    *,
    world: HeadlessVoxelWorld,
    trace: EpisodeTrace,
) -> list[BlockPlacement]:
    if trace.final_world_diff:
        return trace.final_world_diff
    before = deserialize_snapshot(trace.initial_world)
    diff = world.diff(before)
    trace.final_world_diff = [
        BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id=block_id)
        for coord, block_id in sorted(diff.items())
    ]
    return trace.final_world_diff


def _structural_score(task: TaskSpec, world: HeadlessVoxelWorld) -> float:
    checks = task.structural_checks
    target_non_air = {placement.coord() for placement in task.target_blocks if placement.block_id != "minecraft:air"}
    achieved_non_air = {
        coord
        for coord in target_non_air
        if world.block_at(coord) != "minecraft:air"
    }

    components: list[float] = []
    if checks.require_connected:
        components.append(1.0 if is_connected(achieved_non_air) else 0.0)
    if checks.require_grounded:
        components.append(grounding_ratio(world, achieved_non_air))
    if checks.span_axis is not None and checks.min_span is not None:
        components.append(1.0 if span_length(achieved_non_air, checks.span_axis) >= checks.min_span else 0.0)

    if not components:
        return 1.0
    return sum(components) / len(components)
