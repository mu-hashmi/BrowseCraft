from __future__ import annotations

from browsecraft_sim.rl.config import RewardConfig
from browsecraft_sim.rl.grader import grade_task
from browsecraft_sim.rl.task_generator import generate_task
from browsecraft_sim.rl.types import EpisodeTrace, ToolCallRecord
from browsecraft_sim.rl.world_setup import build_world, diff_to_blocks, serialize_snapshot


def _trace_for_world(task, world, before_snapshot):
    return EpisodeTrace(
        task_id=task.task_id,
        tier=task.tier,
        seed=task.seed,
        tool_calls=[ToolCallRecord(name="place_blocks", args={}, success=True)],
        initial_world=serialize_snapshot(before_snapshot),
        final_world_diff=diff_to_blocks(world.diff(before_snapshot)),
    )


def _task_for_family(tier, family):
    for index in range(60):
        task = generate_task(tier=tier, seed=91, index=index)
        if task.family == family:
            return task
    raise AssertionError(f"could not generate family={family} for tier={tier}")


def test_t1_perfect_exact_match_scores_high() -> None:
    task = generate_task(tier="t1_absolute", seed=3, index=0)
    world = build_world(task)
    before = world.snapshot()
    for placement in task.target_blocks:
        world.set_block(placement.coord(), placement.block_id)
    trace = _trace_for_world(task, world, before)
    breakdown = grade_task(task=task, world=world, trace=trace, config=RewardConfig(format_mode="gate"))
    assert breakdown.correctness_score == 1.0
    assert breakdown.reward_normalized > 0.95
    assert breakdown.reward_binary == 1.0


def test_format_failure_zeroes_reward_in_gate_mode() -> None:
    task = generate_task(tier="t2_relative_single_ref", seed=6, index=0)
    world = build_world(task)
    before = world.snapshot()
    trace = EpisodeTrace(
        task_id=task.task_id,
        tier=task.tier,
        seed=task.seed,
        format_valid=False,
        tool_calls=[ToolCallRecord(name="place_blocks", args={}, success=False, error="invalid args")],
        initial_world=serialize_snapshot(before),
        final_world_diff=[],
    )
    breakdown = grade_task(task=task, world=world, trace=trace, config=RewardConfig(format_mode="gate"))
    assert breakdown.reward_raw == 0.0
    assert breakdown.reward_normalized == 0.0
    assert breakdown.reward_binary == 0.0


def test_t6_underbuild_is_penalized() -> None:
    task = _task_for_family("t6_composition", "bridge_between_structures")
    world = build_world(task)
    before = world.snapshot()
    for placement in task.target_blocks[: max(1, len(task.target_blocks) // 3)]:
        world.set_block(placement.coord(), placement.block_id)
    trace = _trace_for_world(task, world, before)
    breakdown = grade_task(task=task, world=world, trace=trace, config=RewardConfig(format_mode="gate"))
    assert breakdown.correctness_score < 0.7
    assert breakdown.reward_normalized < 0.8
    assert breakdown.reward_binary == 0.0


def test_t6_disconnected_shape_fails_structural_check() -> None:
    task = _task_for_family("t6_composition", "bridge_between_structures")
    world = build_world(task)
    before = world.snapshot()
    first = task.target_blocks[0]
    last = task.target_blocks[-1]
    world.set_block(first.coord(), first.block_id)
    world.set_block(last.coord(), last.block_id)
    trace = _trace_for_world(task, world, before)
    breakdown = grade_task(task=task, world=world, trace=trace, config=RewardConfig(format_mode="gate"))
    assert breakdown.structural_score < 1.0
    assert breakdown.reward_normalized < 0.7
    assert breakdown.reward_binary == 0.0
