from __future__ import annotations

from datetime import UTC, datetime

import pytest

from browsecraft_sim.rl.trajectory import read_trajectory_jsonl, trace_to_trajectory_record, validate_anthropic_messages
from browsecraft_sim.rl.types import BlockPlacement, EpisodeTrace


def test_validate_anthropic_messages_accepts_tool_use_and_result() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "call-1", "name": "inspect_area", "input": {"radius": 2}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "{\"ok\": true}"}],
        },
    ]
    validated = validate_anthropic_messages(messages)
    assert len(validated) == 2


def test_trace_to_trajectory_record_round_trip(tmp_path) -> None:
    trace = EpisodeTrace(
        task_id="task-1",
        tier="t1_absolute",
        seed=7,
        model="claude-sonnet-4-6",
        system_prompt="system",
        messages=[
            {"role": "assistant", "content": [{"type": "text", "text": "working"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tc-1", "name": "place_blocks", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tc-1", "content": "{\"placed_count\":1}"}]},
        ],
        final_world_diff=[BlockPlacement(x=0, y=64, z=0, block_id="minecraft:stone")],
        ended_at=datetime.now(UTC),
    )
    record = trace_to_trajectory_record(
        trace=trace,
        model="claude-sonnet-4-6",
        grader={"correctness": 1.0},
        reward_raw=1.0,
        reward_normalized=1.0,
    )
    path = tmp_path / "traj.jsonl"
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    loaded = read_trajectory_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].task_id == "task-1"


def test_validate_anthropic_messages_rejects_non_list_content() -> None:
    with pytest.raises(ValueError):
        validate_anthropic_messages([{"role": "assistant", "content": "plain text"}])
