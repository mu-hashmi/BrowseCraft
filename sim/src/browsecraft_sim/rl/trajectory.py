from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .types import EpisodeTrace


class AnthropicTextBlock(BaseModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class AnthropicToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    input: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    is_error: bool = False


AnthropicBlock = AnthropicTextBlock | AnthropicToolUseBlock | AnthropicToolResultBlock
_BLOCK_ADAPTER = TypeAdapter(AnthropicBlock)


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[AnthropicBlock] = Field(default_factory=list)


class EpisodeTrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    task_mode: str = Field(min_length=1)
    seed: int
    model: str = Field(min_length=1)
    system_prompt: str = ""
    messages: list[AnthropicMessage] = Field(default_factory=list)
    tool_round_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    grader: dict[str, Any] = Field(default_factory=dict)
    reward_raw: float
    reward_normalized: float = Field(ge=0.0, le=1.0)
    reward_binary: float = Field(ge=0.0, le=1.0)
    final_world_diff: list[dict[str, Any]] = Field(default_factory=list)
    started_at: str = Field(min_length=1)
    ended_at: str = Field(min_length=1)


def validate_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise ValueError("message role must be a string")
        if not isinstance(content, list):
            raise ValueError("message content must be a list of Anthropic blocks")
        blocks = [_BLOCK_ADAPTER.validate_python(block).model_dump(mode="json") for block in content]
        validated.append({"role": role, "content": blocks})
    return validated


def trace_to_trajectory_record(
    *,
    trace: EpisodeTrace,
    model: str,
    grader: dict[str, Any],
    reward_raw: float,
    reward_normalized: float,
    reward_binary: float,
) -> EpisodeTrajectoryRecord:
    if trace.ended_at is None:
        raise ValueError("trace ended_at must be set before export")
    validated_messages = validate_anthropic_messages(trace.messages)
    return EpisodeTrajectoryRecord(
        episode_id=trace.episode_id,
        task_id=trace.task_id,
        tier=trace.tier,
        task_mode=trace.task_mode,
        seed=trace.seed,
        model=model,
        system_prompt=trace.system_prompt,
        messages=[AnthropicMessage.model_validate(message) for message in validated_messages],
        tool_round_count=trace.tool_round_count,
        tool_call_count=trace.tool_call_count,
        grader=grader,
        reward_raw=reward_raw,
        reward_normalized=reward_normalized,
        reward_binary=reward_binary,
        final_world_diff=[placement.model_dump(mode="json") for placement in trace.final_world_diff],
        started_at=trace.started_at.isoformat(),
        ended_at=trace.ended_at.isoformat(),
    )


def read_trajectory_jsonl(path: str | Path) -> list[EpisodeTrajectoryRecord]:
    records: list[EpisodeTrajectoryRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if "task_mode" not in payload:
            payload["task_mode"] = "build"
        if "reward_binary" not in payload and "reward_normalized" in payload:
            payload["reward_binary"] = 1.0 if float(payload["reward_normalized"]) >= 0.8 else 0.0
        try:
            record = EpisodeTrajectoryRecord.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"invalid trajectory at line {line_number}: {exc}") from exc
        records.append(record)
    return records


def write_trajectory_jsonl(path: str | Path, records: list[EpisodeTrajectoryRecord]) -> None:
    output = Path(path)
    lines = [record.model_dump_json() for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
