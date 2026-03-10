from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


BuildTier = Literal[
    "t1_absolute",
    "t2_relative_single_ref",
    "t3_primitives",
    "t4_structure_relative",
    "t5_modification",
    "t6_composition",
]

TextQATier = Literal[
    "qa_directional_single_hop",
    "qa_multi_hop_chain",
    "qa_viewpoint_transform",
    "qa_topology",
]

TaskMode = Literal["build", "text_qa"]
AnswerFormat = Literal["single_token", "entity_name", "yes_no", "coordinate"]
Tier = BuildTier

SpanAxis = Literal["x", "y", "z"]

ALL_TIERS: tuple[BuildTier, ...] = (
    "t1_absolute",
    "t2_relative_single_ref",
    "t3_primitives",
    "t4_structure_relative",
    "t5_modification",
    "t6_composition",
)

ALL_TEXT_QA_TIERS: tuple[TextQATier, ...] = (
    "qa_directional_single_hop",
    "qa_multi_hop_chain",
    "qa_viewpoint_transform",
    "qa_topology",
)


class BlockPlacement(BaseModel):
    x: int
    y: int
    z: int
    block_id: str = Field(min_length=1)

    def coord(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)


class PlayerSpec(BaseModel):
    x: int = 0
    y: int = 64
    z: int = 0
    facing: str = Field(default="north", min_length=1)
    dimension: str = Field(default="minecraft:overworld", min_length=1)


class StructuralChecks(BaseModel):
    require_connected: bool = False
    require_grounded: bool = False
    min_span: int | None = Field(default=None, ge=1)
    span_axis: SpanAxis | None = None

    @model_validator(mode="after")
    def validate_span_pair(self) -> "StructuralChecks":
        if (self.min_span is None) != (self.span_axis is None):
            raise ValueError("min_span and span_axis must be set together")
        return self


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tier: Tier
    family: str = Field(min_length=1)
    seed: int
    prompt: str = Field(min_length=1)
    player: PlayerSpec = Field(default_factory=PlayerSpec)
    setup_blocks: list[BlockPlacement] = Field(default_factory=list)
    target_blocks: list[BlockPlacement] = Field(default_factory=list)
    preserved_blocks: list[BlockPlacement] = Field(default_factory=list)
    expected_tool_calls: int = Field(ge=1)
    structural_checks: StructuralChecks = Field(default_factory=StructuralChecks)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextQATaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tier: TextQATier
    family: str = Field(min_length=1)
    seed: int
    prompt: str = Field(min_length=1)
    player: PlayerSpec = Field(default_factory=PlayerSpec)
    setup_blocks: list[BlockPlacement] = Field(default_factory=list)
    expected_answer: str = Field(min_length=1)
    answer_format: AnswerFormat
    canonical_reasoning: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    success: bool
    error: str | None = None


class EpisodeTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    task_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    seed: int
    task_mode: TaskMode = "build"
    model: str = Field(default="", min_length=0)
    system_prompt: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    format_valid: bool = True
    tool_round_count: int = Field(default=0, ge=0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    initial_world: dict[str, str] = Field(default_factory=dict)
    final_world_diff: list[BlockPlacement] = Field(default_factory=list)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    task_mode: TaskMode = "build"
    format_valid: bool
    format_score: float = Field(ge=0.0, le=1.0)
    correctness_score: float = Field(ge=0.0, le=1.0)
    efficiency_score: float = Field(ge=0.0, le=1.0)
    structural_score: float = Field(ge=0.0, le=1.0)
    reward_raw: float
    reward_normalized: float = Field(ge=0.0, le=1.0)
    reward_binary: float = Field(ge=0.0, le=1.0)
    details: dict[str, float | int | str | bool] = Field(default_factory=dict)
