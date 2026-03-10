from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


_REQUIRED_BLOCK_IDS = {
    "minecraft:observer",
    "minecraft:piston",
    "minecraft:sticky_piston",
    "minecraft:repeater",
    "minecraft:comparator",
    "minecraft:hopper",
    "minecraft:dispenser",
    "minecraft:dropper",
    "minecraft:lever",
    "minecraft:lightning_rod",
    "minecraft:end_rod",
    "minecraft:small_amethyst_bud",
    "minecraft:medium_amethyst_bud",
    "minecraft:large_amethyst_bud",
    "minecraft:amethyst_cluster",
}

_REQUIRED_BLOCK_SUFFIXES = (
    "_stairs",
    "_door",
    "_trapdoor",
    "_button",
    "_sign",
    "_wall_sign",
    "_hanging_sign",
    "_wall_hanging_sign",
    "_banner",
    "_wall_banner",
    "_bed",
    "_head",
    "_wall_head",
    "_skull",
    "_wall_skull",
)


class BlockPlacement(BaseModel):
    dx: int
    dy: int
    dz: int
    block_id: str
    block_state: dict[str, str] = Field(default_factory=dict)

    @field_validator("block_state", mode="before")
    @classmethod
    def normalize_block_state(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("block_state must be a mapping")
        return {str(key): str(item) for key, item in value.items()}

    @model_validator(mode="after")
    def validate_required_block_state(self) -> BlockPlacement:
        requires_state = self.block_id in _REQUIRED_BLOCK_IDS or any(
            self.block_id.endswith(suffix) for suffix in _REQUIRED_BLOCK_SUFFIXES
        )
        if requires_state and not self.block_state:
            raise ValueError(f"block_state is required for {self.block_id}")
        return self


class BuildPlan(BaseModel):
    total_blocks: int = Field(ge=0)
    placements: list[BlockPlacement]


class ChatRequest(BaseModel):
    client_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    mode: Literal["build", "plan", "plan_fast"] = "build"
    world_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)


class ChatAcceptedResponse(BaseModel):
    chat_id: str
    status: Literal["accepted"]


class AsyncJobAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["accepted"]


class SearchRequest(BaseModel):
    client_id: str = Field(min_length=1)
    query: str = Field(min_length=1)


class ImagineRequest(BaseModel):
    client_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)


class SessionNewRequest(BaseModel):
    client_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)


class SessionSwitchRequest(BaseModel):
    client_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class SessionCreatedResponse(BaseModel):
    world_id: str
    session_id: str
    status: Literal["created"]


class SessionSwitchedResponse(BaseModel):
    world_id: str
    session_id: str
    status: Literal["active"]


class SessionSummary(BaseModel):
    session_id: str
    message_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    world_id: str
    active_session_id: str | None = None
    sessions: list[SessionSummary]
