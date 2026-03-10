from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .types import TaskSpec


FormatMode = Literal["gate", "weighted"]


class RewardConfig(BaseModel):
    format_mode: FormatMode = "gate"
    weight_correctness: float = Field(default=0.7, ge=0.0)
    weight_efficiency: float = Field(default=0.2, ge=0.0)
    weight_structural: float = Field(default=0.1, ge=0.0)
    weight_format: float = Field(default=0.1, ge=0.0)
    efficiency_min_correctness: float = Field(default=0.1, ge=0.0, le=1.0)
    binary_reward_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> "RewardConfig":
        if self.format_mode == "gate":
            total = self.weight_correctness + self.weight_efficiency + self.weight_structural
            if total <= 0:
                raise ValueError("gate mode requires positive correctness/efficiency/structural weight sum")
        else:
            total = self.weight_correctness + self.weight_efficiency + self.weight_structural + self.weight_format
            if total <= 0:
                raise ValueError("weighted mode requires positive overall weight sum")
        return self

    def expected_tool_calls(self, task: TaskSpec) -> int:
        return task.expected_tool_calls


def load_reward_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> RewardConfig:
    payload: dict[str, Any] = {}
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("reward config file must contain a JSON object")
    if overrides:
        payload.update(overrides)
    return RewardConfig.model_validate(payload)
