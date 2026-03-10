from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PromptVariantRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    family: str = Field(min_length=1)
    seed: int
    original_prompt: str = Field(min_length=1)
    verified_paraphrases: list[str] = Field(default_factory=list)
    shortfall: int = Field(ge=0, le=3)


def read_prompt_variants_jsonl(path: str | Path) -> list[PromptVariantRecord]:
    records: list[PromptVariantRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        try:
            record = PromptVariantRecord.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"invalid prompt variants record at line {line_number}: {exc}") from exc
        records.append(record)
    return records


def write_prompt_variants_jsonl(path: str | Path, records: Iterable[PromptVariantRecord]) -> None:
    output = Path(path)
    lines = [record.model_dump_json() for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
