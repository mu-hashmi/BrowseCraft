from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from browsecraft_sim.rl.trajectory import EpisodeTrajectoryRecord, trace_to_trajectory_record, write_trajectory_jsonl
from browsecraft_sim.rl.types import EpisodeTrace


class RawTrajectoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace: EpisodeTrace
    model: str = Field(min_length=1)
    grader: dict[str, object] = Field(default_factory=dict)
    reward_raw: float
    reward_normalized: float = Field(ge=0.0, le=1.0)
    reward_binary: float = Field(ge=0.0, le=1.0)
    usage: dict[str, int] = Field(default_factory=dict)


def _read_raw_inputs(path: Path) -> list[RawTrajectoryInput]:
    rows = path.read_text(encoding="utf-8").splitlines()
    parsed: list[RawTrajectoryInput] = []
    for line_number, row in enumerate(rows, start=1):
        stripped = row.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        try:
            parsed.append(RawTrajectoryInput.model_validate(payload))
        except ValidationError as exc:
            raise ValueError(f"invalid raw trajectory at line {line_number}: {exc}") from exc
    return parsed


def run(input_path: Path, output_path: Path) -> list[EpisodeTrajectoryRecord]:
    raw_inputs = _read_raw_inputs(input_path)
    records: list[EpisodeTrajectoryRecord] = []
    for row in raw_inputs:
        records.append(
            trace_to_trajectory_record(
                trace=row.trace,
                model=row.model,
                grader=row.grader,
                reward_raw=row.reward_raw,
                reward_normalized=row.reward_normalized,
                reward_binary=row.reward_binary,
            )
        )
    write_trajectory_jsonl(output_path, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and export Anthropic-format trajectory JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = run(input_path=Path(args.input), output_path=Path(args.output))
    print(json.dumps({"records": len(records), "output": str(Path(args.output).resolve())}, indent=2))


if __name__ == "__main__":
    main()
