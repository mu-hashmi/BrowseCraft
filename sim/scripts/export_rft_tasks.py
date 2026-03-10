from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from browsecraft_sim.rl.trajectory import EpisodeTrajectoryRecord, read_trajectory_jsonl


class RftTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: dict[str, Any]
    rubric: dict[str, Any]
    metadata: dict[str, Any]


def _trajectory_to_rft_task(record: EpisodeTrajectoryRecord) -> RftTaskRecord:
    payload = {
        "system_prompt": record.system_prompt,
        "messages": [message.model_dump(mode="json") for message in record.messages],
    }
    rubric = {"reward": record.reward_normalized}
    metadata = {
        "episode_id": record.episode_id,
        "task_id": record.task_id,
        "tier": record.tier,
        "task_mode": record.task_mode,
        "seed": record.seed,
        "model": record.model,
        "tool_call_count": record.tool_call_count,
        "tool_round_count": record.tool_round_count,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "reward_raw": record.reward_raw,
        "reward_normalized": record.reward_normalized,
        "reward_binary": record.reward_binary,
        "grader": record.grader,
        "final_world_diff": record.final_world_diff,
    }
    return RftTaskRecord(input=payload, rubric=rubric, metadata=metadata)


def run(*, trajectories_path: Path, output_path: Path) -> list[RftTaskRecord]:
    trajectories = read_trajectory_jsonl(trajectories_path)
    records = [_trajectory_to_rft_task(record) for record in trajectories]
    lines = [record.model_dump_json() for record in records]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert validated trajectory JSONL into HUD RFT input JSONL.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output).resolve()
    records = run(trajectories_path=Path(args.trajectories), output_path=output)
    print(json.dumps({"output": str(output), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
