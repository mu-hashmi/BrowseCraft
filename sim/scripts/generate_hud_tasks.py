from __future__ import annotations

import argparse
import json
from pathlib import Path

from browsecraft_sim.rl.config import load_reward_config
from browsecraft_sim.rl.task_generator import generate_tasks, tier_counts
from browsecraft_sim.rl.types import ALL_TIERS, Tier


def _parse_tiers(raw: str | None) -> list[Tier]:
    if raw is None or not raw.strip():
        return list(ALL_TIERS)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in ALL_TIERS]
    if invalid:
        raise ValueError(f"unsupported tiers: {', '.join(invalid)}")
    return requested  # type: ignore[return-value]


def _task_record(
    env_name: str,
    task_payload: dict[str, object],
    reward_config: dict[str, object],
) -> dict[str, object]:
    return {
        "env": {"name": env_name},
        "scenario": task_payload["tier"],
        "args": {"task_spec": task_payload, "reward_config": reward_config},
    }


def run(
    *,
    seed: int,
    per_tier: int,
    tiers: list[Tier],
    env_name: str,
    output: Path,
    reward_config: dict[str, object],
) -> None:
    tasks = generate_tasks(seed=seed, per_tier=per_tier, tiers=tiers)
    lines = []
    for task in tasks:
        payload = task.model_dump(mode="json")
        record = _task_record(env_name=env_name, task_payload=payload, reward_config=reward_config)
        lines.append(json.dumps(record))
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    summary = {
        "output": str(output),
        "seed": seed,
        "per_tier": per_tier,
        "tier_counts": tier_counts(tasks),
        "total": len(tasks),
        "reward_config": reward_config,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic HUD task JSONL from BrowseCraft tiers.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=100)
    parser.add_argument("--tiers", default=None, help="Comma-separated list of tiers. Default: all tiers.")
    parser.add_argument("--env-name", default="browsecraft-spatial-rl")
    parser.add_argument("--output", default="remote_tasks.jsonl")
    parser.add_argument("--reward-config-file", default=None)
    parser.add_argument("--format-mode", choices=("gate", "weighted"), default=None)
    parser.add_argument("--weight-correctness", type=float, default=None)
    parser.add_argument("--weight-efficiency", type=float, default=None)
    parser.add_argument("--weight-structural", type=float, default=None)
    parser.add_argument("--weight-format", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tiers = _parse_tiers(args.tiers)
    overrides = {
        key: value
        for key, value in {
            "format_mode": args.format_mode,
            "weight_correctness": args.weight_correctness,
            "weight_efficiency": args.weight_efficiency,
            "weight_structural": args.weight_structural,
            "weight_format": args.weight_format,
        }.items()
        if value is not None
    }
    reward_config = load_reward_config(path=args.reward_config_file, overrides=overrides).model_dump(mode="json")
    run(
        seed=args.seed,
        per_tier=args.per_tier,
        tiers=tiers,
        env_name=str(args.env_name),
        output=Path(args.output),
        reward_config=reward_config,
    )


if __name__ == "__main__":
    main()
