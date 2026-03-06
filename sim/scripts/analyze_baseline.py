from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from browsecraft_sim.rl.metrics import placement_map
from browsecraft_sim.rl.task_generator import reconstruct_task_from_task_id
from browsecraft_sim.rl.types import BlockPlacement, Tier


TIER_ORDER: list[Tier] = [
    "t1_absolute",
    "t2_relative_single_ref",
    "t3_primitives",
    "t4_structure_relative",
    "t5_modification",
    "t6_composition",
]


def parse_task_id(task_id: str) -> tuple[Tier, str, int, int]:
    tier, family, task_seed, index = task_id.split(":", maxsplit=3)
    return tier, family, int(task_seed), int(index)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def count_block_errors(task, row: dict) -> tuple[int, int, int]:
    expected_map = placement_map(task.target_blocks)
    actual_map = placement_map(BlockPlacement.model_validate(item) for item in row["trace"]["final_world_diff"])

    missing_blocks = 0
    wrong_material_blocks = 0
    for coord, expected_block in expected_map.items():
        actual_block = actual_map.get(coord)
        if actual_block is None:
            missing_blocks += 1
            continue
        if actual_block != expected_block:
            wrong_material_blocks += 1

    extra_blocks = sum(1 for coord in actual_map if coord not in expected_map)
    return missing_blocks, extra_blocks, wrong_material_blocks


def error_text(row: dict) -> str:
    errors = [
        f"{tool_call['name']}:{tool_call['error']}"
        for tool_call in row["trace"]["tool_calls"]
        if not tool_call["success"]
    ]
    return "; ".join(errors)


def hit_max_rounds(row: dict, max_rounds: int) -> bool:
    return (
        row["trace"]["tool_round_count"] >= max_rounds
        and not row["grader"]["format_valid"]
        and not any(not tool_call["success"] for tool_call in row["trace"]["tool_calls"])
    )


def tool_sequence(row: dict) -> str:
    return ">".join(tool_call["name"] for tool_call in row["trace"]["tool_calls"])


def write_episode_csv(*, rows: list[dict], output: Path, max_rounds: int) -> dict[str, list[int]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "tier",
        "family",
        "seed",
        "reward",
        "correctness",
        "efficiency_effective",
        "structural",
        "tool_calls",
        "tool_sequence",
        "format_valid",
        "hit_max_rounds",
        "missing_blocks",
        "extra_blocks",
        "wrong_material_blocks",
        "error",
    ]
    tier_tool_calls: dict[str, list[int]] = defaultdict(list)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            tier, family, _, _ = parse_task_id(row["trace"]["task_id"])
            task = reconstruct_task_from_task_id(row["trace"]["task_id"])
            if task.task_id != row["trace"]["task_id"]:
                raise ValueError(f"task reconstruction mismatch for {row['trace']['task_id']}")
            missing_blocks, extra_blocks, wrong_material_blocks = count_block_errors(task, row)
            tool_calls = len(row["trace"]["tool_calls"])
            tier_tool_calls[tier].append(tool_calls)
            writer.writerow(
                {
                    "task_id": row["trace"]["task_id"],
                    "tier": tier,
                    "family": family,
                    "seed": row["trace"]["seed"],
                    "reward": row["reward_normalized"],
                    "correctness": row["grader"]["correctness_score"],
                    "efficiency_effective": row["grader"]["details"].get(
                        "efficiency_effective",
                        row["grader"]["efficiency_score"],
                    ),
                    "structural": row["grader"]["structural_score"],
                    "tool_calls": tool_calls,
                    "tool_sequence": tool_sequence(row),
                    "format_valid": row["grader"]["format_valid"],
                    "hit_max_rounds": hit_max_rounds(row, max_rounds),
                    "missing_blocks": missing_blocks,
                    "extra_blocks": extra_blocks,
                    "wrong_material_blocks": wrong_material_blocks,
                    "error": error_text(row),
                }
            )
    return tier_tool_calls


def write_summary_csv(*, rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        tier, family, _, _ = parse_task_id(row["trace"]["task_id"])
        grouped[(tier, family)].append(row)

    fieldnames = [
        "tier",
        "family",
        "n",
        "mean_reward",
        "std_reward",
        "min_reward",
        "max_reward",
        "mean_correctness",
        "mean_structural",
        "p25_tool_calls",
        "p50_tool_calls",
        "p75_tool_calls",
        "p90_tool_calls",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tier in TIER_ORDER:
            families = sorted(family for candidate_tier, family in grouped if candidate_tier == tier)
            for family in families:
                family_rows = grouped[(tier, family)]
                rewards = [row["reward_normalized"] for row in family_rows]
                tool_calls = [len(row["trace"]["tool_calls"]) for row in family_rows]
                writer.writerow(
                    {
                        "tier": tier,
                        "family": family,
                        "n": len(family_rows),
                        "mean_reward": mean(rewards),
                        "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
                        "min_reward": min(rewards),
                        "max_reward": max(rewards),
                        "mean_correctness": mean(row["grader"]["correctness_score"] for row in family_rows),
                        "mean_structural": mean(row["grader"]["structural_score"] for row in family_rows),
                        "p25_tool_calls": percentile(tool_calls, 0.25),
                        "p50_tool_calls": percentile(tool_calls, 0.50),
                        "p75_tool_calls": percentile(tool_calls, 0.75),
                        "p90_tool_calls": percentile(tool_calls, 0.90),
                    }
                )


def usage_totals(rows: list[dict]) -> dict[str, int]:
    first = rows[0]
    if "usage" not in first:
        return {}
    return {
        key: sum(row["usage"][key] for row in rows)
        for key in first["usage"]
    }


def suggested_budgets(rows: list[dict]) -> dict[str, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["trace"]["tier"]].append(len(row["trace"]["tool_calls"]))
    return {tier: max(1, math.ceil(percentile(grouped[tier], 0.75))) for tier in TIER_ORDER}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze BrowseCraft baseline trajectory output.")
    parser.add_argument("--input", default="runs/claude_baseline_n25.jsonl")
    parser.add_argument("--episodes-csv", default="runs/baseline_episodes.csv")
    parser.add_argument("--summary-csv", default="runs/baseline_summary.csv")
    parser.add_argument("--max-rounds", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.input))
    write_episode_csv(
        rows=rows,
        output=Path(args.episodes_csv),
        max_rounds=args.max_rounds,
    )
    write_summary_csv(rows=rows, output=Path(args.summary_csv))
    print(
        json.dumps(
            {
                "input": str(Path(args.input).resolve()),
                "episodes_csv": str(Path(args.episodes_csv).resolve()),
                "summary_csv": str(Path(args.summary_csv).resolve()),
                "episodes": len(rows),
                "usage": usage_totals(rows),
                "suggested_budgets_p75": suggested_budgets(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
