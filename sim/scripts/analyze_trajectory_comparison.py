from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from browsecraft_sim.rl.config import RewardConfig
from browsecraft_sim.rl.types import ALL_TIERS


EFFICIENCY_MIN_CORRECTNESS = RewardConfig().efficiency_min_correctness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two BrowseCraft trajectory JSONL runs.")
    parser.add_argument("--old", required=True, help="Path to the old trajectory JSONL run.")
    parser.add_argument("--new", required=True, help="Path to the new trajectory JSONL run.")
    return parser


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def group_by_tier(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trace"]["tier"]].append(row)
    return grouped


def mean_reward(rows: list[dict[str, Any]]) -> float:
    return fmean(row["reward_normalized"] for row in rows)


def mean_correctness(rows: list[dict[str, Any]]) -> float:
    return fmean(row["grader"]["correctness_score"] for row in rows)


def mean_efficiency_effective(rows: list[dict[str, Any]]) -> float:
    return fmean(efficiency_effective(row) for row in rows)


def mean_structural(rows: list[dict[str, Any]]) -> float:
    return fmean(row["grader"]["structural_score"] for row in rows)


def mean_tool_calls(rows: list[dict[str, Any]]) -> float:
    return fmean(len(row["trace"]["tool_calls"]) for row in rows)


def efficiency_effective(row: dict[str, Any]) -> float:
    details = row["grader"]["details"]
    if "efficiency_effective" in details:
        return float(details["efficiency_effective"])
    expected_calls = int(details["expected_tool_calls"])
    actual_calls = max(int(details["actual_tool_calls"]), 1)
    correctness = float(row["grader"]["correctness_score"])
    if correctness < EFFICIENCY_MIN_CORRECTNESS:
        return 0.0
    return min(1.0, expected_calls / actual_calls) * correctness


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "reward": mean_reward(rows),
        "correctness": mean_correctness(rows),
        "efficiency_effective": mean_efficiency_effective(rows),
        "structural": mean_structural(rows),
        "tool_calls": mean_tool_calls(rows),
    }


def ordered_tiers(*grouped_runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    seen = set().union(*(grouped.keys() for grouped in grouped_runs))
    tiers = [tier for tier in ALL_TIERS if tier in seen]
    tiers.extend(sorted(seen - set(ALL_TIERS)))
    return tiers


def format_number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    if signed:
        return f"{value:+.3f}"
    return f"{value:.3f}"


def format_pair(old_value: float | None, new_value: float | None) -> str:
    return f"{format_number(old_value)}/{format_number(new_value)}"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    divider = " | ".join("-" * width for width in widths)
    lines = [
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        divider,
    ]
    for row in rows:
        lines.append(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def recover_failed_tools(row: dict[str, Any]) -> list[str]:
    failures = []
    for call in row["trace"]["tool_calls"]:
        if call["success"]:
            continue
        if call["error"]:
            failures.append(f"{call['name']}: {call['error']}")
        else:
            failures.append(str(call["name"]))
    if failures:
        return failures

    tool_names_by_id: dict[str, str] = {}
    for message in row["trace"]["messages"]:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block["type"] == "tool_use":
                tool_names_by_id[block["id"]] = block["name"]

    for message in row["trace"]["messages"]:
        if message["role"] != "user":
            continue
        for block in message["content"]:
            if block["type"] != "tool_result" or not block.get("is_error"):
                continue
            name = tool_names_by_id.get(block["tool_use_id"], block["tool_use_id"])
            failures.append(f"{name}: {block['content']}")
    return failures


def invalid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row["grader"]["format_valid"]]


def print_per_tier_summary(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    old_grouped = group_by_tier(old_rows)
    new_grouped = group_by_tier(new_rows)
    table_rows: list[list[str]] = []
    for tier in ordered_tiers(old_grouped, new_grouped):
        old_summary = summarize(old_grouped[tier]) if tier in old_grouped else None
        new_summary = summarize(new_grouped[tier]) if tier in new_grouped else None
        table_rows.append(
            [
                tier,
                format_number(old_summary["reward"] if old_summary else None),
                format_number(new_summary["reward"] if new_summary else None),
                format_pair(
                    old_summary["correctness"] if old_summary else None,
                    new_summary["correctness"] if new_summary else None,
                ),
                format_pair(
                    old_summary["efficiency_effective"] if old_summary else None,
                    new_summary["efficiency_effective"] if new_summary else None,
                ),
                format_pair(
                    old_summary["structural"] if old_summary else None,
                    new_summary["structural"] if new_summary else None,
                ),
                format_pair(
                    old_summary["tool_calls"] if old_summary else None,
                    new_summary["tool_calls"] if new_summary else None,
                ),
            ]
        )

    print("Per-tier summary")
    print(
        render_table(
            [
                "tier",
                "reward old",
                "reward new",
                "correctness o/n",
                "eff_eff o/n",
                "structural o/n",
                "tool calls o/n",
            ],
            table_rows,
        )
    )


def print_invalid_episodes(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"{label}:")
    invalid = invalid_rows(rows)
    if not invalid:
        print("  none")
        return
    for row in invalid:
        failures = recover_failed_tools(row)
        failure_text = "; ".join(failures) if failures else "no failed tool call recorded in trace"
        print(
            "  - "
            f"{row['trace']['tier']} "
            f"{row['trace']['task_id']} "
            f"(episode {row['trace']['episode_id']}, reward={row['reward_normalized']:.3f}) "
            f"failed={failure_text}"
        )


def reward_means_by_tier(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped = group_by_tier(rows)
    return {tier: mean_reward(grouped[tier]) for tier in grouped}


def print_reward_cliff(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    old_means = reward_means_by_tier(old_rows)
    new_means = reward_means_by_tier(new_rows)
    tiers = ordered_tiers(group_by_tier(old_rows), group_by_tier(new_rows))
    table_rows: list[list[str]] = []
    for previous_tier, current_tier in zip(tiers, tiers[1:]):
        old_drop = None
        new_drop = None
        if previous_tier in old_means and current_tier in old_means:
            old_drop = old_means[previous_tier] - old_means[current_tier]
        if previous_tier in new_means and current_tier in new_means:
            new_drop = new_means[previous_tier] - new_means[current_tier]
        table_rows.append(
            [
                f"{previous_tier} -> {current_tier}",
                format_number(old_drop, signed=True),
                format_number(new_drop, signed=True),
            ]
        )

    print("Reward cliff")
    print(render_table(["transition", "drop old", "drop new"], table_rows))


def main() -> None:
    args = build_parser().parse_args()
    old_rows = load_rows(args.old)
    new_rows = load_rows(args.new)
    print_per_tier_summary(old_rows, new_rows)
    print()
    print("Invalid format episodes")
    print_invalid_episodes("old", old_rows)
    print_invalid_episodes("new", new_rows)
    print()
    print_reward_cliff(old_rows, new_rows)


if __name__ == "__main__":
    main()
