from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def reward_to_success(reward: float, *, threshold: float = 0.8) -> bool:
    return reward >= threshold


def _row_reward(row: dict[str, Any]) -> float:
    if "reward" in row:
        return float(row["reward"])
    if "reward_normalized" in row:
        return float(row["reward_normalized"])
    if "reward_binary" in row:
        return float(row["reward_binary"])
    raise KeyError("row is missing reward, reward_binary, and reward_normalized")


def _task_family_key(row: dict[str, Any]) -> str:
    task_id = str(row["task_id"])
    tier, family, _, _ = task_id.split(":", 3)
    return f"{tier}:{family}"


def _rolling_success_rates(
    rows: Sequence[dict[str, Any]],
    *,
    key_fn,
    window_size: int,
    threshold: float,
    selected_keys: set[str],
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if selected_keys and key not in selected_keys:
            continue
        grouped[key].append(reward_to_success(_row_reward(row), threshold=threshold))

    rates: dict[str, float] = {}
    for key, outcomes in grouped.items():
        recent = outcomes[-window_size:]
        if recent:
            rates[key] = sum(1 for outcome in recent if outcome) / len(recent)
    return rates


def _rolling_mean_rewards(
    rows: Sequence[dict[str, Any]],
    *,
    key_fn,
    window_size: int,
    selected_keys: set[str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if selected_keys and key not in selected_keys:
            continue
        grouped[key].append(_row_reward(row))

    rewards: dict[str, float] = {}
    for key, values in grouped.items():
        recent = values[-window_size:]
        if recent:
            rewards[key] = sum(recent) / len(recent)
    return rewards


def rolling_tier_success_rates(
    rows: Sequence[dict[str, Any]],
    *,
    window_size: int = 100,
    threshold: float = 0.8,
    tiers: Iterable[str] | None = None,
) -> dict[str, float]:
    return _rolling_success_rates(
        rows,
        key_fn=lambda row: str(row["tier"]),
        window_size=window_size,
        threshold=threshold,
        selected_keys=set(tiers or ()),
    )


def rolling_tier_mean_rewards(
    rows: Sequence[dict[str, Any]],
    *,
    window_size: int = 100,
    tiers: Iterable[str] | None = None,
) -> dict[str, float]:
    return _rolling_mean_rewards(
        rows,
        key_fn=lambda row: str(row["tier"]),
        window_size=window_size,
        selected_keys=set(tiers or ()),
    )


def rolling_family_success_rates(
    rows: Sequence[dict[str, Any]],
    *,
    window_size: int = 100,
    threshold: float = 0.8,
    families: Iterable[str] | None = None,
) -> dict[str, float]:
    return _rolling_success_rates(
        rows,
        key_fn=_task_family_key,
        window_size=window_size,
        threshold=threshold,
        selected_keys=set(families or ()),
    )


def rolling_family_mean_rewards(
    rows: Sequence[dict[str, Any]],
    *,
    window_size: int = 100,
    families: Iterable[str] | None = None,
) -> dict[str, float]:
    return _rolling_mean_rewards(
        rows,
        key_fn=_task_family_key,
        window_size=window_size,
        selected_keys=set(families or ()),
    )


def curriculum_weights(
    reward_levels: dict[str, float],
    *,
    low: float = 0.2,
    high: float = 0.7,
) -> dict[str, int]:
    weights: dict[str, int] = {}
    for tier, reward_level in reward_levels.items():
        weights[tier] = 2 if low <= reward_level <= high else 1
    return weights


def weighted_task_counts(
    *,
    total_tasks: int,
    tiers: Sequence[str],
    weights: dict[str, int],
) -> dict[str, int]:
    if total_tasks <= 0:
        raise ValueError("total_tasks must be > 0")
    if not tiers:
        raise ValueError("tiers must not be empty")

    active_weights = {tier: weights.get(tier, 1) for tier in tiers}
    total_weight = sum(active_weights.values())
    counts = {tier: (total_tasks * active_weights[tier]) // total_weight for tier in tiers}
    allocated = sum(counts.values())
    for tier in tiers[: max(0, total_tasks - allocated)]:
        counts[tier] += 1
    return counts


def bootstrap_success_rates(
    *,
    runs_dir: str | Path,
    tiers: Iterable[str] | None = None,
    threshold: float = 0.8,
) -> dict[str, float]:
    runs_path = Path(runs_dir)
    selected_tiers = set(tiers or ())
    episode_files = sorted(runs_path.glob("baseline_episodes*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if episode_files:
        with episode_files[0].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rolling_tier_success_rates(rows, threshold=threshold, tiers=selected_tiers or None)

    summary_files = sorted(runs_path.glob("baseline_summary*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not summary_files:
        return {tier: 0.5 for tier in selected_tiers}

    grouped: dict[str, list[float]] = defaultdict(list)
    with summary_files[0].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tier = str(row["tier"])
            if selected_tiers and tier not in selected_tiers:
                continue
            grouped[tier].append(float(row["mean_reward"]))
    rates = {tier: sum(values) / len(values) for tier, values in grouped.items()}
    for tier in selected_tiers:
        rates.setdefault(tier, 0.5)
    return rates


def bootstrap_mean_rewards(
    *,
    runs_dir: str | Path,
    tiers: Iterable[str] | None = None,
) -> dict[str, float]:
    runs_path = Path(runs_dir)
    selected_tiers = set(tiers or ())
    episode_files = sorted(runs_path.glob("baseline_episodes*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if episode_files:
        with episode_files[0].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rolling_tier_mean_rewards(rows, tiers=selected_tiers or None)

    summary_files = sorted(runs_path.glob("baseline_summary*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not summary_files:
        return {tier: 0.5 for tier in selected_tiers}

    grouped: dict[str, list[float]] = defaultdict(list)
    with summary_files[0].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tier = str(row["tier"])
            if selected_tiers and tier not in selected_tiers:
                continue
            grouped[tier].append(float(row["mean_reward"]))
    rewards = {tier: sum(values) / len(values) for tier, values in grouped.items()}
    for tier in selected_tiers:
        rewards.setdefault(tier, 0.5)
    return rewards


def bootstrap_family_success_rates(
    *,
    runs_dir: str | Path,
    families: Iterable[str] | None = None,
    threshold: float = 0.8,
) -> dict[str, float]:
    runs_path = Path(runs_dir)
    selected_families = set(families or ())
    episode_files = sorted(runs_path.glob("baseline_episodes*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not episode_files:
        return {}
    with episode_files[0].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rolling_family_success_rates(rows, threshold=threshold, families=selected_families or None)


def bootstrap_family_mean_rewards(
    *,
    runs_dir: str | Path,
    families: Iterable[str] | None = None,
) -> dict[str, float]:
    runs_path = Path(runs_dir)
    selected_families = set(families or ())
    episode_files = sorted(runs_path.glob("baseline_episodes*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not episode_files:
        return {}
    with episode_files[0].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rolling_family_mean_rewards(rows, families=selected_families or None)
