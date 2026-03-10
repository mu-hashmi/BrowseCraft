from __future__ import annotations

import pytest

from browsecraft_sim.rl.curriculum import (
    bootstrap_mean_rewards,
    bootstrap_family_success_rates,
    bootstrap_success_rates,
    curriculum_weights,
    rolling_family_mean_rewards,
    rolling_family_success_rates,
    rolling_tier_mean_rewards,
    rolling_tier_success_rates,
    weighted_task_counts,
)


def test_rolling_tier_success_rates_uses_recent_window() -> None:
    rows = [{"tier": "t4_structure_relative", "reward": 0.0} for _ in range(20)]
    rows.extend({"tier": "t4_structure_relative", "reward": 1.0} for _ in range(60))
    rows.extend({"tier": "t4_structure_relative", "reward": 0.0} for _ in range(40))
    rates = rolling_tier_success_rates(rows, window_size=100)
    assert rates["t4_structure_relative"] == 0.6


def test_curriculum_weights_double_only_mid_band_tiers() -> None:
    weights = curriculum_weights(
        {
            "t4_structure_relative": 0.05,
            "t5_modification": 0.2,
            "t6_composition": 0.75,
        }
    )
    assert weights == {
        "t4_structure_relative": 1,
        "t5_modification": 2,
        "t6_composition": 1,
    }


def test_rolling_family_success_rates_uses_task_id_family_key() -> None:
    rows = [{"task_id": "t5_modification:add_window_to_wall:7:0", "reward_binary": 0.0} for _ in range(20)]
    rows.extend({"task_id": "t5_modification:add_window_to_wall:7:0", "reward_binary": 1.0} for _ in range(60))
    rows.extend({"task_id": "t5_modification:add_window_to_wall:7:0", "reward_binary": 0.0} for _ in range(40))
    rates = rolling_family_success_rates(rows, window_size=100)
    assert rates["t5_modification:add_window_to_wall"] == 0.6


def test_rolling_tier_mean_rewards_uses_recent_window() -> None:
    rows = [{"tier": "t5_modification", "reward_normalized": 0.0} for _ in range(20)]
    rows.extend({"tier": "t5_modification", "reward_normalized": 0.4} for _ in range(60))
    rows.extend({"tier": "t5_modification", "reward_normalized": 0.2} for _ in range(40))

    rewards = rolling_tier_mean_rewards(rows, window_size=100)

    assert rewards["t5_modification"] == pytest.approx(0.32)


def test_rolling_family_mean_rewards_uses_task_id_family_key() -> None:
    rows = [{"task_id": "t5_modification:add_window_to_wall:7:0", "reward_normalized": 0.1} for _ in range(20)]
    rows.extend({"task_id": "t5_modification:add_window_to_wall:7:0", "reward_normalized": 0.4} for _ in range(60))
    rows.extend({"task_id": "t5_modification:add_window_to_wall:7:0", "reward_normalized": 0.2} for _ in range(40))

    rewards = rolling_family_mean_rewards(rows, window_size=100)

    assert rewards["t5_modification:add_window_to_wall"] == pytest.approx(0.32)


def test_weighted_task_counts_allocate_more_to_double_weight_tier() -> None:
    counts = weighted_task_counts(
        total_tasks=8,
        tiers=["t4_structure_relative", "t5_modification", "t6_composition"],
        weights={"t4_structure_relative": 1, "t5_modification": 2, "t6_composition": 1},
    )
    assert counts["t5_modification"] > counts["t4_structure_relative"]
    assert counts["t5_modification"] > counts["t6_composition"]
    assert sum(counts.values()) == 8


def test_bootstrap_success_rates_fill_missing_selected_tiers(tmp_path) -> None:
    summary = tmp_path / "baseline_summary_test.csv"
    summary.write_text("tier,mean_reward\n" "t5_modification,0.55\n", encoding="utf-8")
    rates = bootstrap_success_rates(
        runs_dir=tmp_path,
        tiers=("t4_structure_relative", "t5_modification", "t6_composition"),
    )
    assert rates == {
        "t4_structure_relative": 0.5,
        "t5_modification": 0.55,
        "t6_composition": 0.5,
    }


def test_bootstrap_mean_rewards_fill_missing_selected_tiers(tmp_path) -> None:
    summary = tmp_path / "baseline_summary_test.csv"
    summary.write_text("tier,mean_reward\n" "t5_modification,0.35\n", encoding="utf-8")

    rewards = bootstrap_mean_rewards(
        runs_dir=tmp_path,
        tiers=("t4_structure_relative", "t5_modification", "t6_composition"),
    )

    assert rewards == {
        "t4_structure_relative": 0.5,
        "t5_modification": 0.35,
        "t6_composition": 0.5,
    }


def test_bootstrap_family_success_rates_reads_episode_csv(tmp_path) -> None:
    episodes = tmp_path / "baseline_episodes_test.csv"
    episodes.write_text(
        "task_id,reward_binary\n"
        "t5_modification:add_window_to_wall:7:0,0\n"
        "t5_modification:add_window_to_wall:7:1,1\n",
        encoding="utf-8",
    )
    rates = bootstrap_family_success_rates(runs_dir=tmp_path)
    assert rates["t5_modification:add_window_to_wall"] == 0.5
