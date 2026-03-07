from __future__ import annotations

import json

import pytest

from browsecraft_sim.rl.config import RewardConfig, load_reward_config
from browsecraft_sim.rl.reward import compose_reward


def test_gate_mode_zeroes_reward_on_format_failure() -> None:
    config = RewardConfig(format_mode="gate")
    format_score, raw, normalized = compose_reward(
        format_valid=False,
        correctness_score=1.0,
        efficiency_score=1.0,
        structural_score=1.0,
        config=config,
    )
    assert format_score == 0.0
    assert raw == 0.0
    assert normalized == 0.0


def test_weighted_mode_includes_format_weight() -> None:
    config = RewardConfig(
        format_mode="weighted",
        weight_format=0.4,
        weight_correctness=0.4,
        weight_efficiency=0.1,
        weight_structural=0.1,
    )
    _, raw, normalized = compose_reward(
        format_valid=True,
        correctness_score=0.5,
        efficiency_score=0.5,
        structural_score=0.5,
        config=config,
    )
    assert raw == pytest.approx(0.675)
    assert normalized == pytest.approx(0.675)


def test_efficiency_is_zero_when_correctness_below_threshold() -> None:
    config = RewardConfig(format_mode="gate", efficiency_min_correctness=0.1)
    _, raw, normalized = compose_reward(
        format_valid=True,
        correctness_score=0.0,
        efficiency_score=1.0,
        structural_score=1.0,
        config=config,
    )
    assert raw == pytest.approx(0.1)
    assert normalized == pytest.approx(0.1)


def test_efficiency_is_scaled_by_correctness_above_threshold() -> None:
    config = RewardConfig(format_mode="gate", efficiency_min_correctness=0.1)
    _, raw, normalized = compose_reward(
        format_valid=True,
        correctness_score=0.5,
        efficiency_score=1.0,
        structural_score=1.0,
        config=config,
    )
    assert raw == pytest.approx(0.55)
    assert normalized == pytest.approx(0.55)


def test_load_reward_config_applies_overrides(tmp_path) -> None:
    config_path = tmp_path / "reward.json"
    config_path.write_text(json.dumps({"format_mode": "weighted", "weight_format": 0.25}), encoding="utf-8")
    loaded = load_reward_config(
        path=config_path,
        overrides={
            "weight_correctness": 0.5,
            "weight_efficiency": 0.25,
            "weight_structural": 0.0,
            "efficiency_min_correctness": 0.2,
        },
    )
    assert loaded.format_mode == "weighted"
    assert loaded.weight_format == 0.25
    assert loaded.weight_correctness == 0.5
    assert loaded.efficiency_min_correctness == 0.2


def test_default_expected_tool_call_budgets_match_calibrated_p75s() -> None:
    config = RewardConfig()
    assert config.expected_tool_calls_by_tier["t1_absolute"] == 1
    assert config.expected_tool_calls_by_tier["t2_relative_single_ref"] == 2
    assert config.expected_tool_calls_by_tier["t3_primitives"] == 2
    assert config.expected_tool_calls_by_tier["t4_structure_relative"] == 4
    assert config.expected_tool_calls_by_tier["t5_modification"] == 8
    assert config.expected_tool_calls_by_tier["t6_composition"] == 8
