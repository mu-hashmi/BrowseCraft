from __future__ import annotations

from .config import RewardConfig


def effective_efficiency_score(
    *,
    correctness_score: float,
    efficiency_score: float,
    config: RewardConfig,
) -> float:
    if correctness_score < config.efficiency_min_correctness:
        return 0.0
    return efficiency_score * correctness_score


def compose_reward(
    *,
    format_valid: bool,
    correctness_score: float,
    efficiency_score: float,
    structural_score: float,
    config: RewardConfig,
) -> tuple[float, float, float]:
    format_score = 1.0 if format_valid else 0.0
    gated_efficiency = effective_efficiency_score(
        correctness_score=correctness_score,
        efficiency_score=efficiency_score,
        config=config,
    )

    if config.format_mode == "gate":
        if not format_valid:
            return format_score, 0.0, 0.0
        raw = (
            config.weight_correctness * correctness_score
            + config.weight_efficiency * gated_efficiency
            + config.weight_structural * structural_score
        )
        max_raw = config.weight_correctness + config.weight_efficiency + config.weight_structural
        normalized = 0.0 if max_raw == 0 else raw / max_raw
        return format_score, raw, max(0.0, min(1.0, normalized))

    raw = (
        config.weight_format * format_score
        + config.weight_correctness * correctness_score
        + config.weight_efficiency * gated_efficiency
        + config.weight_structural * structural_score
    )
    max_raw = (
        config.weight_format
        + config.weight_correctness
        + config.weight_efficiency
        + config.weight_structural
    )
    normalized = 0.0 if max_raw == 0 else raw / max_raw
    return format_score, raw, max(0.0, min(1.0, normalized))
