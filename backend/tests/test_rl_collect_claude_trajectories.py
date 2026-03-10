from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "sim" / "scripts" / "collect_claude_trajectories.py"
_SPEC = importlib.util.spec_from_file_location("browsecraft_collect_claude_trajectories", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _FakeRandom:
    instances: list["_FakeRandom"] = []

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.calls = 0
        self.weight_history: list[list[int]] = []
        _FakeRandom.instances.append(self)

    def choices(self, population, weights, k=1):
        self.calls += 1
        self.weight_history.append(list(weights))
        if self.calls <= 2:
            return ["t5_modification"]
        max_weight = max(weights)
        return [population[weights.index(max_weight)]]


class _FakeAnthropic:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def close(self) -> None:
        return None


def test_grpo_stage_uses_runtime_curriculum_updates(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(_MODULE, "AsyncAnthropic", _FakeAnthropic)

    async def fake_warmup_prompt_cache(client, *, model: str) -> None:
        return None

    recorded_tiers: list[str] = []

    async def fake_run_episode(client, *, task, model: str, max_rounds: int, reward_config):
        recorded_tiers.append(task.tier)
        reward_binary = 1.0 if task.index % 2 == 0 else 0.0
        return {
            "trace": {
                "task_id": task.task_id,
                "tier": task.tier,
                "tool_call_count": 1,
                "format_valid": True,
            },
            "grader": {"correctness": reward_binary},
            "reward_raw": reward_binary,
            "reward_normalized": reward_binary,
            "reward_binary": reward_binary,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

    def fake_generate_task(*, tier: str, seed: int, index: int):
        return SimpleNamespace(
            tier=tier,
            seed=seed,
            index=index,
            task_id=f"{tier}:fake:{seed}:{index}",
            prompt="prompt",
        )

    monkeypatch.setattr(_MODULE, "_warmup_prompt_cache", fake_warmup_prompt_cache)
    monkeypatch.setattr(_MODULE, "_run_episode", fake_run_episode)
    monkeypatch.setattr(_MODULE, "generate_task", fake_generate_task)
    monkeypatch.setattr(_MODULE.random, "Random", _FakeRandom)
    _FakeRandom.instances.clear()

    args = argparse.Namespace(
        model="claude-sonnet-4-6",
        seed=7,
        stage="grpo_stage2",
        tiers=None,
        sampling="equal",
        per_tier=1,
        total_tasks=5,
        curriculum_runs_dir=str(tmp_path / "runs"),
        curriculum_threshold=0.8,
        curriculum_low_reward=0.2,
        curriculum_high_reward=0.7,
        curriculum_update_every=2,
        max_rounds=4,
        concurrency=1,
        output=str(tmp_path / "episodes.jsonl"),
        log_every=10,
    )

    rows = asyncio.run(_MODULE._run(args))

    assert len(rows) == 5
    assert recorded_tiers[:2] == ["t5_modification", "t5_modification"]
    assert args._sampling_summary["strategy"] == "curriculum"
    assert args._sampling_summary["bootstrap_source"] == "none"
    assert args._sampling_summary["updates"][0]["completed_episodes"] == 2
    assert args._sampling_summary["updates"][0]["family_mean_rewards"]["t5_modification:fake"] == 0.5
    assert args._sampling_summary["updates"][0]["family_success_rates"]["t5_modification:fake"] == 0.5
    assert args._sampling_summary["updates"][0]["mean_rewards"]["t5_modification"] == 0.5
    assert args._sampling_summary["updates"][0]["weights"]["t5_modification"] == 2
    assert args._sampling_summary["final_family_mean_rewards"]["t5_modification:fake"] == 0.5
    assert args._sampling_summary["final_family_success_rates"]["t5_modification:fake"] == 0.5
    assert _FakeRandom.instances[0].weight_history[2] == [1, 2, 1]
