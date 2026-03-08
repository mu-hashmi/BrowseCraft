from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from browsecraft_sim.rl.prompt_variants import PromptVariantRecord, write_prompt_variants_jsonl
from browsecraft_sim.rl.text_qa import generate_text_qa_task, write_text_qa_jsonl
from browsecraft_sim.rl.trajectory import AnthropicMessage, EpisodeTrajectoryRecord, write_trajectory_jsonl


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "sim" / "scripts" / "export_stage_manifests.py"
_SPEC = importlib.util.spec_from_file_location("browsecraft_export_stage_manifests", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
StageManifestRecord = _MODULE.StageManifestRecord


def _trajectory_record(*, task_id: str, tier: str, prompt: str, reward_binary: float) -> EpisodeTrajectoryRecord:
    reward_normalized = 0.9 if reward_binary else 0.2
    return EpisodeTrajectoryRecord(
        episode_id=f"{task_id}-episode",
        task_id=task_id,
        tier=tier,
        task_mode="build",
        seed=7,
        model="claude-sonnet-4-6",
        system_prompt="system",
        messages=[
            AnthropicMessage.model_validate({"role": "user", "content": [{"type": "text", "text": prompt}]}),
            AnthropicMessage.model_validate({"role": "assistant", "content": [{"type": "text", "text": "done"}]}),
        ],
        tool_round_count=1,
        tool_call_count=1,
        grader={"correctness": reward_binary},
        reward_raw=reward_normalized,
        reward_normalized=reward_normalized,
        reward_binary=reward_binary,
        final_world_diff=[],
        started_at="2026-03-07T00:00:00+00:00",
        ended_at="2026-03-07T00:00:10+00:00",
    )


def _read_stage_records(path: Path) -> list[StageManifestRecord]:
    return [StageManifestRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_export_stage_manifests_builds_sft_and_weighted_grpo_outputs(tmp_path) -> None:
    trajectories = [
        _trajectory_record(
            task_id="t1_absolute:absolute_single_block:7:0",
            tier="t1_absolute",
            prompt="Place one minecraft:stone block at absolute coordinates (x=1, y=64, z=2).",
            reward_binary=1.0,
        ),
        _trajectory_record(
            task_id="t2_relative_single_ref:relative_single_reference:7:0",
            tier="t2_relative_single_ref",
            prompt="Reference block prompt",
            reward_binary=0.0,
        ),
        _trajectory_record(
            task_id="t4_structure_relative:marker_chain_place:7:0",
            tier="t4_structure_relative",
            prompt="T4 prompt",
            reward_binary=1.0,
        ),
        _trajectory_record(
            task_id="t5_modification:add_shared_wall_doorway:7:0",
            tier="t5_modification",
            prompt="T5 first prompt",
            reward_binary=1.0,
        ),
        _trajectory_record(
            task_id="t5_modification:add_window_to_wall:7:1",
            tier="t5_modification",
            prompt="T5 second prompt",
            reward_binary=0.0,
        ),
        _trajectory_record(
            task_id="t6_composition:bridge_between_structures:7:0",
            tier="t6_composition",
            prompt="T6 prompt",
            reward_binary=1.0,
        ),
    ]
    trajectories_path = tmp_path / "trajectories.jsonl"
    write_trajectory_jsonl(trajectories_path, trajectories)

    paraphrases_path = tmp_path / "prompt_variants.jsonl"
    write_prompt_variants_jsonl(
        paraphrases_path,
        [
            PromptVariantRecord(
                task_id="t1_absolute:absolute_single_block:7:0",
                tier="t1_absolute",
                family="absolute_single_block",
                seed=7,
                original_prompt="Place one minecraft:stone block at absolute coordinates (x=1, y=64, z=2).",
                verified_paraphrases=[
                    "Put a single stone block at x 1, y 64, z 2.",
                    "Put a single stone block at x 1, y 64, z 2.",
                ],
                shortfall=1,
            )
        ],
    )

    verified_qa_cache_path = tmp_path / "verified_world_qa.jsonl"
    write_text_qa_jsonl(
        verified_qa_cache_path,
        [generate_text_qa_task(tier="qa_multi_hop_chain", seed=19, index=2)],
    )

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "baseline_summary_test.csv").write_text(
        "tier,mean_reward\n"
        "t4_structure_relative,0.2\n"
        "t5_modification,0.5\n"
        "t6_composition,0.9\n",
        encoding="utf-8",
    )

    sft_output = tmp_path / "sft_stage1.jsonl"
    grpo_output = tmp_path / "grpo_stage2.jsonl"
    curriculum_output = tmp_path / "curriculum.json"
    summary = _MODULE.run(
        build_trajectories_path=trajectories_path,
        sft_output=sft_output,
        grpo_output=grpo_output,
        paraphrase_cache_path=str(paraphrases_path),
        text_qa_trajectories_path=None,
        verified_qa_cache_path=str(verified_qa_cache_path),
        text_qa_seed=13,
        text_qa_per_tier=1,
        runs_dir=runs_dir,
        grpo_total_records=4,
        curriculum_output=str(curriculum_output),
        curriculum_low_reward=0.1,
        curriculum_high_reward=0.5,
    )

    assert summary["sft_stage1"] == 7
    assert summary["grpo_stage2"] == 3

    sft_records = _read_stage_records(sft_output)
    build_records = [record for record in sft_records if record.metadata["source"] == "build_trajectory"]
    assert len(build_records) == 2
    build_prompts = [record.input["messages"][0]["content"][0]["text"] for record in build_records]
    assert "Place one minecraft:stone block at absolute coordinates (x=1, y=64, z=2)." in build_prompts
    assert "Put a single stone block at x 1, y 64, z 2." in build_prompts

    grpo_records = _read_stage_records(grpo_output)
    tier_counts = {tier: 0 for tier in ("t4_structure_relative", "t5_modification", "t6_composition")}
    for record in grpo_records:
        tier_counts[record.metadata["tier"]] += 1
        assert record.rubric == {"reward": record.metadata["reward_normalized"]}
    assert tier_counts == {
        "t4_structure_relative": 1,
        "t5_modification": 1,
        "t6_composition": 1,
    }

    curriculum = json.loads(curriculum_output.read_text(encoding="utf-8"))
    assert curriculum["curriculum_low_reward"] == 0.1
    assert curriculum["curriculum_high_reward"] == 0.5
    assert curriculum["weights"]["t4_structure_relative"] == 1
    assert curriculum["weights"]["t5_modification"] == 1
    assert curriculum["trajectory_mean_rewards"]["t5_modification"] == pytest.approx(0.55)
    assert curriculum["trajectory_family_success_rates"]["t5_modification:add_window_to_wall"] == 0.0
