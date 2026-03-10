from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from browsecraft_sim.rl.curriculum import (
    bootstrap_family_mean_rewards,
    bootstrap_family_success_rates,
    bootstrap_mean_rewards,
    bootstrap_success_rates,
    curriculum_weights,
    rolling_family_mean_rewards,
    rolling_family_success_rates,
    rolling_tier_mean_rewards,
    rolling_tier_success_rates,
    weighted_task_counts,
)
from browsecraft_sim.rl.prompt_variants import read_prompt_variants_jsonl
from browsecraft_sim.rl.text_qa import (
    canonical_text_qa_response,
    generate_text_qa_tasks,
    read_text_qa_jsonl,
    read_text_qa_trajectory_jsonl,
    text_qa_full_prompt,
)
from browsecraft_sim.rl.trajectory import EpisodeTrajectoryRecord, read_trajectory_jsonl


class StageManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["sft_stage1", "grpo_stage2"]
    task_mode: str = Field(min_length=1)
    input: dict[str, Any]
    rubric: dict[str, Any] | None = None
    metadata: dict[str, Any]


StageManifestRecord.model_rebuild()


def _messages_json(record: EpisodeTrajectoryRecord) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json") for message in record.messages]


def _clone_messages_with_prompt(messages: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    cloned = json.loads(json.dumps(messages))
    first = cloned[0]
    if first["role"] != "user" or not first["content"]:
        raise ValueError("expected first message to be a user prompt")
    for block in first["content"]:
        if block["type"] == "text":
            block["text"] = prompt
            return cloned
    raise ValueError("expected first user message to contain a text block")


def _first_user_prompt(messages: list[dict[str, Any]]) -> str:
    first = messages[0]
    if first["role"] != "user" or not first["content"]:
        raise ValueError("expected first message to be a user prompt")
    for block in first["content"]:
        if block["type"] == "text":
            return str(block["text"])
    raise ValueError("expected first user message to contain a text block")


def _build_sft_records(
    *,
    build_trajectories: list[EpisodeTrajectoryRecord],
    paraphrase_cache_path: str | None,
    text_qa_trajectories_path: str | None,
    text_qa_seed: int,
    text_qa_per_tier: int,
    verified_qa_cache_path: str | None,
) -> list[StageManifestRecord]:
    records: list[StageManifestRecord] = []
    paraphrases_by_task: dict[str, list[str]] = {}
    seen_build_prompts: set[tuple[str, str]] = set()
    if paraphrase_cache_path is not None:
        for record in read_prompt_variants_jsonl(paraphrase_cache_path):
            paraphrases_by_task[record.task_id] = record.verified_paraphrases

    for record in build_trajectories:
        if record.task_mode != "build":
            continue
        if record.tier not in {"t1_absolute", "t2_relative_single_ref", "t3_primitives"}:
            continue
        if record.reward_binary < 1.0:
            continue
        base_messages = _messages_json(record)
        prompt_variants = [None, *paraphrases_by_task.get(record.task_id, [])]
        for prompt_variant in prompt_variants:
            messages = base_messages if prompt_variant is None else _clone_messages_with_prompt(base_messages, prompt_variant)
            prompt_text = _first_user_prompt(messages)
            prompt_key = (record.task_id, prompt_text)
            if prompt_key in seen_build_prompts:
                continue
            seen_build_prompts.add(prompt_key)
            records.append(
                StageManifestRecord(
                    stage="sft_stage1",
                    task_mode=record.task_mode,
                    input={"system_prompt": record.system_prompt, "messages": messages},
                    metadata={
                        "task_id": record.task_id,
                        "tier": record.tier,
                        "source": "build_trajectory",
                        "is_original_prompt": prompt_variant is None,
                        "prompt_variant": prompt_variant,
                    },
                )
            )

    successful_text_qa_task_ids: set[str] = set()
    if text_qa_trajectories_path is not None:
        for record in read_text_qa_trajectory_jsonl(text_qa_trajectories_path):
            if record.reward_binary < 1.0:
                continue
            successful_text_qa_task_ids.add(record.task_id)
            records.append(
                StageManifestRecord(
                    stage="sft_stage1",
                    task_mode="text_qa",
                    input={"system_prompt": record.system_prompt, "messages": record.messages},
                    metadata={
                        "task_id": record.task_id,
                        "tier": record.tier,
                        "source": "text_qa_trajectory",
                        "answer_format": record.answer_format,
                        "model": record.model,
                        "reward_binary": record.reward_binary,
                    },
                )
            )

    text_qa_tasks = generate_text_qa_tasks(seed=text_qa_seed, per_tier=text_qa_per_tier)
    if verified_qa_cache_path is not None:
        text_qa_tasks.extend(read_text_qa_jsonl(verified_qa_cache_path))

    seen_text_qa_keys: set[tuple[str, str]] = set()
    for task in text_qa_tasks:
        if task.task_id in successful_text_qa_task_ids:
            continue
        key = (task.prompt, task.expected_answer)
        if key in seen_text_qa_keys:
            continue
        seen_text_qa_keys.add(key)
        records.append(
            StageManifestRecord(
                stage="sft_stage1",
                task_mode="text_qa",
                input={
                    "system_prompt": "You answer spatial reasoning questions about a Minecraft-like world.",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": text_qa_full_prompt(task)}]},
                        {"role": "assistant", "content": [{"type": "text", "text": canonical_text_qa_response(task)}]},
                    ],
                },
                metadata={
                    "task_id": task.task_id,
                    "tier": task.tier,
                    "source": "text_qa",
                    "answer_format": task.answer_format,
                },
            )
        )
    return records


def _build_grpo_records(
    *,
    build_trajectories: list[EpisodeTrajectoryRecord],
    runs_dir: str | Path,
    total_records: int | None,
    curriculum_low_reward: float,
    curriculum_high_reward: float,
) -> tuple[list[StageManifestRecord], dict[str, Any]]:
    eligible = [
        record
        for record in build_trajectories
        if record.task_mode == "build" and record.tier in {"t4_structure_relative", "t5_modification", "t6_composition"}
    ]
    eligible.sort(key=lambda record: (record.tier, record.task_id, record.episode_id))
    bootstrap_mean_rewards_by_tier = bootstrap_mean_rewards(
        runs_dir=runs_dir,
        tiers=("t4_structure_relative", "t5_modification", "t6_composition"),
    )
    bootstrap_success_rates_by_tier = bootstrap_success_rates(
        runs_dir=runs_dir,
        tiers=("t4_structure_relative", "t5_modification", "t6_composition"),
    )
    bootstrap_family_mean_rewards_by_family = bootstrap_family_mean_rewards(runs_dir=runs_dir)
    bootstrap_family_rates = bootstrap_family_success_rates(runs_dir=runs_dir)
    trajectory_mean_rewards = rolling_tier_mean_rewards(
        [{"tier": record.tier, "reward_normalized": record.reward_normalized} for record in eligible],
        tiers=("t4_structure_relative", "t5_modification", "t6_composition"),
    )
    trajectory_success_rates = rolling_tier_success_rates(
        [{"tier": record.tier, "reward_binary": record.reward_binary} for record in eligible],
        tiers=("t4_structure_relative", "t5_modification", "t6_composition"),
    )
    trajectory_family_mean_rewards = rolling_family_mean_rewards(
        [{"task_id": record.task_id, "reward_normalized": record.reward_normalized} for record in eligible]
    )
    trajectory_family_rates = rolling_family_success_rates(
        [{"task_id": record.task_id, "reward_binary": record.reward_binary} for record in eligible]
    )
    mean_rewards = dict(bootstrap_mean_rewards_by_tier)
    mean_rewards.update(trajectory_mean_rewards)
    weights = curriculum_weights(
        mean_rewards,
        low=curriculum_low_reward,
        high=curriculum_high_reward,
    )

    selected = eligible
    if total_records is not None and total_records > 0 and eligible:
        per_tier_counts = weighted_task_counts(
            total_tasks=total_records,
            tiers=["t4_structure_relative", "t5_modification", "t6_composition"],
            weights=weights,
        )
        selected = []
        for tier in ["t4_structure_relative", "t5_modification", "t6_composition"]:
            tier_rows = [record for record in eligible if record.tier == tier]
            selected.extend(tier_rows[: per_tier_counts[tier]])

    records = [
            StageManifestRecord(
                stage="grpo_stage2",
                task_mode=record.task_mode,
                input={"system_prompt": record.system_prompt, "messages": _messages_json(record)},
                rubric={"reward": record.reward_normalized},
                metadata={
                    "task_id": record.task_id,
                    "tier": record.tier,
                    "task_mode": record.task_mode,
                    "reward_raw": record.reward_raw,
                    "reward_normalized": record.reward_normalized,
                    "reward_binary": record.reward_binary,
                    "grader": record.grader,
                    "tool_call_count": record.tool_call_count,
                    "tool_round_count": record.tool_round_count,
                },
            )
        for record in selected
    ]
    curriculum = {
        "bootstrap_family_mean_rewards": bootstrap_family_mean_rewards_by_family,
        "bootstrap_family_success_rates": bootstrap_family_rates,
        "bootstrap_mean_rewards": bootstrap_mean_rewards_by_tier,
        "bootstrap_success_rates": bootstrap_success_rates_by_tier,
        "trajectory_family_mean_rewards": trajectory_family_mean_rewards,
        "trajectory_family_success_rates": trajectory_family_rates,
        "trajectory_mean_rewards": trajectory_mean_rewards,
        "trajectory_success_rates": trajectory_success_rates,
        "effective_mean_rewards": mean_rewards,
        "weights": weights,
        "curriculum_low_reward": curriculum_low_reward,
        "curriculum_high_reward": curriculum_high_reward,
        "selected_records": len(records),
        "requested_records": total_records,
    }
    return records, curriculum


def run(
    *,
    build_trajectories_path: Path,
    sft_output: Path,
    grpo_output: Path,
    paraphrase_cache_path: str | None,
    text_qa_trajectories_path: str | None,
    verified_qa_cache_path: str | None,
    text_qa_seed: int,
    text_qa_per_tier: int,
    runs_dir: str | Path,
    grpo_total_records: int | None,
    curriculum_output: str | None,
    curriculum_low_reward: float,
    curriculum_high_reward: float,
) -> dict[str, Any]:
    build_trajectories = read_trajectory_jsonl(build_trajectories_path)
    sft_records = _build_sft_records(
        build_trajectories=build_trajectories,
        paraphrase_cache_path=paraphrase_cache_path,
        text_qa_trajectories_path=text_qa_trajectories_path,
        text_qa_seed=text_qa_seed,
        text_qa_per_tier=text_qa_per_tier,
        verified_qa_cache_path=verified_qa_cache_path,
    )
    grpo_records, curriculum = _build_grpo_records(
        build_trajectories=build_trajectories,
        runs_dir=runs_dir,
        total_records=grpo_total_records,
        curriculum_low_reward=curriculum_low_reward,
        curriculum_high_reward=curriculum_high_reward,
    )

    sft_output.parent.mkdir(parents=True, exist_ok=True)
    grpo_output.parent.mkdir(parents=True, exist_ok=True)
    sft_output.write_text("\n".join(record.model_dump_json() for record in sft_records) + ("\n" if sft_records else ""), encoding="utf-8")
    grpo_output.write_text("\n".join(record.model_dump_json() for record in grpo_records) + ("\n" if grpo_records else ""), encoding="utf-8")

    if curriculum_output is not None:
        curriculum_path = Path(curriculum_output).resolve()
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        curriculum_path.write_text(json.dumps(curriculum, indent=2, sort_keys=True), encoding="utf-8")
    else:
        curriculum_path = None

    return {
        "sft_stage1": len(sft_records),
        "grpo_stage2": len(grpo_records),
        "sft_output": str(sft_output),
        "grpo_output": str(grpo_output),
        "curriculum_output": str(curriculum_path) if curriculum_path is not None else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export staged SFT and GRPO manifests for BrowseCraft spatial training.")
    parser.add_argument("--build-trajectories", required=True)
    parser.add_argument("--sft-output", required=True)
    parser.add_argument("--grpo-output", required=True)
    parser.add_argument("--paraphrase-cache", default=None)
    parser.add_argument("--text-qa-trajectories", default=None)
    parser.add_argument("--verified-qa-cache", default=None)
    parser.add_argument("--text-qa-seed", type=int, default=7)
    parser.add_argument("--text-qa-per-tier", type=int, default=4)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--grpo-total-records", type=int, default=None)
    parser.add_argument("--curriculum-output", default="runs/grpo_curriculum.json")
    parser.add_argument("--curriculum-low-reward", type=float, default=0.2)
    parser.add_argument("--curriculum-high-reward", type=float, default=0.7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        build_trajectories_path=Path(args.build_trajectories).resolve(),
        sft_output=Path(args.sft_output).resolve(),
        grpo_output=Path(args.grpo_output).resolve(),
        paraphrase_cache_path=args.paraphrase_cache,
        text_qa_trajectories_path=args.text_qa_trajectories,
        verified_qa_cache_path=args.verified_qa_cache,
        text_qa_seed=int(args.text_qa_seed),
        text_qa_per_tier=int(args.text_qa_per_tier),
        runs_dir=args.runs_dir,
        grpo_total_records=args.grpo_total_records,
        curriculum_output=args.curriculum_output,
        curriculum_low_reward=float(args.curriculum_low_reward),
        curriculum_high_reward=float(args.curriculum_high_reward),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
