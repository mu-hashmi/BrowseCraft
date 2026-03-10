from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic

from browsecraft_sim.rl.anthropic_batches import create_batch, read_batch_results, wait_for_batch
from browsecraft_sim.rl.augmentation import (
    parse_json_payload,
    paraphrase_request,
    paraphrase_shortfall_report,
    paraphrase_verify_request,
    verified_paraphrase_records,
    verified_world_qa_candidates,
    world_qa_request,
)
from browsecraft_sim.rl.prompt_variants import write_prompt_variants_jsonl
from browsecraft_sim.rl.task_generator import generate_tasks
from browsecraft_sim.rl.text_qa import generate_text_qa_tasks, write_text_qa_jsonl
from browsecraft_sim.rl.types import ALL_TIERS, Tier


def _parse_tiers(raw: str | None) -> list[Tier] | None:
    if raw is None or not raw.strip():
        return None
    tiers = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [tier for tier in tiers if tier not in ALL_TIERS]
    if invalid:
        raise ValueError(f"unsupported tiers: {', '.join(invalid)}")
    return tiers  # type: ignore[return-value]


async def _run_paraphrases(args: argparse.Namespace) -> None:
    tasks = generate_tasks(seed=args.seed, per_tier=args.per_tier, tiers=_parse_tiers(args.tiers))
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    try:
        requests = [
            paraphrase_request(task, variant_index=variant_index, model=args.model)
            for task in tasks
            for variant_index in range(3)
        ]
        created_batch = await create_batch(client, requests=requests)
        await wait_for_batch(
            client,
            message_batch_id=created_batch.id,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        paraphrase_outputs = await read_batch_results(client, message_batch_id=created_batch.id)

        verification_requests = []
        task_by_seed = {task.seed: task for task in tasks}
        for custom_id, text in paraphrase_outputs.items():
            if not custom_id.startswith("paraphrase_"):
                raise ValueError(f"unexpected paraphrase custom id: {custom_id}")
            _, task_seed, variant_index = custom_id.rsplit("_", maxsplit=2)
            task = task_by_seed[int(task_seed)]
            try:
                payload = parse_json_payload(text)
            except ValueError:
                continue
            paraphrase = payload.get("paraphrase")
            if not isinstance(paraphrase, str) or not paraphrase:
                continue
            verification_requests.append(
                paraphrase_verify_request(
                    task,
                    paraphrase=paraphrase,
                    variant_index=int(variant_index),
                    model=args.model,
                )
            )

        verification_outputs: dict[str, str] = {}
        if verification_requests:
            verification_batch = await create_batch(client, requests=verification_requests)
            await wait_for_batch(
                client,
                message_batch_id=verification_batch.id,
                poll_interval_seconds=args.poll_interval_seconds,
            )
            verification_outputs = await read_batch_results(client, message_batch_id=verification_batch.id)
    finally:
        await client.close()

    records = verified_paraphrase_records(
        tasks=tasks,
        paraphrase_outputs=paraphrase_outputs,
        verification_outputs=verification_outputs,
    )
    paraphrase_output = Path(args.paraphrase_output).resolve()
    shortfall_output = Path(args.shortfall_output).resolve()
    paraphrase_output.parent.mkdir(parents=True, exist_ok=True)
    shortfall_output.parent.mkdir(parents=True, exist_ok=True)
    write_prompt_variants_jsonl(paraphrase_output, records)
    shortfall_output.write_text(json.dumps(paraphrase_shortfall_report(records), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"paraphrases": str(paraphrase_output), "shortfall": str(shortfall_output)}, indent=2))


async def _run_world_qa(args: argparse.Namespace) -> None:
    source = args.qa_source
    if source == "build":
        source_tasks = generate_tasks(seed=args.seed, per_tier=args.per_tier, tiers=_parse_tiers(args.tiers))
    else:
        source_tasks = generate_text_qa_tasks(seed=args.seed, per_tier=args.per_tier)

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    try:
        requests = [world_qa_request(task, model=args.model) for task in source_tasks]
        created_batch = await create_batch(client, requests=requests)
        await wait_for_batch(
            client,
            message_batch_id=created_batch.id,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        outputs = await read_batch_results(client, message_batch_id=created_batch.id)
    finally:
        await client.close()

    verified = verified_world_qa_candidates(source_tasks=source_tasks, batch_outputs=outputs)
    qa_output = Path(args.qa_output).resolve()
    qa_output.parent.mkdir(parents=True, exist_ok=True)
    write_text_qa_jsonl(qa_output, verified)
    print(json.dumps({"qa_output": str(qa_output), "verified_count": len(verified)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate offline prompt paraphrases and world-derived spatial QA caches.")
    parser.add_argument("--mode", choices=("paraphrases", "world_qa", "all"), default="all")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=4)
    parser.add_argument("--tiers", default=None)
    parser.add_argument("--qa-source", choices=("build", "text_qa"), default="text_qa")
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--paraphrase-output", default="runs/verified_paraphrases.jsonl")
    parser.add_argument("--qa-output", default="runs/verified_world_qa.jsonl")
    parser.add_argument("--shortfall-output", default="runs/paraphrase_shortfall.json")
    return parser


async def _run(args: argparse.Namespace) -> None:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    if args.mode in {"paraphrases", "all"}:
        await _run_paraphrases(args)
    if args.mode in {"world_qa", "all"}:
        await _run_world_qa(args)


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
