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
    paraphrase_verify_request,
    verified_paraphrase_records,
)
from browsecraft_sim.rl.prompt_variants import PromptVariantRecord, write_prompt_variants_jsonl
from browsecraft_sim.rl.task_generator import generate_tasks
from browsecraft_sim.rl.types import ALL_TIERS, Tier


def _parse_tiers(raw: str | None) -> list[Tier]:
    if raw is None or not raw.strip():
        return list(ALL_TIERS)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in ALL_TIERS]
    if invalid:
        raise ValueError(f"unsupported tiers: {', '.join(invalid)}")
    return requested  # type: ignore[return-value]


async def _run(args: argparse.Namespace) -> list[PromptVariantRecord]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")

    tasks = generate_tasks(seed=args.seed, per_tier=args.per_tier, tiers=_parse_tiers(args.tiers))
    client = AsyncAnthropic(api_key=api_key)
    try:
        paraphrase_batch = await create_batch(
            client,
            requests=[
                paraphrase_request(task, variant_index=variant_index, model=args.model)
                for task in tasks
                for variant_index in range(3)
            ],
        )
        await wait_for_batch(client, message_batch_id=paraphrase_batch.id, poll_interval_seconds=args.poll_interval)
        paraphrase_results = await read_batch_results(client, message_batch_id=paraphrase_batch.id)

        verification_requests = []
        task_by_id = {task.task_id: task for task in tasks}
        for custom_id, text in paraphrase_results.items():
            _, task_id, variant_index = custom_id.split(":", maxsplit=2)
            try:
                payload = parse_json_payload(text)
            except ValueError:
                continue
            paraphrase = payload.get("paraphrase")
            if not isinstance(paraphrase, str) or not paraphrase:
                continue
            verification_requests.append(
                paraphrase_verify_request(
                    task_by_id[task_id],
                    paraphrase=paraphrase,
                    variant_index=int(variant_index),
                    model=args.model,
                )
            )

        verification_results: dict[str, str] = {}
        if verification_requests:
            verification_batch = await create_batch(client, requests=verification_requests)
            await wait_for_batch(client, message_batch_id=verification_batch.id, poll_interval_seconds=args.poll_interval)
            verification_results = await read_batch_results(client, message_batch_id=verification_batch.id)

        records = verified_paraphrase_records(
            tasks=tasks,
            paraphrase_outputs=paraphrase_results,
            verification_outputs=verification_results,
        )
        return records
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate verified prompt paraphrases with Anthropic Message Batches.")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=25)
    parser.add_argument("--tiers", default=None)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--output", default="prompt_variants.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = asyncio.run(_run(args))
    output = Path(args.output).resolve()
    write_prompt_variants_jsonl(output, records)
    print(
        json.dumps(
            {
                "output": str(output),
                "count": len(records),
                "verified_total": sum(len(record.verified_paraphrases) for record in records),
                "shortfall_total": sum(record.shortfall for record in records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
