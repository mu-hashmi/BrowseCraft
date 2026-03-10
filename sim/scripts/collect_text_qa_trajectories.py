from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from anthropic import AsyncAnthropic

from browsecraft_sim.rl.text_qa import (
    TextQATrajectoryRecord,
    generate_text_qa_tasks,
    grade_text_qa_answer,
    text_qa_full_prompt,
)
from browsecraft_sim.rl.trajectory import validate_anthropic_messages


_SYSTEM_PROMPT = (
    "You answer spatial reasoning questions about a Minecraft-like world.\n"
    "Read the user prompt carefully, reason step by step internally, and respond with exactly one line in the form "
    "`Answer: <final answer>`.\n"
    "Do not invent entities that are not described."
)
_CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


def _assistant_text(content_blocks: list[Any]) -> str:
    chunks: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)
    return "\n".join(chunks).strip()


async def _warmup_prompt_cache(client: AsyncAnthropic, *, model: str) -> None:
    await client.messages.create(
        model=model,
        max_tokens=1,
        cache_control=_CACHE_CONTROL_EPHEMERAL,
        temperature=0,
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": _CACHE_CONTROL_EPHEMERAL}],
        messages=[{"role": "user", "content": [{"type": "text", "text": "warmup"}]}],
    )


async def _run_task(
    client: AsyncAnthropic,
    *,
    task: Any,
    model: str,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    full_prompt = text_qa_full_prompt(task)
    messages = [{"role": "user", "content": [{"type": "text", "text": full_prompt}]}]
    response = await client.messages.create(
        model=model,
        max_tokens=256,
        cache_control=_CACHE_CONTROL_EPHEMERAL,
        temperature=0,
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": _CACHE_CONTROL_EPHEMERAL}],
        messages=messages,
    )
    answer = _assistant_text(response.content)
    assistant_message = {"role": "assistant", "content": [{"type": "text", "text": answer or "<empty>"}]}
    validated_messages = validate_anthropic_messages([*messages, assistant_message])
    grade = grade_text_qa_answer(task, answer)
    record = TextQATrajectoryRecord(
        task_id=task.task_id,
        tier=task.tier,
        family=task.family,
        seed=task.seed,
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        messages=validated_messages,
        prompt=full_prompt,
        answer=answer,
        expected_answer=task.expected_answer,
        answer_format=task.answer_format,
        canonical_reasoning=task.canonical_reasoning,
        reward_raw=grade.reward_raw,
        reward_normalized=grade.reward_normalized,
        reward_binary=grade.reward_binary,
        started_at=started_at.isoformat(),
        ended_at=datetime.now(UTC).isoformat(),
        metadata=task.metadata,
    )
    return {
        "record": record.model_dump(mode="json"),
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": response.usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": response.usage.cache_read_input_tokens or 0,
        },
    }


def _append_row(output: Path, row: dict[str, Any]) -> None:
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row["record"]) + "\n")


def _progress_line(
    *,
    completed: int,
    total: int,
    row: dict[str, Any],
    started_at: float,
    usage_totals: dict[str, int],
) -> str:
    record = row["record"]
    elapsed_s = monotonic() - started_at
    return (
        f"[{completed}/{total}] {record['tier']}/{record['family']} "
        f"reward={record['reward_normalized']:.3f} elapsed={elapsed_s:.1f}s "
        f"usage=in:{usage_totals['input_tokens']} out:{usage_totals['output_tokens']}"
    )


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    tasks = generate_text_qa_tasks(seed=args.seed, per_tier=args.per_tier)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    await _warmup_prompt_cache(client, model=args.model)
    started_at = monotonic()

    async def run_one(task: Any) -> dict[str, Any]:
        async with semaphore:
            return await _run_task(client, task=task, model=args.model)

    try:
        rows: list[dict[str, Any]] = []
        usage_totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        pending = [asyncio.create_task(run_one(task)) for task in tasks]
        for completed, done in enumerate(asyncio.as_completed(pending), start=1):
            row = await done
            rows.append(row)
            _append_row(output, row)
            for key in usage_totals:
                usage_totals[key] += int(row["usage"][key])
            if completed == 1 or completed % args.log_every == 0 or completed == len(tasks):
                print(
                    _progress_line(
                        completed=completed,
                        total=len(tasks),
                        row=row,
                        started_at=started_at,
                        usage_totals=usage_totals,
                    ),
                    flush=True,
                )
        return rows
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect text-only QA trajectories for BrowseCraft spatial tasks.")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="text_qa_trajectories.jsonl")
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    rows = asyncio.run(_run(args))
    output = Path(args.output).resolve()
    usage_totals = {
        "input_tokens": sum(row["usage"]["input_tokens"] for row in rows),
        "output_tokens": sum(row["usage"]["output_tokens"] for row in rows),
        "cache_creation_input_tokens": sum(row["usage"]["cache_creation_input_tokens"] for row in rows),
        "cache_read_input_tokens": sum(row["usage"]["cache_read_input_tokens"] for row in rows),
    }
    print(
        json.dumps(
            {"output": str(output), "episodes": len(rows), "model": args.model, "usage": usage_totals},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
