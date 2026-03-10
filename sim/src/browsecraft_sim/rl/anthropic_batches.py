from __future__ import annotations

import asyncio
from typing import Any, Iterable

from anthropic import AsyncAnthropic
from anthropic.types.messages.message_batch import MessageBatch


def build_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
    }


async def create_batch(client: AsyncAnthropic, *, requests: Iterable[dict[str, Any]]) -> MessageBatch:
    return await client.messages.batches.create(requests=list(requests))


async def wait_for_batch(
    client: AsyncAnthropic,
    *,
    message_batch_id: str,
    poll_interval_seconds: float = 10.0,
) -> MessageBatch:
    while True:
        batch = await client.messages.batches.retrieve(message_batch_id)
        if batch.processing_status == "ended":
            return batch
        await asyncio.sleep(poll_interval_seconds)


async def read_batch_results(
    client: AsyncAnthropic,
    *,
    message_batch_id: str,
) -> dict[str, str]:
    decoder = await client.messages.batches.results(message_batch_id)
    results: dict[str, str] = {}
    async for entry in decoder:
        if entry.result.type != "succeeded":
            continue
        results[entry.custom_id] = _message_text(entry.result.message.content)
    return results


def _message_text(content_blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts).strip()
