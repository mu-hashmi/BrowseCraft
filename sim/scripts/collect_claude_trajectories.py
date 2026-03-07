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

from browsecraft_sim.rl.config import RewardConfig
from browsecraft_sim.rl.grader import grade_task
from browsecraft_sim.rl.task_generator import generate_tasks
from browsecraft_sim.rl.trajectory import validate_anthropic_messages
from browsecraft_sim.rl.types import EpisodeTrace, ToolCallRecord
from browsecraft_sim.rl.world_setup import build_world, diff_to_blocks, serialize_snapshot
from browsecraft_sim.tool_dispatch import dispatch_tool


_SYSTEM_PROMPT = (
    "You are a Minecraft building agent operating in a headless simulator.\n"
    "Use tools to inspect state and place blocks.\n"
    "Coordinates use +x east, +y up, +z south.\n"
    "Do not assume missing world information; inspect before complex modifications.\n"
    "Prefer exact coordinate placement when task provides absolute targets.\n"
    "Do not add unrelated blocks."
)
_CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}

_TOOLS = [
    {
        "name": "player_position",
        "description": "Read current player position and facing.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "player_inventory",
        "description": "Read current player inventory summary.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "inspect_area",
        "description": "Inspect blocks around a center with a radius.",
        "input_schema": {
            "type": "object",
            "properties": {
                "center": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}},
                    "required": ["x", "y", "z"],
                    "additionalProperties": False,
                },
                "radius": {"type": "integer"},
                "detailed": {"type": "boolean"},
                "filter_terrain": {"type": "boolean"},
            },
            "required": ["center", "radius"],
            "additionalProperties": False,
        },
    },
    {
        "name": "place_blocks",
        "description": "Place blocks at absolute coordinates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "placements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "z": {"type": "integer"},
                            "block_id": {"type": "string"},
                        },
                        "required": ["x", "y", "z", "block_id"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["placements"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fill_region",
        "description": "Fill a cuboid region with one block type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_corner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}},
                    "required": ["x", "y", "z"],
                    "additionalProperties": False,
                },
                "to_corner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}},
                    "required": ["x", "y", "z"],
                    "additionalProperties": False,
                },
                "block_id": {"type": "string"},
            },
            "required": ["from_corner", "to_corner", "block_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "undo_last",
        "description": "Undo the most recent placement batch.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_active_overlay",
        "description": "Read active overlay state.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "modify_overlay",
        "description": "Modify overlay state.",
        "input_schema": {"type": "object", "properties": {"op": {"type": "string"}}, "required": ["op"]},
    },
    {
        "name": "get_blueprints",
        "description": "List available blueprints.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "save_blueprint",
        "description": "Save active overlay as blueprint.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "load_blueprint",
        "description": "Load blueprint into overlay.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
]
_CACHEABLE_TOOLS = [
    *[dict(tool) for tool in _TOOLS[:-1]],
    {**_TOOLS[-1], "cache_control": _CACHE_CONTROL_EPHEMERAL},
]


def _normalize_assistant_blocks(content_blocks: list[Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
            continue
        if block_type == "tool_use":
            identifier = getattr(block, "id", None)
            name = getattr(block, "name", None)
            payload = getattr(block, "input", None)
            if isinstance(identifier, str) and isinstance(name, str) and isinstance(payload, dict):
                blocks.append({"type": "tool_use", "id": identifier, "name": name, "input": payload})
            continue
    return blocks


async def _warmup_prompt_cache(client: AsyncAnthropic, *, model: str) -> None:
    await client.messages.create(
        model=model,
        max_tokens=1,
        cache_control=_CACHE_CONTROL_EPHEMERAL,
        temperature=0,
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": _CACHE_CONTROL_EPHEMERAL}],
        tools=_CACHEABLE_TOOLS,
        messages=[{"role": "user", "content": [{"type": "text", "text": "warmup"}]}],
    )


async def _run_episode(
    client: AsyncAnthropic,
    *,
    task: Any,
    model: str,
    max_rounds: int,
    reward_config: RewardConfig,
) -> dict[str, Any]:
    world = build_world(task)
    before_snapshot = world.snapshot()
    trace = EpisodeTrace(
        task_id=task.task_id,
        tier=task.tier,
        seed=task.seed,
        model=model,
        system_prompt=_SYSTEM_PROMPT,
        initial_world=serialize_snapshot(before_snapshot),
        started_at=datetime.now(UTC),
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": task.prompt}]}]
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_input_tokens = 0
    total_cache_read_input_tokens = 0

    for round_index in range(max_rounds):
        trace.tool_round_count = round_index + 1
        response = await client.messages.create(
            model=model,
            max_tokens=768,
            cache_control=_CACHE_CONTROL_EPHEMERAL,
            temperature=0,
            system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": _CACHE_CONTROL_EPHEMERAL}],
            tools=_CACHEABLE_TOOLS,
            messages=messages,
        )
        usage = response.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        total_cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        total_cache_read_input_tokens += usage.cache_read_input_tokens or 0
        assistant_blocks = _normalize_assistant_blocks(response.content)
        messages.append({"role": "assistant", "content": assistant_blocks})
        trace.messages = validate_anthropic_messages(messages)

        tool_uses = [block for block in assistant_blocks if block["type"] == "tool_use"]
        if not tool_uses:
            break

        tool_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            try:
                result_payload = dispatch_tool(world, str(tool_use["name"]), dict(tool_use["input"]))
                trace.tool_calls.append(
                    ToolCallRecord(name=str(tool_use["name"]), args=dict(tool_use["input"]), success=True)
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(tool_use["id"]),
                        "content": json.dumps(result_payload),
                    }
                )
            except Exception as exc:
                trace.format_valid = False
                trace.tool_calls.append(
                    ToolCallRecord(name=str(tool_use["name"]), args=dict(tool_use["input"]), success=False, error=str(exc))
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(tool_use["id"]),
                        "is_error": True,
                        "content": f"tool {tool_use['name']} failed: {exc}",
                    }
                )
        messages.append({"role": "user", "content": tool_results})
        trace.messages = validate_anthropic_messages(messages)
    else:
        trace.format_valid = False

    trace.ended_at = datetime.now(UTC)
    trace.final_world_diff = diff_to_blocks(world.diff(before_snapshot))
    breakdown = grade_task(task=task, world=world, trace=trace, config=reward_config)
    return {
        "trace": trace.model_dump(mode="json"),
        "model": model,
        "grader": breakdown.model_dump(mode="json"),
        "reward_raw": breakdown.reward_raw,
        "reward_normalized": breakdown.reward_normalized,
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_creation_input_tokens": total_cache_creation_input_tokens,
            "cache_read_input_tokens": total_cache_read_input_tokens,
        },
    }


def _append_row(output: Path, row: dict[str, Any]) -> None:
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _episode_progress_line(
    *,
    completed: int,
    total: int,
    row: dict[str, Any],
    started_at: float,
    usage_totals: dict[str, int],
) -> str:
    trace = row["trace"]
    grader = row["grader"]
    task_id = str(trace["task_id"])
    _, family, _, _ = task_id.split(":", 3)
    tool_calls = int(trace["tool_call_count"])
    reward = float(row["reward_normalized"])
    format_valid = bool(trace["format_valid"])
    elapsed_s = monotonic() - started_at
    return (
        f"[{completed}/{total}] {trace['tier']}/{family} "
        f"reward={reward:.3f} calls={tool_calls} format={int(format_valid)} "
        f"correctness={grader['correctness']:.3f} elapsed={elapsed_s:.1f}s "
        f"usage=in:{usage_totals['input_tokens']} out:{usage_totals['output_tokens']}"
    )


async def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    reward_config = RewardConfig()
    tasks = generate_tasks(seed=args.seed, per_tier=args.per_tier)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    await _warmup_prompt_cache(client, model=args.model)
    started_at = monotonic()

    async def run_one(task: Any) -> dict[str, Any]:
        async with semaphore:
            return await _run_episode(
                client,
                task=task,
                model=args.model,
                max_rounds=args.max_rounds,
                reward_config=reward_config,
            )

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
                    _episode_progress_line(
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
    parser = argparse.ArgumentParser(description="Collect Claude tool trajectories for BrowseCraft RL tasks.")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="raw_episodes.jsonl")
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
