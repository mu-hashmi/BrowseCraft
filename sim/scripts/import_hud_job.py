from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anyio
import httpx

from browsecraft_sim.rl.agent_config import AGENT_SYSTEM_PROMPT, is_remote_user_prompt
from browsecraft_sim.rl.config import RewardConfig
from browsecraft_sim.rl.grader import grade_task
from browsecraft_sim.rl.trajectory import validate_anthropic_messages
from browsecraft_sim.rl.types import EpisodeTrace, TaskSpec, ToolCallRecord
from browsecraft_sim.rl.world_setup import build_world, diff_to_blocks, serialize_snapshot
from browsecraft_sim.tool_dispatch import dispatch_tool


_HUD_MCP_URL = "https://api.hud.ai/v3/mcp/"
_IGNORED_TOOL_SPANS = {"_hud_submit", "rl_setup_task", "rl_grade_task"}
_HUD_MAX_ATTEMPTS = 5
_WORLD_MODIFYING_TOOLS = {"place_blocks", "fill_region", "undo_last"}


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _extract_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "".join(texts)
    return json.dumps(content, sort_keys=True)


def _normalize_model_name(model_name: str) -> str:
    if "/" not in model_name:
        return model_name
    return model_name.rsplit("/", maxsplit=1)[-1]


def _normalize_message_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [_normalize_message_block(content)]
    if not isinstance(content, list):
        raise ValueError(f"unsupported message content type: {type(content).__name__}")

    normalized: list[dict[str, Any]] = []
    for block in content:
        normalized.append(_normalize_message_block(block))
    return normalized


def _normalize_message_block(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise ValueError(f"unsupported content block type: {type(block).__name__}")
    block_type = block.get("type")
    if block_type == "text":
        return {"type": "text", "text": block["text"]}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": block["id"],
            "name": block["name"],
            "input": block.get("input") or {},
        }
    if block_type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block["tool_use_id"],
            "content": _extract_text_blocks(block.get("content")),
            "is_error": bool(block.get("is_error", False)),
        }
    raise ValueError(f"unsupported content block: {block_type}")


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call["function"]
    arguments = function["arguments"]
    return {
        "type": "tool_use",
        "id": tool_call["id"],
        "name": function["name"],
        "input": json.loads(arguments) if isinstance(arguments, str) else dict(arguments),
    }


def _normalize_request_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message["role"]
    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": _extract_text_blocks(message.get("content")),
                    "is_error": False,
                }
            ],
        }

    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, list):
        blocks.extend(_normalize_message_content(content))
    elif isinstance(content, dict):
        blocks.extend(_normalize_message_content(content))
    elif isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
    elif content is not None:
        raise ValueError(f"unsupported request content type: {type(content).__name__}")

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        blocks.append(_normalize_tool_call(tool_call))
    return {"role": role, "content": blocks}


def _normalize_request_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_request_message(message) for message in raw_messages]


def _normalize_task_prompt_message(messages: list[dict[str, Any]], *, task_prompt: str) -> list[dict[str, Any]]:
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") != "user":
        return messages
    content = first.get("content")
    if not isinstance(content, list):
        return messages
    for block in content:
        if block.get("type") != "text":
            continue
        prompt_text = block.get("text")
        if prompt_text == task_prompt:
            return messages
        if isinstance(prompt_text, str) and is_remote_user_prompt(prompt_text, task_prompt):
            block["text"] = task_prompt
            return messages
        return messages
    return messages


def _classify_failure_mode(error_text: str | None) -> str | None:
    if not error_text:
        return None
    normalized = error_text.casefold()
    if "context_window_exceeded" in normalized or "context window" in normalized:
        return "context_window_exceeded"
    if "timeout" in normalized:
        return "timeout"
    if "invalid arguments" in normalized or "expects" in normalized:
        return "invalid_arguments"
    if "tool error" in normalized:
        return "tool_error"
    return "other_error"


def _tool_diagnostics(tool_calls: list[ToolCallRecord]) -> dict[str, int]:
    inspect_call_count = 0
    redundant_inspect_count = 0
    effective_radius_unchanged_count = 0
    tool_error_count = 0
    current_non_modifying_streak = 0
    max_non_modifying_streak = 0

    for tool_call in tool_calls:
        if not tool_call.success:
            tool_error_count += 1
        if tool_call.name in _WORLD_MODIFYING_TOOLS:
            current_non_modifying_streak = 0
        else:
            current_non_modifying_streak += 1
            max_non_modifying_streak = max(max_non_modifying_streak, current_non_modifying_streak)
        if tool_call.name != "inspect_area":
            continue
        inspect_call_count += 1
        if bool(tool_call.args.get("redundant_with_previous", False)):
            redundant_inspect_count += 1
        if bool(tool_call.args.get("effective_radius_unchanged", False)):
            effective_radius_unchanged_count += 1

    return {
        "inspect_call_count": inspect_call_count,
        "redundant_inspect_count": redundant_inspect_count,
        "effective_radius_unchanged_count": effective_radius_unchanged_count,
        "tool_error_count": tool_error_count,
        "max_non_modifying_streak": max_non_modifying_streak,
    }


def _reward_bucket(reward: float, *, success_threshold: float = 0.8) -> str:
    if reward <= 0.0:
        return "zero"
    if reward >= success_threshold:
        return "success"
    return "partial"


def _final_assistant_blocks(span: dict[str, Any]) -> list[dict[str, Any]]:
    attributes = span["attributes"]
    result = attributes["result"]
    assistant_blocks: list[dict[str, Any]] = []
    content_text = result.get("content") or ""
    if content_text:
        assistant_blocks.append({"type": "text", "text": content_text})
    for tool_call in result.get("tool_calls") or []:
        assistant_blocks.append(_normalize_tool_call(tool_call))
    return assistant_blocks


def _jsonrpc_payload(response_text: str) -> dict[str, Any]:
    stripped = response_text.strip()
    if not stripped:
        raise ValueError("empty HUD MCP response")
    if stripped.startswith("event: message"):
        for line in stripped.splitlines():
            if line.startswith("data: "):
                return json.loads(line[len("data: ") :])
        raise ValueError("HUD MCP SSE response missing data payload")
    return json.loads(stripped)


async def _call_hud_tool(
    *,
    client: httpx.AsyncClient,
    name: str,
    arguments: dict[str, Any],
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, _HUD_MAX_ATTEMPTS + 1):
        try:
            response = await client.post(
                _HUD_MCP_URL,
                headers={"Accept": "application/json, text/event-stream"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": name,
                        "arguments": arguments,
                    },
                },
            )
            response.raise_for_status()
            payload = _jsonrpc_payload(response.text)
            if "error" in payload:
                raise RuntimeError(f"HUD MCP error for {name}: {payload['error']}")
            content = payload["result"]["content"]
            if len(content) != 1:
                raise ValueError(f"expected one HUD MCP content block for {name}")
            result = json.loads(content[0]["text"])
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"HUD tool error for {name}: {result['error']}")
            return result
        except RuntimeError:
            raise
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == _HUD_MAX_ATTEMPTS:
                break
            await anyio.sleep(float(attempt))
    raise RuntimeError(f"HUD MCP call failed for {name} after {_HUD_MAX_ATTEMPTS} attempts") from last_error


async def _fetch_trace_payload(
    *,
    client: httpx.AsyncClient,
    trace_id: str,
) -> dict[str, Any]:
    return await _call_hud_tool(
        client=client,
        name="get_trace",
        arguments={"trace_id": trace_id, "include_trajectory": True},
    )


def _build_raw_row(payload: dict[str, Any], *, reward_tolerance: float) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = payload["trajectory"]
    spans = sorted(trajectory, key=lambda span: _parse_iso8601(span["start_time"]))

    prompt_span = next(
        span for span in spans if span["name"] == "prompts/get.mcp" and span["attributes"]["request"]["method"] == "prompts/get"
    )
    prompt_args = prompt_span["attributes"]["request"]["params"]["arguments"]
    raw_task_spec = prompt_args["task_spec"]
    raw_reward_config = prompt_args["reward_config"]
    task = (
        TaskSpec.model_validate(raw_task_spec)
        if isinstance(raw_task_spec, dict)
        else TaskSpec.model_validate_json(raw_task_spec)
    )
    reward_config = (
        RewardConfig.model_validate(raw_reward_config)
        if isinstance(raw_reward_config, dict)
        else RewardConfig.model_validate_json(raw_reward_config)
    )

    world = build_world(task)
    before_snapshot = world.snapshot()
    inference_spans = [
        span
        for span in spans
        if span["name"] in {"inference.messages", "inference.chat_completion"}
    ]
    if not inference_spans:
        raise ValueError(f"trace {payload['trace_id']} does not contain inference spans")

    messages: list[dict[str, Any]] = []
    tool_calls: list[ToolCallRecord] = []
    model_name = ""
    format_valid = True
    inspect_call_count = 0
    redundant_inspect_count = 0
    effective_radius_unchanged_count = 0
    current_non_modifying_streak = 0
    max_non_modifying_streak = 0

    for span in inference_spans:
        attributes = span["attributes"]
        request_params = attributes["request"]["params"]
        response_model = attributes["result"].get("model") or request_params.get("model")
        if response_model:
            model_name = model_name or _normalize_model_name(response_model)
        request_messages = _normalize_request_messages(request_params["messages"])
        request_messages = _normalize_task_prompt_message(request_messages, task_prompt=task.prompt)
        if messages and request_messages[: len(messages)] != messages:
            raise ValueError(f"trace message history diverged for {payload['trace_id']}")
        messages = request_messages

    final_blocks = _final_assistant_blocks(inference_spans[-1])
    if not final_blocks:
        response_agent_span = next((span for span in spans if span["name"] == "response_agent"), None)
        if response_agent_span is not None:
            final_text = response_agent_span["attributes"]["request"]["agent_message"]
            final_blocks = [{"type": "text", "text": final_text}]
    if final_blocks:
        messages.append({"role": "assistant", "content": final_blocks})

    if not model_name:
        raise ValueError(f"missing model name for trace {payload['trace_id']}")

    queued_tool_uses: list[tuple[str, str, dict[str, Any]]] = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block["type"] == "tool_use":
                if block["name"] in _IGNORED_TOOL_SPANS:
                    continue
                queued_tool_uses.append((block["id"], block["name"], block["input"]))

    for span in spans:
        if span["name"] != "tools/call.mcp":
            continue

        request = span["attributes"]["request"]["params"]
        tool_name = request["name"]
        if tool_name in _IGNORED_TOOL_SPANS:
            continue

        if not queued_tool_uses:
            raise ValueError(f"missing queued tool use for trace {payload['trace_id']} tool {tool_name}")
        tool_use_id, queued_name, queued_input = queued_tool_uses.pop(0)
        arguments = request.get("arguments") or {}
        if queued_name != tool_name or queued_input != arguments:
            raise ValueError(
                f"tool use mismatch for trace {payload['trace_id']}: expected {queued_name} {queued_input}, got {tool_name} {arguments}"
            )
        result = span["attributes"]["result"]
        result_text = _extract_text_blocks(result.get("content"))
        is_error = bool(result.get("isError"))
        if tool_name in _WORLD_MODIFYING_TOOLS:
            current_non_modifying_streak = 0
        else:
            current_non_modifying_streak += 1
            max_non_modifying_streak = max(max_non_modifying_streak, current_non_modifying_streak)
        tool_calls.append(
            ToolCallRecord(
                name=tool_name,
                args=arguments,
                success=not is_error,
                error=result_text if is_error else None,
            )
        )
        if is_error:
            format_valid = False
            continue
        result_payload = dispatch_tool(world, tool_name, arguments)
        if tool_name == "inspect_area":
            inspect_call_count += 1
            if bool(result_payload.get("redundant_with_previous", False)):
                redundant_inspect_count += 1
            if bool(result_payload.get("effective_radius_unchanged", False)):
                effective_radius_unchanged_count += 1

    if queued_tool_uses:
        raise ValueError(f"unmatched tool uses for trace {payload['trace_id']}: {[item[0] for item in queued_tool_uses]}")

    started_at = min(_parse_iso8601(span["start_time"]) for span in spans)
    ended_at = max(_parse_iso8601(span["end_time"]) for span in spans)
    trace = EpisodeTrace(
        episode_id=payload["trace_id"],
        task_id=task.task_id,
        tier=task.tier,
        seed=task.seed,
        model=model_name,
        system_prompt=AGENT_SYSTEM_PROMPT,
        messages=validate_anthropic_messages(messages),
        started_at=started_at,
        ended_at=ended_at,
        format_valid=format_valid,
        tool_round_count=len(inference_spans),
        tool_calls=tool_calls,
        initial_world=serialize_snapshot(before_snapshot),
        final_world_diff=diff_to_blocks(world.diff(before_snapshot)),
    )
    breakdown = grade_task(task=task, world=world, trace=trace, config=reward_config)
    remote_reward = float(payload["reward"])
    if abs(breakdown.reward_normalized - remote_reward) > reward_tolerance:
        raise ValueError(
            f"reward mismatch for {task.task_id}: local={breakdown.reward_normalized} remote={remote_reward}"
        )

    usage = payload["metadata"].get("usage", {})
    diagnostics = {
        "inspect_call_count": inspect_call_count,
        "redundant_inspect_count": redundant_inspect_count,
        "effective_radius_unchanged_count": effective_radius_unchanged_count,
        "tool_error_count": sum(1 for tool_call in tool_calls if not tool_call.success),
        "max_non_modifying_streak": max_non_modifying_streak,
    }
    row = {
        "trace": trace.model_dump(mode="json"),
        "model": model_name,
        "grader": breakdown.model_dump(mode="json"),
        "reward_raw": breakdown.reward_raw,
        "reward_normalized": breakdown.reward_normalized,
        "reward_binary": breakdown.reward_binary,
        "diagnostics": diagnostics,
        "usage": {
            "input_tokens": int(usage.get("total_input_tokens", 0)),
            "output_tokens": int(usage.get("total_output_tokens", 0)),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
    summary = {
        "task_id": task.task_id,
        "tier": task.tier,
        "family": task.family,
        "reward_normalized": breakdown.reward_normalized,
        "reward_binary": breakdown.reward_binary,
        "tool_call_count": trace.tool_call_count,
        "inspect_call_count": diagnostics["inspect_call_count"],
        "redundant_inspect_count": diagnostics["redundant_inspect_count"],
        "effective_radius_unchanged_count": diagnostics["effective_radius_unchanged_count"],
        "tool_error_count": diagnostics["tool_error_count"],
        "max_non_modifying_streak": diagnostics["max_non_modifying_streak"],
    }
    return row, summary


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _read_existing_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], [], set()

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    imported_trace_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        trace = payload["trace"]
        grader = payload["grader"]
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = _tool_diagnostics([ToolCallRecord.model_validate(tool_call) for tool_call in trace["tool_calls"]])
        rows.append(payload)
        imported_trace_ids.add(trace["episode_id"])
        summaries.append(
            {
                "task_id": trace["task_id"],
                "tier": trace["tier"],
                "family": trace["task_id"].split(":", maxsplit=2)[1],
                "reward_normalized": float(payload["reward_normalized"]),
                "reward_binary": float(payload["reward_binary"]),
                "tool_call_count": int(grader["details"]["actual_tool_calls"]),
                "inspect_call_count": int(diagnostics["inspect_call_count"]),
                "redundant_inspect_count": int(diagnostics["redundant_inspect_count"]),
                "effective_radius_unchanged_count": int(diagnostics["effective_radius_unchanged_count"]),
                "tool_error_count": int(diagnostics["tool_error_count"]),
                "max_non_modifying_streak": int(diagnostics["max_non_modifying_streak"]),
            }
        )
    return rows, summaries, imported_trace_ids


async def run(
    *,
    job_id: str,
    output_path: Path,
    summary_path: Path,
    reward_tolerance: float,
) -> dict[str, Any]:
    api_key = os.environ.get("HUD_API_KEY")
    if not api_key:
        raise ValueError("HUD_API_KEY must be set")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows, summaries, imported_trace_ids = _read_existing_rows(output_path)

    async with httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=300.0,
    ) as client:
        job = await _call_hud_tool(client=client, name="get_job", arguments={"job_id": job_id})
        traces = await _call_hud_tool(client=client, name="get_job_traces", arguments={"job_id": job_id})
        status_counts = Counter(trace["status"] for trace in traces["traces"])

        skipped_traces: list[dict[str, Any]] = []
        skipped_failure_modes: Counter[str] = Counter()
        for trace_summary in traces["traces"]:
            if trace_summary["status"] != "completed":
                fetch_error: str | None = None
                failure_mode: str | None = None
                if trace_summary["status"] == "error" or bool(trace_summary.get("has_error")):
                    try:
                        await _fetch_trace_payload(client=client, trace_id=trace_summary["trace_id"])
                    except Exception as exc:
                        fetch_error = str(exc)
                        failure_mode = _classify_failure_mode(fetch_error)
                    else:
                        failure_mode = "unknown_error"
                    if failure_mode is not None:
                        skipped_failure_modes[failure_mode] += 1
                skipped_traces.append(
                    {
                        "trace_id": trace_summary["trace_id"],
                        "status": trace_summary["status"],
                        "reward": trace_summary.get("reward"),
                        "has_error": trace_summary.get("has_error"),
                        "likely_failure_mode": failure_mode,
                        "fetch_error": fetch_error,
                    }
                )
                continue
            if trace_summary["trace_id"] in imported_trace_ids:
                continue
            payload = await _fetch_trace_payload(client=client, trace_id=trace_summary["trace_id"])
            row, row_summary = _build_raw_row(payload, reward_tolerance=reward_tolerance)
            rows.append(row)
            summaries.append(row_summary)
            imported_trace_ids.add(trace_summary["trace_id"])
            _append_jsonl_row(output_path, row)
            if len(rows) == 1 or len(rows) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "imported": len(rows),
                            "skipped": len(skipped_traces),
                            "last_trace_id": trace_summary["trace_id"],
                            "last_task_id": row_summary["task_id"],
                        }
                    ),
                    flush=True,
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        grouped[(item["tier"], item["family"])].append(item)

    family_summary = []
    for (tier, family), items in sorted(grouped.items()):
        zero_count = sum(1 for item in items if _reward_bucket(float(item["reward_normalized"])) == "zero")
        partial_count = sum(1 for item in items if _reward_bucket(float(item["reward_normalized"])) == "partial")
        success_count = sum(1 for item in items if _reward_bucket(float(item["reward_normalized"])) == "success")
        if success_count == len(items):
            family_status = "all_successes"
        elif success_count == 0 and partial_count == 0:
            family_status = "all_zero"
        elif success_count == 0:
            family_status = "partial_only"
        else:
            family_status = "mixed_with_success"
        family_summary.append(
            {
                "tier": tier,
                "family": family,
                "n": len(items),
                "mean_reward": mean(item["reward_normalized"] for item in items),
                "success_rate": mean(item["reward_binary"] for item in items),
                "zero_count": zero_count,
                "partial_count": partial_count,
                "success_count": success_count,
                "status": family_status,
                "mean_tool_calls": mean(item["tool_call_count"] for item in items),
                "mean_inspect_calls": mean(item["inspect_call_count"] for item in items),
                "mean_redundant_inspects": mean(item["redundant_inspect_count"] for item in items),
                "mean_effective_radius_unchanged": mean(item["effective_radius_unchanged_count"] for item in items),
                "mean_tool_errors": mean(item["tool_error_count"] for item in items),
                "max_non_modifying_streak": max(item["max_non_modifying_streak"] for item in items),
            }
        )
    summary = {
        "job": job,
        "imported_traces": len(rows),
        "status_counts": dict(status_counts),
        "skipped_traces": skipped_traces,
        "skipped_failure_modes": dict(skipped_failure_modes),
        "output": str(output_path.resolve()),
        "families": family_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a remote HUD eval job into BrowseCraft raw trajectory JSONL.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--reward-tolerance", type=float, default=1e-6)
    return parser


async def _run_cli(
    job_id: str,
    output_path: Path,
    summary_path: Path,
    reward_tolerance: float,
) -> dict[str, Any]:
    return await run(
        job_id=job_id,
        output_path=output_path,
        summary_path=summary_path,
        reward_tolerance=reward_tolerance,
    )


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output).resolve()
    summary_path = (
        Path(args.summary_output).resolve()
        if args.summary_output is not None
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )
    summary = anyio.run(
        _run_cli,
        str(args.job_id),
        output_path,
        summary_path,
        float(args.reward_tolerance),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
