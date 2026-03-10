from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from browsecraft_sim.rl.agent_config import compose_remote_user_prompt


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "sim" / "scripts" / "import_hud_job.py"
_SPEC = importlib.util.spec_from_file_location("browsecraft_import_hud_job", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_normalize_request_messages_accepts_openai_chat_completion_shape() -> None:
    messages = [
        {"role": "user", "content": "Build the structure."},
        {
            "role": "assistant",
            "content": "I will inspect first.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "inspect_area",
                        "arguments": json.dumps({"center": {"x": 0, "y": 64, "z": 0}, "radius": 6}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"sampled_blocks": 42}',
        },
    ]

    normalized = _MODULE._normalize_request_messages(messages)

    assert normalized == [
        {"role": "user", "content": [{"type": "text", "text": "Build the structure."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will inspect first."},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "inspect_area",
                    "input": {"center": {"x": 0, "y": 64, "z": 0}, "radius": 6},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": '{"sampled_blocks": 42}',
                    "is_error": False,
                }
            ],
        },
    ]


def test_final_assistant_blocks_accepts_chat_completion_result() -> None:
    span = {
        "attributes": {
            "result": {
                "content": "Done.",
                "tool_calls": [
                    {
                        "id": "call-9",
                        "type": "function",
                        "function": {
                            "name": "place_blocks",
                            "arguments": json.dumps(
                                {"placements": [{"x": 1, "y": 64, "z": 2, "block_id": "minecraft:stone"}]}
                            ),
                        },
                    }
                ],
            }
        }
    }

    blocks = _MODULE._final_assistant_blocks(span)

    assert blocks == [
        {"type": "text", "text": "Done."},
        {
            "type": "tool_use",
            "id": "call-9",
            "name": "place_blocks",
            "input": {"placements": [{"x": 1, "y": 64, "z": 2, "block_id": "minecraft:stone"}]},
        },
    ]


def test_normalize_task_prompt_message_rewrites_wrapped_remote_prompt() -> None:
    task_prompt = "Build the structure."
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": compose_remote_user_prompt(task_prompt)}],
        }
    ]

    normalized = _MODULE._normalize_task_prompt_message(messages, task_prompt=task_prompt)

    assert normalized[0]["content"][0]["text"] == task_prompt


def test_classify_failure_mode_detects_context_window_errors() -> None:
    assert (
        _MODULE._classify_failure_mode(
            "Prompt length plus max_tokens exceeds the model's context window: 36421 prompt tokens + 512 max_tokens > 32768."
        )
        == "context_window_exceeded"
    )


@pytest.mark.anyio
async def test_call_hud_tool_raises_on_tool_error_payload() -> None:
    class _FakeResponse:
        text = json.dumps(
            {
                "result": {
                    "content": [
                        {
                            "text": json.dumps({"error": "Failed to list traces"}),
                        }
                    ]
                }
            }
        )

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        async def post(self, *args, **kwargs):
            return _FakeResponse()

    with pytest.raises(RuntimeError, match="HUD tool error for get_job_traces"):
        await _MODULE._call_hud_tool(client=_FakeClient(), name="get_job_traces", arguments={"job_id": "bad-id"})
