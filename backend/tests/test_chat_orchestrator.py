from __future__ import annotations

import asyncio
import copy
import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from browsecraft_backend.chat_orchestrator import (
    CHAT_MODEL,
    ChatOrchestrator,
    _BuildAnchor,
    _BuildRequestTriageModel,
    _PlayerPositionResult,
    _TOOL_SCHEMAS,
    _build_anchor_for_request,
    _compose_system_prompt,
    _relative_placement_guard_for_request,
    _tool_status_message,
    _validate_placement_against_player_position,
)
from browsecraft_backend.convex_client import ConvexHttpClient
from browsecraft_backend.models import ChatRequest
from browsecraft_backend.supermemory_client import SupermemoryProfileContext, SupermemorySearchResult


EXPECTED_TOOL_NAMES = {
    "player_position",
    "player_inventory",
    "inspect_area",
    "place_blocks",
    "fill_region",
    "build_geometry",
    "set_plan",
    "undo_last",
    "get_active_overlay",
    "modify_overlay",
    "get_blueprints",
    "save_blueprint",
    "load_blueprint",
}


def test_build_geometry_tool_schema_has_top_level_object_type() -> None:
    build_geometry_schema = next(schema for schema in _TOOL_SCHEMAS if schema["name"] == "build_geometry")["input_schema"]
    assert build_geometry_schema["type"] == "object"


class FakeAnthropicMessages:
    def __init__(
        self,
        responses: list[Any],
        *,
        wrapped_stream: bool = False,
        create_responses: list[Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._wrapped_stream = wrapped_stream
        self.calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []
        self._create_responses = list(create_responses or [])

    def stream(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise AssertionError("Unexpected anthropic stream call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if self._wrapped_stream:
            return _WrappedFakeAnthropicStreamManager(response)
        return _FakeAnthropicStreamManager(response)

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(copy.deepcopy(kwargs))
        if self._create_responses:
            response = self._create_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        prompt = kwargs["messages"][0]["content"]
        return _triage_response(**_default_triage_for_prompt(prompt))

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(copy.deepcopy(kwargs))
        output_format = kwargs["output_format"]
        if self._create_responses:
            response = self._create_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            parsed_output = getattr(response, "parsed_output", None)
            if parsed_output is not None:
                return response
            content = getattr(response, "content", [])
            payload = json.loads("".join(block.text for block in content if getattr(block, "type", None) == "text"))
            return SimpleNamespace(parsed_output=output_format.model_validate(payload), content=content)
        prompt = kwargs["messages"][0]["content"]
        parsed_output = output_format.model_validate(_default_triage_for_prompt(prompt))
        return SimpleNamespace(parsed_output=parsed_output, content=[])


class FakeAnthropicClient:
    def __init__(
        self,
        responses: list[Any],
        *,
        wrapped_stream: bool = False,
        create_responses: list[Any] | None = None,
    ) -> None:
        self.messages = FakeAnthropicMessages(
            responses,
            wrapped_stream=wrapped_stream,
            create_responses=create_responses,
        )
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeAnthropicStreamManager:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aenter__(self) -> Any:
        return _FakeAnthropicStream(self._response)

    async def __aexit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        return None


class _FakeAnthropicStream:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aiter__(self):
        for index, block in enumerate(getattr(self._response, "content", [])):
            block_type = getattr(block, "type", None)
            if block_type not in {"text", "tool_use"}:
                continue
            if block_type == "text":
                start_block = SimpleNamespace(type="text", text="")
            else:
                start_block = SimpleNamespace(
                    type="tool_use",
                    id=getattr(block, "id", None),
                    name=getattr(block, "name", None),
                    input={},
                )
            yield SimpleNamespace(
                type="content_block_start",
                index=index,
                content_block=start_block,
            )
            if block_type == "text":
                text = getattr(block, "text", "")
                midpoint = max(1, len(text) // 2)
                chunks = [text[:midpoint], text[midpoint:]]
                for chunk in chunks:
                    if not chunk:
                        continue
                    yield SimpleNamespace(
                        type="content_block_delta",
                        index=index,
                        delta=SimpleNamespace(type="text_delta", text=chunk),
                    )
            if block_type == "tool_use":
                tool_input = json.dumps(getattr(block, "input", {}))
                midpoint = max(1, len(tool_input) // 2)
                chunks = [tool_input[:midpoint], tool_input[midpoint:]]
                for chunk in chunks:
                    if not chunk:
                        continue
                    yield SimpleNamespace(
                        type="content_block_delta",
                        index=index,
                        delta=SimpleNamespace(type="input_json_delta", partial_json=chunk),
                    )
            yield SimpleNamespace(type="content_block_stop", index=index)

    async def get_final_message(self) -> Any:
        return self._response


class _WrappedFakeAnthropicStreamManager:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aenter__(self):
        async def wrapped():
            async for event in _FakeAnthropicStream(self._response):
                yield event

        return wrapped()

    async def __aexit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        return None


class FakeWebSocketManager:
    def __init__(self) -> None:
        self.tool_requests: list[tuple[str, str, dict[str, Any]]] = []
        self.sent_payloads: list[tuple[str, dict[str, Any]]] = []

    async def request_tool(self, client_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.tool_requests.append((client_id, tool_name, params))
        if tool_name == "player_position":
            return {
                "x": 10.5,
                "y": 64.0,
                "z": 20.5,
                "yaw": 0.0,
                "pitch": 0.0,
                "block_x": 10,
                "block_y": 64,
                "block_z": 20,
                "ground_y": 63,
                "facing": "south",
                "dimension": "minecraft:overworld",
            }
        if tool_name == "place_blocks":
            return {"placed_count": len(params["placements"])}
        if tool_name == "fill_region":
            return {"placed_count": 1, "fill_region": True}
        return {"tool": tool_name, "ok": True}

    async def send_payload(self, client_id: str, payload: dict[str, Any]) -> None:
        self.sent_payloads.append((client_id, payload))


class SummarizationWebSocketManager(FakeWebSocketManager):
    async def request_tool(self, client_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.tool_requests.append((client_id, tool_name, params))
        if tool_name == "player_position":
            return {
                "x": 0.5,
                "y": 64.0,
                "z": 0.5,
                "yaw": 0.0,
                "pitch": 0.0,
                "block_x": 0,
                "block_y": 64,
                "block_z": 0,
                "ground_y": 63,
                "facing": "south",
                "dimension": "minecraft:overworld",
            }
        if tool_name == "inspect_area":
            return {
                "center": params["center"],
                "radius": params["radius"],
                "detailed": True,
                "filter_terrain": params.get("filter_terrain", True),
                "non_air_blocks": [
                    {"x": idx, "y": 64, "z": 0, "block_id": "minecraft:oak_planks"}
                    for idx in range(120)
                ],
                "block_counts": {"minecraft:oak_planks": 120},
            }
        return {"placed_count": 2, "anchor": {"x": 0, "y": 64, "z": 0}}


class DelayedWebSocketManager(FakeWebSocketManager):
    async def request_tool(self, client_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.tool_requests.append((client_id, tool_name, params))
        if tool_name == "player_position":
            return {
                "x": 10.5,
                "y": 64.0,
                "z": 20.5,
                "yaw": 0.0,
                "pitch": 0.0,
                "block_x": 10,
                "block_y": 64,
                "block_z": 20,
                "ground_y": 63,
                "facing": "south",
                "dimension": "minecraft:overworld",
            }
        await asyncio.sleep(0.05)
        return {"tool": tool_name, "ok": True}


class MutablePositionWebSocketManager(FakeWebSocketManager):
    def __init__(self) -> None:
        super().__init__()
        self.block_x = 10
        self.block_y = 64
        self.block_z = 20
        self.facing = "south"

    async def request_tool(self, client_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.tool_requests.append((client_id, tool_name, params))
        if tool_name == "player_position":
            return {
                "x": self.block_x + 0.5,
                "y": float(self.block_y),
                "z": self.block_z + 0.5,
                "yaw": 0.0,
                "pitch": 0.0,
                "block_x": self.block_x,
                "block_y": self.block_y,
                "block_z": self.block_z,
                "ground_y": self.block_y - 1,
                "facing": self.facing,
                "dimension": "minecraft:overworld",
            }
        return {"tool": tool_name, "ok": True}


class FakeConvexClient(ConvexHttpClient):
    def __init__(self) -> None:
        self.sessions_by_world: dict[str, dict[str, dict[str, Any]]] = {}
        self.query_calls: list[tuple[str, dict[str, Any]]] = []
        self.mutation_calls: list[tuple[str, dict[str, Any]]] = []

    async def query(self, path: str, args: dict[str, Any] | None = None) -> Any:
        payload = args or {}
        self.query_calls.append((path, payload))
        if path == "sessions:get":
            world_id = payload["world_id"]
            session_id = payload["session_id"]
            return self.sessions_by_world.get(world_id, {}).get(session_id)
        if path == "sessions:listByWorld":
            world_id = payload["world_id"]
            world_sessions = self.sessions_by_world.get(world_id, {})
            return list(world_sessions.values())
        raise AssertionError(f"Unexpected query path: {path}")

    async def mutation(self, path: str, args: dict[str, Any] | None = None) -> Any:
        payload = args or {}
        self.mutation_calls.append((path, payload))
        if path != "sessions:upsert":
            raise AssertionError(f"Unexpected mutation path: {path}")
        world_id = payload["world_id"]
        session_id = payload["session_id"]
        self.sessions_by_world.setdefault(world_id, {})[session_id] = copy.deepcopy(payload)
        return "ok"


class FakeSupermemoryClient:
    def __init__(
        self,
        search_results: list[SupermemorySearchResult] | None = None,
        profile_context: SupermemoryProfileContext | None = None,
    ) -> None:
        self.search_results = search_results or []
        self._profile_context = profile_context or SupermemoryProfileContext(static=(), dynamic=())
        self.profile_calls: list[str] = []
        self.search_calls: list[dict[str, Any]] = []
        self.store_calls: list[dict[str, Any]] = []

    async def search_memories(
        self,
        query: str,
        *,
        container_tag: str,
        limit: int = 5,
    ) -> list[SupermemorySearchResult]:
        self.search_calls.append(
            {
                "query": query,
                "container_tag": container_tag,
                "limit": limit,
            }
        )
        return self.search_results

    async def profile_context(self, container_tag: str) -> SupermemoryProfileContext:
        self.profile_calls.append(container_tag)
        return self._profile_context

    async def store_memory(
        self,
        content: str,
        *,
        container_tag: str,
        metadata: dict[str, Any],
    ) -> None:
        self.store_calls.append(
            {
                "content": content,
                "container_tag": container_tag,
                "metadata": metadata,
            }
        )


def _tool_use_response(tool_name: str, tool_input: dict[str, Any]) -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool-call-1",
                name=tool_name,
                input=tool_input,
            )
        ]
    )


def _multi_tool_use_response(tool_uses: list[tuple[str, str, dict[str, Any]]]) -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input,
            )
            for tool_id, tool_name, tool_input in tool_uses
        ]
    )


def _text_response(text: str) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _planner_response(*, name: str = "step-1", goal: str = "build", hint: str = "near anchor", success: str = "blocks placed") -> Any:
    return _text_response(
        json.dumps(
            {
                "steps": [
                    {
                        "name": name,
                        "goal": goal,
                        "relative_location_hint": hint,
                        "success_check": success,
                    }
                ]
            }
        )
    )


def _triage_response(
    *,
    is_build_request: bool = True,
    complexity: str = "complex",
    spatial_reference: str = "default_anchor",
    distance_hint: int | None = None,
    should_undo_first: bool = False,
) -> Any:
    return _text_response(
        json.dumps(
            {
                "is_build_request": is_build_request,
                "complexity": complexity,
                "spatial_reference": spatial_reference,
                "distance_hint": distance_hint,
                "should_undo_first": should_undo_first,
            }
        )
    )


def _default_triage_for_prompt(prompt: str) -> dict[str, Any]:
    lowered = prompt.lower()
    is_build_request = any(
        marker in lowered
        for marker in ("build ", "place ", "add ", "fill ", "replace ", "construct ", "make ", "undo", "revert")
    )
    return {
        "is_build_request": is_build_request,
        "complexity": "complex" if is_build_request else "simple",
        "spatial_reference": "default_anchor" if is_build_request else "none",
        "distance_hint": None,
        "should_undo_first": "undo" in lowered or "revert" in lowered,
    }


def _messages_of_type(ws: FakeWebSocketManager, event_type: str) -> list[dict[str, Any]]:
    return [payload for _, payload in ws.sent_payloads if payload.get("type") == event_type]


@pytest.mark.asyncio
async def test_place_blocks_tool_routes_through_websocket_manager_and_emits_chat_response() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "place_blocks",
                {
                    "placements": [
                        {"x": 10, "y": 64, "z": 20, "block_id": "minecraft:stone"},
                        {"x": 11, "y": 64, "z": 20, "block_id": "minecraft:stone"},
                    ]
                },
            ),
            _text_response("Placed blocks."),
        ],
        create_responses=[
            _triage_response(
                complexity="simple",
                spatial_reference="absolute_coordinates",
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
        enable_build_planner=True,
    )

    await orchestrator._run_chat(chat_id="chat-1", client_id="client-1", user_message="build a short wall")

    assert ws.tool_requests == [
        ("client-1", "player_position", {}),
        (
            "client-1",
            "place_blocks",
            {
                "placements": [
                    {"x": 10, "y": 64, "z": 20, "block_id": "minecraft:stone"},
                    {"x": 11, "y": 64, "z": 20, "block_id": "minecraft:stone"},
                ]
            },
        )
    ]

    first_call = anthropic_client.messages.calls[0]
    assert first_call["model"] == CHAT_MODEL
    assert first_call["max_tokens"] == 2048
    assert isinstance(first_call["system"], list)
    assert first_call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first_call["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert first_call["tool_choice"] == {"type": "any"}
    assert {tool["name"] for tool in first_call["tools"]} == EXPECTED_TOOL_NAMES
    assert "Locked player position for this request (captured at submit time, authoritative)" in first_call["system"][0]["text"]
    assert "block_x=10, block_y=64, block_z=20" in first_call["system"][0]["text"]
    assert "build_anchor_x=10, build_anchor_y=63, build_anchor_z=30" in first_call["system"][0]["text"]

    chat_responses = _messages_of_type(ws, "chat.response")
    assert len(chat_responses) == 1
    assert chat_responses[0]["chat_id"] == "chat-1"
    assert chat_responses[0]["payload"]["message"] == "Placed blocks."
    chat_deltas = _messages_of_type(ws, "chat.delta")
    assert chat_deltas
    assert anthropic_client.closed is True


@pytest.mark.asyncio
async def test_world_tool_routes_through_websocket_manager() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "inspect_area",
                {"center": {"x": 10, "y": 64, "z": 20}, "radius": 4},
            ),
            _text_response("Area inspected."),
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
        enable_build_planner=True,
    )

    await orchestrator._run_chat(chat_id="chat-2", client_id="client-2", user_message="scan nearby blocks")

    assert ws.tool_requests == [
        ("client-2", "player_position", {}),
        (
            "client-2",
            "inspect_area",
            {
                "center": {"x": 10, "y": 64, "z": 20},
                "radius": 4,
                "detailed": False,
                "filter_terrain": True,
            },
        )
    ]
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Area inspected."


@pytest.mark.asyncio
async def test_invalid_tool_args_are_reported_back_to_model() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response("inspect_area", {"radius": 3}),
            _text_response("Please provide center coordinates."),
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-3", client_id="client-3", user_message="inspect around me")

    second_call = anthropic_client.messages.calls[1]
    tool_result = second_call["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tool-call-1"
    assert tool_result["is_error"] is True
    assert "Invalid arguments for inspect_area" in tool_result["content"]
    assert ws.tool_requests == [("client-3", "player_position", {})]
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Please provide center coordinates."


@pytest.mark.asyncio
async def test_detailed_inspect_radius_limit_is_reported_back_to_model() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "inspect_area",
                {"center": {"x": 0, "y": 64, "z": 0}, "radius": 8, "detailed": True},
            ),
            _text_response("Use a smaller radius for detailed inspection."),
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-limit", client_id="client-limit", user_message="inspect room detail")

    second_call = anthropic_client.messages.calls[1]
    tool_result = second_call["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "inspect_area with detailed=true requires radius <= 6" in tool_result["content"]
    assert ws.tool_requests == [("client-limit", "player_position", {})]


@pytest.mark.asyncio
async def test_placement_far_from_live_player_height_is_rejected() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _planner_response(name="far-placement", goal="attempt placement"),
            _tool_use_response(
                "place_blocks",
                {"placements": [{"x": 10, "y": 300, "z": 20, "block_id": "minecraft:stone"}]},
            ),
            _text_response("I'll re-check and use your actual position."),
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-y-range", client_id="client-y-range", user_message="build here")

    third_call = anthropic_client.messages.calls[2]
    tool_result = third_call["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "detached from current player block_y" in tool_result["content"]
    assert ws.tool_requests == [("client-y-range", "player_position", {})]
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "I'll re-check and use your actual position."


@pytest.mark.asyncio
async def test_older_tool_results_are_summarized_in_followup_rounds() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "inspect_area",
                {"center": {"x": 0, "y": 64, "z": 0}, "radius": 4, "detailed": True},
            ),
            _tool_use_response(
                "place_blocks",
                {
                    "placements": [
                        {"x": 0, "y": 64, "z": 0, "block_id": "minecraft:stone"},
                        {"x": 1, "y": 64, "z": 0, "block_id": "minecraft:stone"},
                    ]
                },
            ),
            _text_response("Done."),
        ]
    )
    ws = SummarizationWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-summary", client_id="client-summary", user_message="build")

    third_call = anthropic_client.messages.calls[2]
    tool_result_messages = [
        message
        for message in third_call["messages"]
        if message["role"] == "user" and isinstance(message["content"], list)
    ]
    summarized_found = False
    for message in tool_result_messages:
        for block in message["content"]:
            content = block.get("content")
            if isinstance(content, str) and content.startswith("[summarized tool_result]"):
                summarized_found = True
                break
        if summarized_found:
            break
    assert summarized_found is True


@pytest.mark.asyncio
async def test_session_lifecycle_works_with_in_memory_store() -> None:
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: FakeAnthropicClient([_text_response("unused")]),
    )

    created = await orchestrator.create_session(client_id="client-1", world_id="world-1")
    assert created.world_id == "world-1"
    assert created.status == "created"

    listed = await orchestrator.list_sessions(client_id="client-1", world_id="world-1")
    assert listed.world_id == "world-1"
    assert listed.active_session_id == created.session_id
    assert [summary.session_id for summary in listed.sessions] == [created.session_id]
    assert listed.sessions[0].message_count == 0

    switched = await orchestrator.switch_session(
        client_id="client-1",
        world_id="world-1",
        session_id=created.session_id,
    )
    assert switched.status == "active"
    assert switched.session_id == created.session_id


@pytest.mark.asyncio
async def test_explicit_session_override_uses_requested_convex_session() -> None:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    convex = FakeConvexClient()
    convex.sessions_by_world = {
        "world-1": {
            "session-1": {
                "world_id": "world-1",
                "session_id": "session-1",
                "messages": [
                    {"role": "user", "content": "previous user"},
                    {"role": "assistant", "content": "previous assistant"},
                ],
                "created_at": now_ms - 50_000,
                "updated_at": now_ms - 10_000,
            }
        }
    }

    anthropic_client = FakeAnthropicClient([_text_response("latest assistant")])
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
        convex_client=convex,
    )

    await orchestrator._run_chat(
        chat_id="chat-4",
        client_id="client-1",
        user_message="new question",
        world_id="world-1",
        session_id="session-1",
    )

    first_call = anthropic_client.messages.calls[0]
    assert first_call["messages"] == [
        {"role": "user", "content": "previous user"},
        {"role": "assistant", "content": "previous assistant"},
        {"role": "user", "content": "new question"},
    ]
    persisted = convex.sessions_by_world["world-1"]["session-1"]["messages"]
    assert persisted == [
        {"role": "user", "content": "previous user"},
        {"role": "assistant", "content": "previous assistant"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "latest assistant"},
    ]
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "latest assistant"


@pytest.mark.asyncio
async def test_supermemory_is_used_for_context_and_meaningful_tool_outcomes() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _planner_response(name="memory-build", goal="place remembered block"),
            _tool_use_response(
                "place_blocks",
                {"placements": [{"x": 0, "y": 64, "z": 0, "block_id": "minecraft:oak_planks"}]},
            ),
            _text_response("Placed it."),
        ]
    )
    supermemory = FakeSupermemoryClient(
        search_results=[SupermemorySearchResult(text="User prefers oak builds.", similarity=0.91)],
        profile_context=SupermemoryProfileContext(
            static=("User plays in survival mode.",),
            dynamic=("Currently building with oak and stone.",),
        ),
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
        supermemory_client=supermemory,
    )

    await orchestrator._run_chat(chat_id="chat-5", client_id="client-6", user_message="build something")

    first_call = anthropic_client.messages.calls[1]
    system_text = first_call["system"][0]["text"]
    assert "Relevant long-term memory" in system_text
    assert "Profile static:" in system_text
    assert "Profile dynamic:" in system_text
    assert "User prefers oak builds." in system_text
    assert supermemory.profile_calls == ["default:client-6"]
    assert supermemory.search_calls == [
        {
            "query": "build something",
            "container_tag": "default:client-6",
            "limit": 5,
        }
    ]
    assert len(supermemory.store_calls) == 1
    assert supermemory.store_calls[0]["container_tag"] == "default:client-6"
    assert supermemory.store_calls[0]["metadata"]["tool"] == "place_blocks"


@pytest.mark.asyncio
async def test_set_plan_tool_routes_through_websocket_manager() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "set_plan",
                {
                    "placements": [
                        {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:stone"},
                        {"dx": 1, "dy": 0, "dz": 0, "block_id": "minecraft:stone"},
                    ]
                },
            ),
            _text_response("Plan loaded."),
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-plan",
        client_id="client-plan",
        user_message="plan a tower",
        request_mode="plan",
    )

    assert ws.tool_requests == [
        ("client-plan", "player_position", {}),
        (
            "client-plan",
            "set_plan",
            {
                "placements": [
                    {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:stone", "block_state": {}},
                    {"dx": 1, "dy": 0, "dz": 0, "block_id": "minecraft:stone", "block_state": {}},
                ]
            },
        ),
    ]


@pytest.mark.asyncio
async def test_set_plan_tool_validation_rejects_empty_placements() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response("set_plan", {"placements": []}),
            _text_response("Please provide at least one placement."),
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-plan-invalid",
        client_id="client-plan-invalid",
        user_message="plan",
        request_mode="plan",
    )

    assert ws.tool_requests == [("client-plan-invalid", "player_position", {})]


@pytest.mark.asyncio
async def test_parallel_tool_calls_execute_concurrently() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _multi_tool_use_response(
                [
                    ("tool-call-1", "inspect_area", {"center": {"x": 10, "y": 64, "z": 20}, "radius": 4}),
                    ("tool-call-2", "player_inventory", {}),
                ]
            ),
            _text_response("Done."),
        ]
    )
    ws = DelayedWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    started = time.perf_counter()
    await orchestrator._run_chat(chat_id="chat-parallel", client_id="client-parallel", user_message="build")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.095


@pytest.mark.asyncio
async def test_streaming_delta_events_are_sent_over_websocket() -> None:
    anthropic_client = FakeAnthropicClient(responses=[_text_response("Streaming test response.")])
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-stream", client_id="client-stream", user_message="hi")

    deltas = _messages_of_type(ws, "chat.delta")
    assert len(deltas) >= 2
    assert deltas[-1]["payload"]["partial"] == "Streaming test response."
    first_call = anthropic_client.messages.calls[0]
    assert "tool_choice" not in first_call
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Streaming test response."


@pytest.mark.asyncio
async def test_build_intent_retries_when_model_replies_without_placement_tool() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _text_response("Done! I placed the platform."),
            _tool_use_response(
                "place_blocks",
                {
                    "placements": [
                        {"x": 8, "y": 63, "z": 18, "block_id": "minecraft:oak_planks"},
                        {"x": 9, "y": 63, "z": 18, "block_id": "minecraft:oak_planks"},
                    ],
                },
            ),
            _text_response("Placed it."),
        ],
        create_responses=[
            _triage_response(
                complexity="simple",
                spatial_reference="absolute_coordinates",
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-build-enforced",
        client_id="client-build-enforced",
        user_message="place two oak planks at absolute coordinates",
    )

    assert ws.tool_requests == [
        ("client-build-enforced", "player_position", {}),
        (
            "client-build-enforced",
            "place_blocks",
            {
                "placements": [
                    {"x": 8, "y": 63, "z": 18, "block_id": "minecraft:oak_planks"},
                    {"x": 9, "y": 63, "z": 18, "block_id": "minecraft:oak_planks"},
                ],
            },
        ),
    ]
    assert len(anthropic_client.messages.calls) == 3
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Placed it."


@pytest.mark.asyncio
async def test_directly_in_front_request_uses_one_block_forward_anchor() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _planner_response(name="tower", goal="place tower"),
            _tool_use_response(
                "place_blocks",
                {"placements": [{"x": 10, "y": 64, "z": 21, "block_id": "minecraft:stone"}]},
            ),
            _text_response("Placed."),
        ],
        create_responses=[
            _triage_response(
                complexity="complex",
                spatial_reference="relative_to_player",
                distance_hint=1,
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-front-anchor",
        client_id="client-front-anchor",
        user_message="build a 3 block tower directly in front of me",
    )

    first_executor_call = anthropic_client.messages.calls[1]
    assert "build_anchor_x=10, build_anchor_y=64, build_anchor_z=21" in first_executor_call["system"][0]["text"]


def test_relative_request_without_distance_uses_one_block_forward_anchor() -> None:
    player_position = _PlayerPositionResult(
        x=10.5,
        y=64.0,
        z=20.5,
        block_x=10,
        block_y=64,
        block_z=20,
        ground_y=63,
        facing="south",
        dimension="minecraft:overworld",
    )
    triage = _BuildRequestTriageModel(
        is_build_request=True,
        complexity="simple",
        spatial_reference="relative_to_player",
        distance_hint=None,
        should_undo_first=False,
    )

    assert _build_anchor_for_request(player_position=player_position, triage=triage) == _BuildAnchor(x=10, y=64, z=21)


def test_relative_floor_prompt_keeps_footprint_entirely_ahead_of_player() -> None:
    player_position = _PlayerPositionResult(
        x=10.5,
        y=64.0,
        z=20.5,
        block_x=10,
        block_y=64,
        block_z=20,
        ground_y=63,
        facing="south",
        dimension="minecraft:overworld",
    )
    triage = _BuildRequestTriageModel(
        is_build_request=True,
        complexity="simple",
        spatial_reference="relative_to_player",
        distance_hint=1,
        should_undo_first=False,
    )

    prompt = _compose_system_prompt(
        request_mode="build",
        player_position=player_position,
        build_anchor=_BuildAnchor(x=10, y=64, z=21),
        memory_context="",
        triage=triage,
    )

    assert "keep the entire footprint on the requested side of the player" in prompt
    assert "place the floor over z=21..25" in prompt


def test_relative_front_guard_rejects_platform_that_straddles_player() -> None:
    player_position = _PlayerPositionResult(
        x=10.5,
        y=64.0,
        z=20.5,
        block_x=10,
        block_y=64,
        block_z=20,
        ground_y=63,
        facing="south",
        dimension="minecraft:overworld",
    )
    triage = _BuildRequestTriageModel(
        is_build_request=True,
        complexity="simple",
        spatial_reference="relative_to_player",
        distance_hint=1,
        should_undo_first=False,
    )
    relative_guard = _relative_placement_guard_for_request(
        user_message="Build a 5x5 oak floor in front of me.",
        player_position=player_position,
        triage=triage,
    )

    centered_platform = [
        {"x": x, "y": 64, "z": z, "block_id": "minecraft:oak_planks"}
        for x in range(8, 13)
        for z in range(19, 24)
    ]

    with pytest.raises(ValueError, match="fully in front of the player"):
        _validate_placement_against_player_position(
            tool_name="place_blocks",
            params={"placements": centered_platform},
            player_position=player_position,
            relative_placement_guard=relative_guard,
        )


def test_default_anchor_guard_rejects_platform_centered_on_player() -> None:
    player_position = _PlayerPositionResult(
        x=10.5,
        y=64.0,
        z=20.5,
        block_x=10,
        block_y=64,
        block_z=20,
        ground_y=63,
        facing="south",
        dimension="minecraft:overworld",
    )
    triage = _BuildRequestTriageModel(
        is_build_request=True,
        complexity="simple",
        spatial_reference="default_anchor",
        distance_hint=None,
        should_undo_first=False,
    )
    default_guard = _relative_placement_guard_for_request(
        user_message="Build a 5x5 oak floor.",
        player_position=player_position,
        triage=triage,
    )

    centered_platform = [
        {"x": x, "y": 64, "z": z, "block_id": "minecraft:oak_planks"}
        for x in range(8, 13)
        for z in range(18, 23)
    ]

    with pytest.raises(ValueError, match="fully in front of the player"):
        _validate_placement_against_player_position(
            tool_name="place_blocks",
            params={"placements": centered_platform},
            player_position=player_position,
            relative_placement_guard=default_guard,
        )


@pytest.mark.asyncio
async def test_undo_request_executes_undo_last_before_planning() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _planner_response(name="line", goal="place replacement line"),
            _tool_use_response(
                "place_blocks",
                {"placements": [{"x": 10, "y": 64, "z": 20, "block_id": "minecraft:birch_planks"}]},
            ),
            _text_response("Done."),
        ],
        create_responses=[
            _triage_response(
                complexity="complex",
                spatial_reference="default_anchor",
                should_undo_first=True,
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-undo-first",
        client_id="client-undo-first",
        user_message="Undo that and build a birch line",
    )

    assert ws.tool_requests[0] == ("client-undo-first", "player_position", {})
    assert ws.tool_requests[1] == ("client-undo-first", "undo_last", {})


@pytest.mark.asyncio
async def test_simple_build_request_skips_planner_phase() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "place_blocks",
                {"placements": [{"x": 10, "y": 64, "z": 20, "block_id": "minecraft:torch"}]},
            ),
            _text_response("Placed torch."),
        ],
        create_responses=[
            _triage_response(
                complexity="simple",
                spatial_reference="relative_to_player",
                distance_hint=0,
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-simple-skip-planner",
        client_id="client-simple-skip-planner",
        user_message="place a torch here",
    )

    assert len(anthropic_client.messages.calls) == 2
    assert anthropic_client.messages.calls[0]["model"] == CHAT_MODEL


@pytest.mark.asyncio
async def test_regular_geometry_request_keeps_all_world_modifying_tools_available() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "build_geometry",
                {
                    "shape": "floor",
                    "material": "minecraft:stone",
                    "anchor": {"x": 10, "y": 63, "z": 20},
                    "rotation": "south",
                    "width": 5,
                    "depth": 5,
                    "thickness": 1,
                },
            ),
            _text_response("Built the platform."),
        ],
        create_responses=[
            _triage_response(
                complexity="simple",
                spatial_reference="default_anchor",
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-geometry-tools-visible",
        client_id="client-geometry-tools-visible",
        user_message="build a 5x5 stone platform",
    )

    assert ws.tool_requests == [
        ("client-geometry-tools-visible", "player_position", {}),
        (
            "client-geometry-tools-visible",
            "place_blocks",
            {
                "placements": [
                    {"x": 8, "y": 63, "z": 18, "block_id": "minecraft:stone"},
                    {"x": 8, "y": 63, "z": 19, "block_id": "minecraft:stone"},
                    {"x": 8, "y": 63, "z": 20, "block_id": "minecraft:stone"},
                    {"x": 8, "y": 63, "z": 21, "block_id": "minecraft:stone"},
                    {"x": 8, "y": 63, "z": 22, "block_id": "minecraft:stone"},
                    {"x": 9, "y": 63, "z": 18, "block_id": "minecraft:stone"},
                    {"x": 9, "y": 63, "z": 19, "block_id": "minecraft:stone"},
                    {"x": 9, "y": 63, "z": 20, "block_id": "minecraft:stone"},
                    {"x": 9, "y": 63, "z": 21, "block_id": "minecraft:stone"},
                    {"x": 9, "y": 63, "z": 22, "block_id": "minecraft:stone"},
                    {"x": 10, "y": 63, "z": 18, "block_id": "minecraft:stone"},
                    {"x": 10, "y": 63, "z": 19, "block_id": "minecraft:stone"},
                    {"x": 10, "y": 63, "z": 20, "block_id": "minecraft:stone"},
                    {"x": 10, "y": 63, "z": 21, "block_id": "minecraft:stone"},
                    {"x": 10, "y": 63, "z": 22, "block_id": "minecraft:stone"},
                    {"x": 11, "y": 63, "z": 18, "block_id": "minecraft:stone"},
                    {"x": 11, "y": 63, "z": 19, "block_id": "minecraft:stone"},
                    {"x": 11, "y": 63, "z": 20, "block_id": "minecraft:stone"},
                    {"x": 11, "y": 63, "z": 21, "block_id": "minecraft:stone"},
                    {"x": 11, "y": 63, "z": 22, "block_id": "minecraft:stone"},
                    {"x": 12, "y": 63, "z": 18, "block_id": "minecraft:stone"},
                    {"x": 12, "y": 63, "z": 19, "block_id": "minecraft:stone"},
                    {"x": 12, "y": 63, "z": 20, "block_id": "minecraft:stone"},
                    {"x": 12, "y": 63, "z": 21, "block_id": "minecraft:stone"},
                    {"x": 12, "y": 63, "z": 22, "block_id": "minecraft:stone"},
                ]
            },
        ),
    ]
    first_call = anthropic_client.messages.calls[0]
    first_tool_names = {tool["name"] for tool in first_call["tools"]}
    assert {"place_blocks", "fill_region", "build_geometry"} <= first_tool_names
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Built the platform."


@pytest.mark.asyncio
async def test_build_planner_is_disabled_by_default_for_complex_build_requests() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "build_geometry",
                {
                    "shape": "box",
                    "material": "minecraft:stone",
                    "anchor": {"x": 10, "y": 63, "z": 20},
                    "rotation": "south",
                    "width": 3,
                    "height": 3,
                    "depth": 3,
                    "hollow": False,
                },
            ),
            _text_response("Built the cube."),
        ],
        create_responses=[
            _triage_response(
                complexity="complex",
                spatial_reference="default_anchor",
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
        enable_build_planner=False,
    )

    await orchestrator._run_chat(
        chat_id="chat-no-planner",
        client_id="client-no-planner",
        user_message="build a 3x3x3 stone cube",
    )

    assert len(anthropic_client.messages.calls) == 2
    assert anthropic_client.messages.calls[0]["model"] == CHAT_MODEL
    statuses = _messages_of_type(ws, "chat.tool_status")
    assert all(payload["payload"]["status"] != "🧠 Drafting step plan..." for payload in statuses)


@pytest.mark.asyncio
async def test_build_triage_failure_is_reported_without_heuristic_fallback() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[],
        create_responses=[RuntimeError("triage unavailable")],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-triage-failure",
        client_id="client-triage-failure",
        user_message="place a torch here",
    )

    assert anthropic_client.messages.calls == []
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Unable to process chat request: triage unavailable"


@pytest.mark.asyncio
async def test_step_retry_discards_failed_attempt_message_history(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[_planner_response(name="wall", goal="build wall")],
        create_responses=[
            _triage_response(
                complexity="complex",
                spatial_reference="default_anchor",
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
        enable_build_planner=True,
    )
    attempt_messages: list[list[dict[str, Any]]] = []

    async def fake_execute_tool_loop(**kwargs: Any) -> tuple[str, bool, list[dict[str, Any]]]:
        messages = kwargs["messages"]
        attempt_messages.append(copy.deepcopy(messages))
        if len(attempt_messages) == 1:
            messages.append({"role": "assistant", "content": "failed attempt"})
            raise RuntimeError("step failed")
        assert all(message.get("content") != "failed attempt" for message in messages)
        messages.append({"role": "assistant", "content": "step complete"})
        return "step complete", True, [{"tool_name": "place_blocks", "result": {"placed_count": 1}}]

    monkeypatch.setattr(orchestrator, "_execute_tool_loop", fake_execute_tool_loop)

    await orchestrator._run_chat(
        chat_id="chat-step-retry",
        client_id="client-step-retry",
        user_message="build a wall",
    )

    assert len(attempt_messages) == 2
    assert attempt_messages[1] == attempt_messages[0]


@pytest.mark.asyncio
async def test_direct_build_rejects_preview_only_tools() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "set_plan",
                {
                    "placements": [
                        {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:oak_planks", "block_state": {}},
                    ]
                },
            ),
            _tool_use_response(
                "fill_region",
                {
                    "from_corner": {"x": 8, "y": 63, "z": 18},
                    "to_corner": {"x": 12, "y": 63, "z": 22},
                    "block_id": "minecraft:oak_planks",
                },
            ),
            _text_response("Placed a 5x5 platform."),
        ],
        create_responses=[
            _triage_response(
                complexity="simple",
                spatial_reference="default_anchor",
            )
        ],
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-direct-build-tool-guard",
        client_id="client-direct-build-tool-guard",
        user_message="build a 5x5 oak platform right below me",
    )

    assert ws.tool_requests == [
        ("client-direct-build-tool-guard", "player_position", {}),
        (
            "client-direct-build-tool-guard",
            "fill_region",
            {
                "from_corner": {"x": 8, "y": 63, "z": 18},
                "to_corner": {"x": 12, "y": 63, "z": 22},
                "block_id": "minecraft:oak_planks",
            },
        ),
    ]
    second_call = anthropic_client.messages.calls[1]
    tool_result = second_call["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "preview-only" in tool_result["content"]
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Placed a 5x5 platform."


@pytest.mark.asyncio
async def test_streaming_fallback_handles_wrapped_async_generator_stream() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[_text_response("Wrapped stream response.")],
        wrapped_stream=True,
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-stream-wrapped", client_id="client-stream-wrapped", user_message="hi")

    deltas = _messages_of_type(ws, "chat.delta")
    assert len(deltas) >= 2
    assert deltas[-1]["payload"]["partial"] == "Wrapped stream response."
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Wrapped stream response."


@pytest.mark.asyncio
async def test_wrapped_stream_fallback_reconstructs_tool_use_blocks() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "place_blocks",
                {
                    "placements": [
                        {"x": 10, "y": 64, "z": 20, "block_id": "minecraft:stone"},
                    ]
                },
            ),
            _text_response("Placed one block."),
        ],
        wrapped_stream=True,
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-wrapped-tool", client_id="client-wrapped-tool", user_message="build")

    assert ws.tool_requests == [
        ("client-wrapped-tool", "player_position", {}),
        (
            "client-wrapped-tool",
            "place_blocks",
            {"placements": [{"x": 10, "y": 64, "z": 20, "block_id": "minecraft:stone"}]},
        ),
    ]
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Placed one block."


def test_tool_status_message_formats_expected_strings() -> None:
    assert _tool_status_message("inspect_area", {"radius": 6}) == "🔍 Inspecting area (r=6)..."
    assert _tool_status_message("inspect_area", {}) == "🔍 Inspecting area..."
    assert _tool_status_message(
        "place_blocks",
        {"placements": [{"x": 0, "y": 64, "z": 0, "block_id": "minecraft:stone"}]},
    ) == "🔨 Placing 1 blocks..."
    assert _tool_status_message("fill_region", {}) == "🧱 Filling region..."
    assert _tool_status_message(
        "set_plan",
        {"placements": [{"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:stone"}]},
    ) == "📐 Loading plan (1 blocks)..."
    assert _tool_status_message("player_position", {}) == "📍 Checking player position..."
    assert _tool_status_message("undo_last", {}) == "↩ Undoing last change..."


@pytest.mark.asyncio
async def test_player_position_tool_use_is_served_from_cached_request_position() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response("player_position", {}),
            _text_response("Position confirmed."),
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(chat_id="chat-position", client_id="client-position", user_message="where am i")

    assert ws.tool_requests == [("client-position", "player_position", {})]
    second_call = anthropic_client.messages.calls[1]
    tool_result = second_call["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert '"block_x": 10' in tool_result["content"]


@pytest.mark.asyncio
async def test_submit_chat_mode_plan_propagates_to_system_prompt() -> None:
    anthropic_client = FakeAnthropicClient(responses=[_text_response("Planned.")])
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    accepted = await orchestrator.submit_chat(
        ChatRequest(client_id="client-mode", message="plan a tower", mode="plan")
    )
    await asyncio.gather(*list(orchestrator._tasks))

    assert accepted.status == "accepted"
    first_call = anthropic_client.messages.calls[0]
    assert first_call["tool_choice"] == {"type": "any"}
    assert "Request mode for this turn: PLAN." in first_call["system"][0]["text"]
    assert "build_anchor_x=10, build_anchor_y=63, build_anchor_z=30" in first_call["system"][0]["text"]


@pytest.mark.asyncio
async def test_submit_chat_locks_player_position_at_submit_time() -> None:
    anthropic_client = FakeAnthropicClient(responses=[_text_response("Acknowledged.")])
    ws = MutablePositionWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    accepted = await orchestrator.submit_chat(
        ChatRequest(client_id="client-lock", message="where am i", mode="build")
    )
    ws.block_x = 99
    ws.block_y = 70
    ws.block_z = -20
    ws.facing = "north"
    await asyncio.gather(*list(orchestrator._tasks))

    assert accepted.status == "accepted"
    assert ws.tool_requests == [("client-lock", "player_position", {})]
    first_call = anthropic_client.messages.calls[0]
    assert "block_x=10, block_y=64, block_z=20" in first_call["system"][0]["text"]
    assert "facing=south" in first_call["system"][0]["text"]
    assert "build_anchor_x=10, build_anchor_y=63, build_anchor_z=30" in first_call["system"][0]["text"]


@pytest.mark.asyncio
async def test_plan_fast_forces_single_set_plan_tool_round() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
            _tool_use_response(
                "set_plan",
                {
                    "placements": [
                        {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:oak_planks"},
                    ]
                },
            )
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator._run_chat(
        chat_id="chat-plan-fast",
        client_id="client-plan-fast",
        user_message="plan a market stall",
        request_mode="plan_fast",
    )

    assert ws.tool_requests == [
        ("client-plan-fast", "player_position", {}),
        (
            "client-plan-fast",
            "set_plan",
            {
                "placements": [
                    {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:oak_planks", "block_state": {}},
                ]
            },
        ),
    ]
    first_call = anthropic_client.messages.calls[0]
    assert first_call["tool_choice"] == {"type": "tool", "name": "set_plan"}
    assert {tool["name"] for tool in first_call["tools"]} == {"set_plan"}
    chat_responses = _messages_of_type(ws, "chat.response")
    assert chat_responses[0]["payload"]["message"] == "Preview loaded. Reposition if needed, then confirm to place."


@pytest.mark.asyncio
async def test_warmup_prompt_cache_is_idempotent() -> None:
    anthropic_client = FakeAnthropicClient(responses=[])
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    await orchestrator.warmup_prompt_cache()
    await orchestrator.warmup_prompt_cache()

    assert len(anthropic_client.messages.create_calls) == 2
    assert anthropic_client.messages.create_calls[0]["max_tokens"] == 1
    assert anthropic_client.messages.create_calls[1]["max_tokens"] == 1
