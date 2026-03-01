from __future__ import annotations

import copy
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from browsecraft_backend.chat_orchestrator import CHAT_MODEL, ChatOrchestrator
from browsecraft_backend.convex_client import ConvexHttpClient
from browsecraft_backend.supermemory_client import SupermemoryProfileContext, SupermemorySearchResult


EXPECTED_TOOL_NAMES = {
    "player_position",
    "player_inventory",
    "inspect_area",
    "place_blocks",
    "fill_region",
    "undo_last",
    "get_active_overlay",
    "modify_overlay",
    "get_blueprints",
    "save_blueprint",
    "load_blueprint",
}


class FakeAnthropicMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise AssertionError("Unexpected anthropic create call")
        return self._responses.pop(0)


class FakeAnthropicClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeAnthropicMessages(responses)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


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
                "facing": "south",
                "dimension": "minecraft:overworld",
            }
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


def _text_response(text: str) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


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
        ]
    )
    ws = FakeWebSocketManager()
    orchestrator = ChatOrchestrator(
        anthropic_api_key="test-key",
        websocket_manager=ws,
        anthropic_client_factory=lambda api_key: anthropic_client,
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
    assert first_call["max_tokens"] == 768
    assert isinstance(first_call["system"], list)
    assert first_call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first_call["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert {tool["name"] for tool in first_call["tools"]} == EXPECTED_TOOL_NAMES
    assert "Live player position for this request (authoritative)" in first_call["system"][0]["text"]
    assert "block_x=10, block_y=64, block_z=20" in first_call["system"][0]["text"]

    assert ws.sent_payloads[0][0] == "client-1"
    assert ws.sent_payloads[0][1]["type"] == "chat.response"
    assert ws.sent_payloads[0][1]["chat_id"] == "chat-1"
    assert ws.sent_payloads[0][1]["payload"]["message"] == "Placed blocks."
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
    assert ws.sent_payloads[0][1]["payload"]["message"] == "Area inspected."


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
    assert ws.sent_payloads[0][1]["payload"]["message"] == "Please provide center coordinates."


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

    second_call = anthropic_client.messages.calls[1]
    tool_result = second_call["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "detached from current player block_y" in tool_result["content"]
    assert ws.tool_requests == [("client-y-range", "player_position", {})]
    assert ws.sent_payloads[0][1]["payload"]["message"] == "I'll re-check and use your actual position."


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
    assert ws.sent_payloads[0][1]["payload"]["message"] == "latest assistant"


@pytest.mark.asyncio
async def test_supermemory_is_used_for_context_and_meaningful_tool_outcomes() -> None:
    anthropic_client = FakeAnthropicClient(
        responses=[
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

    first_call = anthropic_client.messages.calls[0]
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
