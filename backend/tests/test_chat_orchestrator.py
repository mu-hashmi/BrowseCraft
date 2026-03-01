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
        return {"tool": tool_name, "ok": True}

    async def send_payload(self, client_id: str, payload: dict[str, Any]) -> None:
        self.sent_payloads.append((client_id, payload))


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
    assert first_call["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert {tool["name"] for tool in first_call["tools"]} == EXPECTED_TOOL_NAMES

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
        (
            "client-2",
            "inspect_area",
            {"center": {"x": 10, "y": 64, "z": 20}, "radius": 4, "detailed": False},
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
    assert ws.tool_requests == []
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
    assert ws.tool_requests == []


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
    assert "Relevant long-term memory" in first_call["system"]
    assert "Profile static:" in first_call["system"]
    assert "Profile dynamic:" in first_call["system"]
    assert "User prefers oak builds." in first_call["system"]
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
