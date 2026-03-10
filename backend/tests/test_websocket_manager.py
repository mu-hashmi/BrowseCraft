from __future__ import annotations

import asyncio
from typing import Any

import pytest

from browsecraft_backend.websocket_manager import WebSocketManager


class DummyWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent_payloads: list[dict[str, Any]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent_payloads.append(payload)


@pytest.mark.asyncio
async def test_request_tool_matches_incoming_response() -> None:
    manager = WebSocketManager()
    websocket = DummyWebSocket()
    await manager.connect("client-1", websocket)

    request_task = asyncio.create_task(
        manager.request_tool(
            client_id="client-1",
            tool_name="player_position",
            params={},
            timeout=1.0,
        )
    )

    await asyncio.sleep(0)
    request_payload = websocket.sent_payloads[0]
    assert request_payload["type"] == "tool.request"
    assert request_payload["tool"] == "player_position"

    await manager.handle_incoming_message(
        "client-1",
        {
            "type": "tool.response",
            "request_id": request_payload["request_id"],
            "result": {"x": 10, "y": 64, "z": 20},
        },
    )

    result = await request_task
    assert result == {"x": 10, "y": 64, "z": 20}


@pytest.mark.asyncio
async def test_request_tool_raises_timeout() -> None:
    manager = WebSocketManager()
    websocket = DummyWebSocket()
    await manager.connect("client-1", websocket)

    with pytest.raises(TimeoutError, match="Timed out waiting for tool response"):
        await manager.request_tool(
            client_id="client-1",
            tool_name="player_inventory",
            params={},
            timeout=0.01,
        )


@pytest.mark.asyncio
async def test_disconnect_fails_pending_request() -> None:
    manager = WebSocketManager()
    websocket = DummyWebSocket()
    await manager.connect("client-1", websocket)

    request_task = asyncio.create_task(
        manager.request_tool(
            client_id="client-1",
            tool_name="inspect_area",
            params={"radius": 4},
            timeout=5.0,
        )
    )
    await asyncio.sleep(0)

    await manager.disconnect("client-1")
    with pytest.raises(ConnectionError, match="WebSocket disconnected"):
        await request_task
