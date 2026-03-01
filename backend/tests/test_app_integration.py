from __future__ import annotations

from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient

from browsecraft_backend.app import create_app
from browsecraft_backend.models import (
    ChatAcceptedResponse,
    ChatRequest,
    SessionCreatedResponse,
    SessionListResponse,
    SessionSummary,
    SessionSwitchedResponse,
)


class DummyChatOrchestrator:
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.new_session_requests: list[tuple[str, str]] = []
        self.list_session_requests: list[tuple[str, str]] = []
        self.switch_session_requests: list[tuple[str, str, str]] = []

    async def submit_chat(self, request: ChatRequest) -> ChatAcceptedResponse:
        self.requests.append(request)
        return ChatAcceptedResponse(chat_id="chat-123", status="accepted")

    async def create_session(self, client_id: str, world_id: str) -> SessionCreatedResponse:
        self.new_session_requests.append((client_id, world_id))
        return SessionCreatedResponse(world_id=world_id, session_id="session-123", status="created")

    async def list_sessions(self, client_id: str, world_id: str) -> SessionListResponse:
        self.list_session_requests.append((client_id, world_id))
        now = datetime.now(UTC)
        return SessionListResponse(
            world_id=world_id,
            active_session_id="session-123",
            sessions=[
                SessionSummary(
                    session_id="session-123",
                    message_count=2,
                    created_at=now,
                    updated_at=now,
                )
            ],
        )

    async def switch_session(self, client_id: str, world_id: str, session_id: str) -> SessionSwitchedResponse:
        self.switch_session_requests.append((client_id, world_id, session_id))
        if session_id == "missing":
            raise LookupError("Session missing not found")
        return SessionSwitchedResponse(world_id=world_id, session_id=session_id, status="active")


def test_chat_endpoint_returns_accepted_with_chat_id() -> None:
    def blocked_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call: {request.url}")

    chat = DummyChatOrchestrator()
    app = create_app(
        http_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(blocked_handler)),
        chat_orchestrator=lambda settings, ws_manager: chat,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "client_id": "chat-client",
                "message": "where am i?",
            },
        )

    response.raise_for_status()
    assert response.json() == {"chat_id": "chat-123", "status": "accepted"}
    assert len(chat.requests) == 1
    assert chat.requests[0].client_id == "chat-client"
    assert chat.requests[0].message == "where am i?"


def test_session_endpoints_route_to_chat_orchestrator() -> None:
    def blocked_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call: {request.url}")

    chat = DummyChatOrchestrator()
    app = create_app(
        http_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(blocked_handler)),
        chat_orchestrator=lambda settings, ws_manager: chat,
    )

    with TestClient(app) as client:
        new_response = client.post(
            "/v1/session/new",
            json={"client_id": "chat-client", "world_id": "world-1"},
        )
        new_response.raise_for_status()
        assert new_response.json() == {
            "world_id": "world-1",
            "session_id": "session-123",
            "status": "created",
        }

        list_response = client.get(
            "/v1/session/list",
            params={"client_id": "chat-client", "world_id": "world-1"},
        )
        list_response.raise_for_status()
        assert list_response.json()["world_id"] == "world-1"
        assert list_response.json()["active_session_id"] == "session-123"
        assert list_response.json()["sessions"][0]["session_id"] == "session-123"

        switch_response = client.post(
            "/v1/session/switch",
            json={"client_id": "chat-client", "world_id": "world-1", "session_id": "session-xyz"},
        )
        switch_response.raise_for_status()
        assert switch_response.json() == {
            "world_id": "world-1",
            "session_id": "session-xyz",
            "status": "active",
        }

        missing_switch_response = client.post(
            "/v1/session/switch",
            json={"client_id": "chat-client", "world_id": "world-1", "session_id": "missing"},
        )
        assert missing_switch_response.status_code == 404
        assert missing_switch_response.json()["detail"] == "Session missing not found"

    assert chat.new_session_requests == [("chat-client", "world-1")]
    assert chat.list_session_requests == [("chat-client", "world-1")]
    assert chat.switch_session_requests == [
        ("chat-client", "world-1", "session-xyz"),
        ("chat-client", "world-1", "missing"),
    ]
