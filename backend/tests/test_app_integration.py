from __future__ import annotations

import io
from datetime import UTC, datetime

import httpx
import nbtlib
from fastapi.testclient import TestClient

from browsecraft_backend.app import create_app
from browsecraft_backend.models import (
    BlockPlacement,
    BuildPlan,
    ChatAcceptedResponse,
    ChatRequest,
    SessionCreatedResponse,
    SessionListResponse,
    SessionSummary,
    SessionSwitchedResponse,
)
from browsecraft_backend.sources import CandidateFile


def _encode_varints(values: list[int]) -> list[int]:
    output: list[int] = []
    for value in values:
        part = value
        while True:
            temp = part & 0x7F
            part >>= 7
            if part:
                output.append(temp | 0x80)
            else:
                output.append(temp)
                break
    return output


def _build_test_schem_bytes() -> bytes:
    root = nbtlib.Compound(
        {
            "Width": nbtlib.Short(2),
            "Height": nbtlib.Short(1),
            "Length": nbtlib.Short(2),
            "Palette": nbtlib.Compound(
                {
                    "minecraft:air": nbtlib.Int(0),
                    "minecraft:stone": nbtlib.Int(1),
                }
            ),
            "BlockData": nbtlib.ByteArray(_encode_varints([1, 0, 1, 1])),
        }
    )
    buf = io.BytesIO()
    nbtlib.File(root, root_name="Schematic").write(buf)
    return buf.getvalue()


class DummyBrowserUse:
    def __init__(self, candidates: list[CandidateFile] | None = None) -> None:
        self._candidates = candidates or []

    async def discover_via_browsing(
        self,
        query: str,
        mc_version: str,
        allowed_exts: tuple[str, ...],
        on_progress=None,
    ) -> list[CandidateFile]:
        del query, mc_version, allowed_exts, on_progress
        return list(self._candidates)


class DummyImagineService:
    def __init__(self, plan: BuildPlan) -> None:
        self._plan = plan
        self.prompts: list[str] = []
        self.modify_prompts: list[str] = []

    async def build_plan(self, prompt: str) -> BuildPlan:
        self.prompts.append(prompt)
        return self._plan

    async def build_plan_result(self, prompt: str):
        self.prompts.append(prompt)
        return _dummy_imagine_result(self._plan)

    async def modify_plan_result(
        self,
        prompt: str,
        reference_image_data: bytes,
        reference_image_mime_type: str,
    ):
        del reference_image_data, reference_image_mime_type
        self.modify_prompts.append(prompt)
        return _dummy_imagine_result(self._plan)


def _dummy_imagine_result(plan: BuildPlan):
    class _Result:
        def __init__(self, plan: BuildPlan) -> None:
            self.plan = plan
            self.image_data = b"image-bytes"
            self.image_mime_type = "image/png"
            self.plan_source = "anthropic_vision"

    return _Result(plan)


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


def test_ready_path_emits_expected_event_sequence() -> None:
    schem_bytes = _build_test_schem_bytes()
    download_url = "https://example.test/test.schem"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == download_url:
            return httpx.Response(200, content=schem_bytes)
        return httpx.Response(404)

    candidate = CandidateFile(
        source="browser_use",
        canonical_url="https://www.planetminecraft.com/project/example/",
        download_url=download_url,
        filename="test.schem",
        title="Test schematic",
        score=0.9,
    )

    app = create_app(
        http_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        browser_use=lambda settings: DummyBrowserUse([candidate]),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/test-client") as websocket:
            response = client.post(
                "/v1/jobs",
                json={
                    "query": "test house",
                    "mc_version": "1.21.11",
                    "client_id": "test-client",
                },
            )
            response.raise_for_status()
            job_id = response.json()["job_id"]

            events = [websocket.receive_json() for _ in range(4)]
            assert [event["type"] for event in events] == [
                "job.status",
                "job.status",
                "job.status",
                "job.ready",
            ]
            assert [event["payload"]["stage"] for event in events[:3]] == [
                "queued",
                "searching",
                "normalizing",
            ]

            status_response = client.get(f"/v1/jobs/{job_id}")
            status_response.raise_for_status()
            body = status_response.json()
            assert body["stage"] == "ready"
            assert body["plan"]["total_blocks"] == 3


def test_no_candidates_emits_error_and_failed_status() -> None:
    def blocked_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call: {request.url}")

    app = create_app(
        http_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(blocked_handler)),
        browser_use=lambda settings: DummyBrowserUse(),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/empty-client") as websocket:
            response = client.post(
                "/v1/jobs",
                json={
                    "query": "missing schematic",
                    "mc_version": "1.21.11",
                    "client_id": "empty-client",
                },
            )
            response.raise_for_status()
            job_id = response.json()["job_id"]

            events: list[dict] = []
            for _ in range(4):
                event = websocket.receive_json()
                events.append(event)
                if event["type"] == "job.error":
                    break

            assert [event["type"] for event in events] == [
                "job.status",
                "job.status",
                "job.error",
            ]
            assert events[-1]["payload"]["code"] == "NO_SCHEMATIC_FOUND"

            status_response = client.get(f"/v1/jobs/{job_id}")
            status_response.raise_for_status()
            body = status_response.json()
            assert body["stage"] == "failed"
            assert body["error_code"] == "NO_SCHEMATIC_FOUND"


def test_imagine_happy_path_emits_expected_event_sequence() -> None:
    def blocked_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call: {request.url}")

    imagine_plan = BuildPlan(
        total_blocks=2,
        placements=[
            BlockPlacement(dx=0, dy=0, dz=0, block_id="minecraft:stone"),
            BlockPlacement(dx=1, dy=0, dz=0, block_id="minecraft:oak_planks"),
        ],
    )
    imagine_service = DummyImagineService(plan=imagine_plan)
    app = create_app(
        http_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(blocked_handler)),
        browser_use=lambda settings: DummyBrowserUse(),
        imagine_service=lambda settings: imagine_service,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/imagine-client") as websocket:
            response = client.post(
                "/v1/imagine",
                json={
                    "prompt": "tiny stone hut",
                    "client_id": "imagine-client",
                },
            )
            response.raise_for_status()
            job_id = response.json()["job_id"]

            events = [websocket.receive_json() for _ in range(4)]
            assert [event["type"] for event in events] == [
                "job.status",
                "job.status",
                "job.status",
                "job.ready",
            ]
            assert [event["payload"]["message"] for event in events[:3]] == [
                "generating image",
                "converting to blocks",
                "ready",
            ]

            status_response = client.get(f"/v1/jobs/{job_id}")
            status_response.raise_for_status()
            body = status_response.json()
            assert body["stage"] == "ready"
            assert body["message"] == "ready"
            assert body["source"]["type"] == "imagine"
            assert body["plan"]["total_blocks"] == 2
            assert imagine_service.prompts == ["tiny stone hut"]


def test_imagine_modify_happy_path_emits_expected_event_sequence() -> None:
    def blocked_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call: {request.url}")

    imagine_plan = BuildPlan(
        total_blocks=2,
        placements=[
            BlockPlacement(dx=0, dy=0, dz=0, block_id="minecraft:stone"),
            BlockPlacement(dx=1, dy=0, dz=0, block_id="minecraft:oak_planks"),
        ],
    )
    imagine_service = DummyImagineService(plan=imagine_plan)
    app = create_app(
        http_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(blocked_handler)),
        browser_use=lambda settings: DummyBrowserUse(),
        imagine_service=lambda settings: imagine_service,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/imagine-client") as websocket:
            first_response = client.post(
                "/v1/imagine",
                json={
                    "prompt": "tiny stone hut",
                    "client_id": "imagine-client",
                },
            )
            first_response.raise_for_status()
            [websocket.receive_json() for _ in range(4)]

            response = client.post(
                "/v1/imagine/modify",
                json={
                    "prompt": "add a tower",
                    "client_id": "imagine-client",
                },
            )
            response.raise_for_status()
            job_id = response.json()["job_id"]

            events = [websocket.receive_json() for _ in range(4)]
            assert [event["type"] for event in events] == [
                "job.status",
                "job.status",
                "job.status",
                "job.ready",
            ]
            assert [event["payload"]["message"] for event in events[:3]] == [
                "editing image",
                "converting to blocks",
                "ready",
            ]

            status_response = client.get(f"/v1/jobs/{job_id}")
            status_response.raise_for_status()
            body = status_response.json()
            assert body["stage"] == "ready"
            assert body["source"]["type"] == "imagine"
            assert body["plan"]["total_blocks"] == 2
            assert imagine_service.prompts == ["tiny stone hut"]
            assert imagine_service.modify_prompts == ["add a tower"]


def test_chat_endpoint_returns_accepted_with_chat_id() -> None:
    def blocked_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call: {request.url}")

    chat = DummyChatOrchestrator()
    app = create_app(
        http_client=lambda: httpx.AsyncClient(transport=httpx.MockTransport(blocked_handler)),
        browser_use=lambda settings: DummyBrowserUse(),
        chat_orchestrator=lambda settings, jobs, ws_manager: chat,
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
        browser_use=lambda settings: DummyBrowserUse(),
        chat_orchestrator=lambda settings, jobs, ws_manager: chat,
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
