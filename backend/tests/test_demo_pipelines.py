from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import nbtlib
import pytest

import browsecraft_backend.demo_pipelines as demo_pipelines
from browsecraft_backend.demo_pipelines import DemoPipelines, _best_candidate
from browsecraft_backend.models import ImagineRequest, SearchRequest


class FakeWebSocketManager:
    def __init__(self) -> None:
        self.sent_payloads: list[tuple[str, dict[str, Any]]] = []
        self.tool_requests: list[tuple[str, str, dict[str, Any]]] = []

    async def send_payload(self, client_id: str, payload: dict[str, Any]) -> None:
        self.sent_payloads.append((client_id, payload))

    async def request_tool(self, client_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        self.tool_requests.append((client_id, tool_name, params))
        return {"ok": True}


def _minimal_schem_bytes() -> bytes:
    root = nbtlib.Compound(
        {
            "Version": nbtlib.Int(3),
            "Width": nbtlib.Short(2),
            "Height": nbtlib.Short(1),
            "Length": nbtlib.Short(2),
            "PaletteMax": nbtlib.Int(2),
            "Palette": nbtlib.Compound(
                {
                    "minecraft:air": nbtlib.Int(0),
                    "minecraft:stone": nbtlib.Int(1),
                }
            ),
            "BlockData": nbtlib.ByteArray([1, 1, 1, 1]),
        }
    )
    with tempfile.TemporaryDirectory(prefix="browsecraft-test-schem-") as temp_dir:
        path = Path(temp_dir) / f"test-{uuid4()}.schem"
        nbtlib.File(root).save(path, gzipped=True)
        return path.read_bytes()


async def _drain_tasks(pipelines: DemoPipelines) -> None:
    while pipelines._tasks:
        await asyncio.gather(*list(pipelines._tasks))


def _statuses(ws: FakeWebSocketManager) -> list[str]:
    return [
        payload["payload"]["status"]
        for _, payload in ws.sent_payloads
        if payload.get("type") == "chat.tool_status"
    ]


def test_best_candidate_prefers_highest_score_valid_extension() -> None:
    output = demo_pipelines._SearchCandidates(
        candidates=[
            demo_pipelines._SearchCandidate(
                canonical_url="https://example.com/a",
                filename="not-a-schematic.zip",
                title="Zip",
                score=0.99,
            ),
            demo_pipelines._SearchCandidate(
                canonical_url="https://example.com/b",
                filename="build.schem",
                title="Schem Low",
                score=0.42,
            ),
            demo_pipelines._SearchCandidate(
                canonical_url="https://example.com/c",
                filename="castle.litematic",
                title="Litematic High",
                score=0.85,
            ),
        ]
    )

    selected = _best_candidate(output)
    assert selected is not None
    assert selected.title == "Litematic High"
    assert selected.filename == "castle.litematic"


def test_best_candidate_returns_none_for_none_or_empty_or_string() -> None:
    assert _best_candidate(None) is None
    assert _best_candidate("raw output") is None
    assert _best_candidate(demo_pipelines._SearchCandidates(candidates=[])) is None


@pytest.mark.asyncio
async def test_imagine_fallback_routes_to_chat_submitter_in_plan_mode() -> None:
    ws = FakeWebSocketManager()
    captured_requests = []

    async def fake_chat_submitter(request: Any) -> Any:
        captured_requests.append(request)
        return SimpleNamespace(chat_id="chat-1", status="accepted")

    pipelines = DemoPipelines(
        websocket_manager=ws,
        browser_use_api_key=None,
        browser_use_llm="browser-use-llm",
        browser_use_skill_id=None,
        tripo_api_key=None,
        chat_submitter=fake_chat_submitter,
    )

    accepted = await pipelines.submit_imagine(ImagineRequest(client_id="client-1", prompt="dragon statue"))
    assert accepted.status == "accepted"
    await _drain_tasks(pipelines)

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.client_id == "client-1"
    assert request.mode == "plan"
    assert "dragon statue" in request.message
    assert "varied materials" in request.message


@pytest.mark.asyncio
async def test_imagine_fallback_without_chat_submitter_emits_error_response() -> None:
    ws = FakeWebSocketManager()
    pipelines = DemoPipelines(
        websocket_manager=ws,
        browser_use_api_key=None,
        browser_use_llm="browser-use-llm",
        browser_use_skill_id=None,
        tripo_api_key=None,
        chat_submitter=None,
    )

    await pipelines.submit_imagine(ImagineRequest(client_id="client-2", prompt="gothic cathedral"))
    await _drain_tasks(pipelines)

    responses = [payload for _, payload in ws.sent_payloads if payload.get("type") == "chat.response"]
    assert responses
    assert responses[0]["payload"]["message"] == "Tripo is unavailable and chat fallback is not configured."


@pytest.mark.asyncio
async def test_search_pipeline_without_api_key_emits_failure_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    ws = FakeWebSocketManager()
    pipelines = DemoPipelines(
        websocket_manager=ws,
        browser_use_api_key=None,
        browser_use_llm="browser-use-llm",
        browser_use_skill_id=None,
        tripo_api_key=None,
        chat_submitter=None,
    )

    await pipelines.submit_search(SearchRequest(client_id="client-3", query="medieval castle"))
    await _drain_tasks(pipelines)

    statuses = _statuses(ws)
    assert statuses[0] == "🔎 Searching Planet Minecraft..."
    assert statuses[-1].startswith("✗ Search failed: BROWSER_USE_API_KEY is required for /search")


@pytest.mark.asyncio
async def test_search_pipeline_emits_expected_status_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schem_bytes = _minimal_schem_bytes()
    ws = FakeWebSocketManager()

    class FakeHttpResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class FakeHttpClient:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        async def __aenter__(self) -> FakeHttpClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
            return None

        async def get(self, url: str) -> FakeHttpResponse:
            assert url == "https://downloads.example/castle.schem"
            return FakeHttpResponse(schem_bytes)

    class FakeTaskRun:
        def __init__(self) -> None:
            self.result = SimpleNamespace(
                output=demo_pipelines._SearchCandidates(
                    candidates=[
                        demo_pipelines._SearchCandidate(
                            canonical_url="https://www.planetminecraft.com/project/castle/",
                            filename="castle.schem",
                            title="Medieval Castle",
                            score=0.88,
                            download_url="https://downloads.example/castle.schem",
                        )
                    ]
                ),
                task=SimpleNamespace(id="task-1", outputFiles=[]),
            )

        async def __aiter__(self):
            yield SimpleNamespace(number=1)

    class FakeBrowserUse:
        def __init__(self, api_key: str, timeout: float) -> None:
            assert api_key == "test-browser-key"
            self.timeout = timeout

        def run(self, **kwargs: Any) -> FakeTaskRun:
            assert kwargs["allowed_domains"] == ["planetminecraft.com", "www.planetminecraft.com"]
            assert kwargs["llm"] == "browser-use-llm"
            return FakeTaskRun()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(demo_pipelines, "AsyncBrowserUse", FakeBrowserUse)
    monkeypatch.setattr(demo_pipelines.httpx, "AsyncClient", FakeHttpClient)

    pipelines = DemoPipelines(
        websocket_manager=ws,
        browser_use_api_key="test-browser-key",
        browser_use_llm="browser-use-llm",
        browser_use_skill_id=None,
        tripo_api_key=None,
        chat_submitter=None,
    )

    await pipelines.submit_search(SearchRequest(client_id="client-4", query="medieval castle schematic"))
    await _drain_tasks(pipelines)

    statuses = _statuses(ws)
    assert statuses == [
        "🔎 Searching Planet Minecraft...",
        "✅ Found: Medieval Castle",
        "📥 Downloading schematic...",
        "📐 Loaded 4 blocks into preview",
        "✓ Done",
    ]
    assert ws.tool_requests
    tool_name = ws.tool_requests[0][1]
    params = ws.tool_requests[0][2]
    assert tool_name == "set_plan"
    assert len(params["placements"]) == 4
