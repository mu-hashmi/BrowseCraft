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
from browsecraft_backend.demo_pipelines import (
    DemoPipelines,
    _CloudflareBlocked,
    _SearchCandidate,
    _absolute_placements,
    _looks_like_cloudflare_block,
)
from browsecraft_backend.models import ImagineRequest, SearchRequest


class FakeWebSocketManager:
    def __init__(self) -> None:
        self.sent_payloads: list[tuple[str, dict[str, Any]]] = []
        self.tool_requests: list[tuple[str, str, dict[str, Any]]] = []

    async def send_payload(self, client_id: str, payload: dict[str, Any]) -> None:
        self.sent_payloads.append((client_id, payload))

    async def request_tool(
        self,
        client_id: str,
        tool_name: str,
        params: dict[str, Any],
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        assert timeout > 0
        self.tool_requests.append((client_id, tool_name, params))
        if tool_name == "player_position":
            return {
                "block_x": 10,
                "block_y": 64,
                "ground_y": 63,
                "block_z": 20,
                "facing": "north",
            }
        if tool_name == "place_blocks":
            return {"placed_count": len(params["placements"])}
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


def test_cloudflare_detector_matches_block_page() -> None:
    blocked_html = "<html><title>Attention Required! | Cloudflare</title><div id='cf-error-details'></div></html>"
    assert _looks_like_cloudflare_block(blocked_html) is True
    assert _looks_like_cloudflare_block("<html><body>ok</body></html>") is False


def test_absolute_placements_anchor_ten_blocks_ahead_and_center_footprint() -> None:
    placements = [
        {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:stone"},
        {"dx": 1, "dy": 0, "dz": 0, "block_id": "minecraft:stone"},
        {"dx": 0, "dy": 1, "dz": 1, "block_id": "minecraft:dirt"},
        {"dx": 1, "dy": 1, "dz": 1, "block_id": "minecraft:dirt"},
    ]
    absolute = _absolute_placements(
        relative_placements=placements,
        player_position={
            "block_x": 100,
            "block_y": 64,
            "ground_y": 63,
            "block_z": 200,
            "facing": "north",
        },
    )

    assert {item["y"] for item in absolute} == {63, 64}
    xs = {item["x"] for item in absolute}
    zs = {item["z"] for item in absolute}
    assert xs == {100, 101}
    assert zs == {190, 191}


@pytest.mark.asyncio
async def test_imagine_fallback_routes_to_chat_submitter_in_fast_plan_mode() -> None:
    ws = FakeWebSocketManager()
    captured_requests = []

    async def fake_chat_submitter(request: Any) -> Any:
        captured_requests.append(request)
        return SimpleNamespace(chat_id="chat-1", status="accepted")

    pipelines = DemoPipelines(
        websocket_manager=ws,
        chat_submitter=fake_chat_submitter,
    )

    accepted = await pipelines.submit_imagine(ImagineRequest(client_id="client-1", prompt="dragon statue"))
    assert accepted.status == "accepted"
    await _drain_tasks(pipelines)

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.client_id == "client-1"
    assert request.mode == "plan_fast"
    assert "dragon statue" in request.message
    statuses = _statuses(ws)
    assert statuses[0] == "🎨 Designing creative structure..."


@pytest.mark.asyncio
async def test_imagine_fallback_without_chat_submitter_emits_error_response() -> None:
    ws = FakeWebSocketManager()
    pipelines = DemoPipelines(
        websocket_manager=ws,
        chat_submitter=None,
    )

    await pipelines.submit_imagine(ImagineRequest(client_id="client-2", prompt="gothic cathedral"))
    await _drain_tasks(pipelines)

    responses = [payload for _, payload in ws.sent_payloads if payload.get("type") == "chat.response"]
    assert responses
    assert responses[0]["payload"]["message"] == "Imagine is unavailable because chat orchestration is not configured."


@pytest.mark.asyncio
async def test_search_pipeline_emits_expected_status_sequence_http_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schem_bytes = _minimal_schem_bytes()
    ws = FakeWebSocketManager()

    async def fake_http_discovery(query: str) -> list[_SearchCandidate]:
        assert "castle" in query
        return [
            _SearchCandidate(
                canonical_url="https://www.planetminecraft.com/project/castle/",
                title="Medieval Castle",
                score=0.91,
            )
        ]

    async def fake_playwright_discovery(query: str) -> list[_SearchCandidate]:
        raise AssertionError("playwright discovery should not be used when HTTP discovery succeeds")

    async def fake_download(candidate: _SearchCandidate) -> demo_pipelines._DownloadedSchematic:
        assert candidate.title == "Medieval Castle"
        target = Path(tempfile.gettempdir()) / f"browsecraft-test-{uuid4()}.schem"
        target.write_bytes(schem_bytes)
        return demo_pipelines._DownloadedSchematic(
            path=target,
            title=candidate.title,
            source_url=candidate.canonical_url,
        )

    monkeypatch.setattr(demo_pipelines, "_discover_candidates_http", fake_http_discovery)
    monkeypatch.setattr(demo_pipelines, "_discover_candidates_with_playwright", fake_playwright_discovery)
    monkeypatch.setattr(demo_pipelines, "_download_candidate_with_playwright", fake_download)

    pipelines = DemoPipelines(
        websocket_manager=ws,
        chat_submitter=None,
    )

    await pipelines.submit_search(SearchRequest(client_id="client-4", query="medieval castle schematic"))
    await _drain_tasks(pipelines)

    statuses = _statuses(ws)
    assert statuses == [
        "🔎 Searching Planet Minecraft...",
        "✅ Found: Medieval Castle",
        "📥 Downloading schematic...",
        "🧭 Using /search position (10 blocks ahead)...",
        "🧱 Placing blocks 1/1...",
        "✓ Done",
    ]
    assert len(ws.tool_requests) == 2
    assert ws.tool_requests[0][1] == "player_position"
    assert ws.tool_requests[1][1] == "place_blocks"


@pytest.mark.asyncio
async def test_search_pipeline_falls_back_to_playwright_discovery_on_cloudflare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schem_bytes = _minimal_schem_bytes()
    ws = FakeWebSocketManager()

    async def blocked_http_discovery(_: str) -> list[_SearchCandidate]:
        raise _CloudflareBlocked("blocked")

    async def fake_playwright_discovery(_: str) -> list[_SearchCandidate]:
        return [
            _SearchCandidate(
                canonical_url="https://www.planetminecraft.com/project/castle-v2/",
                title="Medieval Castle v2",
                score=0.8,
            )
        ]

    async def fake_download(candidate: _SearchCandidate) -> demo_pipelines._DownloadedSchematic:
        target = Path(tempfile.gettempdir()) / f"browsecraft-test-{uuid4()}.schem"
        target.write_bytes(schem_bytes)
        return demo_pipelines._DownloadedSchematic(
            path=target,
            title=candidate.title,
            source_url=candidate.canonical_url,
        )

    monkeypatch.setattr(demo_pipelines, "_discover_candidates_http", blocked_http_discovery)
    monkeypatch.setattr(demo_pipelines, "_discover_candidates_with_playwright", fake_playwright_discovery)
    monkeypatch.setattr(demo_pipelines, "_download_candidate_with_playwright", fake_download)

    pipelines = DemoPipelines(
        websocket_manager=ws,
        chat_submitter=None,
    )

    await pipelines.submit_search(SearchRequest(client_id="client-5", query="medieval castle schematic"))
    await _drain_tasks(pipelines)

    statuses = _statuses(ws)
    assert statuses[0] == "🔎 Searching Planet Minecraft..."
    assert statuses[-1] == "✓ Done"


@pytest.mark.asyncio
async def test_search_pipeline_emits_failure_on_download_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = FakeWebSocketManager()

    async def fake_http_discovery(_: str) -> list[_SearchCandidate]:
        return [
            _SearchCandidate(
                canonical_url="https://www.planetminecraft.com/project/requires-login/",
                title="Requires Login",
                score=0.7,
            )
        ]

    async def failing_download(_: _SearchCandidate) -> demo_pipelines._DownloadedSchematic:
        raise RuntimeError("Login required to download schematic from Planet Minecraft")

    monkeypatch.setattr(demo_pipelines, "_discover_candidates_http", fake_http_discovery)
    monkeypatch.setattr(demo_pipelines, "_download_candidate_with_playwright", failing_download)

    pipelines = DemoPipelines(
        websocket_manager=ws,
        chat_submitter=None,
    )

    await pipelines.submit_search(SearchRequest(client_id="client-6", query="castle schematic"))
    await _drain_tasks(pipelines)

    statuses = _statuses(ws)
    assert statuses[-1].startswith("✗ Search failed: ")
    assert "Login required" in statuses[-1]
