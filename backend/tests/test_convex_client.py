from __future__ import annotations

import json

import httpx
import pytest

from browsecraft_backend.convex_client import ConvexHttpClient


@pytest.mark.asyncio
async def test_convex_query_success_returns_value() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth_header"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "success", "value": {"ok": True}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = ConvexHttpClient("https://convex.example", http_client, access_key="secret-key")
        result = await client.query("jobs:get", {"jobId": "job-123"})

    assert result == {"ok": True}
    assert captured["url"] == "https://convex.example/api/query"
    assert captured["auth_header"] == "Convex secret-key"
    assert captured["payload"] == {
        "path": "jobs:get",
        "args": {"jobId": "job-123"},
        "format": "json",
    }


@pytest.mark.asyncio
async def test_convex_mutation_raises_runtime_error_for_error_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"status": "error", "errorMessage": "mutation failed"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = ConvexHttpClient("https://convex.example", http_client)
        with pytest.raises(RuntimeError, match="mutation failed"):
            await client.mutation("jobs:create", {"query": "castle"})


@pytest.mark.asyncio
async def test_convex_action_raises_for_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, json={"status": "error", "errorMessage": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = ConvexHttpClient("https://convex.example", http_client)
        with pytest.raises(httpx.HTTPStatusError):
            await client.action("jobs:run", {"jobId": "job-123"})
