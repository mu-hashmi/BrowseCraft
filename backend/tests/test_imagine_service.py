from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types
from pydantic import ValidationError

from browsecraft_backend.imagine_service import ANTHROPIC_VISION_MODEL, ImagineService


class FakeGoogleModels:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        return self._response


class FakeGoogleAio:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.models = FakeGoogleModels(response)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeGoogleClient:
    def __init__(self, response: types.GenerateContentResponse) -> None:
        self.aio = FakeGoogleAio(response)


class FakeAnthropicMessages:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class FakeAnthropicClient:
    def __init__(self, response: Any) -> None:
        self.messages = FakeAnthropicMessages(response)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _make_google_image_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=b"fake-image-bytes",
                                mime_type="image/png",
                            )
                        )
                    ],
                )
            )
        ]
    )


def _make_anthropic_response(placements: list[dict[str, Any]]) -> Any:
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="emit_build_plan",
                input={"placements": placements},
            )
        ]
    )


@pytest.mark.asyncio
async def test_build_plan_happy_path_uses_expected_models() -> None:
    google_client = FakeGoogleClient(response=_make_google_image_response())
    anthropic_client = FakeAnthropicClient(
        response=_make_anthropic_response(
            placements=[
                {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:stone", "block_state": {}},
                {"dx": 1, "dy": 0, "dz": 0, "block_id": "minecraft:oak_planks", "block_state": {}},
            ]
        )
    )
    service = ImagineService(
        google_api_key="google-key",
        anthropic_api_key="anthropic-key",
        google_client_factory=lambda api_key: google_client,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    plan = await service.build_plan("small stone house")

    assert plan.total_blocks == 2
    assert plan.placements[0].block_id == "minecraft:stone"
    google_call = google_client.aio.models.calls[0]
    assert google_call["model"] == "gemini-3.1-flash-image-preview"
    anthropic_call = anthropic_client.messages.calls[0]
    assert anthropic_call["model"] == ANTHROPIC_VISION_MODEL
    assert google_client.aio.closed is True
    assert anthropic_client.closed is True


@pytest.mark.asyncio
async def test_build_plan_rejects_500_placements() -> None:
    google_client = FakeGoogleClient(response=_make_google_image_response())
    placements = [
        {"dx": idx, "dy": 0, "dz": 0, "block_id": "minecraft:stone", "block_state": {}}
        for idx in range(500)
    ]
    anthropic_client = FakeAnthropicClient(response=_make_anthropic_response(placements=placements))
    service = ImagineService(
        google_api_key="google-key",
        anthropic_api_key="anthropic-key",
        google_client_factory=lambda api_key: google_client,
        anthropic_client_factory=lambda api_key: anthropic_client,
    )

    with pytest.raises(ValidationError):
        await service.build_plan("large wall")
