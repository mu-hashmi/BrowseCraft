from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types
from lmnr import observe
from pydantic import BaseModel, Field

from .models import BlockPlacement, BuildPlan


logger = logging.getLogger(__name__)

GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
ANTHROPIC_VISION_MODEL = "claude-sonnet-4-20250514"
MAX_IMAGINE_PLACEMENTS = 499
_PLAN_TOOL_NAME = "emit_build_plan"


class ImaginePipeline(Protocol):
    async def build_plan(self, prompt: str) -> BuildPlan:
        ...

    async def build_plan_result(self, prompt: str) -> ImaginePlanResult:
        ...

    async def modify_plan_result(
        self,
        prompt: str,
        reference_image_data: bytes,
        reference_image_mime_type: str,
    ) -> ImaginePlanResult:
        ...


GoogleClientFactory = Callable[[str], genai.Client]
AnthropicClientFactory = Callable[[str], AsyncAnthropic]


@dataclass(slots=True, frozen=True)
class ImaginePlanResult:
    plan: BuildPlan
    image_data: bytes
    image_mime_type: str
    plan_source: str


@dataclass(slots=True)
class _GeneratedImage:
    data: bytes
    mime_type: str


class _ToolPlacement(BaseModel):
    dx: int
    dy: int
    dz: int
    block_id: str
    block_state: dict[str, str] = Field(default_factory=dict)


class _ToolPlanPayload(BaseModel):
    placements: list[_ToolPlacement] = Field(min_length=1, max_length=MAX_IMAGINE_PLACEMENTS)


class ImagineService:
    def __init__(
        self,
        google_api_key: str | None,
        anthropic_api_key: str | None,
        *,
        anthropic_vision_model: str = ANTHROPIC_VISION_MODEL,
        use_gemini_text_plan: bool = True,
        google_client_factory: GoogleClientFactory | None = None,
        anthropic_client_factory: AnthropicClientFactory | None = None,
    ) -> None:
        self._google_api_key = google_api_key
        self._anthropic_api_key = anthropic_api_key
        self._anthropic_vision_model = anthropic_vision_model
        self._use_gemini_text_plan = use_gemini_text_plan
        self._google_client_factory = google_client_factory or (lambda api_key: genai.Client(api_key=api_key))
        self._anthropic_client_factory = anthropic_client_factory or (lambda api_key: AsyncAnthropic(api_key=api_key))

    async def build_plan(self, prompt: str) -> BuildPlan:
        result = await self.build_plan_result(prompt)
        return result.plan

    @observe()
    async def build_plan_result(self, prompt: str) -> ImaginePlanResult:
        self._validate_keys()
        generated_image, maybe_plan_text = await self._generate_image(prompt)
        plan, plan_source = await self._resolve_plan(prompt, generated_image, maybe_plan_text)
        return ImaginePlanResult(
            plan=plan,
            image_data=generated_image.data,
            image_mime_type=generated_image.mime_type,
            plan_source=plan_source,
        )

    @observe()
    async def modify_plan_result(
        self,
        prompt: str,
        reference_image_data: bytes,
        reference_image_mime_type: str,
    ) -> ImaginePlanResult:
        self._validate_keys()
        generated_image, maybe_plan_text = await self._edit_image(
            prompt=prompt,
            reference_image_data=reference_image_data,
            reference_image_mime_type=reference_image_mime_type,
        )
        plan, plan_source = await self._resolve_plan(prompt, generated_image, maybe_plan_text)
        return ImaginePlanResult(
            plan=plan,
            image_data=generated_image.data,
            image_mime_type=generated_image.mime_type,
            plan_source=plan_source,
        )

    def _validate_keys(self) -> None:
        if not self._google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for imagine pipeline")
        if not self._anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for imagine pipeline")

    @observe()
    async def _generate_image(self, prompt: str) -> tuple[_GeneratedImage, str | None]:
        client = self._google_client_factory(self._google_api_key)
        try:
            logger.info(
                "Imagine pipeline: generating reference image",
                extra={
                    "provider": "google",
                    "model": GEMINI_IMAGE_MODEL,
                    "prompt_length": len(prompt),
                    "text_plan": self._use_gemini_text_plan,
                },
            )
            response = await client.aio.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=[prompt, _gemini_plan_instruction(MAX_IMAGINE_PLACEMENTS)],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"] if self._use_gemini_text_plan else ["IMAGE"],
                    image_config=types.ImageConfig(output_mime_type="image/png"),
                ),
            )
        finally:
            await client.aio.aclose()
        return _extract_image_and_text(response)

    @observe()
    async def _edit_image(
        self,
        *,
        prompt: str,
        reference_image_data: bytes,
        reference_image_mime_type: str,
    ) -> tuple[_GeneratedImage, str | None]:
        client = self._google_client_factory(self._google_api_key)
        try:
            logger.info(
                "Imagine pipeline: editing reference image",
                extra={
                    "provider": "google",
                    "model": GEMINI_IMAGE_MODEL,
                    "prompt_length": len(prompt),
                    "reference_mime_type": reference_image_mime_type,
                    "reference_bytes": len(reference_image_data),
                },
            )
            response = await client.aio.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=[
                    types.Part.from_bytes(data=reference_image_data, mime_type=reference_image_mime_type),
                    f"Apply this modification to the existing structure: {prompt}",
                    _gemini_plan_instruction(MAX_IMAGINE_PLACEMENTS),
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"] if self._use_gemini_text_plan else ["IMAGE"],
                    image_config=types.ImageConfig(output_mime_type="image/png"),
                ),
            )
        finally:
            await client.aio.aclose()
        return _extract_image_and_text(response)

    @observe()
    async def _resolve_plan(
        self,
        prompt: str,
        generated_image: _GeneratedImage,
        maybe_plan_text: str | None,
    ) -> tuple[BuildPlan, str]:
        if self._use_gemini_text_plan and maybe_plan_text:
            try:
                plan = _parse_plan_from_text(maybe_plan_text)
                return plan, "gemini_text_plan"
            except (ValueError, json.JSONDecodeError):
                logger.info("Gemini text plan parse failed; using Anthropic vision conversion")

        plan = await self._convert_image_to_plan_with_anthropic(prompt, generated_image)
        return plan, "anthropic_vision"

    @observe()
    async def _convert_image_to_plan_with_anthropic(self, prompt: str, generated_image: _GeneratedImage) -> BuildPlan:
        client = self._anthropic_client_factory(self._anthropic_api_key)
        image_b64 = base64.b64encode(generated_image.data).decode("utf-8")
        try:
            logger.info(
                "Imagine pipeline: converting image to block placements",
                extra={
                    "provider": "anthropic",
                    "model": self._anthropic_vision_model,
                    "mime_type": generated_image.mime_type,
                    "image_bytes": len(generated_image.data),
                },
            )
            message = await client.messages.create(
                model=self._anthropic_vision_model,
                max_tokens=4096,
                temperature=0,
                tools=[
                    {
                        "name": _PLAN_TOOL_NAME,
                        "description": "Return a Minecraft block placement plan as JSON.",
                        "input_schema": _ToolPlanPayload.model_json_schema(),
                    }
                ],
                tool_choice={
                    "type": "tool",
                    "name": _PLAN_TOOL_NAME,
                    "disable_parallel_tool_use": True,
                },
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _build_vision_prompt(prompt),
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": generated_image.mime_type,
                                    "data": image_b64,
                                },
                            },
                        ],
                    }
                ],
            )
        finally:
            await client.close()

        tool_input = _extract_tool_input(message.content)
        parsed = _ToolPlanPayload.model_validate(tool_input)
        placements = [
            BlockPlacement(
                dx=placement.dx,
                dy=placement.dy,
                dz=placement.dz,
                block_id=placement.block_id,
                block_state=placement.block_state,
            )
            for placement in parsed.placements
        ]
        return BuildPlan(total_blocks=len(placements), placements=placements)


def _extract_image_and_text(response: types.GenerateContentResponse) -> tuple[_GeneratedImage, str | None]:
    image: _GeneratedImage | None = None
    text_chunks: list[str] = []

    for candidate in response.candidates or []:
        if candidate.content is None:
            continue
        for part in candidate.content.parts or []:
            if part.text is not None:
                text_chunks.append(part.text)
                continue

            inline_data = part.inline_data
            if inline_data is None:
                continue
            if inline_data.data is None:
                raise RuntimeError("Gemini response image data was empty")
            if inline_data.mime_type is None:
                raise RuntimeError("Gemini response image mime type was missing")
            image = _GeneratedImage(data=inline_data.data, mime_type=inline_data.mime_type)

    if image is None:
        raise RuntimeError("Gemini response did not include image bytes")
    maybe_text = "\n".join(chunk for chunk in text_chunks if chunk.strip()) or None
    return image, maybe_text


def _parse_plan_from_text(text: str) -> BuildPlan:
    payload = _extract_json_payload(text)
    parsed = _ToolPlanPayload.model_validate(payload)
    placements = [
        BlockPlacement(
            dx=placement.dx,
            dy=placement.dy,
            dz=placement.dz,
            block_id=placement.block_id,
            block_state=placement.block_state,
        )
        for placement in parsed.placements
    ]
    return BuildPlan(total_blocks=len(placements), placements=placements)


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("Gemini text plan must be a JSON object")
        return payload

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced is None:
        raise ValueError("Gemini response did not include JSON plan")
    payload = json.loads(fenced.group(1))
    if not isinstance(payload, dict):
        raise ValueError("Gemini text plan must be a JSON object")
    return payload


def _extract_tool_input(content_blocks: list[Any]) -> dict[str, Any]:
    for block in content_blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _PLAN_TOOL_NAME:
            continue
        block_input = getattr(block, "input", None)
        if not isinstance(block_input, dict):
            raise RuntimeError("Anthropic tool output must be a JSON object")
        return block_input
    raise RuntimeError("Anthropic response missing tool output for imagine plan")


def _build_vision_prompt(prompt: str) -> str:
    return (
        "Convert the image into a Minecraft build plan.\n"
        "Return block placements using relative coordinates around origin.\n"
        f"Use fewer than {MAX_IMAGINE_PLACEMENTS + 1} placements.\n"
        "Use minecraft namespace block IDs.\n"
        "Include block_state only when needed for orientation or redstone behavior.\n"
        f"User request: {prompt}"
    )


def _gemini_plan_instruction(max_placements: int) -> str:
    return (
        "After generating the image, output JSON with this exact shape:\n"
        "{\n"
        '  "placements": [\n'
        '    {"dx":0,"dy":0,"dz":0,"block_id":"minecraft:stone","block_state":{}}\n'
        "  ]\n"
        "}\n"
        f"Keep placements between 1 and {max_placements}."
    )
