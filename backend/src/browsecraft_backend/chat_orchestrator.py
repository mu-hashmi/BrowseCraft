from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Annotated, Any, Callable, Literal, Protocol
from uuid import uuid4

from anthropic import AsyncAnthropic
from lmnr import observe
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from .convex_client import ConvexHttpClient
from .geometry_primitives import bounding_box, build_geometry
from .models import (
    ChatAcceptedResponse,
    ChatRequest,
    SessionCreatedResponse,
    SessionListResponse,
    SessionSummary,
    SessionSwitchedResponse,
)
from .supermemory_client import SupermemoryProfileContext, SupermemorySearchResult
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

CHAT_MODEL = "claude-sonnet-4-6"
PLANNER_MODEL = "claude-sonnet-4-6"
TRIAGE_MODEL = "claude-haiku-4-5"
_DEFAULT_WORLD_ID = "default"
_SYSTEM_PROMPT = (
    "You are BrowseCraft's Minecraft in-game assistant.\n"
    "Keep responses concise and action-oriented.\n"
    "Minecraft coordinates: +x=east, -x=west, +y=up, -y=down, +z=south, -z=north.\n"
    "A locked player_position snapshot captured when the user sent this request is already injected in the system prompt. "
    "Do not call player_position to refresh it during this request.\n"
    "Use the injected build_anchor as the default center for new structures (10 blocks in front of player facing). "
    "For follow-up edits to an existing build (replace/add/extend/modify), inspect and anchor to that structure's "
    "actual coordinates rather than re-centering on the player's locked snapshot.\n"
    "Do not assume default world heights like y=64.\n"
    "Treat block_x/block_y/block_z from player_position as context only when no existing target structure "
    "has been identified and the user did not provide explicit absolute coordinates.\n"
    "Treat relative phrasing like 'in front of me', 'behind me', 'next to me', 'left/right of me' as explicit "
    "location instructions derived from locked player_position; do not reinterpret those as the default 10-block build_anchor.\n"
    "When the user gives no explicit location for a new structure, place it in front of the player using the injected build_anchor. "
    "Do not center new floors, platforms, rooms, or other footprints on the player's block unless the user explicitly asks for that.\n"
    "For floors, platforms, walls, and other footprints placed relative to the player, keep the entire footprint on the requested side of the player. "
    "Do not center the footprint on the player or straddle the player's block unless the user explicitly asks for that.\n"
    "If the user says 'directly in front of me' without a distance, use exactly one block forward from block_x/block_z.\n"
    "For immediate relative placements around the player, default vertical placement to block_y unless the user specifies y.\n"
    "build_geometry floor/platform anchors are centers, so shift the anchor far enough that the nearest edge starts at the requested "
    "offset instead of centering the footprint on the player's block.\n"
    "Example: for a 5x5 floor directly in front of a south-facing player at block_z=20, place the floor over z=21..25 rather than "
    "centering it on z=20 or z=21.\n"
    "ground_y is the terrain block beneath the player. For towers, rooms, walls, pillars, platforms, and similar builds "
    "placed at/around/in front of the player, start the build at block_y so it sits on top of the ground rather than "
    "embedding into the ground at ground_y, unless the user explicitly asks to start from the ground block itself.\n"
    "Do not create floating structures unless explicitly requested. Ensure builds are grounded or attached.\n"
    "Before modifying existing structures, inspect first. Use inspect_area with detailed=true when position-level data is needed.\n"
    "When using detailed inspections, keep filter_terrain=true unless terrain layout is directly relevant.\n"
    "When inspecting, start with small radii (4-6). Expand radius only when strictly necessary.\n"
    "If a request references walls/faces, map orientation using coordinates: south face = max z, north face = min z, "
    "east face = max x, west face = min x.\n"
    "For supported regular shapes (boxes, cylinders, spheres, floors, walls, pillars, stairs, flat/gabled/hipped roofs), "
    "build_geometry is often useful.\n"
    "Keep place_blocks for detail or irregular edits, and keep fill_region for bulk cuboid edits when that is the simplest fit.\n"
    "For iterative edits, preserve existing structure and apply minimal diffs instead of rebuilding unrelated parts.\n"
    "For large custom builds, split work into multiple batches; keep each place_blocks batch reasonably small.\n"
    "Default mode is direct building: use build_geometry, place_blocks, or fill_region to modify the world immediately.\n"
    "Use set_plan only when the user explicitly asks for a preview/blueprint/plan, "
    "or when the build is large enough that positioning first is safer.\n"
    "When the user asks for creative structures, use varied materials, depth/layering, "
    "decorative details, and stairs/slabs where appropriate so results feel hand-built.\n"
    "If the user asks to undo/revert the previous build, call undo_last before applying replacement placements.\n"
    "If the latest placement batch is clearly wrong, call undo_last and retry with corrected coordinates. "
    "Large builds and build_geometry placements may require repeated undo_last calls because execution is batched.\n"
    "Use tools for factual game state instead of guessing."
)
_TRIAGE_SYSTEM_PROMPT = (
    "You are a build request triage classifier for Minecraft construction commands.\n"
    "Return strict JSON only. No prose. No markdown.\n"
    "Schema:\n"
    '{'
    '"is_build_request": boolean,'
    '"complexity": "simple" | "complex",'
    '"spatial_reference": "absolute_coordinates" | "relative_to_player" | "relative_to_existing_structure" | "default_anchor" | "none",'
    '"distance_hint": integer | null,'
    '"should_undo_first": boolean'
    '}\n'
    "Guidelines:\n"
    "- simple: single/small edit, one local structure tweak, short linear action.\n"
    "- complex: multi-part structure, staged layout, or interdependent sub-builds.\n"
    "- spatial_reference based on the user's explicit phrasing.\n"
    "- distance_hint only for relative_to_player requests; otherwise null.\n"
    "- should_undo_first true only if the user asks to undo/revert previous work."
)
_MAX_TOOL_ROUNDS = 20
_CONTEXT_MESSAGE_LIMIT = 12
_MAX_MODEL_OUTPUT_TOKENS = 2048
_TRIAGE_MAX_TOKENS = 120
_MAX_Y_DELTA_FROM_PLAYER = 96
_DEFAULT_FORWARD_BUILD_OFFSET = 10
_DEFAULT_RELATIVE_PLAYER_OFFSET = 1
_PLACE_BLOCK_BATCH_SIZE = 256
_MAX_STEP_RETRIES = 2
_CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}
_TOOL_RESULT_SUMMARY_PREFIX = "[summarized tool_result]"
_MEMORY_OUTCOME_TOOLS = {
    "place_blocks",
    "fill_region",
    "build_geometry",
    "set_plan",
    "undo_last",
    "modify_overlay",
    "save_blueprint",
    "load_blueprint",
}


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NoArgs(_ToolArgs):
    pass


class _PlaceBlockArgs(_ToolArgs):
    x: int
    y: int
    z: int
    block_id: str = Field(min_length=1)


class _BlockPositionArgs(_ToolArgs):
    x: int
    y: int
    z: int


class _InspectAreaArgs(_ToolArgs):
    center: _BlockPositionArgs
    radius: int = Field(ge=0, le=12)
    detailed: bool = False
    filter_terrain: bool = True

    @model_validator(mode="after")
    def validate_detailed_radius(self) -> _InspectAreaArgs:
        if self.detailed and self.radius > 6:
            raise ValueError("inspect_area with detailed=true requires radius <= 6")
        return self


class _PlaceBlocksArgs(_ToolArgs):
    placements: list[_PlaceBlockArgs] = Field(min_length=1)


class _PlanPlacementArgs(_ToolArgs):
    dx: int
    dy: int
    dz: int
    block_id: str = Field(min_length=1)
    block_state: dict[str, str] = Field(default_factory=dict)


class _SetPlanArgs(_ToolArgs):
    placements: list[_PlanPlacementArgs] = Field(min_length=1)


class _FillRegionArgs(_ToolArgs):
    from_corner: _BlockPositionArgs
    to_corner: _BlockPositionArgs
    block_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_volume(self) -> _FillRegionArgs:
        width = abs(self.to_corner.x - self.from_corner.x) + 1
        height = abs(self.to_corner.y - self.from_corner.y) + 1
        depth = abs(self.to_corner.z - self.from_corner.z) + 1
        volume = width * height * depth
        if volume > 4096:
            raise ValueError("fill_region volume must be <= 4096 blocks")
        return self


class _BuildGeometryBaseArgs(_ToolArgs):
    material: str = Field(min_length=1)
    anchor: _BlockPositionArgs
    rotation: Literal["north", "east", "south", "west"] = "north"


class _BuildGeometryBoxArgs(_BuildGeometryBaseArgs):
    shape: Literal["box"]
    width: int = Field(ge=1, le=128)
    height: int = Field(ge=1, le=128)
    depth: int = Field(ge=1, le=128)
    hollow: bool = False


class _BuildGeometryCylinderArgs(_BuildGeometryBaseArgs):
    shape: Literal["cylinder"]
    radius: int = Field(ge=0, le=64)
    height: int = Field(ge=1, le=128)
    hollow: bool = False


class _BuildGeometrySphereArgs(_BuildGeometryBaseArgs):
    shape: Literal["sphere"]
    radius: int = Field(ge=0, le=32)
    hollow: bool = False


class _BuildGeometryFloorArgs(_BuildGeometryBaseArgs):
    shape: Literal["floor"]
    width: int = Field(ge=1, le=128)
    depth: int = Field(ge=1, le=128)
    thickness: int = Field(default=1, ge=1, le=32)


class _BuildGeometryWallArgs(_BuildGeometryBaseArgs):
    shape: Literal["wall"]
    width: int = Field(ge=1, le=128)
    height: int = Field(ge=1, le=128)
    thickness: int = Field(default=1, ge=1, le=32)


class _BuildGeometryPillarArgs(_BuildGeometryBaseArgs):
    shape: Literal["pillar"]
    height: int = Field(ge=1, le=256)
    width: int = Field(default=1, ge=1, le=32)
    depth: int = Field(default=1, ge=1, le=32)


class _BuildGeometryStairsArgs(_BuildGeometryBaseArgs):
    shape: Literal["stairs"]
    width: int = Field(ge=1, le=128)
    depth: int = Field(ge=1, le=128)
    height: int = Field(ge=1, le=128)


class _BuildGeometryRoofFlatArgs(_BuildGeometryBaseArgs):
    shape: Literal["roof_flat"]
    width: int = Field(ge=1, le=128)
    depth: int = Field(ge=1, le=128)
    thickness: int = Field(default=1, ge=1, le=32)


class _BuildGeometryRoofGabledArgs(_BuildGeometryBaseArgs):
    shape: Literal["roof_gabled"]
    width: int = Field(ge=1, le=128)
    depth: int = Field(ge=1, le=128)
    height: int | None = Field(default=None, ge=1, le=64)


class _BuildGeometryRoofHippedArgs(_BuildGeometryBaseArgs):
    shape: Literal["roof_hipped"]
    width: int = Field(ge=1, le=128)
    depth: int = Field(ge=1, le=128)
    height: int | None = Field(default=None, ge=1, le=64)


_BuildGeometryUnion = Annotated[
    _BuildGeometryBoxArgs
    | _BuildGeometryCylinderArgs
    | _BuildGeometrySphereArgs
    | _BuildGeometryFloorArgs
    | _BuildGeometryWallArgs
    | _BuildGeometryPillarArgs
    | _BuildGeometryStairsArgs
    | _BuildGeometryRoofFlatArgs
    | _BuildGeometryRoofGabledArgs
    | _BuildGeometryRoofHippedArgs,
    Field(discriminator="shape"),
]
_BUILD_GEOMETRY_ADAPTER = TypeAdapter(_BuildGeometryUnion)
_BUILD_GEOMETRY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shape": {
            "type": "string",
            "enum": [
                "box",
                "cylinder",
                "sphere",
                "floor",
                "wall",
                "pillar",
                "stairs",
                "roof_flat",
                "roof_gabled",
                "roof_hipped",
            ],
        },
        "material": {"type": "string", "minLength": 1},
        "anchor": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "z": {"type": "integer"},
            },
            "required": ["x", "y", "z"],
            "additionalProperties": False,
        },
        "rotation": {"type": "string", "enum": ["north", "east", "south", "west"]},
        "width": {"type": "integer", "minimum": 1, "maximum": 128},
        "height": {"type": "integer", "minimum": 1, "maximum": 256},
        "depth": {"type": "integer", "minimum": 1, "maximum": 128},
        "radius": {"type": "integer", "minimum": 0, "maximum": 64},
        "thickness": {"type": "integer", "minimum": 1, "maximum": 32},
        "hollow": {"type": "boolean"},
    },
    "required": ["shape", "material", "anchor"],
    "additionalProperties": False,
}


class _ModifyOverlayRotateArgs(_ToolArgs):
    op: Literal["rotate"]
    quarters: int = 1


class _ModifyOverlayShiftArgs(_ToolArgs):
    op: Literal["shift"]
    dy: int


class _ModifyOverlaySetAnchorArgs(_ToolArgs):
    op: Literal["set_anchor"]
    x: int
    y: int
    z: int


class _ModifyOverlayReplaceBlockArgs(_ToolArgs):
    op: Literal["replace_block"]
    from_block: str = Field(alias="from", min_length=1)
    to_block: str = Field(alias="to", min_length=1)


_ModifyOverlayUnion = Annotated[
    _ModifyOverlayRotateArgs
    | _ModifyOverlayShiftArgs
    | _ModifyOverlaySetAnchorArgs
    | _ModifyOverlayReplaceBlockArgs,
    Field(discriminator="op"),
]
_MODIFY_OVERLAY_ADAPTER = TypeAdapter(_ModifyOverlayUnion)
_MODIFY_OVERLAY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": ["rotate", "shift", "set_anchor", "replace_block"],
        },
        "quarters": {"type": "integer"},
        "dy": {"type": "integer"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "z": {"type": "integer"},
        "from": {"type": "string", "minLength": 1},
        "to": {"type": "string", "minLength": 1},
    },
    "required": ["op"],
    "additionalProperties": False,
}


class _BlueprintNameArgs(_ToolArgs):
    name: str = Field(min_length=1)


class _PlayerPositionResult(BaseModel):
    x: float
    y: float
    z: float
    block_x: int
    block_y: int
    block_z: int
    ground_y: int | None = None
    facing: str = Field(min_length=1)
    dimension: str = Field(min_length=1)


class _BuildStepModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    relative_location_hint: str = Field(min_length=1)
    success_check: str = Field(min_length=1)


class _BuildStepPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[_BuildStepModel] = Field(min_length=1, max_length=8)


class _BuildRequestTriageModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_build_request: bool
    complexity: Literal["simple", "complex"]
    spatial_reference: Literal[
        "absolute_coordinates",
        "relative_to_player",
        "relative_to_existing_structure",
        "default_anchor",
        "none",
    ]
    distance_hint: int | None = Field(default=None, ge=0, le=64)
    should_undo_first: bool = False


@dataclass(slots=True, frozen=True)
class _BuildAnchor:
    x: int
    y: int
    z: int


@dataclass(slots=True, frozen=True)
class _RelativePlacementGuard:
    label: str
    dx: int
    dz: int
    min_projection: int = 1


@dataclass(slots=True, frozen=True)
class _SessionKey:
    client_id: str
    world_id: str


@dataclass(slots=True)
class _SessionState:
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[dict[str, str]]


@dataclass(slots=True)
class _ToolExecutionResult:
    content: str
    is_error: bool


@dataclass(slots=True)
class _StreamContentBlock:
    block_type: Literal["text", "tool_use"]
    text: str = ""
    tool_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_input_chunks: list[str] = field(default_factory=list)


class _SessionMessageModel(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class _ConvexSessionDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    world_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    messages: list[_SessionMessageModel]
    created_at: int = Field(ge=0)
    updated_at: int = Field(ge=0)


AnthropicClientFactory = Callable[[str], AsyncAnthropic]


class SupermemoryClientProtocol(Protocol):
    async def search_memories(
        self,
        query: str,
        *,
        container_tag: str,
        limit: int = 5,
    ) -> list[SupermemorySearchResult]:
        ...

    async def store_memory(
        self,
        content: str,
        *,
        container_tag: str,
        metadata: dict[str, Any],
    ) -> None:
        ...

    async def profile_context(self, container_tag: str) -> SupermemoryProfileContext:
        ...


_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "player_position",
        "description": "Read current player position and facing.",
        "input_schema": _NoArgs.model_json_schema(),
    },
    {
        "name": "player_inventory",
        "description": "Read current player inventory summary.",
        "input_schema": _NoArgs.model_json_schema(),
    },
    {
        "name": "inspect_area",
        "description": (
            "Inspect blocks around a center position with a radius. "
            "Set detailed=true to include non-air block coordinates. "
            "Set filter_terrain=true to suppress common terrain blocks."
        ),
        "input_schema": _InspectAreaArgs.model_json_schema(),
    },
    {
        "name": "place_blocks",
        "description": "Place blocks at absolute world coordinates for irregular or detail edits. Do not use this for supported regular shapes when build_geometry fits.",
        "input_schema": _PlaceBlocksArgs.model_json_schema(),
    },
    {
        "name": "fill_region",
        "description": "Fill an axis-aligned cuboid region with one block type for bulk cuboid edits. Do not use this for supported regular shapes when build_geometry fits.",
        "input_schema": _FillRegionArgs.model_json_schema(),
    },
    {
        "name": "build_geometry",
        "description": (
            "Build deterministic regular shapes around an anchor (box, cylinder, sphere, floor, wall, pillar, stairs, flat/gabled/hipped roofs). "
            "Anchor is the center with negative-bias centering for even sizes. "
            "For default in-front builds and relative floor/platform requests, shift the anchor so the entire footprint stays on the intended side of the player instead of overlapping the player's block. "
            "Use rotation for cardinal orientation. Use this tool for supported regular geometry even when dimensions must be derived from corners or existing bounds."
        ),
        "input_schema": _BUILD_GEOMETRY_TOOL_SCHEMA,
    },
    {
        "name": "set_plan",
        "description": "Load a relative-coordinate plan into preview mode as a ghost overlay.",
        "input_schema": _SetPlanArgs.model_json_schema(),
    },
    {
        "name": "undo_last",
        "description": "Undo the most recent placement batch. Large builds and multi-batch geometry may require repeated undo_last calls.",
        "input_schema": _NoArgs.model_json_schema(),
    },
    {
        "name": "get_active_overlay",
        "description": "Read the active overlay state.",
        "input_schema": _NoArgs.model_json_schema(),
    },
    {
        "name": "modify_overlay",
        "description": "Modify overlay state with rotate, shift, set_anchor, or replace_block operations.",
        "input_schema": _MODIFY_OVERLAY_TOOL_SCHEMA,
    },
    {
        "name": "get_blueprints",
        "description": "List available local blueprints.",
        "input_schema": _NoArgs.model_json_schema(),
    },
    {
        "name": "save_blueprint",
        "description": "Save the active overlay as a named blueprint.",
        "input_schema": _BlueprintNameArgs.model_json_schema(),
    },
    {
        "name": "load_blueprint",
        "description": "Load a named blueprint into the active overlay.",
        "input_schema": _BlueprintNameArgs.model_json_schema(),
    },
]
_CACHEABLE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    *[dict(schema) for schema in _TOOL_SCHEMAS[:-1]],
    {**_TOOL_SCHEMAS[-1], "cache_control": _CACHE_CONTROL_EPHEMERAL},
]
_PLAN_DISALLOWED_TOOL_NAMES = {
    "place_blocks",
    "fill_region",
    "build_geometry",
    "undo_last",
}
_PLAN_FAST_ALLOWED_TOOL_NAMES = {"set_plan"}


class ChatOrchestrator:
    def __init__(
        self,
        anthropic_api_key: str | None,
        websocket_manager: WebSocketManager,
        chat_model: str = CHAT_MODEL,
        planner_model: str = PLANNER_MODEL,
        triage_model: str = TRIAGE_MODEL,
        enable_build_planner: bool = False,
        anthropic_client_factory: AnthropicClientFactory | None = None,
        convex_client: ConvexHttpClient | None = None,
        supermemory_client: SupermemoryClientProtocol | None = None,
    ) -> None:
        self._anthropic_api_key = anthropic_api_key
        self._chat_model = chat_model
        self._planner_model = planner_model
        self._triage_model = triage_model
        self._enable_build_planner = enable_build_planner
        self._websocket_manager = websocket_manager
        self._anthropic_client_factory = anthropic_client_factory or (lambda api_key: AsyncAnthropic(api_key=api_key))
        self._convex_client = convex_client
        self._supermemory_client = supermemory_client

        self._sessions: dict[str, dict[str, _SessionState]] = {}
        self._active_sessions: dict[_SessionKey, str] = {}
        self._session_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._warmup_lock = asyncio.Lock()
        self._warmup_done = False

    async def submit_chat(self, request: ChatRequest) -> ChatAcceptedResponse:
        world_id = request.world_id or _DEFAULT_WORLD_ID
        session_id = await self._resolve_session_for_chat(
            client_id=request.client_id,
            world_id=world_id,
            requested_session_id=request.session_id,
        )
        locked_player_position: _PlayerPositionResult | None = None
        try:
            locked_player_position = await self._require_player_position(client_id=request.client_id)
        except Exception:
            logger.warning(
                "Unable to lock player position at chat submit time for client=%s",
                request.client_id,
                exc_info=True,
            )

        chat_id = str(uuid4())
        task = asyncio.create_task(
            self._run_chat(
                chat_id=chat_id,
                client_id=request.client_id,
                user_message=request.message,
                request_mode=request.mode,
                world_id=world_id,
                session_id=session_id,
                locked_player_position=locked_player_position,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return ChatAcceptedResponse(chat_id=chat_id, status="accepted")

    def configure_integrations(
        self,
        *,
        convex_client: ConvexHttpClient | None,
        supermemory_client: SupermemoryClientProtocol | None,
    ) -> None:
        self._convex_client = convex_client
        self._supermemory_client = supermemory_client

    async def warmup_prompt_cache(self) -> None:
        if self._warmup_done or not self._anthropic_api_key:
            return
        async with self._warmup_lock:
            if self._warmup_done:
                return
            client = self._anthropic_client_factory(self._anthropic_api_key)
            try:
                await client.messages.create(
                    model=self._chat_model,
                    max_tokens=1,
                    temperature=0,
                    system=[
                        {
                            "type": "text",
                            "text": (
                                f"{_SYSTEM_PROMPT}\n"
                                "Request mode for this turn: BUILD.\n"
                                "Locked player position for this request (captured at submit time, authoritative):\n"
                                "- x=0.000, y=64.000, z=0.000\n"
                                "- block_x=0, block_y=64, block_z=0\n"
                                "- facing=south, dimension=minecraft:overworld\n"
                                "Default build anchor for NEW structures in this request (10 blocks ahead of player facing):\n"
                                "- build_anchor_x=0, build_anchor_y=64, build_anchor_z=10\n"
                                "For new structures, center around this build anchor. "
                                "Only ignore this when the user gives explicit coordinates or requests edits to an existing structure."
                            ),
                            "cache_control": _CACHE_CONTROL_EPHEMERAL,
                        }
                    ],
                    tools=_CACHEABLE_TOOL_SCHEMAS,
                    messages=[{"role": "user", "content": "warmup"}],
                )
                await client.messages.create(
                    model=self._triage_model,
                    max_tokens=1,
                    temperature=0,
                    system=[
                        {
                            "type": "text",
                            "text": _TRIAGE_SYSTEM_PROMPT,
                            "cache_control": _CACHE_CONTROL_EPHEMERAL,
                        }
                    ],
                    messages=[{"role": "user", "content": "warmup"}],
                )
                self._warmup_done = True
            except Exception:
                logger.warning("Prompt cache warmup failed", exc_info=True)
            finally:
                await client.close()

    async def create_session(self, client_id: str, world_id: str) -> SessionCreatedResponse:
        async with self._session_lock:
            session = _new_session_state()
            await self._persist_session(world_id, session)
            self._active_sessions[_SessionKey(client_id=client_id, world_id=world_id)] = session.session_id

        return SessionCreatedResponse(world_id=world_id, session_id=session.session_id, status="created")

    async def list_sessions(self, client_id: str, world_id: str) -> SessionListResponse:
        async with self._session_lock:
            session_states = await self._list_sessions_for_world(world_id)
            session_ids = {session.session_id for session in session_states}
            active_session_id = self._active_sessions.get(_SessionKey(client_id=client_id, world_id=world_id))
            if active_session_id is not None and active_session_id not in session_ids:
                active_session_id = None

        summaries = [
            SessionSummary(
                session_id=session.session_id,
                message_count=len(session.messages),
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
            for session in session_states
        ]
        return SessionListResponse(
            world_id=world_id,
            active_session_id=active_session_id,
            sessions=summaries,
        )

    async def switch_session(self, client_id: str, world_id: str, session_id: str) -> SessionSwitchedResponse:
        async with self._session_lock:
            session = await self._load_session(world_id, session_id)
            if session is None:
                raise LookupError(f"Session {session_id} not found for world {world_id}")
            self._active_sessions[_SessionKey(client_id=client_id, world_id=world_id)] = session.session_id

        return SessionSwitchedResponse(world_id=world_id, session_id=session.session_id, status="active")

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except Exception:
            logger.exception("Chat task failed")

    @observe()
    async def _run_chat(
        self,
        chat_id: str,
        client_id: str,
        user_message: str,
        request_mode: Literal["build", "plan", "plan_fast"] = "build",
        *,
        world_id: str | None = None,
        session_id: str | None = None,
        locked_player_position: _PlayerPositionResult | None = None,
    ) -> None:
        resolved_world_id = world_id or _DEFAULT_WORLD_ID
        try:
            if locked_player_position is None:
                locked_player_position = await self._require_player_position(client_id=client_id)
            resolved_session_id = session_id or await self._resolve_session_for_chat(
                client_id=client_id,
                world_id=resolved_world_id,
                requested_session_id=None,
            )
            assistant_text = await self._complete_chat(
                chat_id=chat_id,
                client_id=client_id,
                world_id=resolved_world_id,
                session_id=resolved_session_id,
                user_message=user_message,
                request_mode=request_mode,
                player_position=locked_player_position,
            )
            await self._append_history(
                world_id=resolved_world_id,
                session_id=resolved_session_id,
                user_message=user_message,
                assistant_message=assistant_text,
            )
            await self._emit_tool_status(client_id=client_id, status="✓ Done")
        except Exception as exc:
            logger.exception("Chat orchestration failed for client=%s chat_id=%s", client_id, chat_id)
            assistant_text = f"Unable to process chat request: {exc}"
            await self._emit_tool_status(client_id=client_id, status=f"✗ {exc}")

        await self._websocket_manager.send_payload(
            client_id,
            {
                "type": "chat.response",
                "chat_id": chat_id,
                "payload": {"message": assistant_text},
            },
        )

    @observe()
    async def _complete_chat(
        self,
        chat_id: str,
        client_id: str,
        world_id: str,
        session_id: str,
        user_message: str,
        request_mode: Literal["build", "plan", "plan_fast"],
        player_position: _PlayerPositionResult,
    ) -> str:
        if not self._anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for chat orchestrator")

        base_messages = await self._conversation_messages(world_id=world_id, session_id=session_id)
        memory_context = await self._search_memory_context(
            client_id=client_id,
            world_id=world_id,
            user_message=user_message,
        )
        client = self._anthropic_client_factory(self._anthropic_api_key)
        try:
            triage: _BuildRequestTriageModel | None = None
            if request_mode == "build":
                await self._emit_tool_status(client_id=client_id, status="⚡ Classifying request...")
                triage = await self._triage_build_request(
                    client=client,
                    user_message=user_message,
                    conversation_messages=base_messages,
                    player_position=player_position,
                )

            build_anchor = _build_anchor_for_request(
                player_position=player_position,
                triage=triage,
            )
            relative_placement_guard = _relative_placement_guard_for_request(
                user_message=user_message,
                player_position=player_position,
                triage=triage,
            )
            system_prompt = _compose_system_prompt(
                request_mode=request_mode,
                player_position=player_position,
                build_anchor=build_anchor,
                memory_context=memory_context,
                triage=triage,
            )

            if request_mode == "build" and triage is not None and triage.is_build_request and triage.should_undo_first:
                await self._emit_tool_status(client_id=client_id, status="↩ Undoing previous placement...")
                undo_result = await self._execute_tool(
                    client_id=client_id,
                    world_id=world_id,
                    session_id=session_id,
                    tool_name="undo_last",
                    raw_input={},
                    player_position=player_position,
                    cached_player_position=player_position,
                    request_mode="build",
                    relative_placement_guard=relative_placement_guard,
                )
                if undo_result.is_error:
                    raise RuntimeError(f"undo_last failed: {undo_result.content}")

            if (
                request_mode == "build"
                and self._enable_build_planner
                and triage is not None
                and triage.is_build_request
                and triage.complexity == "complex"
            ):
                await self._emit_tool_status(client_id=client_id, status="🧠 Drafting step plan...")
                planned_steps = await self._plan_build_steps(
                    client=client,
                    client_id=client_id,
                    chat_id=chat_id,
                    user_message=user_message,
                    conversation_messages=base_messages,
                    player_position=player_position,
                    build_anchor=build_anchor,
                )
                previous_step_outcomes: list[dict[str, Any]] = []
                step_history_messages = [*base_messages]
                final_response = ""
                for step_index, step in enumerate(planned_steps, start=1):
                    await self._emit_tool_status(
                        client_id=client_id,
                        status=f"🧩 Step {step_index}/{len(planned_steps)}: {step.name}",
                    )
                    step_message = _step_execution_message(
                        original_request=user_message,
                        step=step,
                        step_index=step_index,
                        total_steps=len(planned_steps),
                        previous_step_outcomes=previous_step_outcomes,
                    )
                    step_request_messages = [*step_history_messages, {"role": "user", "content": step_message}]
                    for attempt in range(_MAX_STEP_RETRIES + 1):
                        try:
                            step_attempt_messages = [*step_request_messages]
                            final_response, _, step_tool_outcomes = await self._execute_tool_loop(
                                client=client,
                                model=self._chat_model,
                                client_id=client_id,
                                chat_id=chat_id,
                                world_id=world_id,
                                session_id=session_id,
                                player_position=player_position,
                                request_mode="build",
                                system_prompt=system_prompt,
                                messages=step_attempt_messages,
                                force_tool_use=True,
                                require_build_modification=False,
                                enforce_build_intent=True,
                                relative_placement_guard=relative_placement_guard,
                            )
                            step_history_messages = step_attempt_messages
                            break
                        except Exception as exc:
                            if attempt >= _MAX_STEP_RETRIES:
                                raise RuntimeError(
                                    f"Step '{step.name}' failed after {_MAX_STEP_RETRIES + 1} attempts: {exc}"
                                ) from exc
                            await self._emit_tool_status(
                                client_id=client_id,
                                status=f"↻ Retrying step '{step.name}'...",
                            )
                    previous_step_outcomes.append(
                        {
                            "step_name": step.name,
                            "assistant_response": final_response,
                            "tool_outcomes": step_tool_outcomes,
                        }
                    )
                return final_response

            if request_mode == "plan":
                await self._emit_tool_status(client_id=client_id, status="🧠 Drafting preview plan...")
            elif request_mode == "plan_fast":
                await self._emit_tool_status(client_id=client_id, status="🎨 Designing structure preview...")
            messages = [*base_messages, {"role": "user", "content": user_message}]
            force_tool_use = _is_preview_mode(request_mode) or (
                request_mode == "build" and triage is not None and triage.is_build_request
            )
            enforce_build_intent = request_mode == "build" and triage is not None and triage.is_build_request
            assistant_text, _, _ = await self._execute_tool_loop(
                client=client,
                model=self._chat_model,
                client_id=client_id,
                chat_id=chat_id,
                world_id=world_id,
                session_id=session_id,
                player_position=player_position,
                request_mode=request_mode,
                system_prompt=system_prompt,
                messages=messages,
                force_tool_use=force_tool_use,
                require_build_modification=False,
                enforce_build_intent=enforce_build_intent,
                relative_placement_guard=relative_placement_guard,
            )
            return assistant_text
        finally:
            await client.close()

    async def _triage_build_request(
        self,
        *,
        client: AsyncAnthropic,
        user_message: str,
        conversation_messages: list[dict[str, Any]],
        player_position: _PlayerPositionResult,
    ) -> _BuildRequestTriageModel:
        recent_messages = conversation_messages[-6:]
        history_lines: list[str] = []
        for message in recent_messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"

        triage_user_prompt = (
            f"Conversation context:\n{history_text}\n"
            f"Current user request:\n{user_message}\n"
            "Locked player snapshot:\n"
            f"- block_x={player_position.block_x}, block_y={player_position.block_y}, block_z={player_position.block_z}\n"
            f"- facing={player_position.facing}\n"
            "Return JSON only."
        )

        response = await client.messages.parse(
            model=self._triage_model,
            max_tokens=_TRIAGE_MAX_TOKENS,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": _TRIAGE_SYSTEM_PROMPT,
                    "cache_control": _CACHE_CONTROL_EPHEMERAL,
                }
            ],
            messages=[{"role": "user", "content": triage_user_prompt}],
            output_format=_BuildRequestTriageModel,
        )
        parsed_output = response.parsed_output
        if parsed_output is None:
            raw_text = _extract_text_from_message_content(getattr(response, "content", []))
            raise RuntimeError(f"Anthropic triage response did not match structured output: {raw_text}")
        return parsed_output

    async def _plan_build_steps(
        self,
        *,
        client: AsyncAnthropic,
        client_id: str,
        chat_id: str,
        user_message: str,
        conversation_messages: list[dict[str, Any]],
        player_position: _PlayerPositionResult,
        build_anchor: _BuildAnchor,
    ) -> list[_BuildStepModel]:
        planner_system_prompt = (
            "You are a Minecraft build planner. "
            "Decompose the request into concrete, ordered build steps. "
            "Return strict JSON only with this exact schema and no extra keys: "
            '{"steps":[{"name":"...","goal":"...","relative_location_hint":"...","success_check":"..."}]}. '
            "Each step must represent physical world modifications."
        )
        planner_messages = [
            *conversation_messages,
            {
                "role": "user",
                "content": (
                    f"User request: {user_message}\n"
                    "Locked player position:\n"
                    f"- block_x={player_position.block_x}, block_y={player_position.block_y}, block_z={player_position.block_z}\n"
                    f"- facing={player_position.facing}\n"
                    "Default build anchor:\n"
                    f"- x={build_anchor.x}, y={build_anchor.y}, z={build_anchor.z}\n"
                    "Output between 1 and 8 ordered steps."
                ),
            }
        ]
        response = await self._run_model_round(
            client=client,
            model=self._planner_model,
            client_id=client_id,
            chat_id=chat_id,
            system_prompt=planner_system_prompt,
            messages=planner_messages,
            tool_schemas=[],
            tool_choice=None,
            emit_deltas=False,
        )
        planner_blocks = _normalize_assistant_blocks(response.content)
        planner_text = _extract_text_response(planner_blocks)
        parsed_plan = _parse_build_step_plan(planner_text)
        return parsed_plan.steps

    async def _execute_tool_loop(
        self,
        *,
        client: AsyncAnthropic,
        model: str,
        client_id: str,
        chat_id: str,
        world_id: str,
        session_id: str,
        player_position: _PlayerPositionResult,
        request_mode: Literal["build", "plan", "plan_fast"],
        system_prompt: str,
        messages: list[dict[str, Any]],
        force_tool_use: bool,
        require_build_modification: bool,
        enforce_build_intent: bool,
        relative_placement_guard: _RelativePlacementGuard | None,
    ) -> tuple[str, bool, list[dict[str, Any]]]:
        mode_tool_schemas = _tool_schemas_for_mode(request_mode)
        tool_rounds = 0
        no_tool_retries = 0
        applied_build_modification = False
        recorded_tool_outcomes: list[dict[str, Any]] = []
        while True:
            _summarize_historical_tool_results(messages)
            tool_choice = _tool_choice_for_round(
                request_mode=request_mode,
                force_tool_use=force_tool_use,
            )
            response = await self._run_model_round(
                client=client,
                model=model,
                client_id=client_id,
                chat_id=chat_id,
                system_prompt=system_prompt,
                messages=messages,
                tool_schemas=mode_tool_schemas,
                tool_choice=tool_choice,
            )
            assistant_blocks = _normalize_assistant_blocks(response.content)
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [block for block in assistant_blocks if block["type"] == "tool_use"]
            if not tool_uses:
                assistant_text = _extract_text_response(assistant_blocks)
                if (
                    request_mode == "build"
                    and enforce_build_intent
                    and tool_rounds == 0
                    and not applied_build_modification
                    and no_tool_retries < 2
                ):
                    no_tool_retries += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You must execute this build by calling place_blocks, fill_region, or build_geometry. "
                                "Do not claim completion without successful world-modifying tool calls."
                            ),
                        }
                    )
                    continue
                if require_build_modification and not applied_build_modification:
                    raise RuntimeError("Step completed without any world-modifying tool calls")
                return assistant_text, applied_build_modification, recorded_tool_outcomes

            force_tool_use = False
            tool_rounds += 1
            if tool_rounds > _MAX_TOOL_ROUNDS:
                raise RuntimeError(f"Exceeded max tool rounds ({_MAX_TOOL_ROUNDS})")

            tool_results: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                tool_status = _tool_status_message(tool_use["name"], tool_use["input"])
                await self._emit_tool_status(client_id=client_id, status=tool_status)
            executions = await asyncio.gather(
                *[
                    self._execute_tool(
                        client_id=client_id,
                        world_id=world_id,
                        session_id=session_id,
                        tool_name=tool_use["name"],
                        raw_input=tool_use["input"],
                        player_position=player_position,
                        cached_player_position=player_position,
                        request_mode=request_mode,
                        relative_placement_guard=relative_placement_guard,
                    )
                    for tool_use in tool_uses
                ]
            )
            set_plan_succeeded = False
            for tool_use, execution in zip(tool_uses, executions, strict=True):
                if tool_use["name"] in {"place_blocks", "fill_region", "build_geometry"} and not execution.is_error:
                    applied_build_modification = True
                if tool_use["name"] == "set_plan" and not execution.is_error:
                    set_plan_succeeded = True
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": execution.content,
                }
                if execution.is_error:
                    tool_result["is_error"] = True
                tool_results.append(tool_result)
                outcome: dict[str, Any] = {
                    "tool_name": tool_use["name"],
                    "is_error": execution.is_error,
                    "input": tool_use["input"],
                }
                parsed_content = _parse_json_object_maybe(execution.content)
                if parsed_content is not None:
                    outcome["result"] = parsed_content
                else:
                    outcome["result"] = execution.content
                recorded_tool_outcomes.append(outcome)

            messages.append({"role": "user", "content": tool_results})
            if _is_preview_mode(request_mode) and set_plan_succeeded:
                return "Preview loaded. Reposition if needed, then confirm to place.", applied_build_modification, recorded_tool_outcomes

    async def _run_model_round(
        self,
        *,
        client: AsyncAnthropic,
        model: str,
        client_id: str,
        chat_id: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None,
        emit_deltas: bool = True,
    ) -> Any:
        stream_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": _MAX_MODEL_OUTPUT_TOKENS,
            "temperature": 0,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": _CACHE_CONTROL_EPHEMERAL,
                }
            ],
            "tools": tool_schemas,
            "messages": messages,
        }
        if tool_choice is not None:
            stream_kwargs["tool_choice"] = tool_choice

        async with client.messages.stream(**stream_kwargs) as stream:
            partial = ""
            fallback_blocks: dict[int, _StreamContentBlock] = {}
            async for event in stream:
                _accumulate_stream_event(fallback_blocks, event)
                if event.type != "content_block_delta" or event.delta.type != "text_delta":
                    continue
                if not emit_deltas:
                    continue
                delta = event.delta.text
                partial += delta
                await self._websocket_manager.send_payload(
                    client_id,
                    {
                        "type": "chat.delta",
                        "chat_id": chat_id,
                        "payload": {
                            "delta": delta,
                            "partial": partial,
                        },
                    },
                )
            get_final_message = getattr(stream, "get_final_message", None)
            if callable(get_final_message):
                return await get_final_message()

            # Laminar's Anthropic instrumentation currently unwraps the stream
            # into an async generator that drops helper methods.
            return _response_from_stream_events(fallback_blocks)

    async def _emit_tool_status(self, *, client_id: str, status: str) -> None:
        try:
            await self._websocket_manager.send_payload(
                client_id,
                {
                    "type": "chat.tool_status",
                    "payload": {"status": status},
                },
            )
        except Exception:
            logger.warning("Unable to send tool status to client=%s status=%s", client_id, status)

    async def _execute_tool(
        self,
        client_id: str,
        world_id: str,
        session_id: str,
        tool_name: str,
        raw_input: Any,
        player_position: _PlayerPositionResult,
        cached_player_position: _PlayerPositionResult,
        request_mode: Literal["build", "plan", "plan_fast"],
        relative_placement_guard: _RelativePlacementGuard | None,
    ) -> _ToolExecutionResult:
        if request_mode == "build" and tool_name in {"set_plan", "get_active_overlay", "modify_overlay"}:
            return _ToolExecutionResult(
                content=(
                    f"{tool_name} is preview-only. For this direct build request, "
                    "use place_blocks or fill_region to modify the world immediately."
                ),
                is_error=True,
            )
        try:
            params = _validate_tool_args(tool_name, raw_input)
        except (ValidationError, ValueError) as exc:
            return _ToolExecutionResult(content=f"Invalid arguments for {tool_name}: {exc}", is_error=True)

        try:
            if tool_name == "player_position":
                result = cached_player_position.model_dump(mode="json")
            elif tool_name == "build_geometry":
                placements = build_geometry(
                    shape=str(params["shape"]),
                    material=str(params["material"]),
                    anchor=params["anchor"],
                    rotation=str(params["rotation"]),
                    **{
                        key: value
                        for key, value in params.items()
                        if key not in {"shape", "material", "anchor", "rotation"}
                    },
                )
                _validate_placement_against_player_position(
                    tool_name="place_blocks",
                    params={"placements": placements},
                    player_position=player_position,
                    relative_placement_guard=relative_placement_guard,
                )
                placed_count = 0
                batches = 0
                for start in range(0, len(placements), _PLACE_BLOCK_BATCH_SIZE):
                    batch = placements[start:start + _PLACE_BLOCK_BATCH_SIZE]
                    batch_result = await self._dispatch_tool(
                        client_id=client_id,
                        tool_name="place_blocks",
                        params={"placements": batch},
                    )
                    placed_count += int(batch_result["placed_count"])
                    batches += 1
                result = {
                    "shape": params["shape"],
                    "anchor": params["anchor"],
                    "batches": batches,
                    "placed_count": placed_count,
                    "bbox": bounding_box(placements),
                }
            else:
                _validate_placement_against_player_position(
                    tool_name=tool_name,
                    params=params,
                    player_position=player_position,
                    relative_placement_guard=relative_placement_guard,
                )
                result = await self._dispatch_tool(client_id=client_id, tool_name=tool_name, params=params)
            await self._store_tool_memory(
                client_id=client_id,
                world_id=world_id,
                session_id=session_id,
                tool_name=tool_name,
                params=params,
                result=result,
            )
        except Exception as exc:
            return _ToolExecutionResult(content=f"Tool {tool_name} failed: {exc}", is_error=True)

        return _ToolExecutionResult(content=json.dumps(result), is_error=False)

    async def _dispatch_tool(self, client_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._websocket_manager.request_tool(client_id, tool_name, params)

    async def _require_player_position(self, *, client_id: str) -> _PlayerPositionResult:
        try:
            raw_result = await self._dispatch_tool(client_id=client_id, tool_name="player_position", params={})
            return _PlayerPositionResult.model_validate(raw_result)
        except Exception as exc:
            raise RuntimeError(f"Unable to read current player position: {exc}") from exc

    async def _resolve_session_for_chat(
        self,
        client_id: str,
        world_id: str,
        requested_session_id: str | None,
    ) -> str:
        async with self._session_lock:
            key = _SessionKey(client_id=client_id, world_id=world_id)

            if requested_session_id is not None:
                session = await self._load_session(world_id, requested_session_id)
                if session is None:
                    raise LookupError(f"Session {requested_session_id} not found for world {world_id}")
                self._active_sessions[key] = session.session_id
                return session.session_id

            active_session_id = self._active_sessions.get(key)
            if active_session_id is not None:
                return active_session_id

            session = _new_session_state()
            await self._persist_session(world_id, session)
            self._active_sessions[key] = session.session_id
            return session.session_id

    async def _conversation_messages(self, world_id: str, session_id: str) -> list[dict[str, Any]]:
        async with self._session_lock:
            session = await self._load_session(world_id, session_id)
            if session is None:
                raise LookupError(f"Session {session_id} not found for world {world_id}")
            messages = list(session.messages)

        if len(messages) > _CONTEXT_MESSAGE_LIMIT:
            return messages[-_CONTEXT_MESSAGE_LIMIT:]
        return messages

    async def _append_history(
        self,
        world_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        async with self._session_lock:
            session = await self._load_session(world_id, session_id)
            if session is None:
                raise LookupError(f"Session {session_id} not found for world {world_id}")
            session.messages.append({"role": "user", "content": user_message})
            session.messages.append({"role": "assistant", "content": assistant_message})
            session.updated_at = datetime.now(UTC)
            await self._persist_session(world_id, session)

    async def _load_session(self, world_id: str, session_id: str) -> _SessionState | None:
        cached = self._sessions.get(world_id, {}).get(session_id)
        if cached is not None:
            return cached
        if self._convex_client is None:
            return None

        raw_document = await self._convex_client.query(
            "sessions:get",
            {"world_id": world_id, "session_id": session_id},
        )
        if raw_document is None:
            return None
        document = _ConvexSessionDocument.model_validate(raw_document)
        session = _session_from_convex_document(document)
        self._sessions.setdefault(world_id, {})[session.session_id] = session
        return session

    async def _list_sessions_for_world(self, world_id: str) -> list[_SessionState]:
        if self._convex_client is not None:
            raw_documents = await self._convex_client.query(
                "sessions:listByWorld",
                {"world_id": world_id},
            )
            if not isinstance(raw_documents, list):
                raise RuntimeError("Convex sessions:listByWorld must return a list")
            world_sessions: dict[str, _SessionState] = {}
            for raw_document in raw_documents:
                document = _ConvexSessionDocument.model_validate(raw_document)
                session = _session_from_convex_document(document)
                world_sessions[session.session_id] = session
            self._sessions[world_id] = world_sessions

        sessions = list(self._sessions.get(world_id, {}).values())
        sessions.sort(key=lambda session: session.updated_at, reverse=True)
        return sessions

    async def _persist_session(self, world_id: str, session: _SessionState) -> None:
        self._sessions.setdefault(world_id, {})[session.session_id] = session
        if self._convex_client is None:
            return

        await self._convex_client.mutation(
            "sessions:upsert",
            {
                "world_id": world_id,
                "session_id": session.session_id,
                "messages": session.messages,
                "created_at": _to_epoch_millis(session.created_at),
                "updated_at": _to_epoch_millis(session.updated_at),
            },
        )

    async def _search_memory_context(
        self,
        client_id: str,
        world_id: str,
        user_message: str,
    ) -> str:
        if self._supermemory_client is None:
            return ""

        container_tag = _memory_container_tag(client_id=client_id, world_id=world_id)
        profile_context = await self._supermemory_client.profile_context(container_tag=container_tag)
        memories = await self._supermemory_client.search_memories(
            user_message,
            container_tag=container_tag,
            limit=5,
        )
        memory_lines = _format_memory_context(memories)
        profile_lines = _format_profile_context(profile_context)
        if profile_lines and memory_lines:
            return f"{profile_lines}\n{memory_lines}"
        return profile_lines or memory_lines

    async def _store_tool_memory(
        self,
        client_id: str,
        world_id: str,
        session_id: str,
        tool_name: str,
        params: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if self._supermemory_client is None or tool_name not in _MEMORY_OUTCOME_TOOLS:
            return

        content = _format_tool_memory_content(tool_name=tool_name, params=params, result=result)
        await self._supermemory_client.store_memory(
            content,
            container_tag=_memory_container_tag(client_id=client_id, world_id=world_id),
            metadata={
                "tool": tool_name,
                "client_id": client_id,
                "world_id": world_id,
                "session_id": session_id,
                "params": params,
                "result": result,
            },
        )


def _new_session_state() -> _SessionState:
    now = datetime.now(UTC)
    return _SessionState(session_id=str(uuid4()), created_at=now, updated_at=now, messages=[])


def _session_from_convex_document(document: _ConvexSessionDocument) -> _SessionState:
    return _SessionState(
        session_id=document.session_id,
        created_at=datetime.fromtimestamp(document.created_at / 1000, tz=UTC),
        updated_at=datetime.fromtimestamp(document.updated_at / 1000, tz=UTC),
        messages=[message.model_dump(mode="json") for message in document.messages],
    )


def _to_epoch_millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _memory_container_tag(client_id: str, world_id: str) -> str:
    return f"{world_id}:{client_id}"


def _format_memory_context(memories: list[SupermemorySearchResult]) -> str:
    lines: list[str] = []
    for index, memory in enumerate(memories, start=1):
        suffix = ""
        if memory.similarity is not None:
            suffix = f" (similarity={memory.similarity:.2f})"
        lines.append(f"{index}. {memory.text}{suffix}")
    return "\n".join(lines)


def _format_profile_context(profile: SupermemoryProfileContext) -> str:
    lines: list[str] = []
    if profile.static:
        lines.append("Profile static:")
        lines.extend(f"- {item}" for item in profile.static)
    if profile.dynamic:
        lines.append("Profile dynamic:")
        lines.extend(f"- {item}" for item in profile.dynamic)
    return "\n".join(lines)


def _format_tool_memory_content(tool_name: str, params: dict[str, Any], result: dict[str, Any]) -> str:
    params_payload = json.dumps(params, sort_keys=True)
    result_payload = json.dumps(result, sort_keys=True)
    return f"tool={tool_name}; input={params_payload}; output={result_payload}"


def _accumulate_stream_event(blocks: dict[int, _StreamContentBlock], event: Any) -> None:
    event_type = getattr(event, "type", None)
    if event_type == "content_block_start":
        index = getattr(event, "index", None)
        content_block = getattr(event, "content_block", None)
        block_type = getattr(content_block, "type", None)
        if not isinstance(index, int) or block_type not in {"text", "tool_use"}:
            return
        if block_type == "text":
            blocks[index] = _StreamContentBlock(
                block_type="text",
                text=getattr(content_block, "text", "") or "",
            )
            return
        tool_input = getattr(content_block, "input", None)
        blocks[index] = _StreamContentBlock(
            block_type="tool_use",
            tool_id=getattr(content_block, "id", None),
            tool_name=getattr(content_block, "name", None),
            tool_input=tool_input if isinstance(tool_input, dict) else None,
        )
        return

    if event_type != "content_block_delta":
        return

    index = getattr(event, "index", None)
    delta = getattr(event, "delta", None)
    delta_type = getattr(delta, "type", None)
    if not isinstance(index, int):
        return

    if delta_type == "text_delta":
        block = blocks.setdefault(index, _StreamContentBlock(block_type="text"))
        block.text += getattr(delta, "text", "") or ""
        return

    if delta_type == "input_json_delta":
        block = blocks.setdefault(index, _StreamContentBlock(block_type="tool_use"))
        block.tool_input_chunks.append(getattr(delta, "partial_json", "") or "")


def _response_from_stream_events(blocks: dict[int, _StreamContentBlock]) -> Any:
    content: list[Any] = []
    for index in sorted(blocks):
        block = blocks[index]
        if block.block_type == "text":
            content.append(SimpleNamespace(type="text", text=block.text))
            continue

        if block.tool_id is None or block.tool_name is None:
            raise RuntimeError("Anthropic tool_use block missing id or name in streaming fallback")

        if block.tool_input_chunks:
            raw_input = "".join(block.tool_input_chunks).strip()
            parsed_input = json.loads(raw_input) if raw_input else {}
        else:
            parsed_input = block.tool_input if block.tool_input is not None else {}

        if not isinstance(parsed_input, dict):
            raise RuntimeError("Anthropic tool_use block input must be an object")

        content.append(
            SimpleNamespace(
                type="tool_use",
                id=block.tool_id,
                name=block.tool_name,
                input=parsed_input,
            )
        )

    return SimpleNamespace(content=content)


def _normalize_assistant_blocks(content_blocks: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if not isinstance(text, str):
                raise RuntimeError("Anthropic text block missing text field")
            normalized.append({"type": "text", "text": text})
            continue

        if block_type == "tool_use":
            tool_id = getattr(block, "id", None)
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", None)
            if not isinstance(tool_id, str) or not tool_id:
                raise RuntimeError("Anthropic tool_use block missing id")
            if not isinstance(tool_name, str) or not tool_name:
                raise RuntimeError("Anthropic tool_use block missing name")
            if not isinstance(tool_input, dict):
                raise RuntimeError("Anthropic tool_use block input must be an object")
            normalized.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            continue

        raise RuntimeError(f"Unsupported Anthropic content block type: {block_type}")

    return normalized


def _extract_text_response(assistant_blocks: list[dict[str, Any]]) -> str:
    text_chunks = [block["text"] for block in assistant_blocks if block["type"] == "text"]
    full_text = "".join(text_chunks).strip()
    if not full_text:
        raise RuntimeError("Anthropic response did not include assistant text")
    return full_text


def _extract_text_from_message_content(content_blocks: list[Any]) -> str:
    chunks: list[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks).strip()


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    parsed_json = json.loads(raw_text.strip())
    if not isinstance(parsed_json, dict):
        raise ValueError("Expected JSON object")
    return parsed_json


def _parse_build_request_triage(raw_text: str) -> _BuildRequestTriageModel:
    parsed_json = _extract_json_payload(raw_text)
    return _BuildRequestTriageModel.model_validate(parsed_json)


def _build_anchor_for_request(
    *,
    player_position: _PlayerPositionResult,
    triage: _BuildRequestTriageModel | None,
) -> _BuildAnchor:
    distance = _DEFAULT_FORWARD_BUILD_OFFSET
    if triage is not None and triage.spatial_reference == "relative_to_player":
        distance = triage.distance_hint if triage.distance_hint is not None else _DEFAULT_RELATIVE_PLAYER_OFFSET
    build_anchor = _forward_build_anchor(
        player_position=player_position,
        distance=distance,
    )
    if triage is not None and triage.spatial_reference == "relative_to_player":
        return _BuildAnchor(
            x=build_anchor.x,
            y=player_position.block_y,
            z=build_anchor.z,
        )
    return build_anchor


def _parse_json_object_maybe(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _is_preview_mode(request_mode: Literal["build", "plan", "plan_fast"]) -> bool:
    return request_mode in {"plan", "plan_fast"}


def _tool_choice_for_round(
    *,
    request_mode: Literal["build", "plan", "plan_fast"],
    force_tool_use: bool,
) -> dict[str, Any] | None:
    if request_mode == "plan_fast":
        return {"type": "tool", "name": "set_plan"}
    if force_tool_use:
        return {"type": "any"}
    return None


def _tool_schemas_for_mode(request_mode: Literal["build", "plan", "plan_fast"]) -> list[dict[str, Any]]:
    if request_mode == "plan":
        return [
            schema
            for schema in _CACHEABLE_TOOL_SCHEMAS
            if schema["name"] not in _PLAN_DISALLOWED_TOOL_NAMES
        ]
    if request_mode == "plan_fast":
        return [
            schema
            for schema in _CACHEABLE_TOOL_SCHEMAS
            if schema["name"] in _PLAN_FAST_ALLOWED_TOOL_NAMES
        ]
    return _CACHEABLE_TOOL_SCHEMAS


def _compose_system_prompt(
    *,
    request_mode: Literal["build", "plan", "plan_fast"],
    player_position: _PlayerPositionResult,
    build_anchor: _BuildAnchor,
    memory_context: str,
    triage: _BuildRequestTriageModel | None,
) -> str:
    if triage is not None and triage.spatial_reference == "relative_to_player":
        player_position_context = (
            "Locked player position for this request (captured at submit time, authoritative):\n"
            f"- x={player_position.x:.3f}, y={player_position.y:.3f}, z={player_position.z:.3f}\n"
            f"- block_x={player_position.block_x}, block_y={player_position.block_y}, block_z={player_position.block_z}\n"
            f"- facing={player_position.facing}, dimension={player_position.dimension}\n"
            f"- standing_level_y={player_position.block_y}\n"
        )
    else:
        player_position_context = (
            "Locked player position for this request (captured at submit time, authoritative):\n"
            f"- x={player_position.x:.3f}, y={player_position.y:.3f}, z={player_position.z:.3f}\n"
            f"- block_x={player_position.block_x}, block_y={player_position.block_y}, block_z={player_position.block_z}\n"
            f"- ground_y={player_position.ground_y}\n"
            f"- facing={player_position.facing}, dimension={player_position.dimension}\n"
        )

    if triage is not None and triage.spatial_reference == "relative_to_player":
        anchor_context = (
            "Derived relative-to-player anchor for this request:\n"
            f"- build_anchor_x={build_anchor.x}, build_anchor_y={build_anchor.y}, build_anchor_z={build_anchor.z}\n"
            "Use this as the explicit target/base location for unspecified relative coordinates in this request. "
            "This anchor is not the terrain block beneath the player."
        )
    else:
        anchor_context = (
            "Default build anchor for NEW structures in this request (10 blocks ahead of player facing):\n"
            f"- build_anchor_x={build_anchor.x}, build_anchor_y={build_anchor.y}, build_anchor_z={build_anchor.z}\n"
            "For new structures, center around this build anchor. "
            "Only ignore this when the user gives explicit coordinates or requests edits to an existing structure."
        )

    system_prompt = (
        f"{_SYSTEM_PROMPT}\n"
        f"Request mode for this turn: {request_mode.upper()}.\n"
        f"{player_position_context}"
        f"{anchor_context}"
    )
    if triage is not None:
        system_prompt = (
            f"{system_prompt}\n"
            "Request triage for this turn:\n"
            f"- is_build_request={triage.is_build_request}\n"
            f"- complexity={triage.complexity}\n"
            f"- spatial_reference={triage.spatial_reference}\n"
            f"- distance_hint={triage.distance_hint}\n"
            f"- should_undo_first={triage.should_undo_first}"
        )
    if triage is not None and triage.spatial_reference == "relative_to_player":
        system_prompt = (
            f"{system_prompt}\n"
            "Relative-to-player placement rule for this turn:\n"
            f"- Use block_y={player_position.block_y} as the bottom/start level for structures placed at or around the player unless the user explicitly says to start from the ground block.\n"
            "- For an N-block-tall structure placed relative to the player, the vertical span is block_y through block_y+N-1 unless the user specifies otherwise.\n"
            "- For floors and platforms in front of the player, start the nearest edge one block forward unless the user specifies a different offset, and center the width side-to-side rather than centering the whole footprint on the player's block."
        )
        if player_position.ground_y is not None:
            system_prompt = (
                f"{system_prompt}\n"
                f"- Example for this request: a 3-block-tall tower directly in front of the player should span y={player_position.block_y},{player_position.block_y + 1},{player_position.block_y + 2}; do not place the bottom block at ground_y={player_position.ground_y}."
            )
    if memory_context:
        return (
            f"{system_prompt}\n"
            "Relevant long-term memory:\n"
            f"{memory_context}"
        )
    return system_prompt


def _parse_build_step_plan(raw_text: str) -> _BuildStepPlanModel:
    return _BuildStepPlanModel.model_validate(_extract_json_payload(raw_text))


def _step_execution_message(
    *,
    original_request: str,
    step: _BuildStepModel,
    step_index: int,
    total_steps: int,
    previous_step_outcomes: list[dict[str, Any]],
) -> str:
    prior = "none"
    if previous_step_outcomes:
        prior = json.dumps(previous_step_outcomes[-4:], sort_keys=True)
    return (
        f"Original user request: {original_request}\n"
        f"Execute step {step_index}/{total_steps}.\n"
        f"Step name: {step.name}\n"
        f"Goal: {step.goal}\n"
        f"Relative location hint: {step.relative_location_hint}\n"
        f"Success check: {step.success_check}\n"
        "Completed steps so far:\n"
        f"{prior}\n"
        "Use prior tool outcome data (coordinates, anchors, bbox) when building dependent follow-up steps.\n"
        "Modify the world now and keep changes from earlier steps."
    )


def _relative_placement_guard_for_request(
    *,
    user_message: str,
    player_position: _PlayerPositionResult,
    triage: _BuildRequestTriageModel | None,
) -> _RelativePlacementGuard | None:
    if triage is None or triage.spatial_reference not in {"relative_to_player", "default_anchor"}:
        return None

    message = user_message.lower()
    forward_dx, forward_dz = _horizontal_facing_offset(player_position.facing)
    left_dx, left_dz = forward_dz, -forward_dx
    right_dx, right_dz = -forward_dz, forward_dx

    if triage.spatial_reference == "default_anchor":
        if any(
            phrase in message
            for phrase in (
                "centered on me",
                "center it on me",
                "around me",
                "surround me",
                "use my current position",
                "use the player's current block position",
                "at my position",
                "on me",
            )
        ):
            return None
        return _RelativePlacementGuard(label="in front of", dx=forward_dx, dz=forward_dz)

    if "in front of me" in message:
        return _RelativePlacementGuard(label="in front of", dx=forward_dx, dz=forward_dz)
    if "behind me" in message:
        return _RelativePlacementGuard(label="behind", dx=-forward_dx, dz=-forward_dz)
    if "left of me" in message or "to my left" in message:
        return _RelativePlacementGuard(label="to the left of", dx=left_dx, dz=left_dz)
    if "right of me" in message or "to my right" in message:
        return _RelativePlacementGuard(label="to the right of", dx=right_dx, dz=right_dz)
    return None


def _horizontal_facing_offset(facing: str) -> tuple[int, int]:
    facing_normalized = facing.lower()
    if facing_normalized == "north":
        return (0, -1)
    if facing_normalized == "south":
        return (0, 1)
    if facing_normalized == "east":
        return (1, 0)
    if facing_normalized == "west":
        return (-1, 0)
    raise ValueError(f"Unsupported player facing: {facing}")


def _forward_build_anchor(
    *,
    player_position: _PlayerPositionResult,
    distance: int,
) -> _BuildAnchor:
    forward_dx, forward_dz = _horizontal_facing_offset(player_position.facing)
    dx = forward_dx * distance
    dz = forward_dz * distance

    anchor_ground_y = player_position.ground_y
    if anchor_ground_y is None:
        anchor_ground_y = player_position.block_y - 1

    return _BuildAnchor(
        x=player_position.block_x + dx,
        y=anchor_ground_y,
        z=player_position.block_z + dz,
    )


def _validate_tool_args(tool_name: str, raw_input: Any) -> dict[str, Any]:
    if not isinstance(raw_input, dict):
        raise ValueError(f"{tool_name} expects JSON object arguments")

    if tool_name in {"player_position", "player_inventory", "undo_last", "get_active_overlay", "get_blueprints"}:
        parsed = _NoArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "inspect_area":
        parsed = _InspectAreaArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "place_blocks":
        parsed = _PlaceBlocksArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "set_plan":
        parsed = _SetPlanArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "fill_region":
        parsed = _FillRegionArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "build_geometry":
        parsed = _BUILD_GEOMETRY_ADAPTER.validate_python(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "modify_overlay":
        parsed = _MODIFY_OVERLAY_ADAPTER.validate_python(raw_input)
        return parsed.model_dump(mode="json", by_alias=True)
    if tool_name in {"save_blueprint", "load_blueprint"}:
        parsed = _BlueprintNameArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    raise ValueError(f"Unsupported tool: {tool_name}")


def _validate_placement_against_player_position(
    *,
    tool_name: str,
    params: dict[str, Any],
    player_position: _PlayerPositionResult,
    relative_placement_guard: _RelativePlacementGuard | None,
) -> None:
    ys: list[int]
    if tool_name == "place_blocks":
        ys = [placement["y"] for placement in params["placements"]]
    elif tool_name == "fill_region":
        ys = [params["from_corner"]["y"], params["to_corner"]["y"]]
    else:
        return

    min_y = min(ys)
    max_y = max(ys)
    player_y = player_position.block_y
    if max_y < player_y - _MAX_Y_DELTA_FROM_PLAYER or min_y > player_y + _MAX_Y_DELTA_FROM_PLAYER:
        raise ValueError(
            f"{tool_name} y-range {min_y}..{max_y} is detached from current player block_y {player_y}; "
            "derive placement coordinates from the locked player_position for this request"
        )

    if relative_placement_guard is None:
        return

    xz_positions: list[tuple[int, int]]
    if tool_name == "place_blocks":
        xz_positions = [(placement["x"], placement["z"]) for placement in params["placements"]]
    else:
        min_x = min(params["from_corner"]["x"], params["to_corner"]["x"])
        max_x = max(params["from_corner"]["x"], params["to_corner"]["x"])
        min_z = min(params["from_corner"]["z"], params["to_corner"]["z"])
        max_z = max(params["from_corner"]["z"], params["to_corner"]["z"])
        xz_positions = [(min_x, min_z), (min_x, max_z), (max_x, min_z), (max_x, max_z)]

    min_projection = min(
        ((x - player_position.block_x) * relative_placement_guard.dx)
        + ((z - player_position.block_z) * relative_placement_guard.dz)
        for x, z in xz_positions
    )
    if min_projection < relative_placement_guard.min_projection:
        raise ValueError(
            f"{tool_name} placements for a request {relative_placement_guard.label} the player must stay fully "
            f"{relative_placement_guard.label} the player at ({player_position.block_x}, {player_position.block_z}); "
            f"minimum projected offset was {min_projection}"
        )


def _summarize_historical_tool_results(messages: list[dict[str, Any]]) -> None:
    tool_result_indexes: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
            tool_result_indexes.append(index)

    if len(tool_result_indexes) <= 1:
        return

    for index in tool_result_indexes[:-1]:
        content = messages[index]["content"]
        if not isinstance(content, list):
            continue

        updated_blocks: list[dict[str, Any]] = []
        changed = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                updated_blocks.append(block)
                continue
            raw_content = block.get("content")
            if not isinstance(raw_content, str):
                updated_blocks.append(block)
                continue
            summary = _summarize_tool_result_content(raw_content)
            if summary != raw_content:
                changed = True
            updated_blocks.append({**block, "content": summary})

        if changed:
            messages[index] = {
                "role": "user",
                "content": updated_blocks,
            }


def _summarize_tool_result_content(content: str) -> str:
    if content.startswith(_TOOL_RESULT_SUMMARY_PREFIX):
        return content

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        if len(content) <= 1200:
            return content
        return f"{_TOOL_RESULT_SUMMARY_PREFIX} truncated_text={len(content)} chars"

    if not isinstance(payload, dict):
        if len(content) <= 1200:
            return content
        return f"{_TOOL_RESULT_SUMMARY_PREFIX} non_object_payload={type(payload).__name__}"

    if "non_air_blocks" in payload:
        return _summarize_inspect_area_result(payload)
    if "placed_count" in payload and "anchor" in payload:
        anchor = payload.get("anchor", {})
        return (
            f"{_TOOL_RESULT_SUMMARY_PREFIX} place_blocks placed_count={payload.get('placed_count')} "
            f"anchor={anchor}"
        )
    if "placed_count" in payload and payload.get("fill_region"):
        return (
            f"{_TOOL_RESULT_SUMMARY_PREFIX} fill_region placed_count={payload.get('placed_count')} "
            f"from={payload.get('from_corner')} to={payload.get('to_corner')}"
        )
    if "shape" in payload and "placed_count" in payload and "batches" in payload:
        return (
            f"{_TOOL_RESULT_SUMMARY_PREFIX} build_geometry shape={payload.get('shape')} "
            f"placed_count={payload.get('placed_count')} batches={payload.get('batches')} "
            f"bbox={payload.get('bbox')}"
        )
    if "undone" in payload or "undone_count" in payload:
        return f"{_TOOL_RESULT_SUMMARY_PREFIX} undo_last {json.dumps(payload, sort_keys=True)}"

    compact = json.dumps(payload, sort_keys=True)
    if len(compact) <= 1200:
        return content
    keys = ",".join(sorted(payload.keys()))
    return f"{_TOOL_RESULT_SUMMARY_PREFIX} keys={keys} size={len(compact)} chars"


def _summarize_inspect_area_result(payload: dict[str, Any]) -> str:
    non_air_blocks = payload.get("non_air_blocks")
    if not isinstance(non_air_blocks, list):
        return f"{_TOOL_RESULT_SUMMARY_PREFIX} inspect_area no_non_air_blocks"

    count = len(non_air_blocks)
    if count == 0:
        return (
            f"{_TOOL_RESULT_SUMMARY_PREFIX} inspect_area center={payload.get('center')} "
            f"radius={payload.get('radius')} non_air_blocks=0"
        )

    block_ids: list[str] = []
    xs: list[int] = []
    ys: list[int] = []
    zs: list[int] = []
    for block in non_air_blocks:
        if not isinstance(block, dict):
            continue
        block_id = block.get("block_id")
        x = block.get("x")
        y = block.get("y")
        z = block.get("z")
        if isinstance(block_id, str):
            block_ids.append(block_id)
        if isinstance(x, int) and isinstance(y, int) and isinstance(z, int):
            xs.append(x)
            ys.append(y)
            zs.append(z)

    top_blocks = Counter(block_ids).most_common(4)
    bbox = "unknown"
    if xs and ys and zs:
        bbox = f"x={min(xs)}..{max(xs)},y={min(ys)}..{max(ys)},z={min(zs)}..{max(zs)}"

    return (
        f"{_TOOL_RESULT_SUMMARY_PREFIX} inspect_area center={payload.get('center')} "
        f"radius={payload.get('radius')} detailed={payload.get('detailed')} "
        f"filter_terrain={payload.get('filter_terrain')} non_air_blocks={count} "
        f"bbox={bbox} top_blocks={top_blocks}"
    )


def _tool_status_message(tool_name: str, raw_input: dict[str, Any]) -> str:
    if tool_name == "inspect_area":
        radius = raw_input.get("radius")
        if isinstance(radius, int):
            return f"🔍 Inspecting area (r={radius})..."
        return "🔍 Inspecting area..."
    if tool_name == "place_blocks":
        placements = raw_input.get("placements")
        if isinstance(placements, list):
            return f"🔨 Placing {len(placements)} blocks..."
        return "🔨 Placing blocks..."
    if tool_name == "fill_region":
        return "🧱 Filling region..."
    if tool_name == "build_geometry":
        shape = raw_input.get("shape")
        if isinstance(shape, str):
            return f"📏 Building {shape} geometry..."
        return "📏 Building geometry..."
    if tool_name == "set_plan":
        placements = raw_input.get("placements")
        if isinstance(placements, list):
            return f"📐 Loading plan ({len(placements)} blocks)..."
        return "📐 Loading plan..."
    if tool_name == "undo_last":
        return "↩ Undoing last change..."
    if tool_name == "player_position":
        return "📍 Checking player position..."
    if tool_name == "player_inventory":
        return "🎒 Checking inventory..."
    if tool_name == "get_blueprints":
        return "📚 Loading blueprints..."
    if tool_name == "save_blueprint":
        return "💾 Saving blueprint..."
    if tool_name == "load_blueprint":
        return "📂 Loading blueprint..."
    if tool_name == "modify_overlay":
        return "🧭 Updating overlay..."
    return f"⚙ Running {tool_name}..."
