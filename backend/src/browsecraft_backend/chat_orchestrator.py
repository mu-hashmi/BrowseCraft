from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Callable, Literal, Protocol
from uuid import uuid4

from anthropic import AsyncAnthropic
from lmnr import observe
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .convex_client import ConvexHttpClient
from .job_manager import JobManager
from .models import (
    BuildRequest,
    ChatAcceptedResponse,
    ChatRequest,
    ImagineRequest,
    SessionCreatedResponse,
    SessionListResponse,
    SessionSummary,
    SessionSwitchedResponse,
)
from .supermemory_client import SupermemoryProfileContext, SupermemorySearchResult
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

CHAT_MODEL = "claude-sonnet-4-20250514"
CHAT_ESCALATION_MODEL = "claude-opus-4-1-20250805"
DEFAULT_MC_VERSION = "1.21.11"
_DEFAULT_WORLD_ID = "default"
_SYSTEM_PROMPT = (
    "You are BrowseCraft's Minecraft in-game assistant.\n"
    "Keep responses concise.\n"
    "Before requesting world or overlay modifications, use relevant read-only tools first.\n"
    "Use tools for factual game state instead of guessing."
)
_MAX_TOOL_ROUNDS = 12
_CONTEXT_MESSAGE_LIMIT = 12
_MEMORY_OUTCOME_TOOLS = {
    "search_schematics",
    "generate_structure",
    "modify_overlay",
    "save_blueprint",
    "load_blueprint",
}
_OPUS_ESCALATION_HINTS = (
    "castle",
    "blueprint",
    "terrain",
    "cliff",
    "symmetry",
    "multi-floor",
    "redstone",
    "optimize",
    "layout",
)


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NoArgs(_ToolArgs):
    pass


class _SearchSchematicsArgs(_ToolArgs):
    query: str = Field(min_length=1)


class _GenerateStructureArgs(_ToolArgs):
    prompt: str = Field(min_length=1)


class _BlockPositionArgs(_ToolArgs):
    x: int
    y: int
    z: int


class _InspectAreaArgs(_ToolArgs):
    center: _BlockPositionArgs
    radius: int = Field(ge=0, le=16)


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


class _BlueprintNameArgs(_ToolArgs):
    name: str = Field(min_length=1)


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
        "name": "search_schematics",
        "description": "Start a schematic search job from a user query.",
        "input_schema": _SearchSchematicsArgs.model_json_schema(),
    },
    {
        "name": "generate_structure",
        "description": "Start an image-to-structure generation job from a prompt.",
        "input_schema": _GenerateStructureArgs.model_json_schema(),
    },
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
        "description": "Inspect blocks around a center position with a radius.",
        "input_schema": _InspectAreaArgs.model_json_schema(),
    },
    {
        "name": "get_active_overlay",
        "description": "Read the active overlay state.",
        "input_schema": _NoArgs.model_json_schema(),
    },
    {
        "name": "modify_overlay",
        "description": "Modify overlay state with rotate, shift, set_anchor, or replace_block operations.",
        "input_schema": _MODIFY_OVERLAY_ADAPTER.json_schema(),
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


class ChatOrchestrator:
    def __init__(
        self,
        anthropic_api_key: str | None,
        job_manager: JobManager,
        websocket_manager: WebSocketManager,
        chat_model: str = CHAT_MODEL,
        escalation_model: str = CHAT_ESCALATION_MODEL,
        anthropic_client_factory: AnthropicClientFactory | None = None,
        convex_client: ConvexHttpClient | None = None,
        supermemory_client: SupermemoryClientProtocol | None = None,
    ) -> None:
        self._anthropic_api_key = anthropic_api_key
        self._chat_model = chat_model
        self._escalation_model = escalation_model
        self._job_manager = job_manager
        self._websocket_manager = websocket_manager
        self._anthropic_client_factory = anthropic_client_factory or (lambda api_key: AsyncAnthropic(api_key=api_key))
        self._convex_client = convex_client
        self._supermemory_client = supermemory_client

        self._sessions: dict[str, dict[str, _SessionState]] = {}
        self._active_sessions: dict[_SessionKey, str] = {}
        self._session_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit_chat(self, request: ChatRequest) -> ChatAcceptedResponse:
        world_id = request.world_id or _DEFAULT_WORLD_ID
        session_id = await self._resolve_session_for_chat(
            client_id=request.client_id,
            world_id=world_id,
            requested_session_id=request.session_id,
        )

        chat_id = str(uuid4())
        task = asyncio.create_task(
            self._run_chat(
                chat_id=chat_id,
                client_id=request.client_id,
                user_message=request.message,
                world_id=world_id,
                session_id=session_id,
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
        *,
        world_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        resolved_world_id = world_id or _DEFAULT_WORLD_ID
        try:
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
            )
            await self._append_history(
                world_id=resolved_world_id,
                session_id=resolved_session_id,
                user_message=user_message,
                assistant_message=assistant_text,
            )
        except Exception as exc:
            logger.exception("Chat orchestration failed for client=%s chat_id=%s", client_id, chat_id)
            assistant_text = f"Unable to process chat request: {exc}"

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
    ) -> str:
        if not self._anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for chat orchestrator")

        messages = await self._conversation_messages(world_id=world_id, session_id=session_id)
        messages.append({"role": "user", "content": user_message})

        memory_context = await self._search_memory_context(
            client_id=client_id,
            world_id=world_id,
            user_message=user_message,
        )
        system_prompt = _SYSTEM_PROMPT
        if memory_context:
            system_prompt = (
                f"{_SYSTEM_PROMPT}\n"
                "Relevant long-term memory:\n"
                f"{memory_context}"
            )

        selected_model = _select_chat_model(
            user_message=user_message,
            default_model=self._chat_model,
            escalation_model=self._escalation_model,
        )
        client = self._anthropic_client_factory(self._anthropic_api_key)
        try:
            tool_rounds = 0
            while True:
                response = await self._run_model_round(
                    client=client,
                    model=selected_model,
                    client_id=client_id,
                    chat_id=chat_id,
                    system_prompt=system_prompt,
                    messages=messages,
                )
                assistant_blocks = _normalize_assistant_blocks(response.content)
                messages.append({"role": "assistant", "content": assistant_blocks})

                tool_uses = [block for block in assistant_blocks if block["type"] == "tool_use"]
                if not tool_uses:
                    return _extract_text_response(assistant_blocks)

                tool_rounds += 1
                if tool_rounds > _MAX_TOOL_ROUNDS:
                    raise RuntimeError(f"Exceeded max tool rounds ({_MAX_TOOL_ROUNDS})")

                tool_results: list[dict[str, Any]] = []
                for tool_use in tool_uses:
                    execution = await self._execute_tool(
                        client_id=client_id,
                        world_id=world_id,
                        session_id=session_id,
                        tool_name=tool_use["name"],
                        raw_input=tool_use["input"],
                    )
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": execution.content,
                    }
                    if execution.is_error:
                        tool_result["is_error"] = True
                    tool_results.append(tool_result)

                messages.append({"role": "user", "content": tool_results})
        finally:
            await client.close()

    async def _run_model_round(
        self,
        *,
        client: AsyncAnthropic,
        model: str,
        client_id: str,
        chat_id: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
    ) -> Any:
        if not hasattr(client.messages, "stream"):
            return await client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system=system_prompt,
                tools=_TOOL_SCHEMAS,
                messages=messages,
            )

        partial_chunks: list[str] = []
        async with client.messages.stream(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            tools=_TOOL_SCHEMAS,
            messages=messages,
        ) as stream:
            async for chunk in stream.text_stream:
                partial_chunks.append(chunk)
                await self._websocket_manager.send_payload(
                    client_id,
                    {
                        "type": "chat.delta",
                        "chat_id": chat_id,
                        "payload": {
                            "delta": chunk,
                            "partial": "".join(partial_chunks),
                        },
                    },
                )
            return await stream.get_final_message()

    async def _execute_tool(
        self,
        client_id: str,
        world_id: str,
        session_id: str,
        tool_name: str,
        raw_input: Any,
    ) -> _ToolExecutionResult:
        try:
            params = _validate_tool_args(tool_name, raw_input)
        except (ValidationError, ValueError) as exc:
            return _ToolExecutionResult(content=f"Invalid arguments for {tool_name}: {exc}", is_error=True)

        try:
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
        if tool_name == "search_schematics":
            request = BuildRequest(
                query=params["query"],
                mc_version=DEFAULT_MC_VERSION,
                client_id=client_id,
            )
            created = await self._job_manager.create_job(request)
            return created.model_dump(mode="json")

        if tool_name == "generate_structure":
            request = ImagineRequest(prompt=params["prompt"], client_id=client_id)
            created = await self._job_manager.create_imagine_job(request)
            return created.model_dump(mode="json")

        return await self._websocket_manager.request_tool(client_id, tool_name, params)

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


def _select_chat_model(user_message: str, default_model: str, escalation_model: str) -> str:
    lowered = user_message.lower()
    score = sum(1 for hint in _OPUS_ESCALATION_HINTS if hint in lowered)
    if score >= 2:
        return escalation_model
    return default_model


def _format_tool_memory_content(tool_name: str, params: dict[str, Any], result: dict[str, Any]) -> str:
    params_payload = json.dumps(params, sort_keys=True)
    result_payload = json.dumps(result, sort_keys=True)
    return f"tool={tool_name}; input={params_payload}; output={result_payload}"


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


def _validate_tool_args(tool_name: str, raw_input: Any) -> dict[str, Any]:
    if not isinstance(raw_input, dict):
        raise ValueError(f"{tool_name} expects JSON object arguments")

    if tool_name == "search_schematics":
        parsed = _SearchSchematicsArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "generate_structure":
        parsed = _GenerateStructureArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name in {"player_position", "player_inventory", "get_active_overlay", "get_blueprints"}:
        parsed = _NoArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "inspect_area":
        parsed = _InspectAreaArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    if tool_name == "modify_overlay":
        parsed = _MODIFY_OVERLAY_ADAPTER.validate_python(raw_input)
        return parsed.model_dump(mode="json", by_alias=True)
    if tool_name in {"save_blueprint", "load_blueprint"}:
        parsed = _BlueprintNameArgs.model_validate(raw_input)
        return parsed.model_dump(mode="json")
    raise ValueError(f"Unsupported tool: {tool_name}")
