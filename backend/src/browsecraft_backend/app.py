from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from .chat_orchestrator import ChatOrchestrator
from .config import Settings, get_settings
from .convex_client import ConvexHttpClient
from .demo_pipelines import DemoPipelines
from .models import (
    AsyncJobAcceptedResponse,
    ChatAcceptedResponse,
    ChatRequest,
    ImagineRequest,
    SearchRequest,
    SessionCreatedResponse,
    SessionListResponse,
    SessionNewRequest,
    SessionSwitchRequest,
    SessionSwitchedResponse,
)
from .sponsors import initialize_laminar, verify_sponsor_imports
from .supermemory_client import SupermemoryClient
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

HttpClientFactory = Callable[[], httpx.AsyncClient]
ChatOrchestratorFactory = Callable[[Settings, WebSocketManager], ChatOrchestrator]
DemoPipelinesFactory = Callable[[Settings, WebSocketManager, ChatOrchestrator], DemoPipelines]


def _build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(follow_redirects=True, timeout=30.0)


def _build_chat_orchestrator(settings: Settings, ws_manager: WebSocketManager) -> ChatOrchestrator:
    return ChatOrchestrator(
        anthropic_api_key=settings.anthropic_api_key,
        websocket_manager=ws_manager,
        chat_model=settings.anthropic_chat_model,
        planner_model=settings.anthropic_planner_model,
        triage_model=settings.anthropic_triage_model,
        enable_build_planner=settings.anthropic_enable_build_planner,
    )


def _build_demo_pipelines(
    settings: Settings,
    ws_manager: WebSocketManager,
    chat_orchestrator: ChatOrchestrator,
) -> DemoPipelines:
    return DemoPipelines(
        websocket_manager=ws_manager,
        chat_submitter=chat_orchestrator.submit_chat,
    )


def create_app(
    *,
    http_client: HttpClientFactory | None = None,
    chat_orchestrator: ChatOrchestratorFactory | None = None,
    demo_pipelines: DemoPipelinesFactory | None = None,
) -> FastAPI:
    app = FastAPI(title="BrowseCraft Backend", version="0.1.0")

    settings = get_settings()
    verify_sponsor_imports()
    initialize_laminar(settings.laminar_api_key)
    ws_manager = WebSocketManager()

    client_factory = http_client or _build_http_client
    chat_factory = chat_orchestrator or _build_chat_orchestrator
    pipelines_factory = demo_pipelines or _build_demo_pipelines

    client = client_factory()
    convex = (
        ConvexHttpClient(settings.convex_url, client, settings.convex_access_key)
        if settings.convex_url
        else None
    )
    supermemory = SupermemoryClient(settings.supermemory_api_key) if settings.supermemory_api_key else None
    chat = chat_factory(settings, ws_manager)
    pipelines = pipelines_factory(settings, ws_manager, chat)
    if isinstance(chat, ChatOrchestrator):
        chat.configure_integrations(convex_client=convex, supermemory_client=supermemory)

    app.state.settings = settings
    app.state.http_client = client
    app.state.websocket_manager = ws_manager
    app.state.chat_orchestrator = chat
    app.state.demo_pipelines = pipelines
    app.state.convex = convex
    app.state.supermemory = supermemory

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        logger.warning(
            "Request validation failed path=%s errors=%s body=%s",
            request.url.path,
            exc.errors(),
            exc.body,
        )
        return await request_validation_exception_handler(request, exc)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()
        if supermemory is not None:
            await supermemory.close()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat", response_model=ChatAcceptedResponse)
    async def create_chat(payload: ChatRequest) -> ChatAcceptedResponse:
        try:
            return await chat.submit_chat(payload)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/search", response_model=AsyncJobAcceptedResponse)
    async def search_schematics(payload: SearchRequest) -> AsyncJobAcceptedResponse:
        return await pipelines.submit_search(payload)

    @app.post("/v1/imagine", response_model=AsyncJobAcceptedResponse)
    async def imagine_structure(payload: ImagineRequest) -> AsyncJobAcceptedResponse:
        return await pipelines.submit_imagine(payload)

    @app.post("/v1/session/new", response_model=SessionCreatedResponse)
    async def create_session(payload: SessionNewRequest) -> SessionCreatedResponse:
        return await chat.create_session(client_id=payload.client_id, world_id=payload.world_id)

    @app.get("/v1/session/list", response_model=SessionListResponse)
    async def list_sessions(
        client_id: str = Query(min_length=1),
        world_id: str = Query(min_length=1),
    ) -> SessionListResponse:
        return await chat.list_sessions(client_id=client_id, world_id=world_id)

    @app.post("/v1/session/switch", response_model=SessionSwitchedResponse)
    async def switch_session(payload: SessionSwitchRequest) -> SessionSwitchedResponse:
        try:
            return await chat.switch_session(
                client_id=payload.client_id,
                world_id=payload.world_id,
                session_id=payload.session_id,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.websocket("/v1/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
        await ws_manager.connect(client_id, websocket)
        if isinstance(chat, ChatOrchestrator):
            asyncio.create_task(chat.warmup_prompt_cache())
        try:
            while True:
                raw_message = await websocket.receive_text()
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                await ws_manager.handle_incoming_message(client_id, message)
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(client_id)

    return app


app = create_app()
