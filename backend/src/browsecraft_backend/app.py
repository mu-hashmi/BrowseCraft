from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from .browser_use_client import BrowserUseService
from .chat_orchestrator import ChatOrchestrator
from .config import Settings, get_settings
from .convex_client import ConvexHttpClient
from .imagine_service import ImaginePipeline, ImagineService
from .job_manager import JobManager
from .models import (
    BuildJobCreated,
    BuildRequest,
    ChatAcceptedResponse,
    ChatRequest,
    ImagineModifyRequest,
    ImagineRequest,
    JobStatusResponse,
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
BrowserUseFactory = Callable[[Settings], BrowserUseService]
ImagineServiceFactory = Callable[[Settings], ImaginePipeline]
JobManagerFactory = Callable[
    [Settings, httpx.AsyncClient, WebSocketManager, BrowserUseService, ImaginePipeline],
    JobManager,
]
ChatOrchestratorFactory = Callable[[Settings, JobManager, WebSocketManager], ChatOrchestrator]


def _build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(follow_redirects=True, timeout=30.0)


def _build_browser_use(settings: Settings) -> BrowserUseService:
    return BrowserUseService(
        api_key=settings.browser_use_api_key,
        primary_model=settings.browser_use_primary_llm,
        fallback_model=settings.browser_use_fallback_llm,
        timeout_seconds=settings.browser_use_task_timeout_seconds,
        planet_minecraft_skill_id=settings.browser_use_planet_minecraft_skill_id,
        profile_id=settings.browser_use_profile_id,
    )


def _build_imagine_service(settings: Settings) -> ImagineService:
    return ImagineService(
        google_api_key=settings.google_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_vision_model=settings.anthropic_vision_model,
        use_gemini_text_plan=settings.imagine_use_gemini_text_plan,
    )


def _build_job_manager(
    settings: Settings,
    http_client: httpx.AsyncClient,
    ws_manager: WebSocketManager,
    browser_use: BrowserUseService,
    imagine_service: ImaginePipeline,
) -> JobManager:
    return JobManager(
        settings=settings,
        http_client=http_client,
        websocket_manager=ws_manager,
        browser_use=browser_use,
        imagine_service=imagine_service,
    )


def _build_chat_orchestrator(
    settings: Settings,
    jobs: JobManager,
    ws_manager: WebSocketManager,
) -> ChatOrchestrator:
    return ChatOrchestrator(
        anthropic_api_key=settings.anthropic_api_key,
        job_manager=jobs,
        websocket_manager=ws_manager,
        chat_model=settings.anthropic_chat_model,
    )


def create_app(
    *,
    http_client: HttpClientFactory | None = None,
    browser_use: BrowserUseFactory | None = None,
    imagine_service: ImagineServiceFactory | None = None,
    job_manager: JobManagerFactory | None = None,
    chat_orchestrator: ChatOrchestratorFactory | None = None,
) -> FastAPI:
    app = FastAPI(title="BrowseCraft Backend", version="0.1.0")

    settings = get_settings()
    verify_sponsor_imports()
    initialize_laminar(settings.laminar_api_key)
    ws_manager = WebSocketManager()

    client_factory = http_client or _build_http_client
    browser_factory = browser_use or _build_browser_use
    imagine_factory = imagine_service or _build_imagine_service
    job_factory = job_manager or _build_job_manager
    chat_factory = chat_orchestrator or _build_chat_orchestrator

    client = client_factory()
    browser_service = browser_factory(settings)
    imagine_pipeline = imagine_factory(settings)
    convex = (
        ConvexHttpClient(settings.convex_url, client, settings.convex_access_key)
        if settings.convex_url
        else None
    )
    supermemory = SupermemoryClient(settings.supermemory_api_key) if settings.supermemory_api_key else None
    jobs = job_factory(settings, client, ws_manager, browser_service, imagine_pipeline)
    chat = chat_factory(settings, jobs, ws_manager)
    if isinstance(chat, ChatOrchestrator):
        chat.configure_integrations(convex_client=convex, supermemory_client=supermemory)

    app.state.settings = settings
    app.state.http_client = client
    app.state.websocket_manager = ws_manager
    app.state.browser_use = browser_service
    app.state.imagine_service = imagine_pipeline
    app.state.job_manager = jobs
    app.state.chat_orchestrator = chat
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

    @app.post("/v1/jobs", response_model=BuildJobCreated)
    async def create_job(payload: BuildRequest) -> BuildJobCreated:
        return await jobs.create_job(payload)

    @app.post("/v1/imagine", response_model=BuildJobCreated)
    async def create_imagine_job(payload: ImagineRequest) -> BuildJobCreated:
        return await jobs.create_imagine_job(payload)

    @app.post("/v1/imagine/modify", response_model=BuildJobCreated)
    async def create_imagine_modify_job(payload: ImagineModifyRequest) -> BuildJobCreated:
        return await jobs.create_imagine_modify_job(payload)

    @app.post("/v1/chat", response_model=ChatAcceptedResponse)
    async def create_chat(payload: ChatRequest) -> ChatAcceptedResponse:
        try:
            return await chat.submit_chat(payload)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        state = jobs.get_job(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        return state.as_response()

    @app.websocket("/v1/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
        await ws_manager.connect(client_id, websocket)
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
