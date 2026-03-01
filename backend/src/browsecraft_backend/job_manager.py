from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from lmnr import observe

from .browser_use_client import BrowserUseService
from .config import Settings
from .imagine_service import GEMINI_IMAGE_MODEL, ImaginePipeline
from .models import (
    BuildJobCreated,
    BuildRequest,
    EventEnvelope,
    ImagineModifyRequest,
    ImagineRequest,
    JobErrorPayload,
    JobReadyPayload,
    JobState,
    JobStatusPayload,
    SourceInfo,
)
from .schematic_parser import UnsupportedSchematicFormatError, parse_schematic_bytes
from .sources import CandidateFile, SourceDiscovery
from .websocket_manager import WebSocketManager


@dataclass(slots=True)
class JobFailure(Exception):
    code: str
    message: str


@dataclass(slots=True, frozen=True)
class _ImagineReferenceImage:
    data: bytes
    mime_type: str


class JobManager:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient,
        websocket_manager: WebSocketManager,
        source_discovery: SourceDiscovery,
        browser_use: BrowserUseService,
        imagine_service: ImaginePipeline,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._websocket_manager = websocket_manager
        self._source_discovery = source_discovery
        self._browser_use = browser_use
        self._imagine_service = imagine_service

        self._jobs: dict[str, JobState] = {}
        self._job_semaphore = asyncio.Semaphore(1)
        self._browser_semaphore = asyncio.Semaphore(1)
        self._latest_imagine_images: dict[str, _ImagineReferenceImage] = {}

    async def create_job(self, request: BuildRequest) -> BuildJobCreated:
        state = JobState(
            query=request.query,
            mc_version=request.mc_version,
            client_id=request.client_id,
        )
        self._jobs[state.job_id] = state
        asyncio.create_task(self._execute_job(state))
        return BuildJobCreated(job_id=state.job_id, status=state.stage)

    async def create_imagine_job(self, request: ImagineRequest) -> BuildJobCreated:
        state = JobState(
            query=request.prompt,
            mc_version="imagine",
            client_id=request.client_id,
        )
        self._jobs[state.job_id] = state
        asyncio.create_task(self._execute_imagine_job(state))
        return BuildJobCreated(job_id=state.job_id, status=state.stage)

    async def create_imagine_modify_job(self, request: ImagineModifyRequest) -> BuildJobCreated:
        state = JobState(
            query=request.prompt,
            mc_version="imagine_modify",
            client_id=request.client_id,
        )
        self._jobs[state.job_id] = state
        asyncio.create_task(self._execute_imagine_modify_job(state))
        return BuildJobCreated(job_id=state.job_id, status=state.stage)

    def get_job(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    @observe()
    async def _execute_job(self, state: JobState) -> None:
        async with self._job_semaphore:
            await self._emit_status(state, "queued", "Job queued")
            await self._emit_status(state, "searching", "Searching sources")
            try:
                api_task = self._source_discovery.search(
                    query=state.query,
                    mc_version=state.mc_version,
                    allowed_exts=self._settings.allowed_download_exts,
                )
                browser_task = self._discover_via_browser_use(state)
                api_candidates, browser_candidates = await asyncio.gather(api_task, browser_task)

                candidates = self._rank_and_dedupe(api_candidates + browser_candidates)
                if not candidates:
                    raise JobFailure("NO_SCHEMATIC_FOUND", "No schematic candidates found")

                selected_candidate, selected_plan, confidence = await self._download_and_parse(candidates)
                await self._emit_status(state, "normalizing", "Normalizing build plan")
                if selected_plan.total_blocks > self._settings.max_plan_blocks:
                    raise JobFailure(
                        "INVALID_PLAN",
                        f"Plan has {selected_plan.total_blocks} blocks, exceeds max {self._settings.max_plan_blocks}",
                    )

                state.source = SourceInfo(type=selected_candidate.source, url=selected_candidate.canonical_url)
                state.confidence = confidence
                state.plan = selected_plan
                state.set_stage("ready", "Build plan ready")

                payload = JobReadyPayload(source=state.source, confidence=confidence, plan=selected_plan)
                await self._send_event(
                    state.client_id,
                    EventEnvelope(
                        type="job.ready",
                        job_id=state.job_id,
                        payload=payload.model_dump(mode="json"),
                    ),
                )
            except JobFailure as failure:
                await self._emit_error(state, failure.code, failure.message)
            except httpx.HTTPError as exc:
                await self._emit_error(state, "NETWORK_ERROR", str(exc))
            except Exception as exc:
                await self._emit_error(state, "INTERNAL_ERROR", str(exc))

    @observe()
    async def _execute_imagine_job(self, state: JobState) -> None:
        async with self._job_semaphore:
            try:
                await self._emit_status(state, "searching", "generating image")
                imagine_result = await self._imagine_service.build_plan_result(state.query)
                await self._emit_status(state, "normalizing", "converting to blocks")
                await self._publish_imagine_result(state, imagine_result)
            except JobFailure as failure:
                await self._emit_error(state, failure.code, failure.message)
            except Exception as exc:
                await self._emit_error(state, "INTERNAL_ERROR", str(exc))

    @observe()
    async def _execute_imagine_modify_job(self, state: JobState) -> None:
        async with self._job_semaphore:
            reference = self._latest_imagine_images.get(state.client_id)
            if reference is None:
                await self._emit_error(
                    state,
                    "NO_REFERENCE_IMAGE",
                    "No previous imagine image is available for modify. Run /imagine first.",
                )
                return

            try:
                await self._emit_status(state, "searching", "editing image")
                imagine_result = await self._imagine_service.modify_plan_result(
                    prompt=state.query,
                    reference_image_data=reference.data,
                    reference_image_mime_type=reference.mime_type,
                )
                await self._emit_status(state, "normalizing", "converting to blocks")
                await self._publish_imagine_result(state, imagine_result)
            except JobFailure as failure:
                await self._emit_error(state, failure.code, failure.message)
            except Exception as exc:
                await self._emit_error(state, "INTERNAL_ERROR", str(exc))

    async def _publish_imagine_result(self, state: JobState, imagine_result) -> None:
        plan = imagine_result.plan
        if plan.total_blocks > 499:
            raise JobFailure(
                "INVALID_PLAN",
                f"Imagine plan has {plan.total_blocks} blocks, exceeds max 499",
            )

        self._latest_imagine_images[state.client_id] = _ImagineReferenceImage(
            data=imagine_result.image_data,
            mime_type=imagine_result.image_mime_type,
        )
        state.source = SourceInfo(
            type="imagine",
            url=f"imagine://{GEMINI_IMAGE_MODEL}+{imagine_result.plan_source}",
        )
        state.confidence = 0.8
        state.plan = plan
        await self._emit_status(state, "ready", "ready")

        payload = JobReadyPayload(source=state.source, confidence=state.confidence, plan=plan)
        await self._send_event(
            state.client_id,
            EventEnvelope(
                type="job.ready",
                job_id=state.job_id,
                payload=payload.model_dump(mode="json"),
            ),
        )

    async def _discover_via_browser_use(self, state: JobState) -> list[CandidateFile]:
        if not self._settings.browser_use_api_key:
            return []

        async with self._browser_semaphore:
            try:
                return await self._browser_use.discover_via_browsing(
                    query=state.query,
                    mc_version=state.mc_version,
                    allowed_exts=self._settings.allowed_download_exts,
                    on_progress=lambda message: self._emit_status(state, "searching", message),
                )
            except Exception as exc:
                raise JobFailure("CONFIG_ERROR", f"Browser-use discovery failed: {exc}") from exc

    async def _download_and_parse(self, candidates: list[CandidateFile]):
        last_error: JobFailure | None = None

        for rank, candidate in enumerate(candidates):
            try:
                response = await self._http_client.get(candidate.download_url)
                response.raise_for_status()
                plan = parse_schematic_bytes(candidate.filename, response.content)
                confidence = _calculate_confidence(candidate, rank)
                return candidate, plan, confidence
            except UnsupportedSchematicFormatError as exc:
                last_error = JobFailure("UNSUPPORTED_SCHEMATIC_FORMAT", str(exc))
            except httpx.HTTPError as exc:
                last_error = JobFailure("NETWORK_ERROR", str(exc))
            except ValueError as exc:
                last_error = JobFailure("INVALID_PLAN", str(exc))

        if last_error is not None:
            raise last_error
        raise JobFailure("NO_SCHEMATIC_FOUND", "No schematic candidates found")

    def _rank_and_dedupe(self, candidates: list[CandidateFile]) -> list[CandidateFile]:
        priority = {
            ".schem": 0,
            ".litematic": 1,
            ".schematic": 2,
        }
        ordered = sorted(candidates, key=lambda c: (priority.get(c.extension, 100), -c.score))

        seen: set[str] = set()
        deduped: list[CandidateFile] = []
        for candidate in ordered:
            dedupe_key = candidate.download_url.strip().lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped.append(candidate)
        return deduped

    async def _emit_status(self, state: JobState, stage: str, message: str) -> None:
        state.set_stage(stage, message)
        payload = JobStatusPayload(stage=stage, message=message)
        await self._send_event(
            state.client_id,
            EventEnvelope(
                type="job.status",
                job_id=state.job_id,
                payload=payload.model_dump(mode="json"),
            ),
        )

    async def _emit_error(self, state: JobState, code: str, message: str) -> None:
        state.set_error(code, message)
        payload = JobErrorPayload(code=code, message=message)
        await self._send_event(
            state.client_id,
            EventEnvelope(
                type="job.error",
                job_id=state.job_id,
                payload=payload.model_dump(mode="json"),
            ),
        )

    async def _send_event(self, client_id: str, event: EventEnvelope) -> None:
        try:
            await self._websocket_manager.send(client_id, event)
        except KeyError:
            return


def _calculate_confidence(candidate: CandidateFile, rank: int) -> float:
    base_score = candidate.score
    if base_score > 1:
        base_score = min(base_score / 1_000_000.0, 0.95)
    elif base_score <= 0:
        base_score = 0.6

    confidence = base_score - (rank * 0.05)
    return max(0.05, min(confidence, 0.99))
