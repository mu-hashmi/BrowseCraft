from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from browser_use_sdk import AsyncBrowserUse, SessionSettings, TaskOutputFileResponse
from lmnr import observe
from pydantic import BaseModel, Field

from .sources import CandidateFile


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """Go to planetminecraft.com and find downloadable Minecraft Java schematic files.
Use only Planet Minecraft pages.
Search for the query and open project pages.
On each project, use the Schematic download tab/button path.
Never use World Save or map downloads. Never use .zip downloads.
If a project only offers World Save/.zip, skip it and continue.
Download only files with allowed extensions so they appear in task output files.
Stop if you revisit the same page repeatedly and move to the next result.
Return only candidates backed by downloaded files with allowed extensions.
Query: {query}
Minecraft Version: {mc_version}
Allowed extensions: {allowed_exts}
Target site: planetminecraft.com"""

ProgressCallback = Callable[[str], Awaitable[None]]


class BrowsedCandidate(BaseModel):
    canonical_url: str
    filename: str
    title: str
    score: float = Field(default=0.5)
    download_url: str | None = None


class BrowsedCandidates(BaseModel):
    candidates: list[BrowsedCandidate] = Field(default_factory=list)


class BrowserUseService:
    def __init__(
        self,
        api_key: str | None,
        primary_model: str,
        fallback_model: str,
        timeout_seconds: int,
        *,
        planet_minecraft_skill_id: str | None = None,
        profile_id: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._timeout_seconds = timeout_seconds
        self._planet_minecraft_skill_id = planet_minecraft_skill_id
        self._profile_id = profile_id

    @observe()
    async def discover_via_browsing(
        self,
        query: str,
        mc_version: str,
        allowed_exts: tuple[str, ...],
        on_progress: ProgressCallback | None = None,
    ) -> list[CandidateFile]:
        if not self._api_key:
            return []

        if self._planet_minecraft_skill_id:
            return await self._discover_via_skill(query, mc_version, allowed_exts, on_progress)

        return await self._discover_via_agent(query, mc_version, allowed_exts, on_progress)

    @observe()
    async def _discover_via_skill(
        self,
        query: str,
        mc_version: str,
        allowed_exts: tuple[str, ...],
        on_progress: ProgressCallback | None,
    ) -> list[CandidateFile]:
        client = AsyncBrowserUse(api_key=self._api_key, timeout=float(self._timeout_seconds))
        try:
            await _emit_progress(on_progress, "browser-use: executing Planet Minecraft skill")
            execution = await client.skills.execute(
                self._planet_minecraft_skill_id,
                parameters={
                    "query": query,
                    "mc_version": mc_version,
                    "allowed_exts": list(allowed_exts),
                },
            )
            if not execution.success:
                raise RuntimeError(f"browser-use skill execution failed: {execution.error or 'unknown error'}")
            await _emit_progress(on_progress, "browser-use: skill execution complete")
        finally:
            await client.close()

        payload = execution.result
        if isinstance(payload, dict):
            parsed = BrowsedCandidates.model_validate(payload)
            return self._filter_skill_candidates(parsed.candidates, allowed_exts)
        raise RuntimeError("browser-use skill result must be a JSON object with candidates")

    @observe()
    async def _discover_via_agent(
        self,
        query: str,
        mc_version: str,
        allowed_exts: tuple[str, ...],
        on_progress: ProgressCallback | None,
    ) -> list[CandidateFile]:
        allowed_exts_str = ", ".join(allowed_exts)
        prompt = PROMPT_TEMPLATE.format(
            query=query,
            mc_version=mc_version,
            allowed_exts=allowed_exts_str,
        )

        models = [self._primary_model]
        if self._fallback_model and self._fallback_model != self._primary_model:
            models.append(self._fallback_model)

        last_error: Exception | None = None
        for model in models:
            client = AsyncBrowserUse(api_key=self._api_key, timeout=float(self._timeout_seconds))
            try:
                logger.info(
                    "Starting browser-use task",
                    extra={
                        "model": model,
                        "query": query,
                        "mc_version": mc_version,
                        "timeout_seconds": self._timeout_seconds,
                    },
                )
                task = client.run(
                    task=prompt,
                    output_schema=BrowsedCandidates,
                    llm=model,
                    max_steps=24,
                    allowed_domains=["planetminecraft.com", "www.planetminecraft.com"],
                    session_settings=self._session_settings(),
                )
                await _emit_progress(on_progress, f"browser-use: started ({model})")
                async for step in task:
                    await _emit_progress(on_progress, _render_task_step(step))
                if task.result is None:
                    raise RuntimeError("browser-use task finished without result")

                task_id = task.task_id
                if task_id is None:
                    raise RuntimeError("browser-use task id missing")

                output = task.result.output
                if output is None:
                    await _emit_progress(on_progress, "browser-use: no candidates found")
                    return []
                if not isinstance(output, BrowsedCandidates):
                    raise TypeError("browser-use returned unexpected output payload")

                output_files = await self._fetch_output_files(client, task_id, task.result.task.output_files)
                candidates = self._filter_candidates(
                    candidates=output.candidates,
                    output_files=output_files,
                    allowed_exts=allowed_exts,
                    task_id=task_id,
                )
                await _emit_progress(on_progress, f"browser-use: {len(candidates)} downloaded candidates")
                return candidates
            except Exception as exc:
                last_error = exc
                await _emit_progress(on_progress, f"browser-use: model {model} failed ({exc})")
            finally:
                await client.close()

        if last_error is not None:
            raise last_error
        return []

    @observe()
    async def _fetch_output_files(
        self,
        client: AsyncBrowserUse,
        task_id: str,
        output_files: list[Any],
    ) -> dict[str, TaskOutputFileResponse]:
        logger.info(
            "Fetching browser-use output files",
            extra={"task_id": task_id, "output_file_count": len(output_files)},
        )
        files_by_name: dict[str, TaskOutputFileResponse] = {}
        for output_file in output_files:
            file_response = await client.files.task_output(task_id, str(output_file.id))
            files_by_name[_normalize_filename(file_response.file_name)] = file_response
        return files_by_name

    def _session_settings(self) -> SessionSettings | None:
        if not self._profile_id:
            return None
        return SessionSettings(profile_id=self._profile_id)

    def _filter_candidates(
        self,
        candidates: list[BrowsedCandidate],
        output_files: dict[str, TaskOutputFileResponse],
        allowed_exts: tuple[str, ...],
        task_id: str,
    ) -> list[CandidateFile]:
        filtered: list[CandidateFile] = []
        for candidate in candidates:
            candidate_filename = Path(candidate.filename).name
            candidate_ext = Path(candidate_filename).suffix.lower()
            if candidate_ext not in allowed_exts:
                continue

            normalized_name = _normalize_filename(candidate_filename)
            output_file = output_files.get(normalized_name)
            if output_file is None:
                continue

            filtered.append(
                CandidateFile(
                    source="browser_use",
                    canonical_url=candidate.canonical_url,
                    download_url=output_file.download_url,
                    filename=candidate_filename,
                    title=candidate.title,
                    score=candidate.score,
                    browser_task_id=task_id,
                    browser_output_file_id=str(output_file.id),
                )
            )

        return filtered

    def _filter_skill_candidates(
        self,
        candidates: list[BrowsedCandidate],
        allowed_exts: tuple[str, ...],
    ) -> list[CandidateFile]:
        filtered: list[CandidateFile] = []
        for candidate in candidates:
            candidate_filename = Path(candidate.filename).name
            candidate_ext = Path(candidate_filename).suffix.lower()
            if candidate_ext not in allowed_exts:
                continue
            if candidate.download_url is None:
                continue
            filtered.append(
                CandidateFile(
                    source="browser_use",
                    canonical_url=candidate.canonical_url,
                    download_url=candidate.download_url,
                    filename=candidate_filename,
                    title=candidate.title,
                    score=candidate.score,
                )
            )
        return filtered


async def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is None:
        return
    await callback(message)


def _render_task_step(step: Any) -> str:
    number = getattr(step, "number", None)
    goal = getattr(step, "next_goal", "") or ""
    url = getattr(step, "url", "") or ""
    actions = getattr(step, "actions", None) or []
    actions_text = ", ".join(actions[:2]) if actions else "no-actions"
    if number is None:
        return f"browser-use: {goal} [{actions_text}]"
    if url:
        return f"browser-use step {number}: {goal} ({actions_text}) @ {url}"
    return f"browser-use step {number}: {goal} ({actions_text})"


def _normalize_filename(value: str) -> str:
    return re.sub(r"\s+", "", Path(value).name.lower())
