from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4
from urllib.parse import unquote, urlsplit

import httpx
from browser_use_sdk import AsyncBrowserUse as AsyncBrowserUseV2
from pydantic import BaseModel, Field

from .models import AsyncJobAcceptedResponse, ChatAcceptedResponse, ChatRequest, ImagineRequest, SearchRequest
from .schematic_parser import parse_schematic
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = (".schem", ".litematic", ".schematic")
_PLANET_MINECRAFT_ALLOWED_DOMAINS = ("planetminecraft.com", "www.planetminecraft.com")
_PLANET_MINECRAFT_MAX_STEPS = 10
_PLANET_MINECRAFT_START_URL = "https://www.planetminecraft.com/"
_PLANET_MINECRAFT_SYSTEM_GUIDANCE = """Use Planet Minecraft only.

Rules:
1. Stay on Planet Minecraft and only follow schematic project pages.
2. Return only direct schematic files with extensions .schem, .litematic, or .schematic.
3. Never return /download/worldmap/, world-save, or archive/worldmap endpoints.
4. Do not return page links. Every candidate must include a direct downloadable file URL.
5. Return no more than 3 candidates, sorted by quality.
"""
_PLANET_MINECRAFT_PROMPT = """Use Planet Minecraft only.

Search for downloadable Minecraft Java schematics matching this query: {query}
Allowed file extensions: .schem, .litematic, .schematic

Workflow:
1. Search Planet Minecraft.
2. Open promising project pages.
3. Find direct schematic downloads.
4. Skip map/world-save zip downloads.
5. Return best candidates sorted by quality.

Return JSON:
{{
  "candidates": [
    {{
      "canonical_url": "<project_url>",
      "filename": "<download_filename>",
      "title": "<project_title>",
      "score": <float>,
      "download_url": "<direct_download_url>"
    }}
  ]
}}
"""


def _build_search_task(query: str) -> str:
    return f"{_PLANET_MINECRAFT_PROMPT.format(query=query)}\n\n{_PLANET_MINECRAFT_SYSTEM_GUIDANCE}"
_PLANET_MINECRAFT_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}

_V3_MODELS = ("bu-mini", "bu-max")
_V2_FALLBACK_LLM = "browser-use-llm"
_PLACE_BLOCK_BATCH_SIZE = 256
_SEARCH_FORWARD_BLOCKS = 10
_FACING_TO_FORWARD_OFFSET = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
_FACING_TO_ROTATION = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}


def _filename_from_url(value: str | None) -> str:
    if not value:
        return ""
    return unquote(urlsplit(value).path).rsplit("/", maxsplit=1)[-1]


def _normalize_browser_use_model_v2(value: str | None) -> str:
    if value is None:
        return _V2_FALLBACK_LLM
    if value in _V3_MODELS:
        return _V2_FALLBACK_LLM
    return value


def _is_broken_download_url(url: str | None) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return "/worldmap/" in lowered or "world-save" in lowered or "world_save" in lowered


class _SearchCandidate(BaseModel):
    canonical_url: str
    filename: str
    title: str
    score: float = Field(default=0.0)
    download_url: str | None = None


class _SearchCandidates(BaseModel):
    candidates: list[_SearchCandidate] = Field(default_factory=list)


@dataclass(slots=True)
class _DownloadedSchematic:
    path: Path
    title: str
    source_url: str


class DemoPipelines:
    def __init__(
        self,
        *,
        websocket_manager: WebSocketManager,
        browser_use_api_key: str | None,
        browser_use_llm: str,
        browser_use_skill_id: str | None,
        chat_submitter: Callable[[ChatRequest], Awaitable[ChatAcceptedResponse]] | None,
    ) -> None:
        self._websocket_manager = websocket_manager
        self._browser_use_api_key = browser_use_api_key
        self._browser_use_llm = browser_use_llm
        self._browser_use_skill_id = browser_use_skill_id
        self._chat_submitter = chat_submitter
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit_search(self, request: SearchRequest) -> AsyncJobAcceptedResponse:
        job_id = str(uuid4())
        task = asyncio.create_task(
            self._run_search(
                job_id=job_id,
                client_id=request.client_id,
                query=request.query,
            )
        )
        self._track_task(task)
        return AsyncJobAcceptedResponse(job_id=job_id, status="accepted")

    async def submit_imagine(self, request: ImagineRequest) -> AsyncJobAcceptedResponse:
        job_id = str(uuid4())
        task = asyncio.create_task(
            self._run_imagine(
                job_id=job_id,
                client_id=request.client_id,
                prompt=request.prompt,
            )
        )
        self._track_task(task)
        return AsyncJobAcceptedResponse(job_id=job_id, status="accepted")

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except Exception:
            logger.exception("Pipeline task failed")

    async def _run_search(self, *, job_id: str, client_id: str, query: str) -> None:
        try:
            search_origin = await self._websocket_manager.request_tool(
                client_id=client_id,
                tool_name="player_position",
                params={},
            )
            await self._emit_status(client_id, "🔎 Searching Planet Minecraft...")
            downloaded = await self._search_and_download(
                query=query,
                status_callback=lambda status: self._emit_status(client_id, status),
            )
            placements = parse_schematic(downloaded.path)
            placed_count = await self._place_blocks_from_schematic(
                client_id=client_id,
                placements=placements,
                search_origin=search_origin,
                status_callback=lambda status: self._emit_status(client_id, status),
            )
            await self._emit_status(client_id, "✓ Done")
            await self._emit_chat_response(
                client_id=client_id,
                job_id=job_id,
                message=(
                    f"Built {placed_count} blocks from {downloaded.title}."
                ),
            )
        except Exception as exc:
            logger.exception("Search pipeline failed for client=%s", client_id)
            await self._emit_status(client_id, f"✗ Search failed: {exc}")
            await self._emit_chat_response(
                client_id=client_id,
                job_id=job_id,
                message=f"Search failed: {exc}",
            )

    async def _run_imagine(self, *, job_id: str, client_id: str, prompt: str) -> None:
        if self._chat_submitter is not None:
            await self._emit_status(client_id, "🎨 Designing creative structure...")
            await self._chat_submitter(
                ChatRequest(
                    client_id=client_id,
                    mode="plan_fast",
                    message=(
                        f"Plan and preview a creative detailed build for: {prompt}. "
                        "Output the complete structure in one set_plan call. "
                        "Use varied materials, depth, layered silhouettes, decorative trim, windows/overhangs, "
                        "and stairs/slabs for detail."
                    ),
                )
            )
            return

        message = "Imagine is unavailable because chat orchestration is not configured."
        await self._emit_chat_response(client_id=client_id, job_id=job_id, message=message)

    async def _emit_status(self, client_id: str, status: str) -> None:
        await self._websocket_manager.send_payload(
            client_id,
            {
                "type": "chat.tool_status",
                "payload": {"status": status},
            },
        )

    async def _emit_chat_response(self, *, client_id: str, job_id: str, message: str) -> None:
        await self._websocket_manager.send_payload(
            client_id,
            {
                "type": "chat.response",
                "chat_id": job_id,
                "payload": {"message": message},
            },
        )

    async def _search_and_download(
        self,
        *,
        query: str,
        status_callback: Callable[[str], Awaitable[None]],
    ) -> _DownloadedSchematic:
        if not self._browser_use_api_key:
            raise RuntimeError("BROWSER_USE_API_KEY is required for /search")

        browser_v2 = AsyncBrowserUseV2(api_key=self._browser_use_api_key, timeout=300.0)
        try:
            result = await self._search_via_browser_use_v2(
                browser=browser_v2,
                query=query,
            )
            return await _finalize_download(
                browser=browser_v2,
                result=result,
                status_callback=status_callback,
            )
        finally:
            await browser_v2.close()

    async def _search_via_browser_use_v2(
        self,
        *,
        browser: AsyncBrowserUseV2,
        query: str,
    ) -> Any:
        search_kwargs = {
            "task": _build_search_task(query),
            "output_schema": _SearchCandidates,
            "llm": _normalize_browser_use_model_v2(self._browser_use_llm),
            "start_url": _PLANET_MINECRAFT_START_URL,
            "max_steps": _PLANET_MINECRAFT_MAX_STEPS,
            "allowed_domains": list(_PLANET_MINECRAFT_ALLOWED_DOMAINS),
            "flash_mode": True,
            "thinking": False,
            "vision": False,
            "system_prompt_extension": _PLANET_MINECRAFT_SYSTEM_GUIDANCE,
        }
        if self._browser_use_skill_id is not None:
            search_kwargs["skill_ids"] = [self._browser_use_skill_id]
        return await browser.run(**search_kwargs)

    async def _place_blocks_from_schematic(
        self,
        *,
        client_id: str,
        placements: list[dict[str, Any]],
        search_origin: dict[str, Any],
        status_callback: Callable[[str], Awaitable[None]],
    ) -> int:
        if not placements:
            raise RuntimeError("Parsed schematic did not contain placeable blocks")

        await status_callback(f"🧭 Using /search position ({_SEARCH_FORWARD_BLOCKS} blocks ahead)...")
        absolute_placements = _absolute_placements(
            relative_placements=placements,
            player_position=search_origin,
        )

        total_placements = len(absolute_placements)
        batch_count = (total_placements + _PLACE_BLOCK_BATCH_SIZE - 1) // _PLACE_BLOCK_BATCH_SIZE
        placed_count = 0
        for batch_index in range(batch_count):
            start = batch_index * _PLACE_BLOCK_BATCH_SIZE
            end = min(total_placements, start + _PLACE_BLOCK_BATCH_SIZE)
            await status_callback(f"🧱 Placing blocks {batch_index + 1}/{batch_count}...")
            result = await self._websocket_manager.request_tool(
                client_id=client_id,
                tool_name="place_blocks",
                params={"placements": absolute_placements[start:end]},
                timeout=120.0,
            )
            placed_count += int(result["placed_count"])
        return placed_count


async def _finalize_download(
    *,
    browser: Any,
    result: Any,
    status_callback: Callable[[str], Awaitable[None]],
) -> _DownloadedSchematic:
    task_output_urls = await _task_output_urls_by_filename(browser=browser, result=result)
    candidates = _ordered_candidates(result.output, task_output_urls)
    if not candidates:
        raise RuntimeError("No schematic candidates found")
    await status_callback(f"✅ Found: {candidates[0].title}")
    await status_callback("📥 Downloading schematic...")

    last_error: Exception | None = None
    with tempfile.TemporaryDirectory(prefix="browsecraft-search-") as temp_dir:
        for candidate in candidates:
            try:
                downloaded = await _download_candidate(
                    browser=browser,
                    result=result,
                    candidate=candidate,
                    target_dir=Path(temp_dir),
                )
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to download candidate %s (%s): %s",
                    candidate.filename,
                    candidate.canonical_url,
                    exc,
                )
                continue
        else:
            raise RuntimeError(f"All candidates failed download; last error: {last_error}")
        persistent = Path(tempfile.gettempdir()) / f"browsecraft-{uuid4()}{downloaded.path.suffix.lower()}"
        persistent.write_bytes(downloaded.path.read_bytes())
        return _DownloadedSchematic(
            path=persistent,
            title=downloaded.title,
            source_url=downloaded.source_url,
        )


def _best_candidate(
    output: _SearchCandidates | str | None,
    task_output_urls: dict[str, str] | None = None,
) -> _SearchCandidate | None:
    if output is None or isinstance(output, str):
        return None
    candidates = _ordered_candidates(output, task_output_urls)
    if not candidates:
        return None
    return candidates[0]


def _ordered_candidates(
    output: _SearchCandidates | str | None,
    task_output_urls: dict[str, str] | None = None,
) -> list[_SearchCandidate]:
    if output is None or isinstance(output, str):
        return []
    if not output.candidates:
        return []

    sorted_candidates = sorted(output.candidates, key=lambda item: item.score, reverse=True)
    result: list[_SearchCandidate] = []
    for candidate in sorted_candidates:
        candidate_with_download = _candidate_with_task_output_url(candidate, task_output_urls)
        if not _is_candidate_file(candidate_with_download):
            continue
        result.append(candidate_with_download)
    return _single_best_candidate(result)


def _candidate_with_task_output_url(
    candidate: _SearchCandidate,
    task_output_urls: dict[str, str] | None,
) -> _SearchCandidate:
    if candidate.download_url and not _is_broken_download_url(candidate.download_url):
        return candidate
    if not task_output_urls:
        return candidate
    filename = _filename_from_candidate(candidate)
    if not filename:
        return candidate
    download_url = task_output_urls.get(_normalize_filename(filename))
    if download_url is None:
        return candidate
    return candidate.model_copy(update={"download_url": download_url})


def _single_best_candidate(candidates: list[_SearchCandidate]) -> list[_SearchCandidate]:
    if not candidates:
        return []
    best = max(candidates, key=lambda candidate: candidate.score)
    return [best]


async def _task_output_urls_by_filename(
    *,
    browser: Any,
    result: Any,
) -> dict[str, str]:
    task = getattr(result, "task", None)
    task_id = getattr(task, "id", None)
    output_files = getattr(task, "output_files", None)
    if task is None or task_id is None or output_files is None:
        return {}
    if not hasattr(browser, "files"):
        return {}

    urls: dict[str, str] = {}
    for output_file in output_files:
        file_name = _read_output_file_name(output_file)
        if not file_name:
            continue
        file_url = _read_output_file_url(output_file)
        if file_url is None:
            file_id = getattr(output_file, "id", None)
            if file_id is None:
                continue
            task_output = await browser.files.task_output(str(task_id), str(file_id))
            file_url = _read_output_file_url(task_output)
        if file_url is None:
            continue
        urls[_normalize_filename(file_name)] = file_url
    return urls


def _read_output_file_name(output_file: Any) -> str:
    for key in ("file_name", "path", "name", "filename"):
        value = getattr(output_file, key, None)
        if isinstance(value, str):
            return value
    return ""


def _read_output_file_url(output_file: Any) -> str | None:
    for key in ("download_url", "url"):
        value = getattr(output_file, key, None)
        if isinstance(value, str):
            return value
    return None


def _is_candidate_file(candidate: _SearchCandidate) -> bool:
    filename = _filename_from_candidate(candidate).lower()
    if _is_broken_download_url(candidate.download_url):
        return False
    return any(filename.endswith(ext) for ext in _ALLOWED_EXTENSIONS)


async def _download_candidate(
    *,
    browser: Any,
    result: Any,
    candidate: _SearchCandidate,
    target_dir: Path,
) -> _DownloadedSchematic:
    target_dir.mkdir(parents=True, exist_ok=True)

    download_url = candidate.download_url
    filename = _filename_from_candidate(candidate)
    if not filename:
        raise RuntimeError(f"Candidate did not include a usable filename: {candidate.title}")
    if download_url:
        try:
            file_path = await _download_via_url(
                browser_http_client_factory=httpx.AsyncClient,
                download_url=download_url,
                filename=filename,
                target_dir=target_dir,
                referer=candidate.canonical_url,
            )
            return _DownloadedSchematic(path=file_path, title=candidate.title, source_url=candidate.canonical_url)
        except Exception as exc:
            if not isinstance(exc, httpx.HTTPStatusError):
                raise
            logger.warning(
                "Direct schematic download failed for %s, trying session files",
                candidate.filename,
            )

    return await _download_from_session_files(
        browser=browser,
        result=result,
        candidate=candidate,
        filename=filename,
        target_dir=target_dir,
    )


async def _download_via_url(
    *,
    browser_http_client_factory: Callable[..., httpx.AsyncClient],
    download_url: str,
    filename: str,
    target_dir: Path,
    referer: str | None = None,
) -> Path:
    if _is_broken_download_url(download_url):
        raise RuntimeError("Download URL targets map/world endpoint instead of schematic file")

    headers = _PLANET_MINECRAFT_DOWNLOAD_HEADERS.copy()
    if referer is not None:
        headers["Referer"] = referer

    async with browser_http_client_factory(follow_redirects=True, timeout=120.0, headers=headers) as http_client:
        response = await http_client.get(download_url)
        response.raise_for_status()
        file_path = target_dir / filename
        file_path.write_bytes(response.content)
        return file_path


async def _download_from_session_files(
    *,
    browser: Any,
    result: Any,
    candidate: _SearchCandidate,
    filename: str,
    target_dir: Path,
) -> _DownloadedSchematic:
    file_target = None
    result_task = getattr(result, "task", None)
    result_session = getattr(result, "session", None)
    task_output_files = getattr(result_task, "output_files", None) if result_task is not None else None

    if result_session is not None:
        file_list = await browser.sessions.files(str(result_session.id), include_urls=True)
        file_views = list(file_list.files or [])
        file_name_key = "path"
        file_url_key = "url"
    elif task_output_files is not None:
        file_views = list(task_output_files)
        file_name_key = "file_name"
        file_url_key = "download_url"
    else:
        raise RuntimeError("Candidate download failed and session/task output files were unavailable")

    def _read_file_name(file_view: Any) -> str:
        for key in (file_name_key, "path", "name", "filename"):
            value = getattr(file_view, key, None)
            if isinstance(value, str):
                return value
        return ""

    def _read_file_url(file_view: Any) -> str | None:
        for key in (file_url_key, "url", "download_url"):
            value = getattr(file_view, key, None)
            if isinstance(value, str):
                return value
        return None

    def _matches_file(file_view: Any, target: str) -> bool:
        candidate_name = _read_file_name(file_view).lower()
        if not candidate_name:
            return False
        return candidate_name.lower().endswith(target.lower())

    for file_view in file_views:
        if _matches_file(file_view, filename):
            file_target = file_view
            break

    if file_target is None:
        for file_view in file_views:
            file_name = _read_file_name(file_view).lower()
            if any(file_name.endswith(ext) for ext in _ALLOWED_EXTENSIONS):
                file_target = file_view
                break

    if file_target is None:
        raise RuntimeError("No suitable output file was found in browser session")

    download_url = _read_file_url(file_target)
    if download_url is None and result_task is not None:
        file_id = getattr(file_target, "id", None)
        if file_id is None:
            raise RuntimeError("Output file has no downloadable URL")

        if not hasattr(browser, "files"):
            raise RuntimeError("Output file has no downloadable URL")

        task_output = await browser.files.task_output(str(result_task.id), str(file_id))
        download_url = _read_file_url(task_output)

    if download_url is None:
        raise RuntimeError("No suitable output file URL was found in browser session")

    if candidate.filename and candidate.filename.lower().endswith(".zip"):
        raise RuntimeError("Candidate file is a map/world-save archive")

    file_path = await _download_via_url(
        browser_http_client_factory=httpx.AsyncClient,
        download_url=download_url,
        filename=filename,
        target_dir=target_dir,
    )
    return _DownloadedSchematic(path=file_path, title=candidate.title, source_url=candidate.canonical_url)


def _filename_from_candidate(candidate: _SearchCandidate) -> str:
    if candidate.filename.strip():
        extracted = _filename_from_url(candidate.filename)
        if extracted:
            return extracted
    return _filename_from_url(candidate.download_url)


def _normalize_filename(value: str) -> str:
    return "".join(_filename_from_url(value).lower().split())


def _absolute_placements(
    *,
    relative_placements: list[dict[str, Any]],
    player_position: dict[str, Any],
) -> list[dict[str, Any]]:
    base_x = int(player_position["block_x"])
    base_y = int(player_position["block_y"])
    base_z = int(player_position["block_z"])
    facing = str(player_position["facing"]).lower()

    if facing not in _FACING_TO_FORWARD_OFFSET:
        raise RuntimeError(f"Unsupported player facing: {facing}")

    forward_x, forward_z = _FACING_TO_FORWARD_OFFSET[facing]
    rotation = _FACING_TO_ROTATION[facing]
    anchor_x = base_x + (forward_x * _SEARCH_FORWARD_BLOCKS)
    anchor_z = base_z + (forward_z * _SEARCH_FORWARD_BLOCKS)

    absolute: list[dict[str, Any]] = []
    for placement in relative_placements:
        dx = int(placement["dx"])
        dy = int(placement["dy"])
        dz = int(placement["dz"])
        block_id = str(placement["block_id"])
        rotated_x, rotated_z = _rotate(dx, dz, rotation)
        absolute.append(
            {
                "x": anchor_x + rotated_x,
                "y": base_y + dy,
                "z": anchor_z + rotated_z,
                "block_id": block_id,
            }
        )
    return absolute


def _rotate(x: int, z: int, rotation_quarters: int) -> tuple[int, int]:
    if rotation_quarters == 1:
        return -z, x
    if rotation_quarters == 2:
        return -x, -z
    if rotation_quarters == 3:
        return z, -x
    return x, z
