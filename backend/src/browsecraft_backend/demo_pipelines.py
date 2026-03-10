from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, unquote, urljoin, urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from .models import AsyncJobAcceptedResponse, ChatAcceptedResponse, ChatRequest, ImagineRequest, SearchRequest
from .schematic_parser import parse_schematic
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = (".schem", ".litematic", ".schematic")
_PLANET_MINECRAFT_ROOT = "https://www.planetminecraft.com"
_PLANET_MINECRAFT_SEARCH_TEMPLATE = "https://www.planetminecraft.com/resources/projects/?keywords={query}"
_PLANET_MINECRAFT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
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


class _CloudflareBlocked(RuntimeError):
    pass


class _SearchCandidate(BaseModel):
    canonical_url: str
    title: str
    score: float = Field(default=0.0)
    filename: str = ""
    download_url: str | None = None


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
        chat_submitter: Callable[[ChatRequest], Awaitable[ChatAcceptedResponse]] | None,
    ) -> None:
        self._websocket_manager = websocket_manager
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
                message=f"Built {placed_count} blocks from {downloaded.title}.",
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
        candidates = await self._discover_candidates(query)
        if not candidates:
            raise RuntimeError("No schematic candidates found")

        await status_callback(f"✅ Found: {candidates[0].title}")
        await status_callback("📥 Downloading schematic...")

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                return await _download_candidate_with_playwright(candidate)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to download candidate %s (%s): %s",
                    candidate.title,
                    candidate.canonical_url,
                    exc,
                )
        raise RuntimeError(f"All candidates failed download; last error: {last_error}")

    async def _discover_candidates(self, query: str) -> list[_SearchCandidate]:
        try:
            candidates = await _discover_candidates_http(query)
        except _CloudflareBlocked:
            logger.info("HTTP discovery blocked by Cloudflare; falling back to Playwright")
            candidates = await _discover_candidates_with_playwright(query)

        if candidates:
            return candidates

        return await _discover_candidates_with_playwright(query)

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


async def _discover_candidates_http(query: str) -> list[_SearchCandidate]:
    url = _PLANET_MINECRAFT_SEARCH_TEMPLATE.format(query=quote_plus(query))
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0, headers=_PLANET_MINECRAFT_HEADERS) as client:
        response = await client.get(url)
        if response.status_code >= 400:
            raise _CloudflareBlocked(f"HTTP status {response.status_code}")
        html = response.text

    if _looks_like_cloudflare_block(html):
        raise _CloudflareBlocked("Cloudflare challenge page")

    return _parse_project_candidates_from_html(html)


async def _discover_candidates_with_playwright(query: str) -> list[_SearchCandidate]:
    async with _playwright_browser() as page:
        search_url = _PLANET_MINECRAFT_SEARCH_TEMPLATE.format(query=quote_plus(query))
        await page.goto(search_url, wait_until="domcontentloaded", timeout=20_000)
        html = await page.content()
        if _looks_like_cloudflare_block(html):
            raise RuntimeError("Planet Minecraft blocked automated access during discovery")

        candidates = await page.evaluate(
            """
            () => {
              const anchors = Array.from(document.querySelectorAll('a[href*="/project/"]'));
              const seen = new Set();
              const rows = [];
              for (const anchor of anchors) {
                const href = anchor.href;
                if (!href || !href.includes('/project/')) continue;
                if (seen.has(href)) continue;
                seen.add(href);
                const text = (anchor.textContent || anchor.getAttribute('title') || '').trim();
                if (!text) continue;
                rows.push({ href, text });
                if (rows.length >= 10) break;
              }
              return rows;
            }
            """
        )
    parsed: list[_SearchCandidate] = []
    for index, item in enumerate(candidates):
        href = str(item.get("href", ""))
        if not href:
            continue
        parsed.append(
            _SearchCandidate(
                canonical_url=href,
                title=str(item.get("text", "Project")).strip() or "Project",
                score=max(0.0, 1.0 - (index * 0.05)),
            )
        )
    return parsed


async def _download_candidate_with_playwright(candidate: _SearchCandidate) -> _DownloadedSchematic:
    with tempfile.TemporaryDirectory(prefix="browsecraft-search-") as temp_dir:
        temp_path = Path(temp_dir)
        async with _playwright_browser(accept_downloads=True) as page:
            await page.goto(candidate.canonical_url, wait_until="domcontentloaded", timeout=20_000)
            html = await page.content()
            if _looks_like_cloudflare_block(html):
                raise RuntimeError("Planet Minecraft blocked download page")

            downloaded = await _attempt_playwright_download(page, candidate, temp_path)
            if downloaded is None:
                refreshed_html = await page.content()
                if _looks_like_login_gate(refreshed_html):
                    raise RuntimeError("Login required to download schematic from Planet Minecraft")
                raise RuntimeError("Unable to find a downloadable schematic on the project page")

        persistent = Path(tempfile.gettempdir()) / f"browsecraft-{uuid4()}{downloaded.suffix.lower()}"
        persistent.write_bytes(downloaded.read_bytes())
        return _DownloadedSchematic(
            path=persistent,
            title=candidate.title,
            source_url=candidate.canonical_url,
        )


async def _attempt_playwright_download(page: Any, candidate: _SearchCandidate, target_dir: Path) -> Path | None:
    selectors = [
        "a[href*='.schem']",
        "a[href*='.litematic']",
        "a[href*='.schematic']",
        "a[href*='/download/']",
        "a:has-text('Download')",
        "button:has-text('Download')",
    ]

    for selector in selectors:
        locator = page.locator(selector)
        count = await locator.count()
        if count <= 0:
            continue
        for index in range(count):
            try:
                async with page.expect_download(timeout=8_000) as download_info:
                    await locator.nth(index).click(timeout=4_000)
                download = await download_info.value
            except Exception:
                continue

            suggested = _filename_from_url(download.suggested_filename)
            if not _has_allowed_extension(suggested):
                continue
            destination = target_dir / suggested
            await download.save_as(str(destination))
            return destination

    direct_url = await _first_direct_schematic_url(page)
    if direct_url is None:
        return None

    response = await page.request.get(direct_url, headers={"Referer": candidate.canonical_url})
    if not response.ok:
        raise RuntimeError(f"Download request failed with HTTP {response.status}")

    name = _filename_from_url(direct_url)
    if not _has_allowed_extension(name):
        raise RuntimeError("Direct URL did not point to a supported schematic extension")

    destination = target_dir / name
    destination.write_bytes(await response.body())
    return destination


async def _first_direct_schematic_url(page: Any) -> str | None:
    raw = await page.evaluate(
        """
        () => {
          const anchors = Array.from(document.querySelectorAll('a[href]'));
          for (const anchor of anchors) {
            const href = anchor.href;
            if (!href) continue;
            const lower = href.toLowerCase();
            if (lower.includes('.schem') || lower.includes('.litematic') || lower.includes('.schematic')) {
              return href;
            }
          }
          return null;
        }
        """
    )
    if raw is None:
        return None
    return str(raw)


def _parse_project_candidates_from_html(html: str) -> list[_SearchCandidate]:
    anchors = re.findall(r'href=["\'](/project/[^"\']+)["\']', html)
    deduped: list[str] = []
    seen: set[str] = set()
    for href in anchors:
        full = urljoin(_PLANET_MINECRAFT_ROOT, href)
        if full in seen:
            continue
        seen.add(full)
        deduped.append(full)
        if len(deduped) >= 10:
            break

    candidates: list[_SearchCandidate] = []
    for index, canonical_url in enumerate(deduped):
        title = _title_from_project_url(canonical_url)
        candidates.append(
            _SearchCandidate(
                canonical_url=canonical_url,
                title=title,
                score=max(0.0, 1.0 - (index * 0.05)),
            )
        )
    return candidates


def _looks_like_cloudflare_block(html: str) -> bool:
    lowered = html.lower()
    return (
        "attention required! | cloudflare" in lowered
        or "sorry, you have been blocked" in lowered
        or "cf-error-details" in lowered
    )


def _looks_like_login_gate(html: str) -> bool:
    lowered = html.lower()
    return "log in" in lowered and "download" in lowered


def _title_from_project_url(url: str) -> str:
    path = urlsplit(url).path.strip("/")
    if not path:
        return "Project"
    slug = path.split("/")[-1]
    return slug.replace("-", " ").strip() or "Project"


class _BrowserPageContext:
    def __init__(self, *, accept_downloads: bool = False) -> None:
        self._accept_downloads = accept_downloads
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def __aenter__(self) -> Any:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(accept_downloads=self._accept_downloads)
        self._page = await self._context.new_page()
        return self._page

    async def __aexit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()


def _playwright_browser(*, accept_downloads: bool = False) -> _BrowserPageContext:
    return _BrowserPageContext(accept_downloads=accept_downloads)


def _filename_from_url(value: str | None) -> str:
    if not value:
        return ""
    return unquote(urlsplit(value).path).rsplit("/", maxsplit=1)[-1]


def _has_allowed_extension(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in _ALLOWED_EXTENSIONS)


def _absolute_placements(
    *,
    relative_placements: list[dict[str, Any]],
    player_position: dict[str, Any],
) -> list[dict[str, Any]]:
    base_x = int(player_position["block_x"])
    base_y = int(player_position.get("ground_y", int(player_position["block_y"]) - 1))
    base_z = int(player_position["block_z"])
    facing = str(player_position["facing"]).lower()

    if facing not in _FACING_TO_FORWARD_OFFSET:
        raise RuntimeError(f"Unsupported player facing: {facing}")

    forward_x, forward_z = _FACING_TO_FORWARD_OFFSET[facing]
    rotation = _FACING_TO_ROTATION[facing]
    anchor_x = base_x + (forward_x * _SEARCH_FORWARD_BLOCKS)
    anchor_z = base_z + (forward_z * _SEARCH_FORWARD_BLOCKS)

    min_dx = min(int(item["dx"]) for item in relative_placements)
    max_dx = max(int(item["dx"]) for item in relative_placements)
    min_dz = min(int(item["dz"]) for item in relative_placements)
    max_dz = max(int(item["dz"]) for item in relative_placements)
    min_dy = min(int(item["dy"]) for item in relative_placements)

    center_dx = (min_dx + max_dx) // 2
    center_dz = (min_dz + max_dz) // 2

    absolute: list[dict[str, Any]] = []
    for placement in relative_placements:
        dx = int(placement["dx"]) - center_dx
        dy = int(placement["dy"]) - min_dy
        dz = int(placement["dz"]) - center_dz
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
