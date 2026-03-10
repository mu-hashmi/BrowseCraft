from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lmnr import observe
from supermemory import AsyncSupermemory


@dataclass(slots=True, frozen=True)
class SupermemorySearchResult:
    text: str
    similarity: float | None


@dataclass(slots=True, frozen=True)
class SupermemoryProfileContext:
    static: tuple[str, ...]
    dynamic: tuple[str, ...]


class SupermemoryClient:
    def __init__(
        self,
        api_key: str,
    ) -> None:
        self._client = AsyncSupermemory(api_key=api_key)

    @observe()
    async def search_memories(
        self,
        query: str,
        *,
        container_tag: str,
        limit: int = 5,
    ) -> list[SupermemorySearchResult]:
        response = await self._client.search.memories(
            q=query,
            container_tag=container_tag,
            limit=limit,
            search_mode="memories",
        )

        results: list[SupermemorySearchResult] = []
        for hit in response.results:
            text = hit.memory or hit.chunk
            if text is None:
                continue
            results.append(SupermemorySearchResult(text=text, similarity=hit.similarity))
        return results

    @observe()
    async def store_memory(
        self,
        content: str,
        *,
        container_tag: str,
        metadata: dict[str, Any],
    ) -> None:
        await self._client.add(
            content=content,
            container_tag=container_tag,
            metadata=_normalize_metadata(metadata),
        )

    @observe()
    async def profile_context(self, container_tag: str) -> SupermemoryProfileContext:
        response = await self._client.profile(container_tag=container_tag)
        return SupermemoryProfileContext(
            static=tuple(response.profile.static),
            dynamic=tuple(response.profile.dynamic),
        )

    async def close(self) -> None:
        await self._client.close()


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, str | float | bool | list[str]]:
    normalized: dict[str, str | float | bool | list[str]] = {}
    for key, value in metadata.items():
        normalized[str(key)] = _normalize_metadata_value(value)
    return normalized


def _normalize_metadata_value(value: Any) -> str | float | bool | list[str]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return json.dumps(value, sort_keys=True)
