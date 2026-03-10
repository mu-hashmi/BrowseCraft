from __future__ import annotations

from typing import Any, Literal

import httpx


class ConvexHttpClient:
    def __init__(
        self,
        base_url: str,
        http_client: httpx.AsyncClient,
        access_key: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._access_key = access_key

    async def query(self, path: str, args: dict[str, Any] | None = None) -> Any:
        return await self._call("query", path, args or {})

    async def mutation(self, path: str, args: dict[str, Any] | None = None) -> Any:
        return await self._call("mutation", path, args or {})

    async def action(self, path: str, args: dict[str, Any] | None = None) -> Any:
        return await self._call("action", path, args or {})

    async def _call(
        self,
        endpoint: Literal["query", "mutation", "action"],
        path: str,
        args: dict[str, Any],
    ) -> Any:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._access_key:
            headers["Authorization"] = f"Convex {self._access_key}"

        response = await self._http_client.post(
            f"{self._base_url}/api/{endpoint}",
            headers=headers,
            json={
                "path": path,
                "args": args,
                "format": "json",
            },
        )
        response.raise_for_status()

        payload = response.json()
        status = payload.get("status")
        if status == "success":
            return payload.get("value")

        error_message = payload.get("errorMessage") or "Convex request failed"
        raise RuntimeError(error_message)
