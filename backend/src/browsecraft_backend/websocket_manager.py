from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingToolRequest:
    client_id: str
    future: asyncio.Future[dict[str, Any]]


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._pending_tool_requests: dict[str, PendingToolRequest] = {}
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[client_id] = websocket

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            self._connections.pop(client_id, None)
            pending_ids = [
                request_id
                for request_id, pending in self._pending_tool_requests.items()
                if pending.client_id == client_id
            ]
            for request_id in pending_ids:
                pending = self._pending_tool_requests.pop(request_id)
                if not pending.future.done():
                    pending.future.set_exception(ConnectionError(f"WebSocket disconnected for client {client_id}"))

    async def send_payload(self, client_id: str, payload: dict[str, Any]) -> None:
        websocket = self._connections[client_id]
        await websocket.send_json(payload)

    async def request_tool(
        self,
        client_id: str,
        tool_name: str,
        params: dict[str, Any],
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        async with self._lock:
            websocket = self._connections[client_id]
            self._pending_tool_requests[request_id] = PendingToolRequest(client_id=client_id, future=future)

        request_payload = {
            "type": "tool.request",
            "request_id": request_id,
            "tool": tool_name,
            "params": params,
        }
        logger.info("tool.request client=%s tool=%s request_id=%s", client_id, tool_name, request_id)
        try:
            await websocket.send_json(request_payload)
        except Exception:
            async with self._lock:
                self._pending_tool_requests.pop(request_id, None)
            raise

        try:
            response_payload = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Timed out waiting for tool response: {tool_name}") from exc
        finally:
            async with self._lock:
                self._pending_tool_requests.pop(request_id, None)

        if "error" in response_payload:
            logger.info("tool.response client=%s tool=%s request_id=%s status=error", client_id, tool_name, request_id)
            raise RuntimeError(str(response_payload["error"]))
        if "result" not in response_payload:
            raise RuntimeError("Tool response missing result payload")
        result = response_payload["result"]
        if not isinstance(result, dict):
            raise RuntimeError("Tool response result must be a JSON object")
        logger.info("tool.response client=%s tool=%s request_id=%s status=ok", client_id, tool_name, request_id)
        return result

    async def handle_incoming_message(self, client_id: str, message: dict[str, Any]) -> bool:
        if message.get("type") != "tool.response":
            return False

        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return True

        async with self._lock:
            pending = self._pending_tool_requests.get(request_id)
            if pending is None:
                return True
            if pending.client_id != client_id:
                return True
            if pending.future.done():
                return True

            payload: dict[str, Any] = {}
            if "result" in message:
                payload["result"] = message["result"]
            if "error" in message:
                payload["error"] = message["error"]
            pending.future.set_result(payload)
            return True
