from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx
import websockets


Coord = tuple[int, int, int]


@dataclass(slots=True, frozen=True)
class PlayerState:
    x: int = 0
    y: int = 64
    z: int = 0
    facing: str = "north"
    dimension: str = "minecraft:overworld"


class HeadlessVoxelWorld:
    def __init__(self, *, player: PlayerState | None = None) -> None:
        self.player = player or PlayerState()
        self.blocks: dict[Coord, str] = {}
        self._undo_stack: list[list[tuple[Coord, str]]] = []

    def block_at(self, coord: Coord) -> str:
        return self.blocks.get(coord, "minecraft:air")

    def set_block(self, coord: Coord, block_id: str) -> None:
        canonical_block_id = block_id.split("[", 1)[0]
        if canonical_block_id == "minecraft:air":
            self.blocks.pop(coord, None)
            return
        self.blocks[coord] = canonical_block_id

    def place_blocks(self, placements: list[dict[str, Any]]) -> dict[str, Any]:
        history: list[tuple[Coord, str]] = []
        for placement in placements:
            x = int(placement["x"])
            y = int(placement["y"])
            z = int(placement["z"])
            block_id = str(placement["block_id"])
            coord = (x, y, z)
            history.append((coord, self.block_at(coord)))
            self.set_block(coord, block_id)
        self._undo_stack.append(history)
        return {"placed_count": len(placements)}

    def undo_last(self) -> dict[str, Any]:
        if not self._undo_stack:
            raise RuntimeError("No placement batch to undo")
        history = self._undo_stack.pop()
        for coord, previous_block in reversed(history):
            self.set_block(coord, previous_block)
        return {"undone_count": len(history)}

    def inspect_area(self, *, center: dict[str, Any], radius: int) -> dict[str, Any]:
        cx = int(center["x"])
        cy = int(center["y"])
        cz = int(center["z"])
        clamped_radius = max(0, min(16, int(radius)))

        counts: Counter[str] = Counter()
        for dx in range(-clamped_radius, clamped_radius + 1):
            for dy in range(-clamped_radius, clamped_radius + 1):
                for dz in range(-clamped_radius, clamped_radius + 1):
                    counts[self.block_at((cx + dx, cy + dy, cz + dz))] += 1

        return {
            "requested_radius": int(radius),
            "radius": clamped_radius,
            "sampled_blocks": (2 * clamped_radius + 1) ** 3,
            "center": {"x": cx, "y": cy, "z": cz},
            "block_counts": dict(sorted(counts.items())),
        }

    def player_position(self) -> dict[str, Any]:
        return {
            "x": self.player.x,
            "y": self.player.y,
            "z": self.player.z,
            "yaw": 0,
            "pitch": 0,
            "block_x": self.player.x,
            "block_y": self.player.y,
            "block_z": self.player.z,
            "facing": self.player.facing,
            "dimension": self.player.dimension,
        }

    def player_inventory(self) -> dict[str, Any]:
        return {
            "selected_slot": 0,
            "filled_slots": 0,
            "total_item_count": 0,
            "items": [],
        }

    def flat_terrain(
        self,
        *,
        radius: int,
        surface_y: int = 63,
        surface_block: str = "minecraft:grass_block",
        fill_block: str = "minecraft:dirt",
        depth: int = 4,
    ) -> None:
        for x in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                self.set_block((x, surface_y, z), surface_block)
                for y in range(surface_y - depth, surface_y):
                    self.set_block((x, y, z), fill_block)

    def filled_box(self, origin: Coord, size: tuple[int, int, int], block_id: str) -> None:
        ox, oy, oz = origin
        width, height, depth = size
        for x in range(ox, ox + width):
            for y in range(oy, oy + height):
                for z in range(oz, oz + depth):
                    self.set_block((x, y, z), block_id)

    def box_walls(self, origin: Coord, size: tuple[int, int, int], block_id: str) -> None:
        ox, oy, oz = origin
        width, height, depth = size
        max_x = ox + width - 1
        max_y = oy + height - 1
        max_z = oz + depth - 1
        for x in range(ox, max_x + 1):
            for y in range(oy, max_y + 1):
                for z in range(oz, max_z + 1):
                    is_wall = x in {ox, max_x} or z in {oz, max_z}
                    if is_wall:
                        self.set_block((x, y, z), block_id)

    def floor_with_walls(
        self,
        origin: Coord,
        size: tuple[int, int, int],
        *,
        floor_block: str,
        wall_block: str,
    ) -> None:
        ox, oy, oz = origin
        width, height, depth = size
        self.filled_box((ox, oy, oz), (width, 1, depth), floor_block)
        self.box_walls((ox, oy + 1, oz), (width, height - 1, depth), wall_block)


def world_bounding_box(blocks: Iterable[Coord]) -> tuple[Coord, Coord]:
    points = list(blocks)
    if not points:
        raise ValueError("Cannot compute bounding box for empty block set")
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    max_z = max(point[2] for point in points)
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


async def run_chat_simulation(message: str, base_url: str, client_id: str, world: HeadlessVoxelWorld) -> str:
    ws_url = f"{base_url.replace('http://', 'ws://').replace('https://', 'wss://')}/v1/ws/{client_id}"
    chat_id: str

    async with websockets.connect(ws_url) as websocket:
        async with httpx.AsyncClient(base_url=base_url) as client:
            response = await client.post(
                "/v1/chat",
                json={"client_id": client_id, "message": message},
            )
            response.raise_for_status()
            chat_id = str(response.json()["chat_id"])
            print(f"created chat {chat_id}")

        async for raw_message in websocket:
            envelope = json.loads(raw_message)
            event_type = envelope.get("type")

            if event_type == "tool.request":
                await _handle_tool_request(world, websocket, envelope)
                continue

            if event_type == "chat.delta":
                payload = envelope.get("payload", {})
                partial = payload.get("partial", "")
                print(f"delta: {partial}")
                continue

            if event_type == "chat.response" and envelope.get("chat_id") == chat_id:
                payload = envelope.get("payload", {})
                message_text = str(payload.get("message", ""))
                print(f"chat.response: {message_text}")
                return message_text

    raise RuntimeError("WebSocket closed before chat.response")


async def _handle_tool_request(world: HeadlessVoxelWorld, websocket: Any, envelope: dict[str, Any]) -> None:
    request_id = str(envelope["request_id"])
    tool = str(envelope["tool"])
    params = envelope.get("params", {})

    try:
        result = _dispatch_tool(world, tool, params)
        response = {
            "type": "tool.response",
            "request_id": request_id,
            "result": result,
        }
    except Exception as exc:
        response = {
            "type": "tool.response",
            "request_id": request_id,
            "error": str(exc),
        }

    await websocket.send(json.dumps(response))


def _dispatch_tool(world: HeadlessVoxelWorld, tool: str, params: dict[str, Any]) -> dict[str, Any]:
    if tool == "player_position":
        return world.player_position()
    if tool == "player_inventory":
        return world.player_inventory()
    if tool == "inspect_area":
        return world.inspect_area(center=params["center"], radius=int(params["radius"]))
    if tool == "place_blocks":
        placements = params["placements"]
        if not isinstance(placements, list):
            raise RuntimeError("place_blocks expects placements list")
        return world.place_blocks(placements)
    if tool == "undo_last":
        return world.undo_last()
    if tool == "get_active_overlay":
        return {
            "has_plan": False,
            "block_count": 0,
            "anchor": {"x": 0, "y": 0, "z": 0},
            "rotation_quarter_turns": 0,
            "preview_mode": False,
            "confirmed": False,
            "remaining_count": 0,
        }
    if tool == "modify_overlay":
        return {"op": params.get("op")}
    if tool == "get_blueprints":
        return {"names": [], "count": 0}
    if tool == "save_blueprint":
        return {"name": params.get("name"), "saved": True}
    if tool == "load_blueprint":
        return {
            "name": params.get("name"),
            "overlay": {
                "has_plan": False,
                "block_count": 0,
                "anchor": {"x": 0, "y": 0, "z": 0},
                "rotation_quarter_turns": 0,
                "preview_mode": False,
                "confirmed": False,
                "remaining_count": 0,
            },
        }
    raise RuntimeError(f"Unsupported tool: {tool}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BrowseCraft headless simulator")
    parser.add_argument("message", help="chat prompt")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--client-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=32)
    client_id = args.client_id or str(uuid.uuid4())
    asyncio.run(run_chat_simulation(args.message, args.base_url, client_id, world))


if __name__ == "__main__":
    main()
