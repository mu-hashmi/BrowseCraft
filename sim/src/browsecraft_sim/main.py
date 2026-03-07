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

from .tool_dispatch import dispatch_tool


Coord = tuple[int, int, int]
_TERRAIN_BLOCK_IDS = {
    "minecraft:grass_block",
    "minecraft:dirt",
    "minecraft:coarse_dirt",
    "minecraft:podzol",
    "minecraft:mycelium",
    "minecraft:rooted_dirt",
    "minecraft:bedrock",
    "minecraft:sand",
    "minecraft:red_sand",
    "minecraft:gravel",
    "minecraft:deepslate",
    "minecraft:tuff",
}


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

    def snapshot(self) -> dict[Coord, str]:
        return dict(self.blocks)

    def diff(
        self,
        before: dict[Coord, str],
        after: dict[Coord, str] | None = None,
    ) -> dict[Coord, str]:
        compared_after = self.blocks if after is None else after
        changed: dict[Coord, str] = {}
        for coord in set(before) | set(compared_after):
            before_block = before.get(coord, "minecraft:air")
            after_block = compared_after.get(coord, "minecraft:air")
            if before_block != after_block:
                changed[coord] = after_block
        return changed

    def diff_report(
        self,
        before: dict[Coord, str],
        after: dict[Coord, str] | None = None,
    ) -> dict[str, Any]:
        compared_after = self.blocks if after is None else after
        changed = self.diff(before, compared_after)

        added = 0
        removed = 0
        updated = 0
        for coord, after_block in changed.items():
            before_block = before.get(coord, "minecraft:air")
            if before_block == "minecraft:air" and after_block != "minecraft:air":
                added += 1
            elif before_block != "minecraft:air" and after_block == "minecraft:air":
                removed += 1
            else:
                updated += 1

        report: dict[str, Any] = {
            "changed_count": len(changed),
            "added_count": added,
            "removed_count": removed,
            "updated_count": updated,
        }
        if changed:
            report["bbox"] = _bbox_dict(changed.keys())
        return report

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

    def fill_region(
        self,
        *,
        from_corner: dict[str, Any],
        to_corner: dict[str, Any],
        block_id: str,
    ) -> dict[str, Any]:
        min_x = min(int(from_corner["x"]), int(to_corner["x"]))
        max_x = max(int(from_corner["x"]), int(to_corner["x"]))
        min_y = min(int(from_corner["y"]), int(to_corner["y"]))
        max_y = max(int(from_corner["y"]), int(to_corner["y"]))
        min_z = min(int(from_corner["z"]), int(to_corner["z"]))
        max_z = max(int(from_corner["z"]), int(to_corner["z"]))

        volume = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1)
        if volume > 4096:
            raise RuntimeError("fill_region volume must be <= 4096 blocks")

        history: list[tuple[Coord, str]] = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    coord = (x, y, z)
                    history.append((coord, self.block_at(coord)))
                    self.set_block(coord, block_id)
        self._undo_stack.append(history)
        return {
            "placed_count": len(history),
            "fill_region": True,
            "from_corner": {
                "x": int(from_corner["x"]),
                "y": int(from_corner["y"]),
                "z": int(from_corner["z"]),
            },
            "to_corner": {
                "x": int(to_corner["x"]),
                "y": int(to_corner["y"]),
                "z": int(to_corner["z"]),
            },
        }

    def undo_last(self) -> dict[str, Any]:
        if not self._undo_stack:
            raise RuntimeError("No placement batch to undo")
        history = self._undo_stack.pop()
        for coord, previous_block in reversed(history):
            self.set_block(coord, previous_block)
        return {"undone_count": len(history)}

    def inspect_area(
        self,
        *,
        center: dict[str, Any],
        radius: int,
        detailed: bool = False,
        filter_terrain: bool = True,
    ) -> dict[str, Any]:
        cx = int(center["x"])
        cy = int(center["y"])
        cz = int(center["z"])
        max_radius = 6 if detailed else 12
        clamped_radius = max(0, min(max_radius, int(radius)))

        counts: Counter[str] = Counter()
        non_air_blocks: list[dict[str, Any]] = []
        for dx in range(-clamped_radius, clamped_radius + 1):
            for dy in range(-clamped_radius, clamped_radius + 1):
                for dz in range(-clamped_radius, clamped_radius + 1):
                    x = cx + dx
                    y = cy + dy
                    z = cz + dz
                    block_id = self.block_at((x, y, z))
                    counts[block_id] += 1
                    if (
                        detailed
                        and block_id != "minecraft:air"
                        and not (filter_terrain and _is_terrain_block(block_id, y))
                    ):
                        non_air_blocks.append(
                            {
                                "x": x,
                                "y": y,
                                "z": z,
                                "block_id": block_id,
                            }
                        )

        result = {
            "requested_radius": int(radius),
            "radius": clamped_radius,
            "sampled_blocks": (2 * clamped_radius + 1) ** 3,
            "center": {"x": cx, "y": cy, "z": cz},
            "block_counts": dict(sorted(counts.items())),
            "detailed": detailed,
            "filter_terrain": filter_terrain,
        }
        if detailed:
            result["non_air_blocks"] = non_air_blocks
        return result

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

    def ascii_slice(self, *, y: int) -> str:
        layer_coords = [coord for coord in self.blocks if coord[1] == y]
        if not layer_coords:
            return f"y={y} (empty)"

        min_x = min(coord[0] for coord in layer_coords)
        max_x = max(coord[0] for coord in layer_coords)
        min_z = min(coord[2] for coord in layer_coords)
        max_z = max(coord[2] for coord in layer_coords)

        rows = [f"y={y} x={min_x}..{max_x} z={min_z}..{max_z}"]
        for z in range(min_z, max_z + 1):
            chars: list[str] = []
            for x in range(min_x, max_x + 1):
                block_id = self.block_at((x, y, z))
                if block_id == "minecraft:air":
                    chars.append(".")
                    continue
                short = block_id.rsplit(":", maxsplit=1)[-1]
                chars.append(short[0].upper())
            rows.append("".join(chars))
        return "\n".join(rows)

    def validation_report(self) -> dict[str, Any]:
        if not self.blocks:
            return {
                "block_count": 0,
                "height": {"min": None, "max": None},
                "bbox": None,
                "dimensions": {"x": 0, "y": 0, "z": 0},
                "connected": False,
                "component_count": 0,
            }

        coords = list(self.blocks.keys())
        bbox = _bbox_dict(coords)
        ys = [coord[1] for coord in coords]
        components = _component_count(set(coords))
        return {
            "block_count": len(coords),
            "height": {"min": min(ys), "max": max(ys)},
            "bbox": bbox,
            "dimensions": {
                "x": bbox["max"]["x"] - bbox["min"]["x"] + 1,
                "y": bbox["max"]["y"] - bbox["min"]["y"] + 1,
                "z": bbox["max"]["z"] - bbox["min"]["z"] + 1,
            },
            "connected": components == 1,
            "component_count": components,
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


def _bbox_dict(blocks: Iterable[Coord]) -> dict[str, dict[str, int]]:
    minimum, maximum = world_bounding_box(blocks)
    return {
        "min": {"x": minimum[0], "y": minimum[1], "z": minimum[2]},
        "max": {"x": maximum[0], "y": maximum[1], "z": maximum[2]},
    }


def _component_count(coords: set[Coord]) -> int:
    if not coords:
        return 0

    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    remaining = set(coords)
    components = 0
    while remaining:
        components += 1
        queue = [remaining.pop()]
        while queue:
            x, y, z = queue.pop()
            for ox, oy, oz in offsets:
                neighbor = (x + ox, y + oy, z + oz)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return components


async def run_chat_simulation(
    message: str,
    base_url: str,
    client_id: str,
    world: HeadlessVoxelWorld,
    *,
    verbose: bool = True,
) -> str:
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
            if verbose:
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
                if verbose:
                    print(f"delta: {partial}")
                continue

            if event_type == "chat.response" and envelope.get("chat_id") == chat_id:
                payload = envelope.get("payload", {})
                message_text = str(payload.get("message", ""))
                if verbose:
                    print(f"chat.response: {message_text}")
                return message_text

    raise RuntimeError("WebSocket closed before chat.response")


async def _handle_tool_request(world: HeadlessVoxelWorld, websocket: Any, envelope: dict[str, Any]) -> None:
    request_id = str(envelope["request_id"])
    tool = str(envelope["tool"])
    params = envelope.get("params", {})

    try:
        result = dispatch_tool(world, tool, params)
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


def _is_terrain_block(block_id: str, y: int) -> bool:
    if block_id == "minecraft:stone" and y <= 63:
        return True
    return block_id in _TERRAIN_BLOCK_IDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BrowseCraft headless simulator")
    parser.add_argument("message", help="chat prompt")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--report-json", action="store_true")
    parser.add_argument("--slice-y", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=32)
    before = world.snapshot()
    client_id = args.client_id or str(uuid.uuid4())
    assistant_message = asyncio.run(
        run_chat_simulation(
            args.message,
            args.base_url,
            client_id,
            world,
            verbose=not args.report_json,
        )
    )
    if args.report_json:
        report: dict[str, Any] = {
            "assistant_message": assistant_message,
            "validation": world.validation_report(),
            "diff": world.diff_report(before),
        }
        if args.slice_y is not None:
            report["ascii_slice"] = world.ascii_slice(y=args.slice_y)
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
