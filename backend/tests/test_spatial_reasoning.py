from __future__ import annotations

from collections import deque
from typing import Any

import pytest
from fastapi.testclient import TestClient

from browsecraft_backend.app import create_app
from browsecraft_backend.config import get_settings
from browsecraft_sim.main import HeadlessVoxelWorld, world_bounding_box


pytestmark = pytest.mark.spatial


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


def _run_chat_round_trip(
    *,
    client: TestClient,
    websocket: Any,
    world: HeadlessVoxelWorld,
    client_id: str,
    message: str,
    max_events: int = 300,
) -> str:
    response = client.post(
        "/v1/chat",
        json={
            "client_id": client_id,
            "message": message,
        },
    )
    response.raise_for_status()
    chat_id = response.json()["chat_id"]

    for _ in range(max_events):
        envelope = websocket.receive_json()
        event_type = envelope.get("type")

        if event_type == "tool.request":
            request_id = envelope["request_id"]
            tool = envelope["tool"]
            params = envelope.get("params", {})
            try:
                result = _dispatch_tool(world, tool, params)
                websocket.send_json(
                    {
                        "type": "tool.response",
                        "request_id": request_id,
                        "result": result,
                    }
                )
            except Exception as exc:
                websocket.send_json(
                    {
                        "type": "tool.response",
                        "request_id": request_id,
                        "error": str(exc),
                    }
                )
            continue

        if event_type == "chat.response" and envelope.get("chat_id") == chat_id:
            payload = envelope.get("payload", {})
            return str(payload.get("message", ""))

    raise TimeoutError("chat.response was not received before max_events")


def _changed_blocks(before: dict[tuple[int, int, int], str], after: dict[tuple[int, int, int], str]) -> dict[tuple[int, int, int], str]:
    keys = set(before) | set(after)
    changed: dict[tuple[int, int, int], str] = {}
    for coord in keys:
        before_block = before.get(coord, "minecraft:air")
        after_block = after.get(coord, "minecraft:air")
        if before_block != after_block:
            changed[coord] = after_block
    return changed


def _is_connected(coords: set[tuple[int, int, int]]) -> bool:
    if not coords:
        return False

    queue: deque[tuple[int, int, int]] = deque([next(iter(coords))])
    visited: set[tuple[int, int, int]] = set()
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        x, y, z = current
        for ox, oy, oz in offsets:
            neighbor = (x + ox, y + oy, z + oz)
            if neighbor in coords and neighbor not in visited:
                queue.append(neighbor)

    return visited == coords


@pytest.fixture
def _configured_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_CHAT_MODEL", "claude-opus-4-6")
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for spatial tests")
    yield
    get_settings.cache_clear()


def test_builds_3x3x3_stone_cube_near_player(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    before = dict(world.blocks)

    app = create_app()
    client_id = "spatial-cube-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Build a 3x3x3 solid cube of minecraft:stone. "
                    "Use the player's current block position as one bottom corner of the cube."
                ),
            )

    changed = _changed_blocks(before, world.blocks)
    stone_coords = {coord for coord, block_id in changed.items() if block_id == "minecraft:stone"}

    assert len(changed) == 27
    assert len(stone_coords) == 27
    assert _is_connected(stone_coords)

    minimum, maximum = world_bounding_box(stone_coords)
    width = maximum[0] - minimum[0] + 1
    height = maximum[1] - minimum[1] + 1
    depth = maximum[2] - minimum[2] + 1
    assert (width, height, depth) == (3, 3, 3)

    assert minimum[0] - 1 <= world.player.x <= maximum[0] + 1
    assert minimum[1] - 1 <= world.player.y <= maximum[1] + 1
    assert minimum[2] - 1 <= world.player.z <= maximum[2] + 1


def test_adds_door_on_south_wall_of_existing_room(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    world.box_walls(origin=(-2, 64, -2), size=(5, 3, 5), block_id="minecraft:oak_planks")
    before = dict(world.blocks)

    app = create_app()
    client_id = "spatial-door-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "There is an oak plank room with south wall at z=2 and no doorway. "
                    "Place a minecraft:oak_door at (x=0,y=64,z=2) and (x=0,y=65,z=2)."
                ),
            )

    changed = _changed_blocks(before, world.blocks)
    door_coords = [coord for coord, block_id in changed.items() if block_id.endswith("_door")]

    assert door_coords
    assert (0, 64, 2) in door_coords
    assert (0, 65, 2) in door_coords
