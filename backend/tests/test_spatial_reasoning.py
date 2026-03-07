from __future__ import annotations

from collections import deque
import os
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import pytest
from fastapi.testclient import TestClient

from browsecraft_backend.app import create_app
from browsecraft_backend.config import get_settings
from browsecraft_sim.main import HeadlessVoxelWorld, PlayerState, world_bounding_box
from browsecraft_sim.tool_dispatch import dispatch_tool


pytestmark = pytest.mark.spatial
Coord = tuple[int, int, int]
_CHAT_EVENT_TIMEOUT_SECONDS = 30.0
def _run_chat_round_trip(
    *,
    client: TestClient,
    websocket: Any,
    world: HeadlessVoxelWorld,
    client_id: str,
    message: str,
    mode: str = "build",
    max_events: int = 600,
    tool_requests: list[str] | None = None,
    tool_request_payloads: list[tuple[str, dict[str, Any]]] | None = None,
    tool_statuses: list[str] | None = None,
) -> str:
    event_queue: Queue[dict[str, Any]] = Queue()
    stop_event = Event()

    def websocket_loop() -> None:
        try:
            while not stop_event.is_set():
                envelope = websocket.receive_json()
                event_type = envelope.get("type")

                if event_type == "tool.request":
                    request_id = envelope["request_id"]
                    tool = envelope["tool"]
                    params = envelope.get("params", {})
                    if tool_requests is not None:
                        tool_requests.append(str(tool))
                    if tool_request_payloads is not None:
                        tool_request_payloads.append((str(tool), params))
                    try:
                        result = dispatch_tool(world, tool, params)
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

                if event_type == "chat.tool_status":
                    if tool_statuses is not None:
                        payload = envelope.get("payload", {})
                        tool_statuses.append(str(payload.get("status", "")))
                    continue

                event_queue.put(envelope)
                if event_type == "chat.response":
                    return
        except BaseException as exc:
            event_queue.put({"type": "__websocket_error__", "error": exc})

    listener = Thread(target=websocket_loop, daemon=True)
    listener.start()

    try:
        response = client.post(
            "/v1/chat",
            json={
                "client_id": client_id,
                "message": message,
                "mode": mode,
            },
        )
        response.raise_for_status()
        chat_id = response.json()["chat_id"]

        for _ in range(max_events):
            try:
                envelope = event_queue.get(timeout=_CHAT_EVENT_TIMEOUT_SECONDS)
            except Empty as exc:
                raise TimeoutError("chat.response was not received before max_events") from exc

            if envelope.get("type") == "__websocket_error__":
                raise envelope["error"]

            if envelope.get("type") == "chat.response" and envelope.get("chat_id") == chat_id:
                payload = envelope.get("payload", {})
                return str(payload.get("message", ""))

        raise TimeoutError("chat.response was not received before max_events")
    finally:
        stop_event.set()


def assert_region_is(
    world: HeadlessVoxelWorld,
    x_range: range,
    y_range: range,
    z_range: range,
    expected_block: str,
) -> None:
    for x in x_range:
        for y in y_range:
            for z in z_range:
                actual = world.block_at((x, y, z))
                assert actual == expected_block, f"at ({x},{y},{z}): expected {expected_block}, got {actual}"


def assert_footprint_matches(coords: set[Coord], expected_footprint_set: set[tuple[int, int]]) -> None:
    actual = {(x, z) for x, _, z in coords}
    assert actual == expected_footprint_set


def assert_height_profile(coords: set[Coord], expected_min_y: int, expected_max_y: int) -> None:
    y_values = {coord[1] for coord in coords}
    assert y_values
    assert min(y_values) == expected_min_y
    assert max(y_values) == expected_max_y


def assert_is_hollow_box(
    world: HeadlessVoxelWorld,
    *,
    min_corner: Coord,
    max_corner: Coord,
    shell_block: str,
) -> None:
    min_x, min_y, min_z = min_corner
    max_x, max_y, max_z = max_corner
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                is_shell = x in {min_x, max_x} or y in {min_y, max_y} or z in {min_z, max_z}
                expected = shell_block if is_shell else "minecraft:air"
                actual = world.block_at((x, y, z))
                assert actual == expected, f"at ({x},{y},{z}): expected {expected}, got {actual}"


def _forward_offset(facing: str) -> tuple[int, int]:
    if facing == "north":
        return (0, -1)
    if facing == "south":
        return (0, 1)
    if facing == "east":
        return (1, 0)
    if facing == "west":
        return (-1, 0)
    raise ValueError(f"Unsupported facing: {facing}")


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
def _configured_settings(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    spatial_model = (
        os.getenv("ANTHROPIC_SPATIAL_MODEL")
        or os.getenv("ANTHROPIC_CHAT_MODEL")
        or "claude-sonnet-4-6"
    )
    monkeypatch.setenv("ANTHROPIC_CHAT_MODEL", spatial_model)
    monkeypatch.setenv(
        "ANTHROPIC_PLANNER_MODEL",
        os.getenv("ANTHROPIC_SPATIAL_PLANNER_MODEL") or os.getenv("ANTHROPIC_PLANNER_MODEL") or spatial_model,
    )
    monkeypatch.setenv(
        "ANTHROPIC_TRIAGE_MODEL",
        os.getenv("ANTHROPIC_SPATIAL_TRIAGE_MODEL") or os.getenv("ANTHROPIC_TRIAGE_MODEL") or "claude-haiku-4-5",
    )
    monkeypatch.setenv("ANTHROPIC_ENABLE_BUILD_PLANNER", "true" if pytestconfig.getoption("--with-planning") else "false")
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for spatial tests")
    yield
    get_settings.cache_clear()


@pytest.mark.quick_spatial
def test_builds_3x3x3_stone_cube_near_player(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    before = world.snapshot()

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

    changed = world.diff(before)
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
    before = world.snapshot()

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

    changed = world.diff(before)
    door_coords = [coord for coord, block_id in changed.items() if block_id.endswith("_door")]

    assert door_coords
    assert (0, 64, 2) in door_coords
    assert (0, 65, 2) in door_coords


@pytest.mark.quick_spatial
def test_builds_5_block_tall_pillar(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    before = world.snapshot()

    app = create_app()
    client_id = "spatial-pillar-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Build a minecraft:stone pillar exactly 5 blocks tall at x=0,z=0, "
                    "starting from y=64 and ending at y=68."
                ),
            )

    changed = world.diff(before)
    pillar_coords = {coord for coord, block_id in changed.items() if block_id == "minecraft:stone"}
    assert_region_is(world, range(0, 1), range(64, 69), range(0, 1), "minecraft:stone")
    assert _is_connected(pillar_coords)
    assert_footprint_matches(pillar_coords, {(0, 0)})
    assert_height_profile(pillar_coords, 64, 68)


def test_builds_wall_connecting_two_points(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    before = world.snapshot()

    app = create_app()
    client_id = "spatial-wall-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Build a straight minecraft:cobblestone wall connecting (-4,64,0) to (4,64,0). "
                    "Make the wall 3 blocks tall."
                ),
            )

    changed = world.diff(before)
    wall_coords = {coord for coord, block_id in changed.items() if block_id == "minecraft:cobblestone"}
    assert_region_is(world, range(-4, 5), range(64, 67), range(0, 1), "minecraft:cobblestone")
    assert _is_connected(wall_coords)
    assert_height_profile(wall_coords, 64, 66)
    assert_footprint_matches(wall_coords, {(x, 0) for x in range(-4, 5)})


def test_builds_fence_post_wall_with_inclusive_endpoints(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    before = world.snapshot()

    app = create_app()
    client_id = "spatial-fencepost-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Build a straight minecraft:cobblestone row from x=-4 to x=4 at y=64 and z=1. "
                    "Make it exactly one block tall and include both endpoints."
                ),
            )

    changed = world.diff(before)
    wall_coords = {coord for coord, block_id in changed.items() if block_id == "minecraft:cobblestone"}
    assert len(wall_coords) == 9
    assert_region_is(world, range(-4, 5), range(64, 65), range(1, 2), "minecraft:cobblestone")
    assert world.block_at((-5, 64, 1)) != "minecraft:cobblestone"
    assert world.block_at((5, 64, 1)) != "minecraft:cobblestone"
    assert_footprint_matches(wall_coords, {(x, 1) for x in range(-4, 5)})
    assert_height_profile(wall_coords, 64, 64)


def test_builds_hollow_5x5x5_cube_shell(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    before = world.snapshot()

    app = create_app()
    client_id = "spatial-hollow-cube-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Build a hollow 5x5x5 minecraft:stone cube with corners at (-2,64,-2) and (2,68,2). "
                    "Keep every interior block as air."
                ),
            )

    changed = world.diff(before)
    stone_coords = {coord for coord, block_id in changed.items() if block_id == "minecraft:stone"}
    assert len(stone_coords) == 98
    assert _is_connected(stone_coords)
    assert_height_profile(stone_coords, 64, 68)
    assert_footprint_matches(stone_coords, {(x, z) for x in range(-2, 3) for z in range(-2, 3)})
    assert_is_hollow_box(world, min_corner=(-2, 64, -2), max_corner=(2, 68, 2), shell_block="minecraft:stone")


@pytest.mark.quick_spatial
def test_builds_tower_in_front_of_player_from_facing(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld(player=PlayerState(x=5, y=64, z=-3, facing="east"))
    world.flat_terrain(radius=24)
    before = world.snapshot()

    app = create_app()
    client_id = "spatial-forward-tower-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message="Build a 3-block-tall minecraft:stone tower directly in front of me.",
            )

    dx, dz = _forward_offset(world.player.facing)
    expected = {(world.player.x + dx, y, world.player.z + dz) for y in range(world.player.y, world.player.y + 3)}
    changed = world.diff(before)
    stone_coords = {coord for coord, block_id in changed.items() if block_id == "minecraft:stone"}
    assert stone_coords == expected
    assert _is_connected(stone_coords)
    assert_height_profile(stone_coords, world.player.y, world.player.y + 2)
    assert_footprint_matches(stone_coords, {(world.player.x + dx, world.player.z + dz)})


def test_replaces_oak_walls_with_birch(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    world.box_walls(origin=(-2, 64, -2), size=(5, 3, 5), block_id="minecraft:oak_planks")

    app = create_app()
    client_id = "spatial-replace-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Inspect the room around me and replace every minecraft:oak_planks wall block "
                    "with minecraft:birch_planks at the same coordinates."
                ),
            )

    for x in range(-2, 3):
        for y in range(64, 67):
            for z in range(-2, 3):
                is_wall = x in {-2, 2} or z in {-2, 2}
                if not is_wall:
                    continue
                assert world.block_at((x, y, z)) == "minecraft:birch_planks"


def test_adds_roof_to_open_room(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    world.floor_with_walls(
        origin=(-2, 63, -2),
        size=(5, 4, 5),
        floor_block="minecraft:stone",
        wall_block="minecraft:oak_planks",
    )
    before = world.snapshot()

    app = create_app()
    client_id = "spatial-roof-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Add a flat minecraft:oak_planks roof to this open room. "
                    "The room footprint is x=-2..2 and z=-2..2, so place the roof at y=67."
                ),
            )

    changed = world.diff(before)
    assert len(changed) >= 25
    assert_region_is(world, range(-2, 3), range(67, 68), range(-2, 3), "minecraft:oak_planks")


def test_multi_turn_room_modify_sequence_single_session(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)

    app = create_app()
    client_id = "spatial-multiturn-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Build a 5x5 minecraft:stone room centered on me with 3-block-high walls, "
                    "hollow interior, and no roof."
                ),
            )

            wall_positions: list[Coord] = []
            for x in range(-2, 3):
                for y in range(64, 67):
                    for z in range(-2, 3):
                        if x in {-2, 2} or z in {-2, 2}:
                            wall_positions.append((x, y, z))
            assert all(world.block_at(pos) == "minecraft:stone" for pos in wall_positions)

            interior_positions = [(x, y, z) for x in range(-1, 2) for y in range(64, 67) for z in range(-1, 2)]
            assert all(world.block_at(pos) == "minecraft:air" for pos in interior_positions)

            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message="Add a door on the south wall at ground level.",
            )

            door_positions = {(0, 64, 2), (0, 65, 2)}
            assert any(world.block_at(position).endswith("_door") for position in door_positions)
            for position in wall_positions:
                if position in door_positions:
                    continue
                assert world.block_at(position) in {"minecraft:stone", "minecraft:oak_door"}

            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message="Replace all minecraft:stone in that room with minecraft:birch_planks and keep the same shape.",
            )

    for x in range(-2, 3):
        for y in range(64, 67):
            for z in range(-2, 3):
                is_wall = x in {-2, 2} or z in {-2, 2}
                if not is_wall:
                    assert world.block_at((x, y, z)) == "minecraft:air"
                    continue

                block_id = world.block_at((x, y, z))
                if (x, y, z) in {(0, 64, 2), (0, 65, 2)} and block_id.endswith("_door"):
                    continue
                assert block_id == "minecraft:birch_planks"


@pytest.mark.quick_spatial
def test_undo_then_rebuild_uses_undo_last(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)
    before = world.snapshot()

    app = create_app()
    client_id = "spatial-undo-rebuild-client"

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message="Build a 2x2 minecraft:stone platform centered on me at y=64.",
            )
            first_pass = world.diff(before)
            assert any(block_id == "minecraft:stone" for block_id in first_pass.values())

            second_turn_tools: list[str] = []
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                message=(
                    "Undo that and instead build a straight line of minecraft:birch_planks from x=-2 to x=2 "
                    "at y=64 and z=0."
                ),
                tool_requests=second_turn_tools,
            )

    assert "undo_last" in second_turn_tools

    final_changed = world.diff(before)
    birch_coords = {coord for coord, block_id in final_changed.items() if block_id == "minecraft:birch_planks"}
    expected_line = {(x, 64, 0) for x in range(-2, 3)}
    assert birch_coords == expected_line
    assert all(block_id != "minecraft:stone" for block_id in final_changed.values())
    assert_height_profile(birch_coords, 64, 64)
    assert_footprint_matches(birch_coords, {(x, 0) for x in range(-2, 3)})


@pytest.mark.planner_spatial
def test_plan_mode_uses_set_plan_for_3x3_stone_platform(_configured_settings: None) -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=24)

    app = create_app()
    client_id = "spatial-plan-mode-client"
    captured_tool_payloads: list[tuple[str, dict[str, Any]]] = []

    with TestClient(app) as client:
        with client.websocket_connect(f"/v1/ws/{client_id}") as websocket:
            _run_chat_round_trip(
                client=client,
                websocket=websocket,
                world=world,
                client_id=client_id,
                mode="plan",
                message="plan a 3x3 stone platform",
                tool_request_payloads=captured_tool_payloads,
            )

    tool_names = [name for name, _ in captured_tool_payloads]
    assert "set_plan" in tool_names
    assert "place_blocks" not in tool_names
    assert "fill_region" not in tool_names

    set_plan_params = next(params for name, params in captured_tool_payloads if name == "set_plan")
    placements = set_plan_params["placements"]
    assert len(placements) == 9
    assert all(placement["block_id"] == "minecraft:stone" for placement in placements)
