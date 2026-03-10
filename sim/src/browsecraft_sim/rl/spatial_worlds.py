from __future__ import annotations

import random
from collections.abc import Iterable

from .types import BlockPlacement


MARKER_BLOCKS = (
    "minecraft:red_wool",
    "minecraft:blue_wool",
    "minecraft:green_wool",
    "minecraft:yellow_wool",
    "minecraft:purple_wool",
    "minecraft:orange_wool",
    "minecraft:cyan_wool",
    "minecraft:black_wool",
    "minecraft:white_wool",
)

MARKER_NAMES = {
    "minecraft:red_wool": "red marker",
    "minecraft:blue_wool": "blue marker",
    "minecraft:green_wool": "green marker",
    "minecraft:yellow_wool": "yellow marker",
    "minecraft:purple_wool": "purple marker",
    "minecraft:orange_wool": "orange marker",
    "minecraft:cyan_wool": "cyan marker",
    "minecraft:black_wool": "black marker",
    "minecraft:white_wool": "white marker",
}

CARDINAL_OFFSETS: dict[str, tuple[int, int, int]] = {
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
}

OPPOSITE_CARDINAL: dict[str, str] = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


def block_name(block_id: str) -> str:
    return block_id.replace("minecraft:", "").replace("_", " ")


def marker_name(block_id: str) -> str:
    return MARKER_NAMES.get(block_id, block_name(block_id))


def line_blocks(
    *,
    axis: str,
    start: tuple[int, int, int],
    length: int,
    block_id: str,
) -> list[BlockPlacement]:
    dx, dy, dz = {
        "x": (1, 0, 0),
        "y": (0, 1, 0),
        "z": (0, 0, 1),
    }[axis]
    return [
        BlockPlacement(
            x=start[0] + (dx * step),
            y=start[1] + (dy * step),
            z=start[2] + (dz * step),
            block_id=block_id,
        )
        for step in range(length)
    ]


def filled_rect(
    *,
    origin: tuple[int, int, int],
    width: int,
    depth: int,
    block_id: str,
) -> list[BlockPlacement]:
    ox, oy, oz = origin
    return [
        BlockPlacement(x=x, y=oy, z=z, block_id=block_id)
        for x in range(ox, ox + width)
        for z in range(oz, oz + depth)
    ]


def dedupe_blocks(blocks: Iterable[BlockPlacement]) -> list[BlockPlacement]:
    deduped: dict[tuple[int, int, int], BlockPlacement] = {}
    for block in blocks:
        deduped[block.coord()] = block
    return list(deduped.values())


def remove_coords(
    blocks: Iterable[BlockPlacement],
    removed_coords: set[tuple[int, int, int]],
) -> list[BlockPlacement]:
    return [block for block in blocks if block.coord() not in removed_coords]


def room_shell(
    *,
    origin: tuple[int, int, int],
    width: int,
    height: int,
    depth: int,
    wall_block: str,
) -> list[BlockPlacement]:
    ox, oy, oz = origin
    max_x = ox + width - 1
    max_y = oy + height - 1
    max_z = oz + depth - 1
    return [
        BlockPlacement(x=x, y=y, z=z, block_id=wall_block)
        for x in range(ox, max_x + 1)
        for y in range(oy, max_y + 1)
        for z in range(oz, max_z + 1)
        if x in {ox, max_x} or z in {oz, max_z}
    ]


def enclosure_shell(
    *,
    origin: tuple[int, int, int],
    width: int,
    depth: int,
    height: int,
    wall_block: str,
) -> list[BlockPlacement]:
    ox, oy, oz = origin
    max_x = ox + width - 1
    max_z = oz + depth - 1
    return [
        BlockPlacement(x=x, y=y, z=z, block_id=wall_block)
        for x in range(ox, max_x + 1)
        for y in range(oy, oy + height)
        for z in range(oz, max_z + 1)
        if x in {ox, max_x} or z in {oz, max_z}
    ]


def tower(
    *,
    base: tuple[int, int, int],
    height: int,
    block_id: str,
) -> list[BlockPlacement]:
    bx, by, bz = base
    return [BlockPlacement(x=bx, y=by + dy, z=bz, block_id=block_id) for dy in range(height)]


def occupied_coords(*groups: Iterable[BlockPlacement]) -> set[tuple[int, int, int]]:
    occupied: set[tuple[int, int, int]] = set()
    for group in groups:
        for block in group:
            occupied.add(block.coord())
    return occupied


def horizontal_facing_offset(facing: str) -> tuple[int, int]:
    if facing == "north":
        return (0, -1)
    if facing == "south":
        return (0, 1)
    if facing == "east":
        return (1, 0)
    if facing == "west":
        return (-1, 0)
    raise ValueError(f"Unsupported facing: {facing}")


def player_relative_offset(facing: str, relation: str, distance: int) -> tuple[int, int]:
    forward_dx, forward_dz = horizontal_facing_offset(facing)
    left_dx, left_dz = forward_dz, -forward_dx
    right_dx, right_dz = -forward_dz, forward_dx
    if relation == "front":
        return (forward_dx * distance, forward_dz * distance)
    if relation == "behind":
        return (-forward_dx * distance, -forward_dz * distance)
    if relation == "left":
        return (left_dx * distance, left_dz * distance)
    if relation == "right":
        return (right_dx * distance, right_dz * distance)
    raise ValueError(f"Unsupported player-relative relation: {relation}")


def player_relative_direction(facing: str, dx: int, dz: int) -> str:
    forward_dx, forward_dz = horizontal_facing_offset(facing)
    left_dx, left_dz = forward_dz, -forward_dx
    right_dx, right_dz = -forward_dz, forward_dx
    if (dx, dz) == (forward_dx, forward_dz):
        return "front"
    if (dx, dz) == (-forward_dx, -forward_dz):
        return "behind"
    if (dx, dz) == (left_dx, left_dz):
        return "left"
    if (dx, dz) == (right_dx, right_dz):
        return "right"
    raise ValueError("Unsupported player-relative offset")


def chain_positions(
    *,
    rng: random.Random,
    start: tuple[int, int, int],
    hop_count: int,
    step_distance: int,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    positions = [start]
    steps: list[str] = []
    last_direction: str | None = None
    used = {start}
    for _ in range(hop_count):
        candidates = list(CARDINAL_OFFSETS.keys())
        rng.shuffle(candidates)
        chosen: str | None = None
        for direction in candidates:
            if last_direction is not None and direction == OPPOSITE_CARDINAL[last_direction]:
                continue
            dx, _, dz = CARDINAL_OFFSETS[direction]
            candidate = (
                positions[-1][0] + (dx * step_distance),
                positions[-1][1],
                positions[-1][2] + (dz * step_distance),
            )
            if candidate in used:
                continue
            chosen = direction
            positions.append(candidate)
            used.add(candidate)
            break
        if chosen is None:
            raise RuntimeError("could not generate non-overlapping chain positions")
        steps.append(chosen)
        last_direction = chosen
    return positions, steps
