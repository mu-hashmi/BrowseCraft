from __future__ import annotations

from collections import deque
from typing import Iterable

from browsecraft_sim.main import HeadlessVoxelWorld

from .types import BlockPlacement


def placement_map(blocks: Iterable[BlockPlacement]) -> dict[tuple[int, int, int], str]:
    return {placement.coord(): placement.block_id for placement in blocks}


def changed_map(blocks: Iterable[BlockPlacement]) -> dict[tuple[int, int, int], str]:
    return placement_map(blocks)


def typed_set(block_map: dict[tuple[int, int, int], str]) -> set[tuple[int, int, int, str]]:
    return {(coord[0], coord[1], coord[2], block_id) for coord, block_id in block_map.items()}


def iou_score(
    actual: dict[tuple[int, int, int], str],
    expected: dict[tuple[int, int, int], str],
) -> float:
    actual_set = typed_set(actual)
    expected_set = typed_set(expected)
    if not actual_set and not expected_set:
        return 1.0
    union = actual_set | expected_set
    if not union:
        return 0.0
    intersection = actual_set & expected_set
    return len(intersection) / len(union)


def exact_match(actual: dict[tuple[int, int, int], str], expected: dict[tuple[int, int, int], str]) -> float:
    return 1.0 if actual == expected else 0.0


def is_connected(coords: set[tuple[int, int, int]]) -> bool:
    if not coords:
        return False

    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    queue: deque[tuple[int, int, int]] = deque([next(iter(coords))])
    seen: set[tuple[int, int, int]] = set()

    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        x, y, z = current
        for dx, dy, dz in neighbors:
            candidate = (x + dx, y + dy, z + dz)
            if candidate in coords and candidate not in seen:
                queue.append(candidate)

    return seen == coords


def grounding_ratio(world: HeadlessVoxelWorld, coords: set[tuple[int, int, int]]) -> float:
    if not coords:
        return 0.0

    supported = 0
    for x, y, z in coords:
        below = (x, y - 1, z)
        if below in coords:
            supported += 1
            continue
        if world.block_at(below) != "minecraft:air":
            supported += 1
    return supported / len(coords)


def span_length(coords: set[tuple[int, int, int]], axis: str) -> int:
    if not coords:
        return 0
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    values = [coord[axis_index] for coord in coords]
    return max(values) - min(values) + 1


def preservation_score(world: HeadlessVoxelWorld, expected_unchanged: Iterable[BlockPlacement]) -> float:
    expected = list(expected_unchanged)
    if not expected:
        return 1.0

    preserved = 0
    for placement in expected:
        if world.block_at(placement.coord()) == placement.block_id:
            preserved += 1
    return preserved / len(expected)
