from __future__ import annotations

import hashlib
import random
from typing import Callable, Iterable, Sequence

from .types import ALL_TIERS, BlockPlacement, StructuralChecks, TaskSpec, Tier


_BUILD_BLOCKS = (
    "minecraft:stone",
    "minecraft:oak_planks",
    "minecraft:birch_planks",
    "minecraft:cobblestone",
    "minecraft:stone_bricks",
)

_CARDINAL_OFFSETS: dict[str, tuple[int, int, int]] = {
    "north": (0, 0, -1),
    "south": (0, 0, 1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
}


def generate_tasks(
    *,
    seed: int,
    per_tier: int,
    tiers: Sequence[Tier] | None = None,
) -> list[TaskSpec]:
    if per_tier <= 0:
        raise ValueError("per_tier must be > 0")

    selected = list(tiers or ALL_TIERS)
    tasks: list[TaskSpec] = []
    for tier in selected:
        for index in range(per_tier):
            tasks.append(generate_task(tier=tier, seed=seed, index=index))
    return tasks


def generate_task(*, tier: Tier, seed: int, index: int = 0) -> TaskSpec:
    derived = _derive_seed(seed=seed, tier=tier, index=index)
    rng = random.Random(derived)
    builder = _TASK_BUILDERS[tier]
    return builder(seed=derived, index=index, rng=rng)


def _derive_seed(*, seed: int, tier: Tier, index: int) -> int:
    payload = f"{seed}:{tier}:{index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16)


def _task_id(tier: Tier, seed: int, family: str, index: int) -> str:
    return f"{tier}:{family}:{seed}:{index}"


def _line_blocks(
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


def _filled_rect(
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


def _room_shell(
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
    placements: list[BlockPlacement] = []
    for x in range(ox, max_x + 1):
        for y in range(oy, max_y + 1):
            for z in range(oz, max_z + 1):
                if x in {ox, max_x} or z in {oz, max_z}:
                    placements.append(BlockPlacement(x=x, y=y, z=z, block_id=wall_block))
    return placements


def _tower(
    *,
    base: tuple[int, int, int],
    height: int,
    block_id: str,
) -> list[BlockPlacement]:
    bx, by, bz = base
    return [BlockPlacement(x=bx, y=by + dy, z=bz, block_id=block_id) for dy in range(height)]


def _build_t1_absolute(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    block_id = rng.choice(_BUILD_BLOCKS)
    x = rng.randint(-4, 4)
    z = rng.randint(-4, 4)
    y = 64
    target = [BlockPlacement(x=x, y=y, z=z, block_id=block_id)]
    prompt = f"Place one {block_id} block at absolute coordinates (x={x}, y={y}, z={z})."
    family = "absolute_single_block"
    return TaskSpec(
        task_id=_task_id("t1_absolute", seed, family, index),
        tier="t1_absolute",
        family=family,
        seed=seed,
        prompt=prompt,
        target_blocks=target,
        expected_tool_calls=2,
        structural_checks=StructuralChecks(require_grounded=True),
        metadata={"difficulty": "easy"},
    )


def _build_t2_relative(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    ref_x = rng.randint(-4, 4)
    ref_z = rng.randint(-4, 4)
    ref_y = 64
    reference = BlockPlacement(x=ref_x, y=ref_y, z=ref_z, block_id="minecraft:red_wool")
    block_id = rng.choice(tuple(block for block in _BUILD_BLOCKS if block != "minecraft:red_wool"))

    relation = rng.choice(["north", "south", "east", "west", "up"])
    if relation == "up":
        dx, dy, dz = (0, 1, 0)
        distance = 1
    else:
        dx, dy, dz = _CARDINAL_OFFSETS[relation]
        distance = rng.randint(1, 3)
    target = BlockPlacement(
        x=ref_x + (dx * distance),
        y=ref_y + (dy * distance),
        z=ref_z + (dz * distance),
        block_id=block_id,
    )
    prompt = (
        f"A minecraft:red_wool reference block is at ({ref_x}, {ref_y}, {ref_z}). "
        f"Place one {block_id} block {distance} blocks {relation} of that reference."
    )
    family = "relative_single_reference"
    return TaskSpec(
        task_id=_task_id("t2_relative_single_ref", seed, family, index),
        tier="t2_relative_single_ref",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=[reference],
        target_blocks=[target],
        expected_tool_calls=3,
        structural_checks=StructuralChecks(require_grounded=True),
        metadata={"relation": relation, "distance": distance},
    )


def _build_t3_primitives(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(["tower", "wall", "floor"])
    block_id = rng.choice(_BUILD_BLOCKS)
    if family == "tower":
        x = rng.randint(-4, 4)
        z = rng.randint(-4, 4)
        height = rng.randint(3, 5)
        target = _line_blocks(axis="y", start=(x, 64, z), length=height, block_id=block_id)
        prompt = f"Build a {height}-block-tall {block_id} tower at x={x}, z={z}, starting at y=64."
        metadata = {"height": height}
    elif family == "wall":
        start_x = rng.randint(-5, -1)
        length = rng.randint(5, 8)
        z = rng.randint(-3, 3)
        height = 3
        target = [
            BlockPlacement(x=x, y=y, z=z, block_id=block_id)
            for x in range(start_x, start_x + length)
            for y in range(64, 64 + height)
        ]
        prompt = (
            f"Build a straight {block_id} wall from x={start_x} to x={start_x + length - 1} at z={z}. "
            f"Make it {height} blocks tall."
        )
        metadata = {"length": length, "height": height}
    else:
        width = rng.randint(3, 5)
        depth = rng.randint(3, 5)
        ox = rng.randint(-4, 0)
        oz = rng.randint(-4, 0)
        target = _filled_rect(origin=(ox, 64, oz), width=width, depth=depth, block_id=block_id)
        prompt = (
            f"Build a flat {block_id} floor covering x={ox}..{ox + width - 1} "
            f"and z={oz}..{oz + depth - 1} at y=64."
        )
        metadata = {"width": width, "depth": depth}
    return TaskSpec(
        task_id=_task_id("t3_primitives", seed, family, index),
        tier="t3_primitives",
        family=family,
        seed=seed,
        prompt=prompt,
        target_blocks=target,
        expected_tool_calls=4,
        structural_checks=StructuralChecks(require_connected=True, require_grounded=True),
        metadata=metadata,
    )


def _build_t4_structure_relative(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(["top_of_tower", "south_face_marker"])
    if family == "top_of_tower":
        base_x = rng.randint(-4, 4)
        base_z = rng.randint(-4, 4)
        height = 4
        setup = _tower(base=(base_x, 64, base_z), height=height, block_id="minecraft:stone")
        target = [BlockPlacement(x=base_x, y=64 + height, z=base_z, block_id="minecraft:lantern")]
        prompt = (
            "There is a stone tower with base at "
            f"({base_x}, 64, {base_z}). Place one minecraft:lantern on top of the tower."
        )
        checks = StructuralChecks(require_grounded=True)
    else:
        room_origin = (-2, 64, -2)
        setup = _room_shell(origin=room_origin, width=5, height=3, depth=5, wall_block="minecraft:oak_planks")
        target = [BlockPlacement(x=0, y=65, z=2, block_id="minecraft:torch")]
        prompt = (
            "There is an oak plank room centered near you. "
            "Place one minecraft:torch at the center of the south wall (the wall with max z)."
        )
        checks = StructuralChecks(require_grounded=True)
    return TaskSpec(
        task_id=_task_id("t4_structure_relative", seed, family, index),
        tier="t4_structure_relative",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup,
        target_blocks=target,
        expected_tool_calls=6,
        structural_checks=checks,
        metadata={"requires_structure_inspection": True},
    )


def _build_t5_modification(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(["replace_material_preserve_shape", "widen_or_reposition_opening"])
    if family == "replace_material_preserve_shape":
        walls = _room_shell(origin=(-2, 64, -2), width=5, height=3, depth=5, wall_block="minecraft:oak_planks")
        floor = _filled_rect(origin=(-2, 63, -2), width=5, depth=5, block_id="minecraft:stone")
        setup = walls + floor
        target = [BlockPlacement(x=block.x, y=block.y, z=block.z, block_id="minecraft:birch_planks") for block in walls]
        preserved = floor + [
            BlockPlacement(x=x, y=y, z=z, block_id="minecraft:air")
            for x in range(-1, 2)
            for y in range(64, 67)
            for z in range(-1, 2)
        ]
        prompt = (
            "Replace every minecraft:oak_planks wall block in the room with minecraft:birch_planks, "
            "preserving the same wall coordinates and keeping the interior hollow."
        )
        checks = StructuralChecks(require_connected=True, require_grounded=True)
    else:
        setup = [
            BlockPlacement(x=x, y=y, z=0, block_id="minecraft:stone_bricks")
            for x in range(-3, 4)
            for y in range(64, 67)
        ]
        for doorway in ((0, 64, 0), (0, 65, 0)):
            setup.append(BlockPlacement(x=doorway[0], y=doorway[1], z=doorway[2], block_id="minecraft:air"))
        target = [
            BlockPlacement(x=x, y=y, z=0, block_id="minecraft:air")
            for x in (-1, 0, 1)
            for y in (64, 65)
        ]
        preserved = [
            BlockPlacement(x=x, y=y, z=0, block_id="minecraft:stone_bricks")
            for x in range(-3, 4)
            for y in range(64, 67)
            if not (x in {-1, 0, 1} and y in {64, 65})
        ]
        prompt = (
            "A stone_bricks wall has a 1-block-wide doorway centered at x=0. "
            "Widen the doorway to 3 blocks wide while preserving all other wall blocks."
        )
        checks = StructuralChecks(require_grounded=True)
    return TaskSpec(
        task_id=_task_id("t5_modification", seed, family, index),
        tier="t5_modification",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup,
        target_blocks=target,
        preserved_blocks=preserved,
        expected_tool_calls=7,
        structural_checks=checks,
        metadata={"requires_modification": True},
    )


def _build_t6_composition(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(["bridge_between_structures", "connect_rooms_with_corridor"])
    if family == "bridge_between_structures":
        left_tower = _tower(base=(-4, 64, 0), height=4, block_id="minecraft:stone")
        right_tower = _tower(base=(4, 64, 0), height=4, block_id="minecraft:stone")
        setup = left_tower + right_tower
        target = [
            BlockPlacement(x=x, y=67, z=0, block_id="minecraft:cobblestone")
            for x in range(-3, 4)
        ]
        preserved = left_tower + right_tower
        prompt = "Build a minecraft:cobblestone bridge connecting the tops of the two towers."
        checks = StructuralChecks(require_connected=True, min_span=7, span_axis="x")
    else:
        room_a = _room_shell(origin=(-2, 64, -8), width=5, height=3, depth=5, wall_block="minecraft:stone_bricks")
        room_b = _room_shell(origin=(-2, 64, 4), width=5, height=3, depth=5, wall_block="minecraft:stone_bricks")
        setup = room_a + room_b
        for doorway in ((0, 64, -4), (0, 65, -4), (0, 64, 4), (0, 65, 4)):
            setup.append(BlockPlacement(x=doorway[0], y=doorway[1], z=doorway[2], block_id="minecraft:air"))

        target: list[BlockPlacement] = []
        for z in range(-3, 4):
            target.append(BlockPlacement(x=0, y=64, z=z, block_id="minecraft:stone_bricks"))
            target.append(BlockPlacement(x=0, y=66, z=z, block_id="minecraft:stone_bricks"))
            target.append(BlockPlacement(x=-1, y=64, z=z, block_id="minecraft:stone_bricks"))
            target.append(BlockPlacement(x=1, y=64, z=z, block_id="minecraft:stone_bricks"))
            target.append(BlockPlacement(x=-1, y=65, z=z, block_id="minecraft:stone_bricks"))
            target.append(BlockPlacement(x=1, y=65, z=z, block_id="minecraft:stone_bricks"))
        preserved = room_a + room_b
        prompt = (
            "Connect the two rooms with a one-block-wide stone_bricks corridor between the facing doorways."
        )
        checks = StructuralChecks(require_connected=True, require_grounded=True, min_span=7, span_axis="z")
    return TaskSpec(
        task_id=_task_id("t6_composition", seed, family, index),
        tier="t6_composition",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup,
        target_blocks=target,
        preserved_blocks=preserved,
        expected_tool_calls=9,
        structural_checks=checks,
        metadata={"compositional": True},
    )


def tier_counts(tasks: Iterable[TaskSpec]) -> dict[Tier, int]:
    counts: dict[Tier, int] = {tier: 0 for tier in ALL_TIERS}
    for task in tasks:
        counts[task.tier] = counts.get(task.tier, 0) + 1
    return counts


_TASK_BUILDERS: dict[Tier, Callable[..., TaskSpec]] = {
    "t1_absolute": _build_t1_absolute,
    "t2_relative_single_ref": _build_t2_relative,
    "t3_primitives": _build_t3_primitives,
    "t4_structure_relative": _build_t4_structure_relative,
    "t5_modification": _build_t5_modification,
    "t6_composition": _build_t6_composition,
}
