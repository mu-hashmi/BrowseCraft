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


class _ForcedFamilyRandom:
    def __init__(self, seed: int, family: str) -> None:
        self._rng = random.Random(seed)
        self._family = family
        self._forced = False

    def choice(self, seq):
        choice = self._rng.choice(seq)
        if not self._forced and self._family in seq:
            self._forced = True
            return self._family
        return choice

    def __getattr__(self, name: str):
        return getattr(self._rng, name)


def reconstruct_task_from_task_id(task_id: str) -> TaskSpec:
    tier, family, seed_str, index_str = task_id.split(":", maxsplit=3)
    builder = _TASK_BUILDERS[tier]
    derived_seed = int(seed_str)
    return builder(
        seed=derived_seed,
        index=int(index_str),
        rng=_ForcedFamilyRandom(seed=derived_seed, family=family),
    )


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


def _dedupe_blocks(blocks: Iterable[BlockPlacement]) -> list[BlockPlacement]:
    deduped: dict[tuple[int, int, int], BlockPlacement] = {}
    for block in blocks:
        deduped[block.coord()] = block
    return list(deduped.values())


def _remove_coords(
    blocks: Iterable[BlockPlacement],
    removed_coords: set[tuple[int, int, int]],
) -> list[BlockPlacement]:
    return [block for block in blocks if block.coord() not in removed_coords]


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


def _corridor_segment_x(
    *,
    x_start: int,
    x_end: int,
    z: int,
    y: int,
    block_id: str,
) -> list[BlockPlacement]:
    min_x = min(x_start, x_end)
    max_x = max(x_start, x_end)
    blocks: list[BlockPlacement] = []
    for x in range(min_x, max_x + 1):
        blocks.append(BlockPlacement(x=x, y=y, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 2, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y, z=z - 1, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 1, z=z - 1, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y, z=z + 1, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 1, z=z + 1, block_id=block_id))
    return blocks


def _corridor_segment_z(
    *,
    x: int,
    z_start: int,
    z_end: int,
    y: int,
    block_id: str,
) -> list[BlockPlacement]:
    min_z = min(z_start, z_end)
    max_z = max(z_start, z_end)
    blocks: list[BlockPlacement] = []
    for z in range(min_z, max_z + 1):
        blocks.append(BlockPlacement(x=x, y=y, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 2, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x - 1, y=y, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x - 1, y=y + 1, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x + 1, y=y, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x + 1, y=y + 1, z=z, block_id=block_id))
    return blocks


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
        expected_tool_calls=1,
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
        expected_tool_calls=2,
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
        expected_tool_calls=2,
        structural_checks=StructuralChecks(require_connected=True, require_grounded=True),
        metadata=metadata,
    )


def _build_t4_structure_relative(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(
        [
            "top_of_tower",
            "south_face_marker",
            "inside_room_through_doorway",
            "shorter_tower_marker",
        ]
    )
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
    elif family == "south_face_marker":
        room_origin = (rng.randint(-6, 2), 64, rng.randint(-6, 2))
        room_width = 5
        room_depth = 5
        setup = _room_shell(origin=room_origin, width=5, height=3, depth=5, wall_block="minecraft:oak_planks")
        south_face_center_x = room_origin[0] + (room_width // 2)
        south_face_z = room_origin[2] + room_depth - 1
        target = [BlockPlacement(x=south_face_center_x, y=65, z=south_face_z, block_id="minecraft:torch")]
        prompt = (
            "There is an oak plank room centered near you. "
            "Replace the center block of the south wall (the wall with max z) with one minecraft:torch."
        )
        checks = StructuralChecks(require_grounded=True)
    elif family == "inside_room_through_doorway":
        room_origin = (rng.randint(-6, 1), 64, rng.randint(-6, 1))
        doorway_wall = rng.choice(["north", "south", "east", "west"])
        doorway_coords = {
            "north": {(room_origin[0] + 2, 64, room_origin[2]), (room_origin[0] + 2, 65, room_origin[2])},
            "south": {(room_origin[0] + 2, 64, room_origin[2] + 4), (room_origin[0] + 2, 65, room_origin[2] + 4)},
            "west": {(room_origin[0], 64, room_origin[2] + 2), (room_origin[0], 65, room_origin[2] + 2)},
            "east": {(room_origin[0] + 4, 64, room_origin[2] + 2), (room_origin[0] + 4, 65, room_origin[2] + 2)},
        }
        inside_targets = {
            "north": (room_origin[0] + 2, 64, room_origin[2] + 1),
            "south": (room_origin[0] + 2, 64, room_origin[2] + 3),
            "west": (room_origin[0] + 1, 64, room_origin[2] + 2),
            "east": (room_origin[0] + 3, 64, room_origin[2] + 2),
        }
        setup = _remove_coords(
            _room_shell(origin=room_origin, width=5, height=3, depth=5, wall_block="minecraft:oak_planks"),
            doorway_coords[doorway_wall],
        )
        target_coord = inside_targets[doorway_wall]
        target = [BlockPlacement(x=target_coord[0], y=target_coord[1], z=target_coord[2], block_id="minecraft:lantern")]
        prompt = (
            "There is an oak plank room with exactly one doorway. "
            "Place one minecraft:lantern on the floor tile immediately inside the doorway."
        )
        checks = StructuralChecks(require_grounded=True)
    else:
        left_base = (rng.randint(-8, -4), 64, rng.randint(-3, 3))
        right_base = (rng.randint(4, 8), 64, left_base[2] + rng.randint(-1, 1))
        short_height = rng.randint(3, 4)
        tall_height = short_height + rng.randint(1, 2)
        shorter_side = rng.choice(["left", "right"])
        left_height = short_height if shorter_side == "left" else tall_height
        right_height = short_height if shorter_side == "right" else tall_height
        left_tower = _tower(base=left_base, height=left_height, block_id="minecraft:stone")
        right_tower = _tower(base=right_base, height=right_height, block_id="minecraft:stone")
        setup = left_tower + right_tower
        shorter_base = left_base if shorter_side == "left" else right_base
        target = [
            BlockPlacement(
                x=shorter_base[0],
                y=64 + short_height,
                z=shorter_base[2],
                block_id="minecraft:torch",
            )
        ]
        prompt = (
            "There are two nearby stone towers with different heights. "
            "Place one minecraft:torch on top of the shorter tower."
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
        expected_tool_calls=4,
        structural_checks=checks,
        metadata={"requires_structure_inspection": True},
    )


def _build_t5_modification(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(
        [
            "replace_material_preserve_shape",
            "widen_or_reposition_opening",
            "add_window_to_wall",
            "move_window_to_opposite_wall",
        ]
    )
    if family == "replace_material_preserve_shape":
        origin = (rng.randint(-5, 1), 64, rng.randint(-5, 1))
        walls = _room_shell(origin=origin, width=5, height=3, depth=5, wall_block="minecraft:oak_planks")
        floor = _filled_rect(origin=(origin[0], 63, origin[2]), width=5, depth=5, block_id="minecraft:stone")
        setup = walls + floor
        target = [BlockPlacement(x=block.x, y=block.y, z=block.z, block_id="minecraft:birch_planks") for block in walls]
        preserved = floor + [
            BlockPlacement(x=x, y=y, z=z, block_id="minecraft:air")
            for x in range(origin[0] + 1, origin[0] + 4)
            for y in range(64, 67)
            for z in range(origin[2] + 1, origin[2] + 4)
        ]
        prompt = (
            "Replace every minecraft:oak_planks wall block in the room with minecraft:birch_planks, "
            "preserving the same wall coordinates and keeping the interior hollow."
        )
        checks = StructuralChecks(require_connected=True, require_grounded=True)
    elif family == "widen_or_reposition_opening":
        wall_z = rng.randint(-4, 4)
        center_x = rng.randint(-1, 1)
        existing_doorway = {(center_x, 64, wall_z), (center_x, 65, wall_z)}
        setup = [
            BlockPlacement(x=x, y=y, z=wall_z, block_id="minecraft:stone_bricks")
            for x in range(center_x - 3, center_x + 4)
            for y in range(64, 67)
            if (x, y, wall_z) not in existing_doorway
        ]
        target = [
            BlockPlacement(x=x, y=y, z=wall_z, block_id="minecraft:air")
            for x in (center_x - 1, center_x + 1)
            for y in (64, 65)
        ]
        preserved = [
            BlockPlacement(x=x, y=y, z=wall_z, block_id="minecraft:stone_bricks")
            for x in range(center_x - 3, center_x + 4)
            for y in range(64, 67)
            if not (x in {center_x - 1, center_x, center_x + 1} and y in {64, 65})
        ]
        prompt = (
            "A stone_bricks wall has a 1-block-wide doorway near the center. "
            "Widen the doorway to 3 blocks wide while preserving all other wall blocks."
        )
        checks = StructuralChecks()
    elif family == "add_window_to_wall":
        origin = (rng.randint(-5, 1), 64, rng.randint(-5, 1))
        shell = _room_shell(origin=origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks")
        floor = _filled_rect(origin=(origin[0], 63, origin[2]), width=5, depth=5, block_id="minecraft:cobblestone")
        target = [
            BlockPlacement(x=origin[0] + 4, y=y, z=z, block_id="minecraft:air")
            for y in (64, 65)
            for z in (origin[2] + 1, origin[2] + 2)
        ]
        preserved = floor + _remove_coords(shell, {block.coord() for block in target})
        setup = shell + floor
        prompt = "Add a 2-by-2 window to the east wall of the stone_bricks room and keep the rest unchanged."
        checks = StructuralChecks()
    else:
        origin = (rng.randint(-5, 1), 64, rng.randint(-5, 1))
        west_window = {
            (origin[0], 64, origin[2] + 1),
            (origin[0], 64, origin[2] + 2),
            (origin[0], 65, origin[2] + 1),
            (origin[0], 65, origin[2] + 2),
        }
        east_window = {
            (origin[0] + 4, 64, origin[2] + 1),
            (origin[0] + 4, 64, origin[2] + 2),
            (origin[0] + 4, 65, origin[2] + 1),
            (origin[0] + 4, 65, origin[2] + 2),
        }
        shell = _remove_coords(
            _room_shell(origin=origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks"),
            west_window,
        )
        floor = _filled_rect(origin=(origin[0], 63, origin[2]), width=5, depth=5, block_id="minecraft:cobblestone")
        target = [
            *[
                BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id="minecraft:stone_bricks")
                for coord in sorted(west_window)
            ],
            *[
                BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id="minecraft:air")
                for coord in sorted(east_window)
            ],
        ]
        preserved = floor + _remove_coords(shell, east_window)
        setup = shell + floor
        prompt = (
            "Move the existing 2-by-2 window from the west wall to the east wall of the room. "
            "Seal the old opening with stone_bricks."
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
        expected_tool_calls=8,
        structural_checks=checks,
        metadata={"requires_modification": True},
    )


def _build_t6_composition(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(
        [
            "bridge_between_structures",
            "connect_rooms_with_corridor",
            "bridge_between_offset_towers",
            "l_shaped_corridor_offset_rooms",
        ]
    )
    if family == "bridge_between_structures":
        left_base = (rng.randint(-8, -5), 64, rng.randint(-2, 2))
        right_base = (left_base[0] + rng.randint(6, 8), 64, left_base[2])
        left_tower = _tower(base=left_base, height=4, block_id="minecraft:stone")
        right_tower = _tower(base=right_base, height=4, block_id="minecraft:stone")
        setup = left_tower + right_tower
        target = [
            BlockPlacement(x=x, y=68, z=left_base[2], block_id="minecraft:cobblestone")
            for x in range(left_base[0], right_base[0] + 1)
        ]
        preserved = left_tower + right_tower
        prompt = "Build a minecraft:cobblestone bridge connecting the tops of the two towers."
        checks = StructuralChecks(require_connected=True, min_span=right_base[0] - left_base[0] + 1, span_axis="x")
    elif family == "connect_rooms_with_corridor":
        origin_x = rng.randint(-4, 0)
        top_room_origin = (origin_x, 64, rng.randint(-10, -7))
        bottom_room_origin = (origin_x, 64, top_room_origin[2] + rng.randint(11, 13))
        top_doorway = {
            (top_room_origin[0] + 2, 64, top_room_origin[2] + 4),
            (top_room_origin[0] + 2, 65, top_room_origin[2] + 4),
        }
        bottom_doorway = {
            (bottom_room_origin[0] + 2, 64, bottom_room_origin[2]),
            (bottom_room_origin[0] + 2, 65, bottom_room_origin[2]),
        }
        room_a = _remove_coords(
            _room_shell(origin=top_room_origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks"),
            top_doorway,
        )
        room_b = _remove_coords(
            _room_shell(origin=bottom_room_origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks"),
            bottom_doorway,
        )
        setup = room_a + room_b
        target = _corridor_segment_z(
            x=top_room_origin[0] + 2,
            z_start=top_room_origin[2] + 5,
            z_end=bottom_room_origin[2] - 1,
            y=64,
            block_id="minecraft:stone_bricks",
        )
        preserved = room_a + room_b
        prompt = (
            "Connect the two rooms with a one-block-wide stone_bricks corridor between the facing doorways."
        )
        checks = StructuralChecks(
            require_connected=True,
            require_grounded=True,
            min_span=bottom_room_origin[2] - top_room_origin[2] - 3,
            span_axis="z",
        )
    elif family == "bridge_between_offset_towers":
        left_base = (rng.randint(-8, -5), 64, rng.randint(-5, -2))
        right_base = (rng.randint(4, 8), 64, rng.randint(2, 5))
        left_tower = _tower(base=left_base, height=4, block_id="minecraft:stone")
        right_tower = _tower(base=right_base, height=4, block_id="minecraft:stone")
        setup = left_tower + right_tower
        x_path = [
            BlockPlacement(x=x, y=67, z=left_base[2], block_id="minecraft:cobblestone")
            for x in range(left_base[0] + 1, right_base[0] + 1)
        ]
        z_path = [
            BlockPlacement(x=right_base[0], y=67, z=z, block_id="minecraft:cobblestone")
            for z in range(min(left_base[2], right_base[2]), max(left_base[2], right_base[2]) + 1)
        ]
        target = _dedupe_blocks([*x_path, *z_path])
        preserved = left_tower + right_tower
        prompt = "Build an L-shaped minecraft:cobblestone bridge connecting the tops of the two offset towers."
        checks = StructuralChecks(require_connected=True)
    else:
        room_a_origin = (rng.randint(-10, -7), 64, rng.randint(-9, -6))
        room_b_origin = (room_a_origin[0] + rng.randint(8, 10), 64, room_a_origin[2] + rng.randint(8, 10))
        room_a_doorway = {
            (room_a_origin[0] + 4, 64, room_a_origin[2] + 2),
            (room_a_origin[0] + 4, 65, room_a_origin[2] + 2),
        }
        room_b_doorway = {
            (room_b_origin[0] + 2, 64, room_b_origin[2]),
            (room_b_origin[0] + 2, 65, room_b_origin[2]),
        }
        room_a = _remove_coords(
            _room_shell(origin=room_a_origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks"),
            room_a_doorway,
        )
        room_b = _remove_coords(
            _room_shell(origin=room_b_origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks"),
            room_b_doorway,
        )
        setup = room_a + room_b
        turn_x = room_b_origin[0] + 2
        segment_x = _corridor_segment_x(
            x_start=room_a_origin[0] + 5,
            x_end=turn_x,
            z=room_a_origin[2] + 2,
            y=64,
            block_id="minecraft:stone_bricks",
        )
        segment_z = _corridor_segment_z(
            x=turn_x,
            z_start=room_a_origin[2] + 2,
            z_end=room_b_origin[2] - 1,
            y=64,
            block_id="minecraft:stone_bricks",
        )
        target = _dedupe_blocks([*segment_x, *segment_z])
        preserved = room_a + room_b
        prompt = (
            "Connect the two offset rooms with an L-shaped one-block-wide stone_bricks corridor between their doorways."
        )
        checks = StructuralChecks(require_connected=True, require_grounded=True)
    return TaskSpec(
        task_id=_task_id("t6_composition", seed, family, index),
        tier="t6_composition",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup,
        target_blocks=target,
        preserved_blocks=preserved,
        expected_tool_calls=8,
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
