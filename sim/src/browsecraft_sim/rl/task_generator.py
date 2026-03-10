from __future__ import annotations

import hashlib
import random
from typing import Callable, Iterable, Sequence

from .spatial_worlds import (
    CARDINAL_OFFSETS as _CARDINAL_OFFSETS,
    MARKER_BLOCKS as _MARKER_BLOCKS,
    OPPOSITE_CARDINAL as _OPPOSITE_CARDINAL,
    block_name as _shared_block_name,
    chain_positions as _shared_chain_positions,
    dedupe_blocks as _shared_dedupe_blocks,
    enclosure_shell as _shared_enclosure_shell,
    filled_rect as _shared_filled_rect,
    horizontal_facing_offset as _shared_horizontal_facing_offset,
    line_blocks as _shared_line_blocks,
    marker_name as _shared_marker_name,
    occupied_coords as _shared_occupied_coords,
    player_relative_offset as _shared_player_relative_offset,
    remove_coords as _shared_remove_coords,
    room_shell as _shared_room_shell,
    tower as _shared_tower,
)
from .types import ALL_TIERS, BlockPlacement, PlayerSpec, StructuralChecks, TaskSpec, Tier


_BUILD_BLOCKS = (
    "minecraft:stone",
    "minecraft:oak_planks",
    "minecraft:birch_planks",
    "minecraft:cobblestone",
    "minecraft:stone_bricks",
    "minecraft:sandstone",
    "minecraft:deepslate_bricks",
)
_DISTRACTOR_SENTENCES = (
    "There is also an unrelated birch plank pillar nearby. Ignore it.",
    "A separate cobblestone marker sits off to the side and is irrelevant to this task.",
    "An extra stone pillar is nearby, but it does not matter for this task.",
)


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


def sample_weighted_tasks(
    *,
    seed: int,
    total_tasks: int,
    tier_weights: dict[Tier, int],
    tiers: Sequence[Tier] | None = None,
) -> list[TaskSpec]:
    if total_tasks <= 0:
        raise ValueError("total_tasks must be > 0")

    selected = list(tiers or ALL_TIERS)
    missing = [tier for tier in selected if tier not in tier_weights]
    if missing:
        raise ValueError(f"missing tier weights for: {', '.join(missing)}")

    rng = random.Random(_derive_seed(seed=seed, tier=selected[0], index=total_tasks))
    next_index = {tier: 0 for tier in selected}
    weights = [tier_weights[tier] for tier in selected]
    tasks: list[TaskSpec] = []
    for _ in range(total_tasks):
        tier = rng.choices(selected, weights=weights, k=1)[0]
        index = next_index[tier]
        next_index[tier] += 1
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


def _canonical_intent(*, family: str, **fields: object) -> dict[str, object]:
    return {"family": family, **fields}


def _block_name(block_id: str) -> str:
    return _shared_block_name(block_id)


def _marker_name(block_id: str) -> str:
    return _shared_marker_name(block_id)


def _line_blocks(
    *,
    axis: str,
    start: tuple[int, int, int],
    length: int,
    block_id: str,
) -> list[BlockPlacement]:
    return _shared_line_blocks(axis=axis, start=start, length=length, block_id=block_id)


def _filled_rect(
    *,
    origin: tuple[int, int, int],
    width: int,
    depth: int,
    block_id: str,
) -> list[BlockPlacement]:
    return _shared_filled_rect(origin=origin, width=width, depth=depth, block_id=block_id)


def _dedupe_blocks(blocks: Iterable[BlockPlacement]) -> list[BlockPlacement]:
    return _shared_dedupe_blocks(blocks)


def _remove_coords(
    blocks: Iterable[BlockPlacement],
    removed_coords: set[tuple[int, int, int]],
) -> list[BlockPlacement]:
    return _shared_remove_coords(blocks, removed_coords)


def _room_shell(
    *,
    origin: tuple[int, int, int],
    width: int,
    height: int,
    depth: int,
    wall_block: str,
) -> list[BlockPlacement]:
    return _shared_room_shell(origin=origin, width=width, height=height, depth=depth, wall_block=wall_block)


def _enclosure_shell(
    *,
    origin: tuple[int, int, int],
    width: int,
    depth: int,
    height: int,
    wall_block: str,
) -> list[BlockPlacement]:
    return _shared_enclosure_shell(origin=origin, width=width, depth=depth, height=height, wall_block=wall_block)


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
        blocks.append(BlockPlacement(x=x, y=y - 1, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 2, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 2, z=z - 1, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 2, z=z + 1, block_id=block_id))
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
        blocks.append(BlockPlacement(x=x, y=y - 1, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x, y=y + 2, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x - 1, y=y + 2, z=z, block_id=block_id))
        blocks.append(BlockPlacement(x=x + 1, y=y + 2, z=z, block_id=block_id))
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
    return _shared_tower(base=base, height=height, block_id=block_id)


def _occupied_coords(*groups: Iterable[BlockPlacement]) -> set[tuple[int, int, int]]:
    return _shared_occupied_coords(*groups)


def _horizontal_facing_offset(facing: str) -> tuple[int, int]:
    return _shared_horizontal_facing_offset(facing)


def _player_relative_offset(facing: str, relation: str, distance: int) -> tuple[int, int]:
    return _shared_player_relative_offset(facing, relation, distance)


def _chain_positions(
    *,
    rng: random.Random,
    start: tuple[int, int, int],
    hop_count: int,
    step_distance: int,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    return _shared_chain_positions(rng=rng, start=start, hop_count=hop_count, step_distance=step_distance)


def _add_optional_distractor(
    *,
    rng: random.Random,
    tier: Tier,
    prompt: str,
    setup: list[BlockPlacement],
    target: list[BlockPlacement],
    preserved: list[BlockPlacement],
    metadata: dict[str, object],
) -> tuple[str, list[BlockPlacement], dict[str, object]]:
    if tier not in {"t4_structure_relative", "t5_modification", "t6_composition"}:
        return prompt, setup, metadata
    if rng.random() >= 0.6:
        return prompt, setup, metadata

    occupied = _occupied_coords(setup, target, preserved)
    candidate_bases = [(-18, 64, -18), (-18, 64, 18), (18, 64, -18), (18, 64, 18)]
    distractor_base: tuple[int, int, int] | None = None
    distractor_blocks: list[BlockPlacement] = []
    for base in candidate_bases:
        blocks = _tower(base=base, height=3, block_id="minecraft:birch_planks")
        if _occupied_coords(blocks).isdisjoint(occupied):
            distractor_base = base
            distractor_blocks = blocks
            break
    if distractor_base is None:
        return prompt, setup, metadata

    sentence = rng.choice(_DISTRACTOR_SENTENCES)
    updated_metadata = dict(metadata)
    updated_metadata["distractor"] = {
        "kind": "pillar",
        "base": {"x": distractor_base[0], "y": distractor_base[1], "z": distractor_base[2]},
        "height": 3,
        "block_id": "minecraft:birch_planks",
    }
    return f"{prompt} {sentence}", setup + distractor_blocks, updated_metadata


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
        metadata={
            "difficulty": "easy",
            "canonical_intent": _canonical_intent(
                family=family,
                block_id=block_id,
                coordinate={"x": x, "y": y, "z": z},
            ),
        },
    )


def _build_t2_relative(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(["relative_single_reference", "egocentric_relative"])
    if family == "relative_single_reference":
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
        metadata = {
            "relation": relation,
            "distance": distance,
            "canonical_intent": _canonical_intent(
                family=family,
                block_id=block_id,
                reference={"x": ref_x, "y": ref_y, "z": ref_z},
                relation=relation,
                distance=distance,
            ),
        }
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
            metadata=metadata,
        )

    facing = rng.choice(["north", "south", "east", "west"])
    relation = rng.choice(["front", "behind", "left", "right"])
    relation_text = {
        "front": "in front of you",
        "behind": "behind you",
        "left": "to your left",
        "right": "to your right",
    }[relation]
    distance = rng.randint(1, 3)
    block_id = rng.choice(_BUILD_BLOCKS)
    dx, dz = _player_relative_offset(facing, relation, distance)
    target = [
        BlockPlacement(
            x=dx,
            y=64,
            z=dz,
            block_id=block_id,
        )
    ]
    prompt = f"Place one {block_id} block {distance} blocks {relation_text}."
    return TaskSpec(
        task_id=_task_id("t2_relative_single_ref", seed, family, index),
        tier="t2_relative_single_ref",
        family=family,
        seed=seed,
        prompt=prompt,
        player=PlayerSpec(facing=facing),
        target_blocks=target,
        expected_tool_calls=1,
        structural_checks=StructuralChecks(require_grounded=True),
        metadata={
            "frame": "player",
            "relation": relation,
            "distance": distance,
            "canonical_intent": _canonical_intent(
                family=family,
                block_id=block_id,
                player_facing=facing,
                relation=relation,
                distance=distance,
            ),
        },
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
        metadata = {
            "height": height,
            "canonical_intent": _canonical_intent(
                family=family,
                block_id=block_id,
                base={"x": x, "y": 64, "z": z},
                height=height,
            ),
        }
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
            f"The wall should run from y=64 through y={64 + height - 1}."
        )
        metadata = {
            "length": length,
            "height": height,
            "canonical_intent": _canonical_intent(
                family=family,
                block_id=block_id,
                start_x=start_x,
                end_x=start_x + length - 1,
                z=z,
                height=height,
            ),
        }
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
        metadata = {
            "width": width,
            "depth": depth,
            "canonical_intent": _canonical_intent(
                family=family,
                block_id=block_id,
                origin={"x": ox, "y": 64, "z": oz},
                width=width,
                depth=depth,
            ),
        }
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
            "marker_chain_place",
            "structure_chain_place",
            "mark_structure_inside_enclosure",
        ]
    )
    setup: list[BlockPlacement]
    target: list[BlockPlacement]
    checks: StructuralChecks
    expected_tool_calls = 4
    metadata: dict[str, object]
    prompt: str

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
        metadata = {
            "requires_structure_inspection": True,
            "canonical_intent": _canonical_intent(
                family=family,
                tower_base={"x": base_x, "y": 64, "z": base_z},
                tower_height=height,
                target_block="minecraft:lantern",
            ),
        }
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
        metadata = {
            "requires_structure_inspection": True,
            "canonical_intent": _canonical_intent(
                family=family,
                room_origin={"x": room_origin[0], "y": room_origin[1], "z": room_origin[2]},
                target_wall="south",
                target_block="minecraft:torch",
            ),
        }
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
        metadata = {
            "requires_structure_inspection": True,
            "canonical_intent": _canonical_intent(
                family=family,
                room_origin={"x": room_origin[0], "y": room_origin[1], "z": room_origin[2]},
                doorway_wall=doorway_wall,
                target_block="minecraft:lantern",
            ),
        }
    elif family == "shorter_tower_marker":
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
            "Two stone towers are already built nearby, and one is shorter than the other. "
            "Each tower is a single vertical stone column, and both tower bases are within 10 blocks of you on the same flat ground level. "
            "Do not build or modify either tower. "
            "Inspect them, identify the shorter existing tower, and place exactly one minecraft:torch "
            "in the air block directly above that tower's highest stone block. Do not place a torch anywhere else."
        )
        checks = StructuralChecks(require_grounded=True)
        metadata = {
            "requires_structure_inspection": True,
            "canonical_intent": _canonical_intent(
                family=family,
                left_base={"x": left_base[0], "y": left_base[1], "z": left_base[2]},
                right_base={"x": right_base[0], "y": right_base[1], "z": right_base[2]},
                left_height=left_height,
                right_height=right_height,
                target_block="minecraft:torch",
            ),
        }
    elif family == "marker_chain_place":
        hop_count = rng.randint(2, 6)
        step_distance = 3
        positions, steps = _chain_positions(
            rng=rng,
            start=(rng.randint(-8, -2), 64, rng.randint(-8, -2)),
            hop_count=hop_count,
            step_distance=step_distance,
        )
        markers = list(_MARKER_BLOCKS[: hop_count + 1])
        setup = [
            BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id=block_id)
            for coord, block_id in zip(positions, markers, strict=True)
        ]
        final_coord = positions[-1]
        target = [BlockPlacement(x=final_coord[0], y=final_coord[1] + 1, z=final_coord[2], block_id="minecraft:gold_block")]
        steps_text = ", then ".join(f"move to the marker {step_distance} blocks {step} of that marker" for step in steps)
        prompt = (
            "Colored wool markers are placed nearby. "
            f"Start at the {_marker_name(markers[0])}, {steps_text}, and then place one minecraft:gold_block "
            "one block above the final marker."
        )
        checks = StructuralChecks(require_grounded=True)
        expected_tool_calls = 6
        metadata = {
            "requires_structure_inspection": True,
            "hop_count": hop_count,
            "canonical_intent": _canonical_intent(
                family=family,
                step_distance=step_distance,
                start_marker=markers[0],
                steps=steps,
                markers=[
                    {
                        "name": _marker_name(block_id),
                        "block_id": block_id,
                        "x": coord[0],
                        "y": coord[1],
                        "z": coord[2],
                    }
                    for coord, block_id in zip(positions, markers, strict=True)
                ],
                target_block="minecraft:gold_block",
            ),
        }
    elif family == "structure_chain_place":
        hop_count = rng.randint(2, 6)
        step_distance = 5
        positions, steps = _chain_positions(
            rng=rng,
            start=(rng.randint(-10, -4), 64, rng.randint(-10, -4)),
            hop_count=hop_count,
            step_distance=step_distance,
        )
        structure_blocks = list(_BUILD_BLOCKS[: hop_count + 1])
        heights = [3 + (idx % 2) for idx in range(hop_count + 1)]
        setup = _dedupe_blocks(
            block
            for coord, block_id, height in zip(positions, structure_blocks, heights, strict=True)
            for block in _tower(base=coord, height=height, block_id=block_id)
        )
        final_coord = positions[-1]
        final_height = heights[-1]
        target = [BlockPlacement(x=final_coord[0], y=final_coord[1] + final_height, z=final_coord[2], block_id="minecraft:torch")]
        steps_text = ", then ".join(f"move to the tower {step_distance} blocks {step} of that tower" for step in steps)
        prompt = (
            "Several material-coded towers are nearby. "
            f"Start at the {_block_name(structure_blocks[0])} tower, {steps_text}, and then place one minecraft:torch "
            "on top of the final tower."
        )
        checks = StructuralChecks(require_grounded=True)
        expected_tool_calls = 6
        metadata = {
            "requires_structure_inspection": True,
            "hop_count": hop_count,
            "canonical_intent": _canonical_intent(
                family=family,
                step_distance=step_distance,
                start_structure=structure_blocks[0],
                steps=steps,
                structures=[
                    {
                        "name": f"{_block_name(block_id)} tower",
                        "block_id": block_id,
                        "height": height,
                        "x": coord[0],
                        "y": coord[1],
                        "z": coord[2],
                    }
                    for coord, block_id, height in zip(positions, structure_blocks, heights, strict=True)
                ],
                target_block="minecraft:torch",
            ),
        }
    else:
        enclosure_origin = (rng.randint(-7, -2), 64, rng.randint(-7, -2))
        enclosure = _enclosure_shell(
            origin=enclosure_origin,
            width=7,
            depth=7,
            height=2,
            wall_block="minecraft:stone_bricks",
        )
        inside_base = (enclosure_origin[0] + 3, 64, enclosure_origin[2] + 3)
        outside_a = (enclosure_origin[0] - 4, 64, enclosure_origin[2] + 1)
        outside_b = (enclosure_origin[0] + 9, 64, enclosure_origin[2] + 5)
        inside_tower = _tower(base=inside_base, height=3, block_id="minecraft:stone")
        outside_tower_a = _tower(base=outside_a, height=3, block_id="minecraft:stone")
        outside_tower_b = _tower(base=outside_b, height=3, block_id="minecraft:stone")
        setup = enclosure + inside_tower + outside_tower_a + outside_tower_b
        target = [BlockPlacement(x=inside_base[0], y=67, z=inside_base[2], block_id="minecraft:lantern")]
        prompt = (
            "Three nearby stone towers are visible and one of them is inside a stone_bricks enclosure. "
            "Place one minecraft:lantern on top of the tower that is inside the enclosure."
        )
        checks = StructuralChecks(require_grounded=True)
        expected_tool_calls = 5
        metadata = {
            "requires_structure_inspection": True,
            "canonical_intent": _canonical_intent(
                family=family,
                enclosure_origin={"x": enclosure_origin[0], "y": enclosure_origin[1], "z": enclosure_origin[2]},
                inside_tower={"x": inside_base[0], "y": inside_base[1], "z": inside_base[2]},
                enclosure_block_id="minecraft:stone_bricks",
                target_block="minecraft:lantern",
            ),
        }

    prompt, setup, metadata = _add_optional_distractor(
        rng=rng,
        tier="t4_structure_relative",
        prompt=prompt,
        setup=setup,
        target=target,
        preserved=[],
        metadata=metadata,
    )
    return TaskSpec(
        task_id=_task_id("t4_structure_relative", seed, family, index),
        tier="t4_structure_relative",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup,
        target_blocks=target,
        expected_tool_calls=expected_tool_calls,
        structural_checks=checks,
        metadata=metadata,
    )


def _build_t5_modification(*, seed: int, index: int, rng: random.Random) -> TaskSpec:
    family = rng.choice(
        [
            "replace_material_preserve_shape",
            "widen_or_reposition_opening",
            "add_window_to_wall",
            "move_window_to_opposite_wall",
            "add_shared_wall_doorway",
        ]
    )
    setup: list[BlockPlacement]
    target: list[BlockPlacement]
    preserved: list[BlockPlacement]
    checks: StructuralChecks
    prompt: str
    metadata: dict[str, object]
    expected_tool_calls = 8

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
        metadata = {
            "requires_modification": True,
            "canonical_intent": _canonical_intent(
                family=family,
                room_origin={"x": origin[0], "y": origin[1], "z": origin[2]},
                from_block="minecraft:oak_planks",
                to_block="minecraft:birch_planks",
            ),
        }
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
        metadata = {
            "requires_modification": True,
            "canonical_intent": _canonical_intent(
                family=family,
                wall_z=wall_z,
                center_x=center_x,
            ),
        }
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
        metadata = {
            "requires_modification": True,
            "canonical_intent": _canonical_intent(
                family=family,
                room_origin={"x": origin[0], "y": origin[1], "z": origin[2]},
                target_wall="east",
            ),
        }
    elif family == "move_window_to_opposite_wall":
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
        metadata = {
            "requires_modification": True,
            "canonical_intent": _canonical_intent(
                family=family,
                room_origin={"x": origin[0], "y": origin[1], "z": origin[2]},
                from_wall="west",
                to_wall="east",
            ),
        }
    else:
        first_origin = (rng.randint(-8, -4), 64, rng.randint(-4, 0))
        second_origin = (first_origin[0] + 4, 64, first_origin[2])
        room_a = _room_shell(origin=first_origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks")
        room_b = _room_shell(origin=second_origin, width=5, height=3, depth=5, wall_block="minecraft:stone_bricks")
        setup = _dedupe_blocks([*room_a, *room_b])
        doorway_coords = {
            (first_origin[0] + 4, 64, first_origin[2] + 2),
            (first_origin[0] + 4, 65, first_origin[2] + 2),
        }
        target = [
            BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id="minecraft:air")
            for coord in sorted(doorway_coords)
        ]
        preserved = _remove_coords(setup, doorway_coords)
        prompt = (
            "Two adjacent stone_bricks rooms share a wall. "
            "Cut a 1-block-wide doorway through the shared wall at the center where they touch."
        )
        checks = StructuralChecks()
        expected_tool_calls = 7
        metadata = {
            "requires_modification": True,
            "canonical_intent": _canonical_intent(
                family=family,
                first_room_origin={"x": first_origin[0], "y": first_origin[1], "z": first_origin[2]},
                second_room_origin={"x": second_origin[0], "y": second_origin[1], "z": second_origin[2]},
                room_width=5,
                room_depth=5,
                room_height=3,
            ),
        }

    prompt, setup, metadata = _add_optional_distractor(
        rng=rng,
        tier="t5_modification",
        prompt=prompt,
        setup=setup,
        target=target,
        preserved=preserved,
        metadata=metadata,
    )
    return TaskSpec(
        task_id=_task_id("t5_modification", seed, family, index),
        tier="t5_modification",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup,
        target_blocks=target,
        preserved_blocks=preserved,
        expected_tool_calls=expected_tool_calls,
        structural_checks=checks,
        metadata=metadata,
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
    setup: list[BlockPlacement]
    target: list[BlockPlacement]
    preserved: list[BlockPlacement]
    prompt: str
    checks: StructuralChecks
    metadata: dict[str, object]

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
        metadata = {
            "compositional": True,
            "canonical_intent": _canonical_intent(
                family=family,
                left_base={"x": left_base[0], "y": left_base[1], "z": left_base[2]},
                right_base={"x": right_base[0], "y": right_base[1], "z": right_base[2]},
                block_id="minecraft:cobblestone",
            ),
        }
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
            "Connect the two rooms with a one-block-wide hollow stone_bricks corridor shell between the facing doorways. "
            "Build the corridor floor, side walls, and roof, and leave the interior passage empty."
        )
        checks = StructuralChecks()
        metadata = {
            "compositional": True,
            "canonical_intent": _canonical_intent(
                family=family,
                top_room_origin={"x": top_room_origin[0], "y": top_room_origin[1], "z": top_room_origin[2]},
                bottom_room_origin={"x": bottom_room_origin[0], "y": bottom_room_origin[1], "z": bottom_room_origin[2]},
                block_id="minecraft:stone_bricks",
            ),
        }
    elif family == "bridge_between_offset_towers":
        left_base = (rng.randint(-8, -5), 64, rng.randint(-5, -2))
        right_base = (rng.randint(4, 8), 64, rng.randint(2, 5))
        left_tower = _tower(base=left_base, height=4, block_id="minecraft:stone")
        right_tower = _tower(base=right_base, height=4, block_id="minecraft:stone")
        setup = left_tower + right_tower
        x_path = [
            BlockPlacement(x=x, y=68, z=left_base[2], block_id="minecraft:cobblestone")
            for x in range(left_base[0], right_base[0] + 1)
        ]
        z_path = [
            BlockPlacement(x=right_base[0], y=68, z=z, block_id="minecraft:cobblestone")
            for z in range(min(left_base[2], right_base[2]), max(left_base[2], right_base[2]) + 1)
        ]
        target = _dedupe_blocks([*x_path, *z_path])
        preserved = left_tower + right_tower
        prompt = (
            "Two stone towers are already built nearby. Do not build new towers or change the existing towers. "
            "Build only the missing one-block-wide L-shaped minecraft:cobblestone bridge in the air one block above "
            "the tower tops. Start directly above the west tower top, run straight east until you are aligned with "
            "the east tower, then turn once and continue along z until the bridge ends directly above the east tower top. "
            "Do not place any cobblestone on or inside the towers themselves."
        )
        checks = StructuralChecks(require_connected=True)
        metadata = {
            "compositional": True,
            "canonical_intent": _canonical_intent(
                family=family,
                left_base={"x": left_base[0], "y": left_base[1], "z": left_base[2]},
                right_base={"x": right_base[0], "y": right_base[1], "z": right_base[2]},
                block_id="minecraft:cobblestone",
            ),
        }
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
            "Two stone_bricks rooms with existing doorways are already built nearby. Do not rebuild or modify the rooms. "
            "Build only the missing one-block-wide L-shaped stone_bricks corridor shell between the two doorways, "
            "including the corridor floor, side walls, and roof. Leave the interior passage empty."
        )
        checks = StructuralChecks()
        metadata = {
            "compositional": True,
            "canonical_intent": _canonical_intent(
                family=family,
                first_room_origin={"x": room_a_origin[0], "y": room_a_origin[1], "z": room_a_origin[2]},
                second_room_origin={"x": room_b_origin[0], "y": room_b_origin[1], "z": room_b_origin[2]},
                block_id="minecraft:stone_bricks",
            ),
        }

    prompt, setup, metadata = _add_optional_distractor(
        rng=rng,
        tier="t6_composition",
        prompt=prompt,
        setup=setup,
        target=target,
        preserved=preserved,
        metadata=metadata,
    )
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
        metadata=metadata,
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
