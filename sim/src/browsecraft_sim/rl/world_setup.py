from __future__ import annotations

from typing import Iterable

from browsecraft_sim.main import HeadlessVoxelWorld, PlayerState

from .types import BlockPlacement, PlayerSpec, TaskSpec, TextQATaskSpec


def build_world_from_setup(
    *,
    player: PlayerSpec,
    setup_blocks: Iterable[BlockPlacement],
    terrain_radius: int = 24,
) -> HeadlessVoxelWorld:
    world = HeadlessVoxelWorld(
        player=PlayerState(
            x=player.x,
            y=player.y,
            z=player.z,
            facing=player.facing,
            dimension=player.dimension,
        )
    )
    world.flat_terrain(radius=terrain_radius)
    apply_blocks(world, setup_blocks)
    return world


def build_world(task: TaskSpec | TextQATaskSpec, terrain_radius: int = 24) -> HeadlessVoxelWorld:
    return build_world_from_setup(player=task.player, setup_blocks=task.setup_blocks, terrain_radius=terrain_radius)


def apply_blocks(world: HeadlessVoxelWorld, blocks: Iterable[BlockPlacement]) -> None:
    for placement in blocks:
        world.set_block(placement.coord(), placement.block_id)


def serialize_snapshot(snapshot: dict[tuple[int, int, int], str]) -> dict[str, str]:
    return {coord_key(coord): block_id for coord, block_id in snapshot.items()}


def deserialize_snapshot(serialized: dict[str, str]) -> dict[tuple[int, int, int], str]:
    return {parse_coord_key(key): block_id for key, block_id in serialized.items()}


def coord_key(coord: tuple[int, int, int]) -> str:
    return f"{coord[0]},{coord[1]},{coord[2]}"


def parse_coord_key(key: str) -> tuple[int, int, int]:
    x_str, y_str, z_str = key.split(",", maxsplit=2)
    return (int(x_str), int(y_str), int(z_str))


def diff_to_blocks(diff: dict[tuple[int, int, int], str]) -> list[BlockPlacement]:
    placements = [
        BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id=block_id)
        for coord, block_id in sorted(diff.items())
    ]
    return placements
