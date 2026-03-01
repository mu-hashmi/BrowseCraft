from __future__ import annotations

from browsecraft_sim.rl.task_generator import generate_task, generate_tasks
from browsecraft_sim.rl.types import ALL_TIERS


def test_generate_tasks_is_deterministic_for_same_seed() -> None:
    first = generate_tasks(seed=17, per_tier=2)
    second = generate_tasks(seed=17, per_tier=2)
    assert [task.model_dump(mode="json") for task in first] == [task.model_dump(mode="json") for task in second]


def test_generate_tasks_covers_all_tiers() -> None:
    tasks = generate_tasks(seed=9, per_tier=1)
    assert {task.tier for task in tasks} == set(ALL_TIERS)


def test_t5_has_required_families() -> None:
    families = {generate_task(tier="t5_modification", seed=33, index=index).family for index in range(20)}
    assert families >= {"replace_material_preserve_shape", "widen_or_reposition_opening"}


def test_t6_has_required_families() -> None:
    families = {generate_task(tier="t6_composition", seed=33, index=index).family for index in range(20)}
    assert families >= {"bridge_between_structures", "connect_rooms_with_corridor"}


def _find_family_task(*, tier: str, family: str, seed: int, max_index: int = 500):
    for index in range(max_index):
        task = generate_task(tier=tier, seed=seed, index=index)
        if task.family == family:
            return task
    raise AssertionError(f"family {family} not found for tier={tier} seed={seed}")


def test_t5_widen_opening_setup_has_no_air_and_no_coordinate_overlap() -> None:
    task = _find_family_task(tier="t5_modification", family="widen_or_reposition_opening", seed=2026)
    assert all(block.block_id != "minecraft:air" for block in task.setup_blocks)
    setup_coords = {(block.x, block.y, block.z) for block in task.setup_blocks}
    assert len(setup_coords) == len(task.setup_blocks)
    assert (0, 64, 0) not in setup_coords
    assert (0, 65, 0) not in setup_coords


def test_t6_corridor_setup_has_no_air_and_preserves_unique_blocks() -> None:
    task = _find_family_task(tier="t6_composition", family="connect_rooms_with_corridor", seed=2026)
    assert all(block.block_id != "minecraft:air" for block in task.setup_blocks)
    setup_coords = {(block.x, block.y, block.z) for block in task.setup_blocks}
    assert len(setup_coords) == len(task.setup_blocks)
    for doorway in ((0, 64, -4), (0, 65, -4), (0, 64, 4), (0, 65, 4)):
        assert doorway not in setup_coords


def test_t4_south_face_marker_varies_room_position() -> None:
    target_positions = set()
    for index in range(400):
        task = generate_task(tier="t4_structure_relative", seed=2026, index=index)
        if task.family == "south_face_marker":
            target = task.target_blocks[0]
            target_positions.add((target.x, target.z))
            if len(target_positions) >= 2:
                break
    assert len(target_positions) >= 2
