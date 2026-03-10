from __future__ import annotations

from browsecraft_sim.rl.task_generator import generate_task, generate_tasks, reconstruct_task_from_task_id
from browsecraft_sim.rl.types import ALL_TIERS


def test_generate_tasks_is_deterministic_for_same_seed() -> None:
    first = generate_tasks(seed=17, per_tier=2)
    second = generate_tasks(seed=17, per_tier=2)
    assert [task.model_dump(mode="json") for task in first] == [task.model_dump(mode="json") for task in second]


def test_reconstruct_task_from_task_id_is_stable() -> None:
    task = generate_task(tier="t6_composition", seed=17, index=3)
    reconstructed = reconstruct_task_from_task_id(task.task_id)
    assert reconstructed.model_dump(mode="json") == task.model_dump(mode="json")


def test_generate_tasks_covers_all_tiers() -> None:
    tasks = generate_tasks(seed=9, per_tier=1)
    assert {task.tier for task in tasks} == set(ALL_TIERS)


def test_t4_has_required_families() -> None:
    families = {generate_task(tier="t4_structure_relative", seed=33, index=index).family for index in range(200)}
    assert families >= {
        "top_of_tower",
        "south_face_marker",
        "inside_room_through_doorway",
        "shorter_tower_marker",
        "marker_chain_place",
        "structure_chain_place",
        "mark_structure_inside_enclosure",
    }


def test_t5_has_required_families() -> None:
    families = {generate_task(tier="t5_modification", seed=33, index=index).family for index in range(200)}
    assert families >= {
        "replace_material_preserve_shape",
        "widen_or_reposition_opening",
        "add_window_to_wall",
        "move_window_to_opposite_wall",
        "add_shared_wall_doorway",
    }


def test_t6_has_required_families() -> None:
    families = {generate_task(tier="t6_composition", seed=33, index=index).family for index in range(200)}
    assert families >= {
        "bridge_between_structures",
        "connect_rooms_with_corridor",
        "bridge_between_offset_towers",
        "l_shaped_corridor_offset_rooms",
    }


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
    target_coords = {(block.x, block.y, block.z) for block in task.target_blocks}
    target_xs = {block.x for block in task.target_blocks}
    target_ys = {block.y for block in task.target_blocks}
    center_x = sum(target_xs) // len(target_xs)
    wall_z = next(iter({block.z for block in task.target_blocks}))
    assert len(task.target_blocks) == 4
    assert target_xs == {center_x - 1, center_x + 1}
    assert target_ys == {64, 65}
    assert (center_x, 64, wall_z) not in target_coords
    assert (center_x, 65, wall_z) not in target_coords


def test_t6_corridor_setup_has_no_air_and_preserves_unique_blocks() -> None:
    task = _find_family_task(tier="t6_composition", family="connect_rooms_with_corridor", seed=2026)
    assert all(block.block_id != "minecraft:air" for block in task.setup_blocks)
    setup_coords = {(block.x, block.y, block.z) for block in task.setup_blocks}
    assert len(setup_coords) == len(task.setup_blocks)
    target_coords = {(block.x, block.y, block.z) for block in task.target_blocks}
    assert setup_coords.isdisjoint(target_coords)


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


def test_t4_south_face_marker_prompt_clarifies_wall_block() -> None:
    task = _find_family_task(tier="t4_structure_relative", family="south_face_marker", seed=2026)
    assert "Replace the center block of the south wall" in task.prompt


def test_t6_bridge_target_uses_walkable_top_span() -> None:
    task = _find_family_task(tier="t6_composition", family="bridge_between_structures", seed=2026)
    tower_xs = sorted({block.x for block in task.setup_blocks})
    target_xs = sorted({block.x for block in task.target_blocks})
    target_ys = {block.y for block in task.target_blocks}
    assert target_ys == {68}
    assert target_xs == list(range(tower_xs[0], tower_xs[-1] + 1))


def test_t2_generates_egocentric_relative_family() -> None:
    families = {generate_task(tier="t2_relative_single_ref", seed=33, index=index).family for index in range(60)}
    assert "egocentric_relative" in families


def test_t2_egocentric_relative_target_matches_player_facing() -> None:
    task = _find_family_task(tier="t2_relative_single_ref", family="egocentric_relative", seed=2026)
    target = task.target_blocks[0]
    relation = task.metadata["relation"]
    facing = task.player.facing
    expected_offsets = {
        ("north", "front"): (0, -1),
        ("north", "behind"): (0, 1),
        ("north", "left"): (-1, 0),
        ("north", "right"): (1, 0),
        ("south", "front"): (0, 1),
        ("south", "behind"): (0, -1),
        ("south", "left"): (1, 0),
        ("south", "right"): (-1, 0),
        ("east", "front"): (1, 0),
        ("east", "behind"): (-1, 0),
        ("east", "left"): (0, -1),
        ("east", "right"): (0, 1),
        ("west", "front"): (-1, 0),
        ("west", "behind"): (1, 0),
        ("west", "left"): (0, 1),
        ("west", "right"): (0, -1),
    }
    distance = task.metadata["distance"]
    expected_dx, expected_dz = expected_offsets[(facing, relation)]
    assert (target.x, target.z) == (expected_dx * distance, expected_dz * distance)


def test_t4_marker_chain_places_block_above_final_marker() -> None:
    task = _find_family_task(tier="t4_structure_relative", family="marker_chain_place", seed=2026)
    markers = task.metadata["canonical_intent"]["markers"]
    final_marker = markers[-1]
    target = task.target_blocks[0]
    assert task.metadata["hop_count"] >= 2
    assert target.x == final_marker["x"]
    assert target.y == final_marker["y"] + 1
    assert target.z == final_marker["z"]


def test_t4_structure_chain_can_reach_six_hops() -> None:
    hop_counts = set()
    for index in range(800):
        task = generate_task(tier="t4_structure_relative", seed=2026, index=index)
        if task.family != "structure_chain_place":
            continue
        hop_counts.add(task.metadata["hop_count"])
        if 6 in hop_counts:
            break
    assert 6 in hop_counts


def test_t4_inside_enclosure_targets_only_internal_tower() -> None:
    task = _find_family_task(tier="t4_structure_relative", family="mark_structure_inside_enclosure", seed=2026)
    inside = task.metadata["canonical_intent"]["inside_tower"]
    target = task.target_blocks[0]
    assert target.x == inside["x"]
    assert target.z == inside["z"]


def test_t5_shared_wall_doorway_targets_center_of_touching_wall() -> None:
    task = _find_family_task(tier="t5_modification", family="add_shared_wall_doorway", seed=2026)
    doorway_coords = {(block.x, block.y, block.z) for block in task.target_blocks}
    assert len(doorway_coords) == 2
    assert {coord[1] for coord in doorway_coords} == {64, 65}
    assert len({coord[0] for coord in doorway_coords}) == 1
    assert len({coord[2] for coord in doorway_coords}) == 1


def test_harder_tiers_can_include_logged_distractors() -> None:
    distracted = []
    for tier in ("t4_structure_relative", "t5_modification", "t6_composition"):
        for index in range(200):
            task = generate_task(tier=tier, seed=2026, index=index)
            distractor = task.metadata.get("distractor")
            if distractor is None:
                continue
            target_coords = {(block.x, block.y, block.z) for block in task.target_blocks}
            distractor_base = distractor["base"]
            distractor_coords = {
                (distractor_base["x"], distractor_base["y"] + dy, distractor_base["z"])
                for dy in range(distractor["height"])
            }
            assert distractor_coords.isdisjoint(target_coords)
            distracted.append(tier)
            break
    assert set(distracted) == {"t4_structure_relative", "t5_modification", "t6_composition"}
