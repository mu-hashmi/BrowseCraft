from __future__ import annotations

from browsecraft_backend.geometry_primitives import bounding_box, build_geometry


def _coords(placements: list[dict[str, int | str]]) -> set[tuple[int, int, int]]:
    return {
        (int(item["x"]), int(item["y"]), int(item["z"]))
        for item in placements
    }


def test_box_solid_single_block() -> None:
    placements = build_geometry(
        shape="box",
        material="minecraft:stone",
        anchor={"x": 0, "y": 64, "z": 0},
        width=1,
        height=1,
        depth=1,
        hollow=False,
    )
    assert placements == [{"x": 0, "y": 64, "z": 0, "block_id": "minecraft:stone"}]


def test_box_hollow_3x3x3_shell() -> None:
    placements = build_geometry(
        shape="box",
        material="minecraft:stone",
        anchor={"x": 0, "y": 64, "z": 0},
        width=3,
        height=3,
        depth=3,
        hollow=True,
    )
    assert len(placements) == 26
    assert (0, 64, 0) not in _coords(placements)


def test_box_hollow_degenerate_matches_solid_for_thin_axis() -> None:
    hollow = build_geometry(
        shape="box",
        material="minecraft:stone",
        anchor={"x": 10, "y": 64, "z": 10},
        width=1,
        height=4,
        depth=3,
        hollow=True,
    )
    solid = build_geometry(
        shape="box",
        material="minecraft:stone",
        anchor={"x": 10, "y": 64, "z": 10},
        width=1,
        height=4,
        depth=3,
        hollow=False,
    )
    assert _coords(hollow) == _coords(solid)


def test_cylinder_solid_and_hollow_counts() -> None:
    solid = build_geometry(
        shape="cylinder",
        material="minecraft:stone",
        anchor={"x": 0, "y": 64, "z": 0},
        radius=2,
        height=1,
        hollow=False,
    )
    hollow = build_geometry(
        shape="cylinder",
        material="minecraft:stone",
        anchor={"x": 0, "y": 64, "z": 0},
        radius=2,
        height=1,
        hollow=True,
    )
    assert len(solid) > len(hollow)
    assert (0, 64, 0) in _coords(solid)
    assert (0, 64, 0) not in _coords(hollow)


def test_sphere_solid_and_hollow_counts() -> None:
    solid = build_geometry(
        shape="sphere",
        material="minecraft:stone",
        anchor={"x": 0, "y": 64, "z": 0},
        radius=2,
        hollow=False,
    )
    hollow = build_geometry(
        shape="sphere",
        material="minecraft:stone",
        anchor={"x": 0, "y": 64, "z": 0},
        radius=2,
        hollow=True,
    )
    assert len(solid) > len(hollow)
    assert (0, 64, 0) in _coords(solid)
    assert (0, 64, 0) not in _coords(hollow)


def test_floor_and_wall_and_pillar_dimensions() -> None:
    floor = build_geometry(
        shape="floor",
        material="minecraft:oak_planks",
        anchor={"x": 0, "y": 64, "z": 0},
        width=4,
        depth=2,
        thickness=1,
    )
    wall = build_geometry(
        shape="wall",
        material="minecraft:stone",
        anchor={"x": 0, "y": 64, "z": 0},
        width=5,
        height=3,
        thickness=1,
    )
    pillar = build_geometry(
        shape="pillar",
        material="minecraft:cobblestone",
        anchor={"x": 0, "y": 64, "z": 0},
        width=1,
        depth=1,
        height=5,
    )
    assert len(floor) == 8
    assert len(wall) == 15
    assert len(pillar) == 5


def test_stairs_height_progression() -> None:
    stairs = build_geometry(
        shape="stairs",
        material="minecraft:stone_bricks",
        anchor={"x": 0, "y": 64, "z": 0},
        width=1,
        depth=4,
        height=4,
    )
    coords = _coords(stairs)
    assert (0, 63, -1) in coords
    assert (0, 66, 2) in coords
    assert (0, 66, -1) not in coords


def test_roof_flat_gabled_and_hipped_generate_blocks() -> None:
    roof_flat = build_geometry(
        shape="roof_flat",
        material="minecraft:oak_planks",
        anchor={"x": 0, "y": 70, "z": 0},
        width=5,
        depth=5,
        thickness=1,
    )
    roof_gabled = build_geometry(
        shape="roof_gabled",
        material="minecraft:oak_planks",
        anchor={"x": 0, "y": 70, "z": 0},
        width=5,
        depth=5,
        height=3,
    )
    roof_hipped = build_geometry(
        shape="roof_hipped",
        material="minecraft:oak_planks",
        anchor={"x": 0, "y": 70, "z": 0},
        width=5,
        depth=5,
        height=3,
    )
    assert len(roof_flat) == 25
    assert len(roof_gabled) > len(roof_flat)
    assert len(roof_hipped) > 0


def test_rotation_cardinals_produce_expected_orientations_for_wall() -> None:
    north = build_geometry(
        shape="wall",
        material="minecraft:stone",
        anchor={"x": 10, "y": 64, "z": 20},
        rotation="north",
        width=3,
        height=2,
        thickness=1,
    )
    east = build_geometry(
        shape="wall",
        material="minecraft:stone",
        anchor={"x": 10, "y": 64, "z": 20},
        rotation="east",
        width=3,
        height=2,
        thickness=1,
    )
    south = build_geometry(
        shape="wall",
        material="minecraft:stone",
        anchor={"x": 10, "y": 64, "z": 20},
        rotation="south",
        width=3,
        height=2,
        thickness=1,
    )
    west = build_geometry(
        shape="wall",
        material="minecraft:stone",
        anchor={"x": 10, "y": 64, "z": 20},
        rotation="west",
        width=3,
        height=2,
        thickness=1,
    )

    north_bbox = bounding_box(north)
    east_bbox = bounding_box(east)
    south_bbox = bounding_box(south)
    west_bbox = bounding_box(west)

    assert north_bbox["max"]["x"] - north_bbox["min"]["x"] == 2
    assert north_bbox["max"]["z"] - north_bbox["min"]["z"] == 0
    assert east_bbox["max"]["x"] - east_bbox["min"]["x"] == 0
    assert east_bbox["max"]["z"] - east_bbox["min"]["z"] == 2
    assert south_bbox == north_bbox
    assert west_bbox == east_bbox


def test_even_size_centering_uses_negative_bias() -> None:
    placements = build_geometry(
        shape="floor",
        material="minecraft:stone",
        anchor={"x": 10, "y": 64, "z": 10},
        width=4,
        depth=2,
        thickness=1,
    )
    bbox = bounding_box(placements)
    assert bbox == {
        "min": {"x": 9, "y": 64, "z": 10},
        "max": {"x": 12, "y": 64, "z": 11},
    }
