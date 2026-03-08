from __future__ import annotations

from browsecraft_sim.main import HeadlessVoxelWorld


def test_validation_report_and_ascii_slice() -> None:
    world = HeadlessVoxelWorld()
    world.place_blocks(
        [
            {"x": 0, "y": 64, "z": 0, "block_id": "minecraft:stone"},
            {"x": 1, "y": 64, "z": 0, "block_id": "minecraft:stone"},
            {"x": 1, "y": 65, "z": 0, "block_id": "minecraft:stone"},
        ]
    )

    report = world.validation_report()
    assert report["block_count"] == 3
    assert report["height"] == {"min": 64, "max": 65}
    assert report["component_count"] == 1
    assert report["connected"] is True
    assert report["dimensions"] == {"x": 2, "y": 2, "z": 1}

    slice_text = world.ascii_slice(y=64)
    assert "y=64" in slice_text
    assert "SS" in slice_text


def test_diff_report_counts_added_removed_and_updated() -> None:
    world = HeadlessVoxelWorld()
    world.place_blocks([{"x": 0, "y": 64, "z": 0, "block_id": "minecraft:stone"}])
    before = world.snapshot()

    world.place_blocks(
        [
            {"x": 0, "y": 64, "z": 0, "block_id": "minecraft:oak_planks"},
            {"x": 1, "y": 64, "z": 0, "block_id": "minecraft:oak_planks"},
        ]
    )
    world.place_blocks([{"x": 2, "y": 64, "z": 0, "block_id": "minecraft:air"}])

    report = world.diff_report(before)
    assert report["changed_count"] == 2
    assert report["added_count"] == 1
    assert report["updated_count"] == 1
    assert report["removed_count"] == 0
    assert report["bbox"]["min"] == {"x": 0, "y": 64, "z": 0}


def test_inspect_area_filters_air_and_terrain_in_non_detailed_mode() -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=2)
    world.place_blocks([{"x": 0, "y": 64, "z": 0, "block_id": "minecraft:stone"}])

    result = world.inspect_area(center={"x": 0, "y": 64, "z": 0}, radius=2, filter_terrain=True)

    assert result["detailed"] is False
    assert result["block_counts"] == {"minecraft:stone": 1}
    assert "minecraft:air" not in result["block_counts"]
    assert "minecraft:grass_block" not in result["block_counts"]
    assert "minecraft:dirt" not in result["block_counts"]


def test_detailed_inspect_area_returns_compact_summary_and_redundancy_feedback() -> None:
    world = HeadlessVoxelWorld()
    world.flat_terrain(radius=2)
    world.place_blocks([{"x": 0, "y": 64, "z": 0, "block_id": "minecraft:stone"}])

    first = world.inspect_area(center={"x": 0, "y": 64, "z": 0}, radius=20, detailed=True, filter_terrain=True)
    second = world.inspect_area(center={"x": 0, "y": 64, "z": 0}, radius=30, detailed=True, filter_terrain=True)

    assert first["radius"] == 6
    assert first["radius_clamped"] is True
    assert "block_counts" not in first
    assert first["retained_block_count"] == 1
    assert first["top_block_ids"] == [{"block_id": "minecraft:stone", "count": 1}]
    assert first["redundant_with_previous"] is False
    assert first["effective_radius_unchanged"] is False
    assert second["redundant_with_previous"] is True
    assert second["effective_radius_unchanged"] is True
