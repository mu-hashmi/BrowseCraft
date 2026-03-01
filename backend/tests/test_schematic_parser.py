from __future__ import annotations

from pathlib import Path

import nbtlib
import pytest

import browsecraft_backend.schematic_parser as schematic_parser
from browsecraft_backend.schematic_parser import (
    _decode_packed_longs,
    _decode_varints,
    _split_block_state,
    parse_schematic,
)


def _pack_values(values: list[int], bits_per_block: int) -> list[int]:
    payload = 0
    for index, value in enumerate(values):
        payload |= value << (index * bits_per_block)
    return [payload]


def test_parse_sponge_schem_returns_relative_placements(tmp_path: Path) -> None:
    root = nbtlib.Compound(
        {
            "Version": nbtlib.Int(3),
            "Width": nbtlib.Short(2),
            "Height": nbtlib.Short(1),
            "Length": nbtlib.Short(2),
            "PaletteMax": nbtlib.Int(2),
            "Palette": nbtlib.Compound(
                {
                    "minecraft:air": nbtlib.Int(0),
                    "minecraft:stone": nbtlib.Int(1),
                }
            ),
            "BlockData": nbtlib.ByteArray([1, 0, 1, 0]),
        }
    )
    file = nbtlib.File(root)
    schem_path = tmp_path / "sample.schem"
    file.save(schem_path, gzipped=True)

    placements = parse_schematic(schem_path)

    assert placements == [
        {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:stone", "block_state": {}},
        {"dx": 0, "dy": 0, "dz": 1, "block_id": "minecraft:stone", "block_state": {}},
    ]


def test_parse_sponge_schem_with_all_air_returns_empty_list(tmp_path: Path) -> None:
    root = nbtlib.Compound(
        {
            "Version": nbtlib.Int(3),
            "Width": nbtlib.Short(2),
            "Height": nbtlib.Short(1),
            "Length": nbtlib.Short(2),
            "PaletteMax": nbtlib.Int(1),
            "Palette": nbtlib.Compound({"minecraft:air": nbtlib.Int(0)}),
            "BlockData": nbtlib.ByteArray([0, 0, 0, 0]),
        }
    )
    schem_path = tmp_path / "all-air.schem"
    nbtlib.File(root).save(schem_path, gzipped=True)

    assert parse_schematic(schem_path) == []


def test_parse_litematic_returns_normalized_relative_placements(tmp_path: Path) -> None:
    palette = nbtlib.List[nbtlib.Compound](
        [
            nbtlib.Compound({"Name": nbtlib.String("minecraft:air")}),
            nbtlib.Compound({"Name": nbtlib.String("minecraft:stone")}),
            nbtlib.Compound({"Name": nbtlib.String("minecraft:oak_planks")}),
        ]
    )
    # 2x1x2 volume, flattened indices:
    # 0:(0,0,0)=air, 1:(1,0,0)=stone, 2:(0,0,1)=oak_planks, 3:(1,0,1)=air
    block_states = nbtlib.LongArray(_pack_values([0, 1, 2, 0], bits_per_block=2))
    region = nbtlib.Compound(
        {
            "Size": nbtlib.Compound(
                {
                    "x": nbtlib.Int(2),
                    "y": nbtlib.Int(1),
                    "z": nbtlib.Int(2),
                }
            ),
            "Position": nbtlib.Compound(
                {
                    "x": nbtlib.Int(10),
                    "y": nbtlib.Int(64),
                    "z": nbtlib.Int(-5),
                }
            ),
            "BlockStatePalette": palette,
            "BlockStates": block_states,
        }
    )
    root = nbtlib.Compound({"Regions": nbtlib.Compound({"main": region})})
    path = tmp_path / "tiny.litematic"
    nbtlib.File(root).save(path, gzipped=True)

    placements = parse_schematic(path)
    assert placements == [
        {"dx": 1, "dy": 0, "dz": 0, "block_id": "minecraft:stone", "block_state": {}},
        {"dx": 0, "dy": 0, "dz": 1, "block_id": "minecraft:oak_planks", "block_state": {}},
    ]


def test_parse_legacy_schematic_maps_numeric_blocks_and_filters_air(tmp_path: Path) -> None:
    root = nbtlib.Compound(
        {
            "Width": nbtlib.Short(2),
            "Height": nbtlib.Short(1),
            "Length": nbtlib.Short(2),
            # (0,0,0)=stone(1), (1,0,0)=air(0), (0,0,1)=oak_planks(5), (1,0,1)=air(0)
            "Blocks": nbtlib.ByteArray([1, 0, 5, 0]),
            "Data": nbtlib.ByteArray([0, 0, 0, 0]),
        }
    )
    path = tmp_path / "legacy.schematic"
    nbtlib.File(root).save(path, gzipped=True)

    placements = parse_schematic(path)
    assert placements == [
        {"dx": 0, "dy": 0, "dz": 0, "block_id": "minecraft:stone", "block_state": {}},
        {"dx": 0, "dy": 0, "dz": 1, "block_id": "minecraft:oak_planks", "block_state": {}},
    ]


def test_split_block_state_parses_properties() -> None:
    block_id, block_state = _split_block_state("minecraft:oak_stairs[facing=east,half=top]")
    assert block_id == "minecraft:oak_stairs"
    assert block_state == {"facing": "east", "half": "top"}


def test_split_block_state_without_properties_returns_empty_state() -> None:
    block_id, block_state = _split_block_state("minecraft:stone")
    assert block_id == "minecraft:stone"
    assert block_state == {}


def test_decode_varints_supports_multibyte_values() -> None:
    payload = bytes([0x01, 0x82, 0x01, 0x05])  # [1, 130, 5]
    assert _decode_varints(payload, expected_count=3) == [1, 130, 5]


def test_decode_varints_raises_on_truncated_payload() -> None:
    with pytest.raises(ValueError, match="Unexpected end of varint payload"):
        _decode_varints(bytes([0x80]), expected_count=1)


def test_decode_packed_longs_decodes_values() -> None:
    # 4 values at 3 bits each -> 0b001_111_010_101 (LSB first) = 981
    assert _decode_packed_longs([981], bits_per_block=3, count=4) == [5, 2, 7, 1]


def test_parse_schematic_raises_for_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.txt"
    path.write_text("not an nbt payload")
    with pytest.raises(ValueError, match="Unsupported schematic extension"):
        parse_schematic(path)


def test_parse_legacy_schematic_all_air_returns_empty_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schematic_parser, "_LEGACY_BLOCKS", {"0:0": "minecraft:air"})
    root = nbtlib.Compound(
        {
            "Width": nbtlib.Short(1),
            "Height": nbtlib.Short(1),
            "Length": nbtlib.Short(1),
            "Blocks": nbtlib.ByteArray([0]),
            "Data": nbtlib.ByteArray([0]),
        }
    )
    path = tmp_path / "all-air.schematic"
    nbtlib.File(root).save(path, gzipped=True)

    assert parse_schematic(path) == []
