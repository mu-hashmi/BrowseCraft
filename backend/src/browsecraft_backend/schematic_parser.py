from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import nbtlib


_LEGACY_BLOCKS: dict[str, str] | None = None
_TERRAIN_PLATFORM_BLOCK_IDS = {
    "minecraft:stone",
    "minecraft:grass_block",
    "minecraft:dirt",
    "minecraft:coarse_dirt",
    "minecraft:podzol",
    "minecraft:mycelium",
    "minecraft:rooted_dirt",
    "minecraft:sand",
    "minecraft:red_sand",
    "minecraft:gravel",
    "minecraft:deepslate",
    "minecraft:tuff",
}


def parse_schematic(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".schem":
        placements = _parse_sponge_schem(path)
        return _strip_bottom_terrain_platform(placements)
    if suffix == ".litematic":
        placements = _parse_litematic(path)
        return _strip_bottom_terrain_platform(placements)
    if suffix == ".schematic":
        placements = _parse_legacy_schematic(path)
        return _strip_bottom_terrain_platform(placements)
    raise ValueError(f"Unsupported schematic extension: {suffix}")


def _parse_sponge_schem(path: Path) -> list[dict[str, Any]]:
    file = nbtlib.load(path)
    root = file

    width = int(root["Width"])
    height = int(root["Height"])
    length = int(root["Length"])
    palette_tag = root["Palette"]
    block_data = bytes(root["BlockData"])
    expected_count = width * height * length
    block_indices = _decode_varints(block_data, expected_count)

    palette_by_index: dict[int, str] = {}
    for block_name, palette_index in palette_tag.items():
        palette_by_index[int(palette_index)] = str(block_name)

    placements: list[dict[str, Any]] = []
    for index, palette_index in enumerate(block_indices):
        block_value = palette_by_index.get(palette_index)
        if block_value is None:
            raise ValueError(f"Palette index {palette_index} missing from Palette")
        block_id, block_state = _split_block_state(block_value)
        if block_id == "minecraft:air":
            continue

        x = index % width
        z = (index // width) % length
        y = index // (width * length)
        placements.append(
            {
                "dx": x,
                "dy": y,
                "dz": z,
                "block_id": block_id,
                "block_state": block_state,
            }
        )

    return placements


def _parse_litematic(path: Path) -> list[dict[str, Any]]:
    file = nbtlib.load(path)
    root = file
    if "Regions" not in root:
        raise ValueError("Invalid litematic: missing Regions")

    placements: list[dict[str, Any]] = []
    min_x = 0
    min_y = 0
    min_z = 0
    initialized_min = False

    for _, region in root["Regions"].items():
        size_tag = region["Size"]
        pos_tag = region["Position"]

        size_x_raw = int(size_tag["x"])
        size_y_raw = int(size_tag["y"])
        size_z_raw = int(size_tag["z"])
        size_x = abs(size_x_raw)
        size_y = abs(size_y_raw)
        size_z = abs(size_z_raw)
        if size_x == 0 or size_y == 0 or size_z == 0:
            continue

        region_x = int(pos_tag["x"])
        region_y = int(pos_tag["y"])
        region_z = int(pos_tag["z"])

        palette_list = list(region["BlockStatePalette"])
        palette_size = len(palette_list)
        if palette_size == 0:
            continue
        bits_per_block = max(2, math.ceil(math.log2(palette_size)))
        state_longs = [int(value) for value in region["BlockStates"]]
        block_count = size_x * size_y * size_z
        block_indices = _decode_packed_longs(state_longs, bits_per_block, block_count)

        for index, palette_index in enumerate(block_indices):
            if palette_index >= palette_size:
                raise ValueError(f"Palette index {palette_index} exceeds palette size {palette_size}")
            palette_entry = palette_list[palette_index]
            block_id = str(palette_entry["Name"])
            if block_id == "minecraft:air":
                continue
            block_state = {
                str(key): str(value)
                for key, value in dict(palette_entry.get("Properties", {})).items()
            }

            local_x = index % size_x
            local_z = (index // size_x) % size_z
            local_y = index // (size_x * size_z)
            if size_x_raw < 0:
                local_x = size_x - 1 - local_x
            if size_y_raw < 0:
                local_y = size_y - 1 - local_y
            if size_z_raw < 0:
                local_z = size_z - 1 - local_z

            world_x = region_x + local_x
            world_y = region_y + local_y
            world_z = region_z + local_z
            if not initialized_min:
                min_x = world_x
                min_y = world_y
                min_z = world_z
                initialized_min = True
            else:
                min_x = min(min_x, world_x)
                min_y = min(min_y, world_y)
                min_z = min(min_z, world_z)

            placements.append(
                {
                    "dx": world_x,
                    "dy": world_y,
                    "dz": world_z,
                    "block_id": block_id,
                    "block_state": block_state,
                }
            )

    if not placements:
        return placements

    normalized: list[dict[str, Any]] = []
    for placement in placements:
        normalized.append(
            {
                **placement,
                "dx": placement["dx"] - min_x,
                "dy": placement["dy"] - min_y,
                "dz": placement["dz"] - min_z,
            }
        )
    return normalized


def _parse_legacy_schematic(path: Path) -> list[dict[str, Any]]:
    file = nbtlib.load(path)
    root = file

    width = int(root["Width"])
    height = int(root["Height"])
    length = int(root["Length"])
    blocks = bytes(root["Blocks"])
    data = bytes(root["Data"])
    add_blocks = bytes(root["AddBlocks"]) if "AddBlocks" in root else b""
    if len(blocks) != len(data):
        raise ValueError("Legacy schematic Blocks and Data lengths mismatch")
    if len(blocks) != width * height * length:
        raise ValueError("Legacy schematic dimensions do not match block arrays")

    legacy_map = _load_legacy_map()
    placements: list[dict[str, Any]] = []
    for index, block_value in enumerate(blocks):
        extra = _legacy_add_value(add_blocks, index)
        block_id_numeric = (extra << 8) | int(block_value)
        data_value = int(data[index])
        mapped = legacy_map.get(f"{block_id_numeric}:{data_value}") or legacy_map.get(f"{block_id_numeric}:0")
        if mapped is None:
            continue
        block_id, block_state = _split_block_state(mapped)
        if block_id == "minecraft:air":
            continue

        x = index % width
        z = (index // width) % length
        y = index // (width * length)
        placements.append(
            {
                "dx": x,
                "dy": y,
                "dz": z,
                "block_id": block_id,
                "block_state": block_state,
            }
        )

    return placements


def _decode_varints(payload: bytes, expected_count: int) -> list[int]:
    values: list[int] = []
    index = 0
    length = len(payload)
    while index < length and len(values) < expected_count:
        shift = 0
        value = 0
        while True:
            if index >= length:
                raise ValueError("Unexpected end of varint payload")
            current = payload[index]
            index += 1
            value |= (current & 0x7F) << shift
            if current & 0x80 == 0:
                break
            shift += 7
            if shift > 35:
                raise ValueError("Invalid varint encoding")
        values.append(value)

    if len(values) != expected_count:
        raise ValueError(f"BlockData count mismatch: expected {expected_count}, got {len(values)}")
    return values


def _decode_packed_longs(longs: list[int], bits_per_block: int, count: int) -> list[int]:
    mask = (1 << bits_per_block) - 1
    unsigned = [value & ((1 << 64) - 1) for value in longs]
    values: list[int] = []
    for block_index in range(count):
        bit_index = block_index * bits_per_block
        long_index = bit_index // 64
        bit_offset = bit_index % 64
        if long_index >= len(unsigned):
            raise ValueError("BlockStates payload ended before expected block count")
        value = (unsigned[long_index] >> bit_offset) & mask
        if bit_offset + bits_per_block > 64:
            if long_index + 1 >= len(unsigned):
                raise ValueError("BlockStates payload ended while reading spanning value")
            carried = unsigned[long_index + 1] << (64 - bit_offset)
            value = (value | carried) & mask
        values.append(int(value))
    return values


def _split_block_state(value: str) -> tuple[str, dict[str, str]]:
    if "[" not in value:
        return value, {}
    block_id, state_payload = value.split("[", 1)
    state_payload = state_payload.rstrip("]")
    if not state_payload:
        return block_id, {}
    state: dict[str, str] = {}
    for pair in state_payload.split(","):
        if "=" not in pair:
            continue
        key, item = pair.split("=", 1)
        state[key] = item
    return block_id, state


def _legacy_add_value(add_blocks: bytes, index: int) -> int:
    if not add_blocks:
        return 0
    nibble_index = index // 2
    if nibble_index >= len(add_blocks):
        return 0
    value = add_blocks[nibble_index]
    if index % 2 == 0:
        return value & 0x0F
    return (value >> 4) & 0x0F


def _load_legacy_map() -> dict[str, str]:
    global _LEGACY_BLOCKS
    if _LEGACY_BLOCKS is None:
        legacy_path = Path(__file__).with_name("legacy_blocks.json")
        _LEGACY_BLOCKS = json.loads(legacy_path.read_text())
    return _LEGACY_BLOCKS


def _strip_bottom_terrain_platform(placements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not placements:
        return placements

    min_y = min(int(item["dy"]) for item in placements)
    max_y = max(int(item["dy"]) for item in placements)
    if max_y <= min_y:
        return placements
    footprint = {(int(item["dx"]), int(item["dz"])) for item in placements}
    bottom_by_footprint: dict[tuple[int, int], dict[str, Any]] = {
        (int(item["dx"]), int(item["dz"])): item
        for item in placements
        if int(item["dy"]) == min_y
    }
    if len(bottom_by_footprint) != len(footprint):
        return placements

    bottom_blocks = list(bottom_by_footprint.values())
    if not bottom_blocks:
        return placements
    if any(str(item["block_id"]) not in _TERRAIN_PLATFORM_BLOCK_IDS for item in bottom_blocks):
        return placements

    return [item for item in placements if int(item["dy"]) != min_y]
