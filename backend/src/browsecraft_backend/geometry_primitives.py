from __future__ import annotations

from math import ceil
from typing import Literal


Cardinal = Literal["north", "east", "south", "west"]
Coord = tuple[int, int, int]


def build_geometry(
    *,
    shape: str,
    material: str,
    anchor: dict[str, int],
    rotation: Cardinal = "north",
    **dimensions: int | bool | None,
) -> list[dict[str, int | str]]:
    anchor_x = int(anchor["x"])
    anchor_y = int(anchor["y"])
    anchor_z = int(anchor["z"])
    quarters = _rotation_quarters(rotation)

    if shape == "box":
        coords = _box(
            anchor=(anchor_x, anchor_y, anchor_z),
            width=int(dimensions["width"]),
            height=int(dimensions["height"]),
            depth=int(dimensions["depth"]),
            hollow=bool(dimensions.get("hollow", False)),
        )
    elif shape == "cylinder":
        coords = _cylinder(
            anchor=(anchor_x, anchor_y, anchor_z),
            radius=int(dimensions["radius"]),
            height=int(dimensions["height"]),
            hollow=bool(dimensions.get("hollow", False)),
        )
    elif shape == "sphere":
        coords = _sphere(
            anchor=(anchor_x, anchor_y, anchor_z),
            radius=int(dimensions["radius"]),
            hollow=bool(dimensions.get("hollow", False)),
        )
    elif shape == "floor":
        coords = _floor(
            anchor=(anchor_x, anchor_y, anchor_z),
            width=int(dimensions["width"]),
            depth=int(dimensions["depth"]),
            thickness=int(dimensions.get("thickness", 1)),
        )
    elif shape == "wall":
        coords = _wall(
            anchor=(anchor_x, anchor_y, anchor_z),
            width=int(dimensions["width"]),
            height=int(dimensions["height"]),
            thickness=int(dimensions.get("thickness", 1)),
        )
    elif shape == "pillar":
        coords = _pillar(
            anchor=(anchor_x, anchor_y, anchor_z),
            height=int(dimensions["height"]),
            width=int(dimensions.get("width", 1)),
            depth=int(dimensions.get("depth", 1)),
        )
    elif shape == "stairs":
        coords = _stairs(
            anchor=(anchor_x, anchor_y, anchor_z),
            width=int(dimensions["width"]),
            depth=int(dimensions["depth"]),
            height=int(dimensions["height"]),
        )
    elif shape == "roof_flat":
        coords = _roof_flat(
            anchor=(anchor_x, anchor_y, anchor_z),
            width=int(dimensions["width"]),
            depth=int(dimensions["depth"]),
            thickness=int(dimensions.get("thickness", 1)),
        )
    elif shape == "roof_gabled":
        coords = _roof_gabled(
            anchor=(anchor_x, anchor_y, anchor_z),
            width=int(dimensions["width"]),
            depth=int(dimensions["depth"]),
            height=(int(dimensions["height"]) if dimensions.get("height") is not None else None),
        )
    elif shape == "roof_hipped":
        coords = _roof_hipped(
            anchor=(anchor_x, anchor_y, anchor_z),
            width=int(dimensions["width"]),
            depth=int(dimensions["depth"]),
            height=(int(dimensions["height"]) if dimensions.get("height") is not None else None),
        )
    else:
        raise ValueError(f"Unsupported geometry shape: {shape}")

    rotated = _rotate_about_anchor(coords, anchor=(anchor_x, anchor_y, anchor_z), quarters=quarters)
    placements = [
        {"x": x, "y": y, "z": z, "block_id": material}
        for x, y, z in sorted(rotated, key=lambda item: (item[1], item[0], item[2]))
    ]
    return placements


def bounding_box(placements: list[dict[str, int | str]]) -> dict[str, dict[str, int]]:
    xs = [int(item["x"]) for item in placements]
    ys = [int(item["y"]) for item in placements]
    zs = [int(item["z"]) for item in placements]
    return {
        "min": {"x": min(xs), "y": min(ys), "z": min(zs)},
        "max": {"x": max(xs), "y": max(ys), "z": max(zs)},
    }


def _box(*, anchor: Coord, width: int, height: int, depth: int, hollow: bool) -> set[Coord]:
    ax, ay, az = anchor
    min_x = _centered_min(ax, width)
    min_y = _centered_min(ay, height)
    min_z = _centered_min(az, depth)
    max_x = min_x + width - 1
    max_y = min_y + height - 1
    max_z = min_z + depth - 1

    coords: set[Coord] = set()
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                if not hollow or x in {min_x, max_x} or y in {min_y, max_y} or z in {min_z, max_z}:
                    coords.add((x, y, z))
    return coords


def _cylinder(*, anchor: Coord, radius: int, height: int, hollow: bool) -> set[Coord]:
    ax, ay, az = anchor
    min_y = _centered_min(ay, height)
    max_y = min_y + height - 1
    outer_sq = radius * radius
    inner_sq = (radius - 1) * (radius - 1)
    coords: set[Coord] = set()
    for x in range(ax - radius, ax + radius + 1):
        for z in range(az - radius, az + radius + 1):
            dist_sq = (x - ax) * (x - ax) + (z - az) * (z - az)
            if dist_sq > outer_sq:
                continue
            if hollow and radius > 0 and dist_sq < inner_sq:
                continue
            for y in range(min_y, max_y + 1):
                coords.add((x, y, z))
    return coords


def _sphere(*, anchor: Coord, radius: int, hollow: bool) -> set[Coord]:
    ax, ay, az = anchor
    outer_sq = radius * radius
    inner_sq = (radius - 1) * (radius - 1)
    coords: set[Coord] = set()
    for x in range(ax - radius, ax + radius + 1):
        for y in range(ay - radius, ay + radius + 1):
            for z in range(az - radius, az + radius + 1):
                dist_sq = (x - ax) * (x - ax) + (y - ay) * (y - ay) + (z - az) * (z - az)
                if dist_sq > outer_sq:
                    continue
                if hollow and radius > 0 and dist_sq < inner_sq:
                    continue
                coords.add((x, y, z))
    return coords


def _floor(*, anchor: Coord, width: int, depth: int, thickness: int) -> set[Coord]:
    ax, ay, az = anchor
    min_x = _centered_min(ax, width)
    min_y = _centered_min(ay, thickness)
    min_z = _centered_min(az, depth)
    coords: set[Coord] = set()
    for x in range(min_x, min_x + width):
        for y in range(min_y, min_y + thickness):
            for z in range(min_z, min_z + depth):
                coords.add((x, y, z))
    return coords


def _wall(*, anchor: Coord, width: int, height: int, thickness: int) -> set[Coord]:
    ax, ay, az = anchor
    min_x = _centered_min(ax, width)
    min_y = _centered_min(ay, height)
    min_z = _centered_min(az, thickness)
    coords: set[Coord] = set()
    for x in range(min_x, min_x + width):
        for y in range(min_y, min_y + height):
            for z in range(min_z, min_z + thickness):
                coords.add((x, y, z))
    return coords


def _pillar(*, anchor: Coord, height: int, width: int, depth: int) -> set[Coord]:
    ax, ay, az = anchor
    min_x = _centered_min(ax, width)
    min_y = _centered_min(ay, height)
    min_z = _centered_min(az, depth)
    coords: set[Coord] = set()
    for x in range(min_x, min_x + width):
        for y in range(min_y, min_y + height):
            for z in range(min_z, min_z + depth):
                coords.add((x, y, z))
    return coords


def _stairs(*, anchor: Coord, width: int, depth: int, height: int) -> set[Coord]:
    ax, ay, az = anchor
    min_x = _centered_min(ax, width)
    min_y = _centered_min(ay, height)
    min_z = _centered_min(az, depth)
    coords: set[Coord] = set()
    for z_index in range(depth):
        step_height = ceil(((z_index + 1) * height) / depth)
        for x in range(min_x, min_x + width):
            for y in range(min_y, min_y + step_height):
                coords.add((x, y, min_z + z_index))
    return coords


def _roof_flat(*, anchor: Coord, width: int, depth: int, thickness: int) -> set[Coord]:
    return _floor(anchor=anchor, width=width, depth=depth, thickness=thickness)


def _roof_gabled(*, anchor: Coord, width: int, depth: int, height: int | None) -> set[Coord]:
    roof_height = height if height is not None else max(1, ceil(depth / 2))
    ax, ay, az = anchor
    min_x = _centered_min(ax, width)
    min_y = _centered_min(ay, roof_height)
    min_z = _centered_min(az, depth)
    center_z = min_z + ((depth - 1) // 2)
    coords: set[Coord] = set()
    for z in range(min_z, min_z + depth):
        band_height = roof_height - abs(z - center_z)
        if band_height <= 0:
            continue
        for x in range(min_x, min_x + width):
            for y in range(min_y, min_y + band_height):
                coords.add((x, y, z))
    return coords


def _roof_hipped(*, anchor: Coord, width: int, depth: int, height: int | None) -> set[Coord]:
    roof_height = height if height is not None else max(1, min(ceil(width / 2), ceil(depth / 2)))
    ax, ay, az = anchor
    min_x = _centered_min(ax, width)
    min_y = _centered_min(ay, roof_height)
    min_z = _centered_min(az, depth)
    center_x = min_x + ((width - 1) // 2)
    center_z = min_z + ((depth - 1) // 2)
    coords: set[Coord] = set()
    for x in range(min_x, min_x + width):
        for z in range(min_z, min_z + depth):
            band_height = roof_height - max(abs(x - center_x), abs(z - center_z))
            if band_height <= 0:
                continue
            for y in range(min_y, min_y + band_height):
                coords.add((x, y, z))
    return coords


def _centered_min(anchor_axis: int, size: int) -> int:
    return anchor_axis - ((size - 1) // 2)


def _rotation_quarters(rotation: Cardinal) -> int:
    if rotation == "east":
        return 1
    if rotation == "south":
        return 2
    if rotation == "west":
        return 3
    return 0


def _rotate_about_anchor(coords: set[Coord], *, anchor: Coord, quarters: int) -> set[Coord]:
    if quarters % 4 == 0:
        return coords
    ax, _, az = anchor
    rotated: set[Coord] = set()
    for x, y, z in coords:
        dx = x - ax
        dz = z - az
        rx, rz = _rotate(dx, dz, quarters)
        rotated.add((ax + rx, y, az + rz))
    return rotated


def _rotate(x: int, z: int, quarters: int) -> tuple[int, int]:
    normalized = quarters % 4
    if normalized == 1:
        return -z, x
    if normalized == 2:
        return -x, -z
    if normalized == 3:
        return z, -x
    return x, z
