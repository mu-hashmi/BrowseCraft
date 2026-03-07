from __future__ import annotations

from typing import Any, Protocol


class ToolDispatchWorld(Protocol):
    def player_position(self) -> dict[str, Any]:
        ...

    def player_inventory(self) -> dict[str, Any]:
        ...

    def inspect_area(
        self,
        *,
        center: dict[str, Any],
        radius: int,
        detailed: bool = False,
        filter_terrain: bool = True,
    ) -> dict[str, Any]:
        ...

    def place_blocks(self, placements: list[dict[str, Any]]) -> dict[str, Any]:
        ...

    def fill_region(
        self,
        *,
        from_corner: dict[str, Any],
        to_corner: dict[str, Any],
        block_id: str,
    ) -> dict[str, Any]:
        ...

    def undo_last(self) -> dict[str, Any]:
        ...


def dispatch_tool(world: ToolDispatchWorld, tool: str, params: dict[str, Any]) -> dict[str, Any]:
    if tool == "player_position":
        return world.player_position()
    if tool == "player_inventory":
        return world.player_inventory()
    if tool == "inspect_area":
        return world.inspect_area(
            center=params["center"],
            radius=int(params["radius"]),
            detailed=bool(params.get("detailed", False)),
            filter_terrain=bool(params.get("filter_terrain", True)),
        )
    if tool == "place_blocks":
        placements = params["placements"]
        if not isinstance(placements, list):
            raise RuntimeError("place_blocks expects placements list")
        return world.place_blocks(placements)
    if tool == "fill_region":
        return world.fill_region(
            from_corner=params["from_corner"],
            to_corner=params["to_corner"],
            block_id=str(params["block_id"]),
        )
    if tool == "undo_last":
        return world.undo_last()
    if tool == "get_active_overlay":
        return _empty_overlay()
    if tool == "modify_overlay":
        return {"op": params.get("op")}
    if tool == "get_blueprints":
        return {"names": [], "count": 0}
    if tool == "save_blueprint":
        return {"name": params.get("name"), "saved": True}
    if tool == "load_blueprint":
        return _load_blueprint_result(params.get("name"))
    raise RuntimeError(f"Unsupported tool: {tool}")


def _empty_overlay() -> dict[str, Any]:
    return {
        "has_plan": False,
        "block_count": 0,
        "anchor": {"x": 0, "y": 0, "z": 0},
        "rotation_quarter_turns": 0,
        "preview_mode": False,
        "confirmed": False,
        "remaining_count": 0,
    }


def _load_blueprint_result(name: Any) -> dict[str, Any]:
    return {
        "name": name,
        "overlay": _empty_overlay(),
    }
