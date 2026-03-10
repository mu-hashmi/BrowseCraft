import pytest

from browsecraft_backend.models import BlockPlacement, ChatRequest


def test_block_state_none_normalizes_to_empty_dict() -> None:
    placement = BlockPlacement(dx=0, dy=0, dz=0, block_id="minecraft:stone", block_state=None)
    assert placement.block_state == {}


def test_block_state_values_are_stringified() -> None:
    placement = BlockPlacement(
        dx=0,
        dy=0,
        dz=0,
        block_id="minecraft:stone",
        block_state={"power": 1, "enabled": True, "unknown": None},
    )
    assert placement.block_state == {
        "power": "1",
        "enabled": "True",
        "unknown": "None",
    }


def test_mechanical_blocks_require_non_empty_state() -> None:
    with pytest.raises(ValueError):
        BlockPlacement(dx=0, dy=0, dz=0, block_id="minecraft:observer", block_state={})


def test_suffix_required_blocks_require_non_empty_state() -> None:
    with pytest.raises(ValueError):
        BlockPlacement(dx=0, dy=0, dz=0, block_id="minecraft:oak_stairs", block_state={})


def test_required_block_with_state_is_valid() -> None:
    placement = BlockPlacement(
        dx=1,
        dy=2,
        dz=3,
        block_id="minecraft:oak_stairs",
        block_state={"facing": "north", "half": "bottom"},
    )
    assert placement.block_state["facing"] == "north"


def test_chat_request_accepts_world_and_session_ids() -> None:
    request = ChatRequest(
        client_id="client-1",
        message="hello",
        world_id="world-1",
        session_id="session-1",
    )
    assert request.world_id == "world-1"
    assert request.session_id == "session-1"
