from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .spatial_worlds import (
    MARKER_BLOCKS as _MARKER_BLOCKS,
    chain_positions as _shared_chain_positions,
    enclosure_shell as _shared_enclosure_shell,
    horizontal_facing_offset as _shared_horizontal_facing_offset,
    marker_name as _marker_name,
    player_relative_direction as _shared_player_relative_direction,
    room_shell as _shared_room_shell,
    tower as _shared_tower,
)
from .types import ALL_TEXT_QA_TIERS, AnswerFormat, BlockPlacement, PlayerSpec, TextQATaskSpec, TextQATier


_NOISE_SENTENCES = (
    "There is also an unrelated marker farther away; ignore it.",
    "One extra structure is present but does not affect the answer.",
)


class TextQAGradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    tier: TextQATier
    task_mode: Literal["text_qa"] = "text_qa"
    answer: str
    expected_answer: str
    normalized_answer: str
    normalized_expected_answer: str
    answer_format: AnswerFormat
    correct: bool
    reward_raw: float
    reward_normalized: float = Field(ge=0.0, le=1.0)
    reward_binary: float = Field(ge=0.0, le=1.0)


class TextQATrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    task_id: str
    tier: TextQATier
    family: str
    seed: int
    task_mode: Literal["text_qa"] = "text_qa"
    model: str
    system_prompt: str
    messages: list[dict[str, Any]]
    prompt: str
    answer: str
    expected_answer: str
    answer_format: AnswerFormat
    canonical_reasoning: list[str]
    reward_raw: float
    reward_normalized: float = Field(ge=0.0, le=1.0)
    reward_binary: float = Field(ge=0.0, le=1.0)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    ended_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


def generate_text_qa_tasks(
    *,
    seed: int,
    per_tier: int,
    tiers: Sequence[TextQATier] | None = None,
) -> list[TextQATaskSpec]:
    if per_tier <= 0:
        raise ValueError("per_tier must be > 0")

    selected = list(tiers or ALL_TEXT_QA_TIERS)
    tasks: list[TextQATaskSpec] = []
    for tier in selected:
        for index in range(per_tier):
            tasks.append(generate_text_qa_task(tier=tier, seed=seed, index=index))
    return tasks


def generate_text_qa_task(*, tier: TextQATier, seed: int, index: int = 0) -> TextQATaskSpec:
    derived = _derive_seed(seed=seed, tier=tier, index=index)
    rng = random.Random(derived)
    builder = _TEXT_QA_BUILDERS[tier]
    return builder(seed=derived, index=index, rng=rng)


def reconstruct_text_qa_task_from_task_id(task_id: str) -> TextQATaskSpec:
    tier, family, seed_str, index_str = task_id.split(":", maxsplit=3)
    builder = _TEXT_QA_BUILDERS[tier]
    derived_seed = int(seed_str)
    return builder(
        seed=derived_seed,
        index=int(index_str),
        rng=_ForcedFamilyRandom(seed=derived_seed, family=family),
    )


def normalize_text_qa_answer(answer: str, answer_format: AnswerFormat) -> str:
    stripped = answer.strip().lower()
    stripped = stripped.replace("**", "")
    if "answer:" in stripped:
        stripped = stripped.rsplit("answer:", maxsplit=1)[-1].strip()
    if "\n" in stripped:
        stripped = [line.strip() for line in stripped.splitlines() if line.strip()][-1]
    if answer_format in {"single_token", "entity_name", "yes_no"}:
        stripped = stripped.replace("_", " ")
        stripped = stripped.replace("the ", "")
        if stripped.endswith(" wool"):
            stripped = stripped.removesuffix(" wool") + " marker"
        stripped = re.sub(r"\s+", " ", stripped)
        stripped = stripped.rstrip(".")
        if answer_format == "yes_no":
            matches = re.findall(r"\b(?:yes|no)\b", stripped)
            if matches:
                return matches[-1]
        if answer_format == "entity_name":
            marker_matches = re.findall(
                r"\b(?:red|blue|green|yellow|purple|orange|cyan|black|white) marker\b",
                stripped,
            )
            if marker_matches:
                return marker_matches[-1]
        return stripped
    if answer_format == "coordinate":
        numbers = [int(match) for match in re.findall(r"-?\d+", stripped)]
        if len(numbers) != 3:
            return stripped
        return f"{numbers[0]},{numbers[1]},{numbers[2]}"
    return stripped


def grade_text_qa_answer(task: TextQATaskSpec, answer: str) -> TextQAGradeResult:
    normalized_answer = normalize_text_qa_answer(answer, task.answer_format)
    normalized_expected = normalize_text_qa_answer(task.expected_answer, task.answer_format)
    correct = normalized_answer == normalized_expected
    reward = 1.0 if correct else 0.0
    return TextQAGradeResult(
        task_id=task.task_id,
        tier=task.tier,
        answer=answer,
        expected_answer=task.expected_answer,
        normalized_answer=normalized_answer,
        normalized_expected_answer=normalized_expected,
        answer_format=task.answer_format,
        correct=correct,
        reward_raw=reward,
        reward_normalized=reward,
        reward_binary=reward,
    )


def canonical_text_qa_response(task: TextQATaskSpec) -> str:
    reasoning_lines = "\n".join(f"{index}. {step}" for index, step in enumerate(task.canonical_reasoning, start=1))
    return f"{reasoning_lines}\nAnswer: {task.expected_answer}"


def text_qa_full_prompt(task: TextQATaskSpec) -> str:
    metadata = task.metadata
    lines: list[str] = []

    if task.family == "generated_world_candidate":
        source_task_id = metadata.get("source_task_id")
        if isinstance(source_task_id, str) and source_task_id.startswith("qa_"):
            source_task = reconstruct_text_qa_task_from_task_id(source_task_id)
            source_lines = text_qa_full_prompt(source_task).splitlines()
            if source_lines and source_lines[-1].startswith("Question:"):
                source_lines = source_lines[:-1]
            return "\n".join([*source_lines, f"Question: {task.prompt}"])
        return task.prompt
    if task.family in {"furthest_cardinal_marker", "resolve_marker_chain"}:
        entities = metadata["entities"]
        lines.append("World state:")
        for entity in entities:
            lines.append(
                f"- {entity['name']} is at ({entity['x']}, {entity['y']}, {entity['z']})."
            )
    elif task.family == "relative_to_player_marker":
        block_by_coord = {(block.x, block.y, block.z): block.block_id for block in task.setup_blocks}
        lines.append(
            f"World state: the player is at ({task.player.x}, {task.player.y}, {task.player.z}) facing {task.player.facing}."
        )
        for direction, offset in metadata["world_offsets"].items():
            coord = (task.player.x + offset["x"], task.player.y, task.player.z + offset["z"])
            block_id = block_by_coord[coord]
            lines.append(f"- {_marker_name(block_id)} is {direction} of the player at {coord}.")
    elif task.family == "inside_enclosure":
        origin = metadata["enclosure_origin"]
        width = metadata["enclosure_width"]
        depth = metadata["enclosure_depth"]
        lines.append(
            "World state: "
            f"a stone_bricks enclosure starts at ({origin['x']}, {origin['y']}, {origin['z']}) "
            f"with width {width} and depth {depth}."
        )
        grouped: dict[str, list[tuple[int, int, int]]] = {}
        for block in task.setup_blocks:
            grouped.setdefault(block.block_id, []).append((block.x, block.y, block.z))
        for block_id in metadata["candidate_entity_block_ids"]:
            coords = grouped[block_id]
            base = min(coords)
            lines.append(f"- {_marker_name(block_id)} tower base is at {base}.")
    else:
        first = metadata["left_room_origin"]
        second = metadata["right_room_origin"]
        lines.append(
            "World state: "
            f"room A starts at ({first['x']}, {first['y']}, {first['z']}) and "
            f"room B starts at ({second['x']}, {second['y']}, {second['z']})."
        )
        lines.append(
            f"Each room has width {metadata['room_width']}, depth {metadata['room_depth']}, and height {metadata['room_height']}."
        )

    noise = metadata.get("noise")
    if isinstance(noise, dict) and noise.get("kind") == "tower":
        base = noise["base"]
        lines.append(
            "Ignore the unrelated birch_planks tower "
            f"at ({base['x']}, {base['y']}, {base['z']}) with height {noise['height']}."
        )

    lines.append(f"Question: {task.prompt}")
    return "\n".join(lines)


def read_text_qa_jsonl(path: str | Path) -> list[TextQATaskSpec]:
    records: list[TextQATaskSpec] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        try:
            record = TextQATaskSpec.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"invalid text QA task at line {line_number}: {exc}") from exc
        records.append(record)
    return records


def read_text_qa_trajectory_jsonl(path: str | Path) -> list[TextQATrajectoryRecord]:
    records: list[TextQATrajectoryRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        try:
            record = TextQATrajectoryRecord.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"invalid text QA trajectory at line {line_number}: {exc}") from exc
        records.append(record)
    return records


def write_text_qa_jsonl(path: str | Path, records: Sequence[TextQATaskSpec]) -> None:
    output = Path(path)
    lines = [record.model_dump_json() for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _derive_seed(*, seed: int, tier: TextQATier, index: int) -> int:
    payload = f"{seed}:{tier}:{index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16)


def _task_id(tier: TextQATier, seed: int, family: str, index: int) -> str:
    return f"{tier}:{family}:{seed}:{index}"


class _ForcedFamilyRandom:
    def __init__(self, seed: int, family: str) -> None:
        self._rng = random.Random(seed)
        self._family = family
        self._forced = False

    def choice(self, seq):
        choice = self._rng.choice(seq)
        if not self._forced and self._family in seq:
            self._forced = True
            return self._family
        return choice

    def __getattr__(self, name: str):
        return getattr(self._rng, name)


def _horizontal_facing_offset(facing: str) -> tuple[int, int]:
    return _shared_horizontal_facing_offset(facing)


def _player_relative_direction(facing: str, dx: int, dz: int) -> str:
    return _shared_player_relative_direction(facing, dx, dz)


def _chain_positions(
    *,
    rng: random.Random,
    start: tuple[int, int, int],
    hop_count: int,
    step_distance: int,
) -> tuple[list[tuple[int, int, int]], list[str]]:
    return _shared_chain_positions(rng=rng, start=start, hop_count=hop_count, step_distance=step_distance)


def _tower(base: tuple[int, int, int], *, height: int, block_id: str) -> list[BlockPlacement]:
    return _shared_tower(base=base, height=height, block_id=block_id)


def _enclosure(origin: tuple[int, int, int], *, width: int, depth: int, height: int, block_id: str) -> list[BlockPlacement]:
    return _shared_enclosure_shell(origin=origin, width=width, depth=depth, height=height, wall_block=block_id)


def _room_shell(origin: tuple[int, int, int], *, width: int, depth: int, height: int, block_id: str) -> list[BlockPlacement]:
    return _shared_room_shell(origin=origin, width=width, height=height, depth=depth, wall_block=block_id)


def _with_noise(
    *,
    rng: random.Random,
    prompt: str,
    setup_blocks: list[BlockPlacement],
    metadata: dict[str, Any],
) -> tuple[str, list[BlockPlacement], dict[str, Any]]:
    if rng.random() >= 0.6:
        return prompt, setup_blocks, metadata
    noise_base = (18, 64, 18)
    noise_blocks = _tower(noise_base, height=2, block_id="minecraft:birch_planks")
    updated_metadata = dict(metadata)
    updated_metadata["noise"] = {
        "kind": "tower",
        "base": {"x": noise_base[0], "y": noise_base[1], "z": noise_base[2]},
        "height": 2,
    }
    return f"{prompt} {rng.choice(_NOISE_SENTENCES)}", setup_blocks + noise_blocks, updated_metadata


def _build_directional_single_hop(*, seed: int, index: int, rng: random.Random) -> TextQATaskSpec:
    family = "furthest_cardinal_marker"
    coords = [(-4, 64, -2), (0, 64, 3), (3, 64, -6)]
    rng.shuffle(coords)
    markers = list(_MARKER_BLOCKS[:3])
    setup_blocks = [
        BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id=block_id)
        for coord, block_id in zip(coords, markers, strict=True)
    ]
    answer_marker = min(zip(coords, markers, strict=True), key=lambda item: item[0][2])[1]
    prompt = (
        "Three colored wool markers are nearby: "
        f"{_marker_name(markers[0])}, {_marker_name(markers[1])}, and {_marker_name(markers[2])}. "
        "Which marker is furthest north?"
    )
    metadata = {
        "entities": [
            {
                "name": _marker_name(block_id),
                "block_id": block_id,
                "x": coord[0],
                "y": coord[1],
                "z": coord[2],
            }
            for coord, block_id in zip(coords, markers, strict=True)
        ]
    }
    prompt, setup_blocks, metadata = _with_noise(
        rng=rng,
        prompt=prompt,
        setup_blocks=setup_blocks,
        metadata=metadata,
    )
    return TextQATaskSpec(
        task_id=_task_id("qa_directional_single_hop", seed, family, index),
        tier="qa_directional_single_hop",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup_blocks,
        expected_answer=_marker_name(answer_marker),
        answer_format="entity_name",
        canonical_reasoning=[
            "North corresponds to the smallest z coordinate.",
            f"The furthest north marker is {_marker_name(answer_marker)}.",
        ],
        metadata=metadata,
    )


def _build_multi_hop_chain(*, seed: int, index: int, rng: random.Random) -> TextQATaskSpec:
    family = "resolve_marker_chain"
    hop_count = rng.randint(2, 8)
    step_distance = 3
    positions, steps = _chain_positions(
        rng=rng,
        start=(rng.randint(-9, -3), 64, rng.randint(-9, -3)),
        hop_count=hop_count,
        step_distance=step_distance,
    )
    markers = list(_MARKER_BLOCKS[: hop_count + 1])
    setup_blocks = [
        BlockPlacement(x=coord[0], y=coord[1], z=coord[2], block_id=block_id)
        for coord, block_id in zip(positions, markers, strict=True)
    ]
    final_marker = _marker_name(markers[-1])
    steps_text = ", then ".join(f"move to the marker {step_distance} blocks {step} of that marker" for step in steps)
    prompt = (
        "Colored wool markers are placed nearby. "
        f"Start at the {_marker_name(markers[0])}, {steps_text}. Which marker do you end on?"
    )
    reasoning = [f"Start at {_marker_name(markers[0])}."] + [
        f"Step {index + 1}: move {step_distance} blocks {step} to {_marker_name(markers[index + 1])}."
        for index, step in enumerate(steps)
    ]
    reasoning.append(f"The final marker is {final_marker}.")
    metadata = {
        "step_distance": step_distance,
        "steps": steps,
        "entities": [
            {
                "name": _marker_name(block_id),
                "block_id": block_id,
                "x": coord[0],
                "y": coord[1],
                "z": coord[2],
            }
            for coord, block_id in zip(positions, markers, strict=True)
        ],
    }
    prompt, setup_blocks, metadata = _with_noise(
        rng=rng,
        prompt=prompt,
        setup_blocks=setup_blocks,
        metadata=metadata,
    )
    return TextQATaskSpec(
        task_id=_task_id("qa_multi_hop_chain", seed, family, index),
        tier="qa_multi_hop_chain",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup_blocks,
        expected_answer=final_marker,
        answer_format="entity_name",
        canonical_reasoning=reasoning,
        metadata=metadata,
    )


def _build_viewpoint_transform(*, seed: int, index: int, rng: random.Random) -> TextQATaskSpec:
    family = "relative_to_player_marker"
    facing = rng.choice(["north", "south", "east", "west"])
    player = PlayerSpec(facing=facing)
    forward_dx, forward_dz = _horizontal_facing_offset(facing)
    left_dx, left_dz = forward_dz, -forward_dx
    right_dx, right_dz = -forward_dz, forward_dx
    world_offsets = {
        "front": (forward_dx, forward_dz),
        "behind": (-forward_dx, -forward_dz),
        "left": (left_dx, left_dz),
        "right": (right_dx, right_dz),
    }
    markers = list(_MARKER_BLOCKS[:4])
    directions = ["front", "behind", "left", "right"]
    setup_blocks = [
        BlockPlacement(x=world_offsets[direction][0], y=64, z=world_offsets[direction][1], block_id=block_id)
        for direction, block_id in zip(directions, markers, strict=True)
    ]
    asked_direction = rng.choice(directions)
    answer_marker = _marker_name(markers[directions.index(asked_direction)])
    prompt = (
        f"You are facing {facing}. "
        "Four colored markers surround you. "
        f"Which marker is {asked_direction} of you?"
    )
    reasoning = [
        f"When you face {facing}, {asked_direction} maps to offset {world_offsets[asked_direction]}.",
        f"The marker at that relative position is {answer_marker}.",
    ]
    metadata = {
        "facing": facing,
        "world_offsets": {direction: {"x": offset[0], "z": offset[1]} for direction, offset in world_offsets.items()},
    }
    return TextQATaskSpec(
        task_id=_task_id("qa_viewpoint_transform", seed, family, index),
        tier="qa_viewpoint_transform",
        family=family,
        seed=seed,
        prompt=prompt,
        player=player,
        setup_blocks=setup_blocks,
        expected_answer=answer_marker,
        answer_format="entity_name",
        canonical_reasoning=reasoning,
        metadata=metadata,
    )


def _build_topology(*, seed: int, index: int, rng: random.Random) -> TextQATaskSpec:
    family = rng.choice(["inside_enclosure", "shared_wall_yes_no"])
    if family == "inside_enclosure":
        enclosure_width = rng.choice([7, 9])
        enclosure_depth = rng.choice([7, 9])
        enclosure_origin = (rng.randint(-10, -2), 64, rng.randint(-10, -2))
        inside_base = (
            enclosure_origin[0] + rng.randint(1, enclosure_width - 2),
            64,
            enclosure_origin[2] + rng.randint(1, enclosure_depth - 2),
        )
        outside_base = (
            enclosure_origin[0] + enclosure_width + rng.randint(2, 4),
            64,
            enclosure_origin[2] + rng.randint(1, enclosure_depth - 2),
        )
        far_base = (
            enclosure_origin[0] - rng.randint(4, 6),
            64,
            enclosure_origin[2] + rng.randint(1, enclosure_depth - 2),
        )
        markers = list(_MARKER_BLOCKS[:3])
        rng.shuffle(markers)
        inside_marker, outside_marker, far_marker = markers
        setup_blocks = (
            _enclosure(
                enclosure_origin,
                width=enclosure_width,
                depth=enclosure_depth,
                height=2,
                block_id="minecraft:stone_bricks",
            )
            + _tower(inside_base, height=3, block_id=inside_marker)
            + _tower(outside_base, height=3, block_id=outside_marker)
            + _tower(far_base, height=3, block_id=far_marker)
        )
        prompt = (
            "Three material-coded towers are nearby and one of them is inside a stone_bricks enclosure. "
            "Which tower is inside the enclosure?"
        )
        expected_answer = _marker_name(inside_marker)
        return TextQATaskSpec(
            task_id=_task_id("qa_topology", seed, family, index),
            tier="qa_topology",
            family=family,
            seed=seed,
            prompt=prompt,
            setup_blocks=setup_blocks,
            expected_answer=expected_answer,
            answer_format="entity_name",
            canonical_reasoning=[
                "The enclosure bounds the interior region between its walls.",
                f"The {_marker_name(inside_marker)} sits inside those bounds while the other towers do not.",
                f"The tower inside the enclosure is {expected_answer}.",
            ],
            metadata={
                "inside_tower": expected_answer,
                "inside_entity_block_id": inside_marker,
                "candidate_entity_block_ids": [inside_marker, outside_marker, far_marker],
                "enclosure_block_id": "minecraft:stone_bricks",
                "enclosure_width": enclosure_width,
                "enclosure_depth": enclosure_depth,
                "enclosure_origin": {"x": enclosure_origin[0], "y": enclosure_origin[1], "z": enclosure_origin[2]},
            },
        )

    room_width = 5
    room_depth = 5
    room_height = 3
    axis = rng.choice(["x", "z"])
    share_wall = rng.choice([True, False])
    first_origin = (rng.randint(-8, -2), 64, rng.randint(-8, -2))
    gap = room_width - 1 if share_wall else room_width + rng.choice([1, 2, 3])
    if axis == "x":
        second_origin = (first_origin[0] + gap, 64, first_origin[2])
    else:
        second_origin = (first_origin[0], 64, first_origin[2] + gap)
    setup_blocks = _room_shell(
        first_origin,
        width=room_width,
        depth=room_depth,
        height=room_height,
        block_id="minecraft:stone_bricks",
    ) + _room_shell(
        second_origin,
        width=room_width,
        depth=room_depth,
        height=room_height,
        block_id="minecraft:stone_bricks",
    )
    prompt = "Two stone_bricks rooms are nearby. Do they share a wall?"
    expected_answer = "yes" if share_wall else "no"
    return TextQATaskSpec(
        task_id=_task_id("qa_topology", seed, family, index),
        tier="qa_topology",
        family=family,
        seed=seed,
        prompt=prompt,
        setup_blocks=setup_blocks,
        expected_answer=expected_answer,
        answer_format="yes_no",
        canonical_reasoning=[
            "Rooms share a wall only when their wall coordinates overlap on one face.",
            (
                "These two rooms touch along one face, so they share a wall."
                if share_wall
                else "There is a gap between the rooms, so their wall coordinates do not overlap."
            ),
            f"The correct answer is {expected_answer}.",
        ],
        metadata={
            "left_room_origin": {"x": first_origin[0], "y": first_origin[1], "z": first_origin[2]},
            "right_room_origin": {"x": second_origin[0], "y": second_origin[1], "z": second_origin[2]},
            "room_width": room_width,
            "room_depth": room_depth,
            "room_height": room_height,
            "wall_block_id": "minecraft:stone_bricks",
            "share_wall": share_wall,
            "axis": axis,
        },
    )


_TEXT_QA_BUILDERS = {
    "qa_directional_single_hop": _build_directional_single_hop,
    "qa_multi_hop_chain": _build_multi_hop_chain,
    "qa_viewpoint_transform": _build_viewpoint_transform,
    "qa_topology": _build_topology,
}
