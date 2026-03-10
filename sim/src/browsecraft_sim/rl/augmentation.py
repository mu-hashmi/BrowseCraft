from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Literal, Sequence

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator

from .prompt_variants import PromptVariantRecord
from .spatial_worlds import player_relative_direction as _player_relative_direction
from .text_qa import TextQATaskSpec, normalize_text_qa_answer
from .types import AnswerFormat, BlockPlacement, TaskSpec
from .world_setup import build_world_from_setup


_PARAPHRASE_SYSTEM_PROMPT = (
    "You paraphrase Minecraft spatial instructions while preserving exact spatial semantics.\n"
    "Return JSON only."
)
_PARAPHRASE_VERIFY_SYSTEM_PROMPT = (
    "You extract the canonical spatial intent of a Minecraft instruction.\n"
    "Return JSON only."
)
_WORLD_QA_SYSTEM_PROMPT = (
    "You generate Minecraft spatial QA tasks from a world snapshot.\n"
    "Use only the supported question types, do not invent unsupported entities, and return plain JSON only."
)
_ENTITY_NAMES = {
    "minecraft:red_wool": "red marker",
    "minecraft:blue_wool": "blue marker",
    "minecraft:green_wool": "green marker",
    "minecraft:yellow_wool": "yellow marker",
    "minecraft:purple_wool": "purple marker",
    "minecraft:orange_wool": "orange marker",
    "minecraft:cyan_wool": "cyan marker",
    "minecraft:black_wool": "black marker",
    "minecraft:white_wool": "white marker",
}


class WorldQACandidatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_type: Literal[
        "furthest_north_marker",
        "marker_chain",
        "player_relative_marker",
        "inside_enclosure",
        "shared_wall_yes_no",
    ]
    prompt: str = Field(min_length=1, validation_alias=AliasChoices("prompt", "question"))
    expected_answer: str = Field(min_length=1, validation_alias=AliasChoices("expected_answer", "answer"))
    answer_format: AnswerFormat
    canonical_reasoning: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("canonical_reasoning", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        return value


class WorldQACandidateBatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[WorldQACandidatePayload] = Field(min_length=0, max_length=5)


def task_canonical_intent(task: TaskSpec) -> dict[str, Any]:
    intent = task.metadata["canonical_intent"]
    if not isinstance(intent, dict):
        raise ValueError(f"task {task.task_id} has invalid canonical intent")
    return intent


def parse_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object response")
    return payload


def _json_canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def verify_paraphrase_intent(task: TaskSpec, extracted_intent: dict[str, Any]) -> bool:
    return _json_canonical(extracted_intent) == _json_canonical(task_canonical_intent(task))


def paraphrase_request(task: TaskSpec, *, variant_index: int, model: str) -> dict[str, Any]:
    prompt = (
        "Paraphrase the following Minecraft build instruction into a different natural-language variant.\n"
        f"Variant index: {variant_index}\n"
        f"Canonical intent JSON: {json.dumps(task_canonical_intent(task), sort_keys=True)}\n"
        f"Original instruction: {task.prompt}\n"
        'Return JSON with exactly one key: {"paraphrase": "..."}'
    )
    return {
        "custom_id": f"paraphrase_{task.seed}_{variant_index}",
        "params": {
            "model": model,
            "max_tokens": 256,
            "temperature": 0,
            "system": _PARAPHRASE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def paraphrase_verify_request(task: TaskSpec, *, paraphrase: str, variant_index: int, model: str) -> dict[str, Any]:
    prompt = (
        "Extract the canonical spatial intent from this Minecraft instruction.\n"
        f"Expected JSON shape example: {json.dumps(task_canonical_intent(task), sort_keys=True)}\n"
        f"Instruction: {paraphrase}\n"
        "Return JSON only."
    )
    return {
        "custom_id": f"paraphrase_verify_{task.seed}_{variant_index}",
        "params": {
            "model": model,
            "max_tokens": 512,
            "temperature": 0,
            "system": _PARAPHRASE_VERIFY_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def world_qa_request(task: TaskSpec | TextQATaskSpec, *, model: str) -> dict[str, Any]:
    world_payload = {
        "task_id": task.task_id,
        "player": task.player.model_dump(mode="json"),
        "setup_blocks": [block.model_dump(mode="json") for block in task.setup_blocks],
    }
    prompt = (
        "Given this world snapshot, generate up to five candidate spatial QA tasks that are fully supported by the world.\n"
        "Do not invent markers, rooms, enclosures, or chains that are not present. If a question type is unsupported, omit it.\n"
        "Return plain JSON only, with no markdown fences.\n"
        "Supported question types and required shapes:\n"
        "- furthest_north_marker: answer_format must be entity_name; metadata must contain entity_block_ids as a list of block ids present as single markers.\n"
        "- marker_chain: answer_format must be entity_name; metadata must contain start_entity, steps (north/south/east/west), and step_distance.\n"
        "- player_relative_marker: answer_format must be entity_name; metadata must contain target_direction (front/behind/left/right).\n"
        "- inside_enclosure: answer_format must be entity_name or coordinate; ask which entity or coordinate is inside the enclosure.\n"
        "- shared_wall_yes_no: answer_format must be yes_no; ask whether the two rooms share a wall.\n"
        "canonical_reasoning must be a JSON array of short strings.\n"
        f"World JSON: {json.dumps(world_payload, sort_keys=True)}\n"
        'Return JSON as {"candidates": [...]} with at most five candidates.'
    )
    return {
        "custom_id": f"world_qa_{task.seed}",
        "params": {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": _WORLD_QA_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def verified_paraphrase_records(
    *,
    tasks: Sequence[TaskSpec],
    paraphrase_outputs: dict[str, str],
    verification_outputs: dict[str, str],
) -> list[PromptVariantRecord]:
    task_by_seed = {task.seed: task for task in tasks}
    grouped: dict[str, list[str]] = defaultdict(list)
    for custom_id, paraphrase_text in paraphrase_outputs.items():
        if not custom_id.startswith("paraphrase_"):
            raise ValueError(f"unexpected paraphrase custom id: {custom_id}")
        _, task_seed, variant_index = custom_id.rsplit("_", maxsplit=2)
        verification_key = f"paraphrase_verify_{task_seed}_{variant_index}"
        verification_text = verification_outputs.get(verification_key)
        task = task_by_seed.get(int(task_seed))
        if verification_text is None or task is None:
            continue
        try:
            extracted = parse_json_payload(verification_text)
        except ValueError:
            continue
        if not verify_paraphrase_intent(task, extracted):
            continue
        try:
            payload = parse_json_payload(paraphrase_text)
        except ValueError:
            continue
        paraphrase = payload.get("paraphrase")
        if not isinstance(paraphrase, str) or not paraphrase.strip():
            continue
        grouped[task.task_id].append(paraphrase.strip())

    return [
        PromptVariantRecord(
            task_id=task.task_id,
            tier=task.tier,
            family=task.family,
            seed=task.seed,
            original_prompt=task.prompt,
            verified_paraphrases=grouped.get(task.task_id, []),
            shortfall=max(0, 3 - len(grouped.get(task.task_id, []))),
        )
        for task in tasks
    ]


def paraphrase_shortfall_report(records: Sequence[PromptVariantRecord]) -> dict[str, Any]:
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "verified": 0, "shortfall": 0})
    for record in records:
        key = f"{record.tier}:{record.family}"
        by_family[key]["tasks"] += 1
        by_family[key]["verified"] += len(record.verified_paraphrases)
        by_family[key]["shortfall"] += record.shortfall
    return {
        "families": dict(sorted(by_family.items())),
        "totals": {
            "tasks": len(records),
            "verified_paraphrases": sum(len(record.verified_paraphrases) for record in records),
            "shortfall": sum(record.shortfall for record in records),
        },
    }


def _entity_name(block_id: str) -> str:
    return _ENTITY_NAMES.get(block_id, block_id.replace("minecraft:", "").replace("_", " "))


def _snapshot_by_block_id(blocks: dict[tuple[int, int, int], str]) -> dict[str, list[tuple[int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for coord, block_id in blocks.items():
        grouped[block_id].append(coord)
    for coords in grouped.values():
        coords.sort()
    return dict(grouped)


def _single_block_coord(
    positions_by_block_id: dict[str, list[tuple[int, int, int]]],
    block_id: str,
) -> tuple[int, int, int] | None:
    coords = positions_by_block_id.get(block_id)
    if coords is None or len(coords) != 1:
        return None
    return coords[0]


def _world_coord_answer(coord: tuple[int, int, int]) -> str:
    return f"{coord[0]},{coord[1]},{coord[2]}"


def _canonical_intent(source_task: TaskSpec | TextQATaskSpec) -> dict[str, Any]:
    intent = source_task.metadata.get("canonical_intent")
    if not isinstance(intent, dict):
        return {}
    return intent


def _origin_coord(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, dict):
        return None
    return (int(value["x"]), int(value["y"]), int(value["z"]))


def _coord_inside_xz(coord: tuple[int, int, int], *, min_x: int, max_x: int, min_z: int, max_z: int) -> bool:
    return min_x < coord[0] < max_x and min_z < coord[2] < max_z


def _answer_from_furthest_north(
    positions_by_block_id: dict[str, list[tuple[int, int, int]]],
    metadata: dict[str, Any],
) -> str | None:
    entities = metadata.get("entity_block_ids")
    if not isinstance(entities, list) or not entities:
        return None
    coords: list[tuple[str, tuple[int, int, int]]] = []
    for entity in entities:
        if not isinstance(entity, str):
            return None
        coord = _single_block_coord(positions_by_block_id, entity)
        if coord is None:
            return None
        coords.append((entity, coord))
    return _entity_name(min(coords, key=lambda item: item[1][2])[0])


def _answer_from_marker_chain(
    world_blocks: dict[tuple[int, int, int], str],
    positions_by_block_id: dict[str, list[tuple[int, int, int]]],
    metadata: dict[str, Any],
) -> str | None:
    start_entity = metadata.get("start_entity")
    steps = metadata.get("steps")
    step_distance = metadata.get("step_distance")
    if not isinstance(start_entity, str) or not isinstance(steps, list) or not isinstance(step_distance, int):
        return None
    current = _single_block_coord(positions_by_block_id, start_entity)
    if current is None:
        return None
    for raw_step in steps:
        if raw_step == "north":
            current = (current[0], current[1], current[2] - step_distance)
        elif raw_step == "south":
            current = (current[0], current[1], current[2] + step_distance)
        elif raw_step == "east":
            current = (current[0] + step_distance, current[1], current[2])
        elif raw_step == "west":
            current = (current[0] - step_distance, current[1], current[2])
        else:
            return None
    final_entity = world_blocks.get(current)
    if final_entity is None:
        return None
    return _entity_name(final_entity)


def _answer_from_player_relative(
    *,
    player_facing: str,
    player_x: int,
    player_z: int,
    positions_by_block_id: dict[str, list[tuple[int, int, int]]],
    metadata: dict[str, Any],
) -> str | None:
    target_direction = metadata.get("target_direction")
    if target_direction not in {"front", "behind", "left", "right"}:
        return None
    for block_id, coords in positions_by_block_id.items():
        if len(coords) != 1:
            continue
        coord = coords[0]
        dx = coord[0] - player_x
        dz = coord[2] - player_z
        if abs(dx) + abs(dz) != 1:
            continue
        if _player_relative_direction(player_facing, dx, dz) == target_direction:
            return _entity_name(block_id)
    return None


def _answer_from_inside_enclosure(
    *,
    source_task: TaskSpec | TextQATaskSpec,
    positions_by_block_id: dict[str, list[tuple[int, int, int]]],
    answer_format: AnswerFormat,
) -> str | None:
    metadata = source_task.metadata
    intent = _canonical_intent(source_task)
    enclosure_block_id = metadata.get("enclosure_block_id") or intent.get("enclosure_block_id") or "minecraft:stone_bricks"
    if not isinstance(enclosure_block_id, str):
        return None
    enclosure_coords = positions_by_block_id.get(enclosure_block_id)
    if not enclosure_coords:
        return None

    min_x = min(coord[0] for coord in enclosure_coords)
    max_x = max(coord[0] for coord in enclosure_coords)
    min_z = min(coord[2] for coord in enclosure_coords)
    max_z = max(coord[2] for coord in enclosure_coords)

    if answer_format == "coordinate":
        inside_coord = _origin_coord(intent.get("inside_tower"))
        if inside_coord is None or not _coord_inside_xz(inside_coord, min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z):
            return None
        return _world_coord_answer(inside_coord)

    candidate_entities = metadata.get("candidate_entity_block_ids")
    if isinstance(candidate_entities, list):
        inside_entities: list[str] = []
        for entity in candidate_entities:
            if not isinstance(entity, str):
                return None
            coords = positions_by_block_id.get(entity)
            if coords is None:
                return None
            if any(_coord_inside_xz(coord, min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z) for coord in coords):
                inside_entities.append(entity)
        if len(inside_entities) != 1:
            return None
        return _entity_name(inside_entities[0])

    inside_entity_block_id = metadata.get("inside_entity_block_id")
    if not isinstance(inside_entity_block_id, str):
        return None
    coords = positions_by_block_id.get(inside_entity_block_id)
    if coords is None:
        return None
    if not any(_coord_inside_xz(coord, min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z) for coord in coords):
        return None
    return _entity_name(inside_entity_block_id)


def _answer_from_shared_wall(
    *,
    source_task: TaskSpec | TextQATaskSpec,
    world_blocks: dict[tuple[int, int, int], str],
) -> str | None:
    metadata = source_task.metadata
    intent = _canonical_intent(source_task)
    first_room_origin = _origin_coord(metadata.get("left_room_origin")) or _origin_coord(intent.get("first_room_origin"))
    second_room_origin = _origin_coord(metadata.get("right_room_origin")) or _origin_coord(intent.get("second_room_origin"))
    if first_room_origin is None or second_room_origin is None:
        return None

    room_width = int(metadata.get("room_width") or intent.get("room_width") or 5)
    room_depth = int(metadata.get("room_depth") or intent.get("room_depth") or 5)
    room_height = int(metadata.get("room_height") or intent.get("room_height") or 3)
    wall_block_id = metadata.get("wall_block_id") or intent.get("wall_block_id") or "minecraft:stone_bricks"
    if not isinstance(wall_block_id, str):
        return None

    x1, y1, z1 = first_room_origin
    x2, y2, z2 = second_room_origin
    shared_coords: list[tuple[int, int, int]] = []
    if x1 + room_width - 1 == x2 or x2 + room_width - 1 == x1:
        shared_x = max(x1, x2)
        z_start = max(z1, z2)
        z_end = min(z1 + room_depth - 1, z2 + room_depth - 1)
        for y in range(max(y1, y2), max(y1, y2) + room_height):
            for z in range(z_start, z_end + 1):
                shared_coords.append((shared_x, y, z))
    elif z1 + room_depth - 1 == z2 or z2 + room_depth - 1 == z1:
        shared_z = max(z1, z2)
        x_start = max(x1, x2)
        x_end = min(x1 + room_width - 1, x2 + room_width - 1)
        for y in range(max(y1, y2), max(y1, y2) + room_height):
            for x in range(x_start, x_end + 1):
                shared_coords.append((x, y, shared_z))
    else:
        return "no"

    if not shared_coords:
        return "no"
    return "yes" if all(world_blocks.get(coord) == wall_block_id for coord in shared_coords) else "no"


def verify_world_qa_candidate(
    *,
    source_task: TaskSpec | TextQATaskSpec,
    candidate: WorldQACandidatePayload,
    candidate_index: int,
) -> TextQATaskSpec | None:
    world = build_world_from_setup(player=source_task.player, setup_blocks=source_task.setup_blocks)
    world_blocks = world.snapshot()
    positions_by_block_id = _snapshot_by_block_id(world_blocks)
    reasoning = candidate.canonical_reasoning
    metadata = dict(candidate.metadata)
    metadata["source_task_id"] = source_task.task_id
    metadata["generated_from_world"] = True

    if candidate.question_type == "furthest_north_marker":
        expected_answer = _answer_from_furthest_north(positions_by_block_id, candidate.metadata)
        tier = "qa_directional_single_hop"
    elif candidate.question_type == "marker_chain":
        expected_answer = _answer_from_marker_chain(world_blocks, positions_by_block_id, candidate.metadata)
        tier = "qa_multi_hop_chain"
    elif candidate.question_type == "player_relative_marker":
        expected_answer = _answer_from_player_relative(
            player_facing=source_task.player.facing,
            player_x=source_task.player.x,
            player_z=source_task.player.z,
            positions_by_block_id=positions_by_block_id,
            metadata=candidate.metadata,
        )
        tier = "qa_viewpoint_transform"
    elif candidate.question_type == "inside_enclosure":
        expected_answer = _answer_from_inside_enclosure(
            source_task=source_task,
            positions_by_block_id=positions_by_block_id,
            answer_format=candidate.answer_format,
        )
        tier = "qa_topology"
    else:
        expected_answer = _answer_from_shared_wall(source_task=source_task, world_blocks=world_blocks)
        tier = "qa_topology"
    if expected_answer is None:
        return None

    normalized_candidate = normalize_text_qa_answer(candidate.expected_answer, candidate.answer_format)
    normalized_expected = normalize_text_qa_answer(expected_answer, candidate.answer_format)
    if normalized_candidate != normalized_expected:
        return None

    return TextQATaskSpec(
        task_id=f"{tier}:generated_world_candidate:{source_task.seed}:{candidate_index}",
        tier=tier,
        family="generated_world_candidate",
        seed=source_task.seed,
        prompt=candidate.prompt,
        player=source_task.player,
        setup_blocks=source_task.setup_blocks,
        expected_answer=expected_answer,
        answer_format=candidate.answer_format,
        canonical_reasoning=reasoning,
        metadata=metadata,
    )


def verified_world_qa_candidates(
    *,
    source_tasks: Sequence[TaskSpec | TextQATaskSpec],
    batch_outputs: dict[str, str],
) -> list[TextQATaskSpec]:
    task_by_seed = {task.seed: task for task in source_tasks}
    verified: list[TextQATaskSpec] = []
    for custom_id, text in batch_outputs.items():
        if not custom_id.startswith("world_qa_"):
            raise ValueError(f"unexpected world QA custom id: {custom_id}")
        task_seed = custom_id.removeprefix("world_qa_")
        task = task_by_seed.get(int(task_seed))
        if task is None:
            continue
        try:
            payload = WorldQACandidateBatchPayload.model_validate(parse_json_payload(text))
        except (ValidationError, ValueError):
            continue
        for candidate_index, candidate in enumerate(payload.candidates):
            verified_candidate = verify_world_qa_candidate(
                source_task=task,
                candidate=candidate,
                candidate_index=candidate_index,
            )
            if verified_candidate is not None:
                verified.append(verified_candidate)
    return verified
