from __future__ import annotations

import json

from browsecraft_sim.main import HeadlessVoxelWorld, PlayerState
from browsecraft_sim.rl import augmentation
from browsecraft_sim.rl.augmentation import (
    paraphrase_shortfall_report,
    verified_paraphrase_records,
    verified_world_qa_candidates,
)
from browsecraft_sim.rl.task_generator import generate_task
from browsecraft_sim.rl.text_qa import generate_text_qa_task


def test_verified_paraphrase_records_keep_tasks_with_zero_verified_outputs() -> None:
    first = generate_task(tier="t1_absolute", seed=7, index=0)
    second = generate_task(tier="t1_absolute", seed=7, index=1)

    records = verified_paraphrase_records(
        tasks=[first, second],
        paraphrase_outputs={
            f"paraphrase_{first.seed}_0": json.dumps({"paraphrase": "Put the same block at the same coordinates."}),
        },
        verification_outputs={
            f"paraphrase_verify_{first.seed}_0": json.dumps(first.metadata["canonical_intent"]),
        },
    )

    assert len(records) == 2
    first_record = next(record for record in records if record.task_id == first.task_id)
    second_record = next(record for record in records if record.task_id == second.task_id)
    assert first_record.verified_paraphrases == ["Put the same block at the same coordinates."]
    assert first_record.shortfall == 2
    assert second_record.verified_paraphrases == []
    assert second_record.shortfall == 3

    report = paraphrase_shortfall_report(records)
    assert report["totals"]["tasks"] == 2
    assert report["totals"]["verified_paraphrases"] == 1
    assert report["totals"]["shortfall"] == 5


def test_verified_world_qa_candidates_filter_unverifiable_outputs() -> None:
    source_task = generate_text_qa_task(tier="qa_directional_single_hop", seed=9, index=0)
    entity_block_ids = [block.block_id for block in source_task.setup_blocks]
    outputs = {
        f"world_qa_{source_task.seed}": json.dumps(
            {
                "candidates": [
                    {
                        "question_type": "furthest_north_marker",
                        "prompt": "Which marker is furthest north?",
                        "expected_answer": source_task.expected_answer,
                        "answer_format": "entity_name",
                        "canonical_reasoning": ["North is the smallest z coordinate."],
                        "metadata": {"entity_block_ids": entity_block_ids},
                    },
                    {
                        "question_type": "furthest_north_marker",
                        "prompt": "Which marker is furthest north?",
                        "expected_answer": "wrong marker",
                        "answer_format": "entity_name",
                        "canonical_reasoning": ["North is the smallest z coordinate."],
                        "metadata": {"entity_block_ids": entity_block_ids},
                    },
                ]
            }
        )
    }

    verified = verified_world_qa_candidates(source_tasks=[source_task], batch_outputs=outputs)
    assert len(verified) == 1
    assert verified[0].expected_answer == source_task.expected_answer
    assert verified[0].metadata["generated_from_world"] is True


def test_verified_world_qa_candidates_use_simulator_world_state(monkeypatch) -> None:
    source_task = generate_text_qa_task(tier="qa_directional_single_hop", seed=9, index=0)
    entity_block_ids = [block.block_id for block in source_task.setup_blocks]
    expected_block_id = source_task.setup_blocks[1].block_id
    expected_answer = expected_block_id.removeprefix("minecraft:").replace("_wool", "").replace("_", " ") + " marker"
    calls: list[int] = []

    def fake_build_world_from_setup(*, player, setup_blocks, terrain_radius=24):
        calls.append(terrain_radius)
        world = HeadlessVoxelWorld(
            player=PlayerState(
                x=player.x,
                y=player.y,
                z=player.z,
                facing=player.facing,
                dimension=player.dimension,
            )
        )
        world.flat_terrain(radius=terrain_radius)
        for index, block in enumerate(setup_blocks):
            z = -20 if index == 1 else 20 + index
            world.set_block((block.x, block.y, z), block.block_id)
        return world

    monkeypatch.setattr(augmentation, "build_world_from_setup", fake_build_world_from_setup)
    outputs = {
        f"world_qa_{source_task.seed}": json.dumps(
            {
                "candidates": [
                    {
                        "question_type": "furthest_north_marker",
                        "prompt": "Which marker is furthest north?",
                        "expected_answer": expected_answer,
                        "answer_format": "entity_name",
                        "canonical_reasoning": ["North is the smallest z coordinate."],
                        "metadata": {"entity_block_ids": entity_block_ids},
                    }
                ]
            }
        )
    }

    verified = verified_world_qa_candidates(source_tasks=[source_task], batch_outputs=outputs)
    assert calls == [24]
    assert len(verified) == 1
    assert verified[0].expected_answer == expected_answer
