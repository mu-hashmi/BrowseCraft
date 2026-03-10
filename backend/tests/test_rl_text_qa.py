from __future__ import annotations

from browsecraft_sim.rl.text_qa import (
    canonical_text_qa_response,
    generate_text_qa_task,
    generate_text_qa_tasks,
    grade_text_qa_answer,
    read_text_qa_jsonl,
    reconstruct_text_qa_task_from_task_id,
    write_text_qa_jsonl,
)
from browsecraft_sim.rl.types import ALL_TEXT_QA_TIERS


def test_generate_text_qa_tasks_is_deterministic_for_same_seed() -> None:
    first = generate_text_qa_tasks(seed=17, per_tier=2)
    second = generate_text_qa_tasks(seed=17, per_tier=2)
    assert [task.model_dump(mode="json") for task in first] == [task.model_dump(mode="json") for task in second]


def test_generate_text_qa_tasks_covers_all_tiers() -> None:
    tasks = generate_text_qa_tasks(seed=9, per_tier=1)
    assert {task.tier for task in tasks} == set(ALL_TEXT_QA_TIERS)


def test_reconstruct_text_qa_task_from_task_id_is_stable() -> None:
    task = generate_text_qa_task(tier="qa_multi_hop_chain", seed=17, index=3)
    reconstructed = reconstruct_text_qa_task_from_task_id(task.task_id)
    assert reconstructed.model_dump(mode="json") == task.model_dump(mode="json")


def test_grade_text_qa_answer_normalizes_entity_names() -> None:
    task = generate_text_qa_task(tier="qa_directional_single_hop", seed=7, index=0)
    result = grade_text_qa_answer(task, f"The {task.expected_answer.upper()}")
    assert result.correct
    assert result.reward_binary == 1.0


def test_multi_hop_chain_records_hop_count_and_final_answer() -> None:
    task = generate_text_qa_task(tier="qa_multi_hop_chain", seed=7, index=0)
    assert 2 <= len(task.metadata["steps"]) <= 8
    response = canonical_text_qa_response(task)
    assert response.endswith(f"Answer: {task.expected_answer}")


def test_topology_tasks_vary_world_geometry() -> None:
    geometries = set()
    for index in range(80):
        task = generate_text_qa_task(tier="qa_topology", seed=17, index=index)
        coords = tuple(sorted((block.x, block.y, block.z, block.block_id) for block in task.setup_blocks))
        geometries.add((task.family, coords))
        if len(geometries) >= 4:
            break
    assert len(geometries) >= 4


def test_shared_wall_topology_generates_yes_and_no_cases() -> None:
    answers = set()
    for index in range(200):
        task = generate_text_qa_task(tier="qa_topology", seed=23, index=index)
        if task.family != "shared_wall_yes_no":
            continue
        answers.add(task.expected_answer)
        if answers == {"yes", "no"}:
            break
    assert answers == {"yes", "no"}


def test_text_qa_jsonl_round_trip(tmp_path) -> None:
    tasks = generate_text_qa_tasks(seed=21, per_tier=1)
    path = tmp_path / "text_qa.jsonl"
    write_text_qa_jsonl(path, tasks)
    loaded = read_text_qa_jsonl(path)
    assert [task.model_dump(mode="json") for task in loaded] == [task.model_dump(mode="json") for task in tasks]
