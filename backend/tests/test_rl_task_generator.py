from __future__ import annotations

from browsecraft_sim.rl.task_generator import generate_task, generate_tasks
from browsecraft_sim.rl.types import ALL_TIERS


def test_generate_tasks_is_deterministic_for_same_seed() -> None:
    first = generate_tasks(seed=17, per_tier=2)
    second = generate_tasks(seed=17, per_tier=2)
    assert [task.model_dump(mode="json") for task in first] == [task.model_dump(mode="json") for task in second]


def test_generate_tasks_covers_all_tiers() -> None:
    tasks = generate_tasks(seed=9, per_tier=1)
    assert {task.tier for task in tasks} == set(ALL_TIERS)


def test_t5_has_required_families() -> None:
    families = {generate_task(tier="t5_modification", seed=33, index=index).family for index in range(20)}
    assert families >= {"replace_material_preserve_shape", "widen_or_reposition_opening"}


def test_t6_has_required_families() -> None:
    families = {generate_task(tier="t6_composition", seed=33, index=index).family for index in range(20)}
    assert families >= {"bridge_between_structures", "connect_rooms_with_corridor"}
