from __future__ import annotations

from browsecraft_sim.rl.agent_config import (
    AGENT_SYSTEM_PROMPT,
    TEACHER_TRAJECTORY_SYSTEM_PROMPT,
    compose_remote_user_prompt,
    is_remote_user_prompt,
)


def test_compose_remote_user_prompt_wraps_raw_task_prompt() -> None:
    task_prompt = "Build a short stone bridge between the towers."

    prompt = compose_remote_user_prompt(task_prompt)

    assert prompt.startswith(AGENT_SYSTEM_PROMPT)
    assert prompt.endswith(task_prompt)
    assert "\n\nTask:\n" in prompt
    assert is_remote_user_prompt(prompt, task_prompt) is True


def test_teacher_prompt_extends_shared_policy_with_brief_plan_instruction() -> None:
    assert TEACHER_TRAJECTORY_SYSTEM_PROMPT.startswith(AGENT_SYSTEM_PROMPT)
    assert "brief plan" in TEACHER_TRAJECTORY_SYSTEM_PROMPT
