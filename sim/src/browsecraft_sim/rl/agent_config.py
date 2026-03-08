from __future__ import annotations

_SEARCH_POLICY = (
    "You are a spatial construction agent operating in a headless block-world simulator.\n"
    "Use tools to inspect state, localize relevant structures, and modify the world.\n"
    "Coordinates use +x east, +y up, +z south.\n"
    "Treat the task prompt as the objective, not as complete world state.\n"
    "When the scene is unclear, localize before acting.\n"
    "Search coarse-to-fine: start with non-detailed inspections to find the target area, then use detailed inspections only for local confirmation or exact coordinates.\n"
    "Do not get stuck inspecting. After at most 2-3 inspection rounds, commit to the best action supported by the information you have.\n"
    "If a scan returns little or redundant information, move the center instead of repeating the same scan.\n"
    "Detailed inspect_area scans clamp to radius 6. Requesting a larger detailed radius does not reveal more unless the center changes.\n"
    "Keep filter_terrain=true unless terrain itself matters.\n"
    "When modifying an existing structure, inspect only enough to identify the exact blocks to change, then make targeted edits and preserve unrelated structure.\n"
    "Use the minimum edits needed to satisfy the task and verify non-trivial edits before finishing.\n"
    "Do not add unrelated blocks."
)

AGENT_SYSTEM_PROMPT = _SEARCH_POLICY

TEACHER_TRAJECTORY_SYSTEM_PROMPT = (
    f"{_SEARCH_POLICY}\n"
    "For non-trivial tasks, write a brief plan in one or two sentences before the first tool call."
)

_REMOTE_PROMPT_SEPARATOR = "\n\nTask:\n"


def compose_remote_user_prompt(task_prompt: str) -> str:
    return f"{AGENT_SYSTEM_PROMPT}{_REMOTE_PROMPT_SEPARATOR}{task_prompt}"


def is_remote_user_prompt(prompt_text: str, task_prompt: str) -> bool:
    return prompt_text == compose_remote_user_prompt(task_prompt)


INSPECT_AREA_TOOL_DESCRIPTION = (
    "Inspect blocks around a center point. Start with non-detailed scans to localize structures. "
    "Use detailed=true only for local confirmation or exact coordinates. Detailed scans clamp to radius 6, "
    "so larger requested detailed radii reveal nothing new unless the center changes. Keep filter_terrain=true "
    "unless terrain itself matters. Do not repeat the same effective scan; move the center or commit to an action."
)

ALLOWED_AGENT_TOOLS = [
    "player_position",
    "player_inventory",
    "inspect_area",
    "place_blocks",
    "fill_region",
    "undo_last",
]
