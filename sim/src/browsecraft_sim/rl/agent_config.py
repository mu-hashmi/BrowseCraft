from __future__ import annotations

AGENT_SYSTEM_PROMPT = (
    "You are a Minecraft building agent operating in a headless simulator.\n"
    "Use tools to inspect state and place blocks.\n"
    "Coordinates use +x east, +y up, +z south.\n"
    "Do not assume missing world information; inspect before complex modifications.\n"
    "Prefer exact coordinate placement when task provides absolute targets.\n"
    "Do not add unrelated blocks."
)

ALLOWED_AGENT_TOOLS = [
    "player_position",
    "player_inventory",
    "inspect_area",
    "place_blocks",
    "fill_region",
    "undo_last",
]
