# Instructions For Coding Agents Working With This Repo

## Preferred Codex Test Workflow

Validate chat tool use and spatial behavior programmatically.

### Non-negotiables

- Do not use GUI automation unless explicitly required to validate HUD elements.
- Do not rely on manual clicking/typing as the primary validation path.
- Validate world reasoning and block placement behavior, not just HTTP acceptance.
- Keep all world-modifying tools available during spatial validation. Evaluate spatial tests on world outcome, not on whether the model picked a preferred tool.

### Required env keys

Set in `backend/.env`:

| Key                                | Required for                              |
| ---------------------------------- | ----------------------------------------- |
| `ANTHROPIC_API_KEY`                | `/chat` model reasoning + tool calls      |
| `ANTHROPIC_CHAT_MODEL`             | chat model (default `claude-sonnet-4-6`)  |
| `ANTHROPIC_PLANNER_MODEL`          | optional planner model override           |
| `ANTHROPIC_TRIAGE_MODEL`           | optional triage model override            |
| `ANTHROPIC_ENABLE_BUILD_PLANNER`   | optional plan-then-execute for build mode |
| `CONVEX_URL` + `CONVEX_ACCESS_KEY` | optional session persistence              |
| `SUPERMEMORY_API_KEY`              | optional long-term memory                 |
| `LAMINAR_API_KEY`                  | optional tracing/observability            |

### Fast checks

```bash
cd ~/BrowseCraft/backend && uv run pytest -q
cd ~/BrowseCraft/mod && gradle test
cd ~/BrowseCraft/mod && gradle build
```

### Spatial reliability checks (real API)

Run spatial tests explicitly (these are slower and cost money):

```bash
cd ~/BrowseCraft/backend && uv run pytest -q -m spatial
```

Default live iteration should stay Sonnet-only and planner-free unless you are explicitly validating planner behavior:

```bash
# default live pass: no Opus, no build planner, no planning tests
cd ~/BrowseCraft/backend && ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6 ANTHROPIC_PLANNER_MODEL=claude-sonnet-4-6 ANTHROPIC_TRIAGE_MODEL=claude-haiku-4-5 ANTHROPIC_ENABLE_BUILD_PLANNER=false uv run pytest -q -m spatial --quick

# broader Sonnet-only live pass
cd ~/BrowseCraft/backend && ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6 ANTHROPIC_PLANNER_MODEL=claude-sonnet-4-6 ANTHROPIC_TRIAGE_MODEL=claude-haiku-4-5 ANTHROPIC_ENABLE_BUILD_PLANNER=false uv run pytest -q -m spatial

# explicitly include planning coverage
cd ~/BrowseCraft/backend && ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6 ANTHROPIC_PLANNER_MODEL=claude-sonnet-4-6 ANTHROPIC_TRIAGE_MODEL=claude-haiku-4-5 ANTHROPIC_ENABLE_BUILD_PLANNER=true uv run pytest -q -m spatial --with-planning

# only use Opus if you intentionally want planner-model coverage
cd ~/BrowseCraft/backend && ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6 ANTHROPIC_PLANNER_MODEL=claude-opus-4-6 ANTHROPIC_TRIAGE_MODEL=claude-haiku-4-5 ANTHROPIC_ENABLE_BUILD_PLANNER=true uv run pytest -q -m spatial --with-planning
```

These tests use the headless voxel simulator (no Minecraft client required), send real `/v1/chat` requests, service real tool calls, and assert structural outcomes.
They are for spatial correctness, not for enforcing specific tool-choice policies.

### Optional manual live run

If manual confirmation is requested:

```bash
cd ~/BrowseCraft/backend && uv run browsecraft-backend
cd ~/BrowseCraft/mod && gradle runClient
```

Then run `/chat <message>` in-game.

### HUD-only visual verification

Use the HUD capture gametest when you need rendered screenshots of the client HUD without manual interaction:

```bash
cd ~/BrowseCraft/mod
env JAVA_TOOL_OPTIONS='-Dbrowsecraft.clientGameTestSuite=hud' gradle runClientGameTest
```

This path is exclusively for validating HUD elements:
- HUD mode rendering
- INPUT mode rendering
- text wrapping
- panel/input layout
- cursor placement
- screenshot/metadata capture

It is not a build validation path.
- It does not verify planner/executor behavior.
- It does not verify backend tool use.
- It does not verify block placement correctness in the world.

Use spatial backend tests or a real `/chat` integration run when the goal is to verify that structures build correctly.

### Iteration loop for spatial improvements

1. Update system prompt/tool descriptions in `chat_orchestrator.py`.
2. Run `uv run pytest -q -m spatial --quick` first. Add `--with-planning` only when you are intentionally validating planner behavior.
3. Inspect failing scenario assertions and adjust prompt/tool behavior.
4. Re-run until stable.

### Headless simulator iteration

Use the simulator to validate structural outcomes without launching Minecraft:

```bash
cd ~/BrowseCraft/sim
uv run browsecraft-sim \"build a 5x5 stone platform\" --report-json --slice-y 64
```

The JSON report includes world diff metrics, block-count/height/connectivity validation, and optional ASCII slice output for quick diagnosis.

### Token-cost controls

- Prompt caching is enabled in chat calls; keep system/tool definitions stable to maximize cache hits.
- Prefer `inspect_area` with small radii (4-6) first.
- Use `detailed=true` only when coordinate-level block positions are required.
- Large `place_blocks` payloads are expensive; batch large builds into multiple calls.

### Common failure signals

- `Invalid arguments for inspect_area`: missing `center`/`radius` or malformed `detailed` flag.
- `Exceeded max tool rounds`: model is stuck in inspect/modify loop; tighten prompt guidance.
- Spatial assertions failing with zero placements: model did not call a world-modifying tool.
- Spatial assertions failing due wrong coordinates: improve orientation/grounding instructions in the system prompt.

## RL Workflow Notes

- RL modules live under `sim/src/browsecraft_sim/rl`.
- Shared simulator dispatch logic lives in `sim/src/browsecraft_sim/tool_dispatch.py`.
- Use deterministic task generation through `sim/scripts/generate_hud_tasks.py`.
- Keep reward settings configurable via JSON/CLI, defaulting to format-gated scoring.
- Use Claude-only baseline models first (`claude-sonnet-4-6`, `claude-opus-4-6`) before adding provider adapters.
- Run `backend` regression tests after dispatch changes before adding additional RL code.
