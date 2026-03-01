# Instructions For Coding Agents Working With This Repo

This is a hackathon project. Production-readiness is unnecessary.

## Preferred Codex Test Workflow

BrowseCraft is now `/chat`-first. Validate chat tool use and spatial behavior programmatically.

### Non-negotiables

- Do not use GUI automation.
- Do not rely on manual clicking/typing as the primary validation path.
- Validate world reasoning and block placement behavior, not just HTTP acceptance.
- Treat `place_blocks` + `undo_last` tool flow as the core build path.

### Required env keys

Set in `backend/.env`:

| Key                                | Required for                              |
| ---------------------------------- | ----------------------------------------- |
| `ANTHROPIC_API_KEY`                | `/chat` model reasoning + tool calls      |
| `ANTHROPIC_CHAT_MODEL`             | chat model (default `claude-sonnet-4-6`)  |
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

Use Sonnet for day-to-day iteration and Opus only for final validation:

```bash
# iteration (default)
cd ~/BrowseCraft/backend && ANTHROPIC_CHAT_MODEL=claude-sonnet-4-6 uv run pytest -q -m spatial

# final validation
cd ~/BrowseCraft/backend && ANTHROPIC_CHAT_MODEL=claude-opus-4-6 uv run pytest -q -m spatial
```

These tests use the headless voxel simulator (no Minecraft client required), send real `/v1/chat` requests, service real tool calls, and assert structural outcomes.

### Optional manual live run

If manual confirmation is requested:

```bash
cd ~/BrowseCraft/backend && uv run browsecraft-backend
cd ~/BrowseCraft/mod && gradle runClient
```

Then run `/chat <message>` in-game.

### Iteration loop for spatial improvements

1. Update system prompt/tool descriptions in `chat_orchestrator.py`.
2. Run `uv run pytest -q -m spatial`.
3. Inspect failing scenario assertions and adjust prompt/tool behavior.
4. Re-run until stable.

### Token-cost controls

- Prompt caching is enabled in chat calls; keep system/tool definitions stable to maximize cache hits.
- Prefer `inspect_area` with small radii (4-6) first.
- Use `detailed=true` only when coordinate-level block positions are required.
- Large `place_blocks` payloads are expensive; batch large builds into multiple calls.

### Common failure signals

- `Invalid arguments for inspect_area`: missing `center`/`radius` or malformed `detailed` flag.
- `Exceeded max tool rounds`: model is stuck in inspect/modify loop; tighten prompt guidance.
- Spatial assertions failing with zero placements: model did not call `place_blocks`.
- Spatial assertions failing due wrong coordinates: improve orientation/grounding instructions in the system prompt.
