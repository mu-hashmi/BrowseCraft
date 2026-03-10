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

## HUD Workflow Notes

### Verified programmatic surfaces

- `hud.datasets.save_tasks(...)` is the reliable way to create a platform taskset programmatically.
- `hud rft run ...` is the verified CLI surface for HUD reinforcement fine-tuning.
- HUD MCP is useful for read/debug only:
  - `list_jobs`
  - `get_job`
  - `get_job_traces`
  - `get_trace`
  - `list_environments`
  - `get_environment`
- Do not assume HUD MCP can launch training or upload tasksets. It cannot.
- Do not assume there is a documented standalone `hud sft` CLI. We did not find one. The generic training UI exists, but the only verified programmatic training surface in this repo workflow is `hud rft run`.

### Two task formats that look similar but are not interchangeable

- `generate_hud_tasks.py --format v5-hub` emits v5 HUD tasks:
  - shape: `{"env":{"name":...},"scenario":...,"args":...}`
  - use these for:
    - platform tasksets
    - HUD evals
    - the training UI taskset selector
- `generate_hud_tasks.py --format local-eval` emits legacy/local-eval tasks:
  - shape includes `id`, `prompt`, `mcp_config`, `setup_tool`, `evaluate_tool`, `agent_config`, `metadata`
  - use these for:
    - `hud rft run ...`
- Do not feed `v5-hub` output directly to `hud rft run`.
  - That path produced a trainer failure with:
    - `ValueError: Task is missing required id`
    - `prompt: null`
- If `hud rft run` shows `Task is missing required id`, the input file is almost certainly the wrong task format.

### Environment and image prerequisites

- Build and push the HUD environment before remote eval or RFT:

```bash
cd ~/BrowseCraft/sim
hud push --yes
```

- `generate_hud_tasks.py` uses repo state to resolve the correct environment/image:
  - `.hud/deploy.json` is used to resolve the linked remote environment name for `v5-hub`
  - `hud.lock.yaml` is used to resolve the local image for `local-eval`
- If the environment is not linked or published yet, fix that first instead of trying to patch around missing image/env metadata.

### Creating a platform taskset that the training UI can see

- The HUD training wizard is blocked until a platform taskset exists.
- If the UI says there are no tasksets, upload one from Python instead of clicking around.

```bash
cd ~/BrowseCraft/sim
uv run python scripts/generate_hud_tasks.py \
  --seed 42 \
  --per-tier 5 \
  --env-name browsecraft-spatial-rl-1 \
  --format v5-hub \
  --output runs/train_taskset_seed42_30.jsonl
```

```bash
set -a && source ~/BrowseCraft/backend/.env && set +a
cd ~/BrowseCraft/sim
uv run python - <<'PY'
from pathlib import Path
from hud.datasets.loader import save_tasks
from hud.eval.task import Task

path = Path("runs/train_taskset_seed42_30.jsonl")
tasks = [Task.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
taskset_id = save_tasks("ranoobaba/browsecraft-spatial-rl-train-smoke-v1", tasks)
print(taskset_id)
PY
```

- After that, `Train Model` / `New -> Training Job` can select the taskset normally.

### Recommended smoke eval before training

- Before spending training credits, run a small eval on the uploaded taskset.
- Recommended smoke settings in the HUD UI:
  - `Max Steps = 20`
  - `Group Size = 1`
- Reason:
  - catches schema/runtime/tooling failures cheaply
  - avoids multiplying traces for no benefit during wiring checks

### Running HUD RFT from CLI

- Use `local-eval` task files for `hud rft run`.

```bash
cd ~/BrowseCraft/sim
uv run python scripts/generate_hud_tasks.py \
  --seed 123 \
  --per-tier 5 \
  --tiers t1_absolute,t2_relative_single_ref \
  --env-name browsecraft-spatial-rl-1 \
  --format local-eval \
  --output rft_smoke_local_10.jsonl
```

```bash
cd ~/BrowseCraft/sim
hud rft run rft_smoke_local_10.jsonl \
  --model-id <forked_model_id> \
  --reasoning-effort medium \
  --yes
```

- There is also a thin wrapper script in `sim/scripts/run_hud_rft.py` if explicit failure messaging is useful.

### Getting the right job id

- Use `list_jobs` before trying to import or inspect a job. Misreading one character in the UUID caused a long false debugging path.
- A wrong job id can produce confusing HUD tool-side failures like:
  - `'NoneType' object has no attribute 'data'`
- Do not trust hand-copied ids from screenshots. Resolve the exact id first.

### Importing traces from HUD

- Use `sim/scripts/import_hud_job.py` to pull a HUD job into local RL JSONL.

```bash
set -a && source ~/BrowseCraft/backend/.env && set +a
cd ~/BrowseCraft/sim
uv run python scripts/import_hud_job.py \
  --job-id <job-id> \
  --output runs/hud_job_raw.jsonl \
  --summary-output runs/hud_job_summary.json
```

- The importer now supports both trace schemas we observed:
  - Anthropic-style spans: `inference.messages`
  - Tinker/Qwen/OpenAI-style spans: `inference.chat_completion`
- The importer normalizes OpenAI-style:
  - `assistant.tool_calls`
  - `tool` role messages
  into the Anthropic-style internal message representation used by the RL pipeline.
- The importer skips traces with `status != completed` and records them in `skipped_traces`.
- A HUD job with `status = error` can still contain many usable completed traces. Import the completed traces instead of discarding the whole job.

### Importer and MCP failure semantics

- `get_job` / `get_job_traces` can return a successful HTTP response whose JSON payload is actually a tool-side error:
  - shape: `{"error": "..."}`
- Treat that as a real failure immediately. Do not continue and assume fields like `traces["traces"]` exist.
- `list_jobs` was more reliable than `get_job` when the underlying problem was simply a wrong id.

### Model-specific trace and context quirks

- Smaller/non-Claude models can blow their context window on large `inspect_area(..., detailed=true)` results.
- We saw a real Qwen eval trace fail on:
  - `context_window_exceeded_error`
  - `39538 prompt tokens + 1024 max_tokens > 32768`
- That came from a modification task after repeated large detailed inspections.
- When evaluating smaller models on BrowseCraft tasks:
  - keep `inspect_area` radii small
  - avoid unnecessary `detailed=true`
  - expect T5/T6 tasks to fail from context pressure before they fail from pure spatial reasoning

### What worked and what did not

- Worked:
  - `save_tasks(...)` for taskset upload
  - `v5-hub` for tasksets and evals
  - `local-eval` for `hud rft run`
  - `import_hud_job.py` for completed-trace export
  - HUD MCP for read/debug
- Did not work:
  - using browser/UI as the primary trace-inspection path
  - using `v5-hub` tasks with `hud rft run`
  - assuming HUD MCP exposes training controls
  - assuming all traces use the Anthropic span schema
