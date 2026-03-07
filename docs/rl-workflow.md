# BrowseCraft RL Workflow

This repository runs RL experiments against the headless voxel simulator and a HUD MCP environment.

## 1. Task Generation

Generate deterministic tasks for all tiers:

```bash
cd ~/BrowseCraft-rl/sim
uv run python scripts/generate_hud_tasks.py --seed 7 --per-tier 100 --output remote_tasks.jsonl
```

Limit to specific tiers with `--tiers t1_absolute,t2_relative_single_ref`.

## 2. Local Environment Validation

Basic smoke checks:

```bash
cd ~/BrowseCraft-rl/sim
uv run python scripts/run_hud_smoke.py --skip-debug
```

HUD readiness check:

```bash
cd ~/BrowseCraft-rl/sim
uv run python scripts/run_hud_smoke.py
```

## 3. Local MCP Development

```bash
cd ~/BrowseCraft-rl/sim
hud dev env:env
```

## 4. Baseline Evaluation (Claude-Only)

```bash
cd ~/BrowseCraft-rl/sim
uv run python scripts/run_baseline_eval.py --tasks-file remote_tasks.jsonl
```

Default models:

- `claude-sonnet-4-6`
- `claude-opus-4-6`

## 5. Trajectory Export

Collect raw episodes from real Claude tool loops:

```bash
cd ~/BrowseCraft-rl/sim
uv run python scripts/collect_claude_trajectories.py --model claude-sonnet-4-6 --per-tier 1 --output raw_episodes.jsonl
```

Validate and export trajectory records:

```bash
cd ~/BrowseCraft-rl/sim
uv run python scripts/export_trajectories.py --input raw_episodes.jsonl --output trajectories.jsonl
```

This validates Anthropic-format message blocks (`text`, `tool_use`, `tool_result`) before writing JSONL.

## 6. RFT Task Export + Launch

```bash
cd ~/BrowseCraft-rl/sim
uv run python scripts/export_rft_tasks.py --trajectories trajectories.jsonl --output rft_tasks.jsonl
uv run python scripts/run_hud_rft.py --tasks-file rft_tasks.jsonl
```

`run_hud_rft.py` fails loudly on invite-only/access-denied signals.
