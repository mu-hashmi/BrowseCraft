from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from browsecraft_sim.rl.task_generator import generate_tasks


DEFAULT_MODELS = ("claude-sonnet-4-6", "claude-opus-4-6")


def _ensure_tasks_file(path: Path, seed: int, per_tier: int, env_name: str) -> None:
    if path.exists():
        return
    tasks = generate_tasks(seed=seed, per_tier=per_tier)
    lines = []
    for task in tasks:
        record = {
            "env": {"name": env_name},
            "scenario": task.tier,
            "args": {"task_spec": task.model_dump(mode="json")},
        }
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _run_hud_eval(tasks_file: Path, model: str, cwd: Path) -> dict[str, object]:
    started = datetime.now(UTC)
    command = [
        "hud",
        "eval",
        str(tasks_file),
        "claude",
        "--model",
        model,
        "--full",
        "--yes",
        "--quiet",
    ]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    ended = datetime.now(UTC)
    return {
        "model": model,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_seconds": round((ended - started).total_seconds(), 3),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Claude-only baseline HUD evals for BrowseCraft RL tasks.")
    parser.add_argument("--tasks-file", default="remote_tasks.jsonl")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=100)
    parser.add_argument("--env-name", default="browsecraft-spatial-rl")
    parser.add_argument("--summary-out", default="baseline_eval_summary.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sim_dir = Path(__file__).resolve().parents[1]
    tasks_file = (sim_dir / args.tasks_file).resolve()
    _ensure_tasks_file(
        path=tasks_file,
        seed=int(args.seed),
        per_tier=int(args.per_tier),
        env_name=str(args.env_name),
    )

    models = [model.strip() for model in str(args.models).split(",") if model.strip()]
    if not models:
        raise ValueError("at least one model is required")

    results = [_run_hud_eval(tasks_file=tasks_file, model=model, cwd=sim_dir) for model in models]
    summary = {
        "tasks_file": str(tasks_file),
        "models": models,
        "results": results,
    }

    summary_path = (sim_dir / args.summary_out).resolve()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_out": str(summary_path), "models": models}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
