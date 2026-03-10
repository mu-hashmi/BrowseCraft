from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


_INVITE_ONLY_MARKERS = (
    "invite only",
    "invite-only",
    "access required",
    "not authorized",
    "forbidden",
    "permission denied",
)


def _contains_invite_only(output: str) -> bool:
    normalized = output.casefold()
    return any(marker in normalized for marker in _INVITE_ONLY_MARKERS)


def run(tasks_file: Path, model_id: str | None, reasoning_effort: str) -> dict[str, object]:
    command = ["hud", "rft", "run", str(tasks_file), "--reasoning-effort", reasoning_effort, "--yes"]
    if model_id:
        command.extend(["--model-id", model_id])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    combined = f"{result.stdout}\n{result.stderr}".strip()
    if _contains_invite_only(combined):
        raise RuntimeError(
            "HUD RFT access appears unavailable (invite-only or unauthorized). "
            "Request account access, then retry this command."
        )
    if result.returncode != 0:
        raise RuntimeError(f"hud rft run failed (exit={result.returncode}): {combined}")

    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HUD RFT with explicit invite-only/access checks.")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--reasoning-effort", default="medium", choices=("low", "medium", "high"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        tasks_file=Path(args.tasks_file).resolve(),
        model_id=args.model_id,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
