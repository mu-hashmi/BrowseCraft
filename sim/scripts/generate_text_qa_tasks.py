from __future__ import annotations

import argparse
import json
from pathlib import Path

from browsecraft_sim.rl.text_qa import generate_text_qa_tasks, write_text_qa_jsonl
from browsecraft_sim.rl.types import ALL_TEXT_QA_TIERS, TextQATier


def _parse_tiers(raw: str | None) -> list[TextQATier]:
    if raw is None or not raw.strip():
        return list(ALL_TEXT_QA_TIERS)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in ALL_TEXT_QA_TIERS]
    if invalid:
        raise ValueError(f"unsupported text QA tiers: {', '.join(invalid)}")
    return requested  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic text-only spatial QA tasks.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=100)
    parser.add_argument("--tiers", default=None)
    parser.add_argument("--output", default="text_qa_tasks.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tiers = _parse_tiers(args.tiers)
    tasks = generate_text_qa_tasks(seed=args.seed, per_tier=args.per_tier, tiers=tiers)
    output = Path(args.output).resolve()
    write_text_qa_jsonl(output, tasks)
    print(
        json.dumps(
            {
                "output": str(output),
                "seed": args.seed,
                "per_tier": args.per_tier,
                "tiers": tiers,
                "count": len(tasks),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
