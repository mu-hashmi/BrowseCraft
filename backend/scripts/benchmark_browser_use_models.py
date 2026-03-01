from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass

from browser_use_sdk import AsyncBrowserUse, SessionSettings
from pydantic import BaseModel, Field


_PROMPT_TEMPLATE = """Go to planetminecraft.com and find downloadable Minecraft Java schematic files.
Use only Planet Minecraft pages.
Search for the query and open project pages.
On each project, use the Schematic download tab/button path.
Never use World Save or map downloads. Never use .zip downloads.
If a project only offers World Save/.zip, skip it and continue.
Download only files with allowed extensions so they appear in task output files.
Stop if you revisit the same page repeatedly and move to the next result.
Return only candidates backed by downloaded files with allowed extensions.
Query: {query}
Minecraft Version: {mc_version}
Allowed extensions: {allowed_exts}
Target site: planetminecraft.com"""


class _Candidate(BaseModel):
    canonical_url: str
    filename: str
    title: str
    score: float = Field(default=0.5)
    download_url: str | None = None


class _Candidates(BaseModel):
    candidates: list[_Candidate] = Field(default_factory=list)


@dataclass(slots=True)
class _RunResult:
    model: str
    run_index: int
    elapsed_seconds: float
    steps: int
    candidate_count: int
    success: bool
    error: str | None = None


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


async def _benchmark_one(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_steps: int,
    timeout_seconds: int,
    profile_id: str | None,
    run_index: int,
) -> _RunResult:
    start = time.perf_counter()
    steps = 0
    client = AsyncBrowserUse(api_key=api_key, timeout=float(timeout_seconds))
    try:
        task = client.run(
            task=prompt,
            output_schema=_Candidates,
            llm=model,
            max_steps=max_steps,
            allowed_domains=["planetminecraft.com", "www.planetminecraft.com"],
            session_settings=SessionSettings(profile_id=profile_id) if profile_id else None,
        )
        async for _ in task:
            steps += 1

        if task.result is None:
            raise RuntimeError("Task completed without result")
        output = task.result.output
        if output is None:
            return _RunResult(
                model=model,
                run_index=run_index,
                elapsed_seconds=round(time.perf_counter() - start, 3),
                steps=steps,
                candidate_count=0,
                success=True,
            )
        if not isinstance(output, _Candidates):
            raise TypeError(f"Unexpected output type: {type(output)!r}")
        return _RunResult(
            model=model,
            run_index=run_index,
            elapsed_seconds=round(time.perf_counter() - start, 3),
            steps=steps,
            candidate_count=len(output.candidates),
            success=True,
        )
    except Exception as exc:
        return _RunResult(
            model=model,
            run_index=run_index,
            elapsed_seconds=round(time.perf_counter() - start, 3),
            steps=steps,
            candidate_count=0,
            success=False,
            error=str(exc),
        )
    finally:
        await client.close()


async def _run(args: argparse.Namespace) -> None:
    api_key = _required_env("BROWSER_USE_API_KEY")
    allowed_exts = ".schem, .litematic, .schematic"
    prompt = _PROMPT_TEMPLATE.format(
        query=args.query,
        mc_version=args.mc_version,
        allowed_exts=allowed_exts,
    )

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise RuntimeError("At least one model is required")

    all_results: list[_RunResult] = []
    for model in models:
        for run_index in range(1, args.runs + 1):
            result = await _benchmark_one(
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_steps=args.max_steps,
                timeout_seconds=args.timeout_seconds,
                profile_id=args.profile_id,
                run_index=run_index,
            )
            all_results.append(result)
            print(json.dumps(asdict(result), sort_keys=True))

    summary: dict[str, dict[str, float]] = {}
    for model in models:
        rows = [result for result in all_results if result.model == model and result.success]
        if not rows:
            summary[model] = {"successful_runs": 0}
            continue
        elapsed = [result.elapsed_seconds for result in rows]
        candidate_counts = [result.candidate_count for result in rows]
        summary[model] = {
            "successful_runs": float(len(rows)),
            "avg_elapsed_seconds": round(sum(elapsed) / len(elapsed), 3),
            "avg_candidate_count": round(sum(candidate_counts) / len(candidate_counts), 3),
        }

    print("SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Browser Use models for PlanetMinecraft schematic search.")
    parser.add_argument("--query", default="medieval castle")
    parser.add_argument("--mc-version", default="1.21.11")
    parser.add_argument("--models", default="browser-use-llm,browser-use-2.0")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--profile-id", default=os.getenv("BROWSER_USE_PROFILE_ID"))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
