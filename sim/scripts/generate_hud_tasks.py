from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anyio
from hud.cli.utils.lockfile import get_local_image, load_lock
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from browsecraft_sim.rl.agent_config import AGENT_SYSTEM_PROMPT, ALLOWED_AGENT_TOOLS
from browsecraft_sim.rl.config import load_reward_config
from browsecraft_sim.rl.hud_env import ENV_NAME as SCENARIO_ENV_NAME
from browsecraft_sim.rl.task_generator import generate_tasks, tier_counts
from browsecraft_sim.rl.types import ALL_TIERS, Tier

LOCAL_EVAL_FORMAT = "local-eval"
V5_HUB_FORMAT = "v5-hub"
DEFAULT_ENV_NAME = "browsecraft-spatial-rl"
_HUD_MCP_URL = "https://api.hud.ai/v3/mcp/"


def _parse_tiers(raw: str | None) -> list[Tier]:
    if raw is None or not raw.strip():
        return list(ALL_TIERS)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in requested if item not in ALL_TIERS]
    if invalid:
        raise ValueError(f"unsupported tiers: {', '.join(invalid)}")
    return requested  # type: ignore[return-value]


def _task_record(
    env_name: str,
    task_payload: dict[str, object],
    reward_config: dict[str, object],
) -> dict[str, object]:
    return {
        "env": {"name": env_name},
        "scenario": f"{SCENARIO_ENV_NAME}:{task_payload['tier']}",
        "args": {"task_spec": task_payload, "reward_config": reward_config},
    }


def _local_eval_task_record(
    *,
    server_name: str,
    image: str,
    task_payload: dict[str, object],
    reward_config: dict[str, object],
) -> dict[str, object]:
    return {
        "id": task_payload["task_id"],
        "prompt": task_payload["prompt"],
        "mcp_config": {
            server_name: {
                "command": "docker",
                "args": ["run", "--rm", "-i", image],
            }
        },
        "setup_tool": {
            "name": "rl_setup_task",
            "arguments": {
                "task_spec": task_payload,
                "reward_config": reward_config,
            },
        },
        "evaluate_tool": {
            "name": "rl_grade_task",
            "arguments": {},
        },
        "agent_config": {
            "system_prompt": AGENT_SYSTEM_PROMPT,
            "allowed_tools": ALLOWED_AGENT_TOOLS,
        },
        "metadata": {
            "task_id": task_payload["task_id"],
            "tier": task_payload["tier"],
            "family": task_payload["family"],
            "seed": task_payload["seed"],
        },
    }


def _resolve_local_image(*, output: Path, explicit_image: str | None) -> str:
    if explicit_image:
        return explicit_image

    lock_path = output.parent / "hud.lock.yaml"
    if not lock_path.exists():
        raise FileNotFoundError(
            f"hud.lock.yaml not found at {lock_path}. Run 'hud build' first or pass --image."
        )

    lock_data = load_lock(lock_path)
    image = get_local_image(lock_data)
    if not image:
        raise ValueError("hud.lock.yaml does not contain a local image reference")
    return image


def _linked_deploy_registry_id(root: Path) -> str | None:
    deploy_path = root / ".hud" / "deploy.json"
    if not deploy_path.exists():
        return None
    payload = json.loads(deploy_path.read_text(encoding="utf-8"))
    registry_id = payload.get("registryId")
    if not isinstance(registry_id, str) or not registry_id:
        raise ValueError(f"invalid registryId in {deploy_path}")
    return registry_id


async def _fetch_environment_name(environment_id: str) -> str:
    api_key = os.environ.get("HUD_API_KEY")
    if not api_key:
        raise ValueError("HUD_API_KEY must be set to resolve the linked remote environment name")

    headers = {"Authorization": f"Bearer {api_key}"}
    async with streamablehttp_client(_HUD_MCP_URL, headers=headers) as streams:
        read_stream, write_stream, _ = streams
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("get_environment", {"environment_id": environment_id})
            texts = [block.text for block in result.content if hasattr(block, "text")]
            if len(texts) != 1:
                raise ValueError("expected exactly one text block from get_environment")
            payload = json.loads(texts[0])
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"environment {environment_id} is missing a name")
            return name


def _resolve_remote_env_name(*, requested_name: str, root: Path) -> str:
    registry_id = _linked_deploy_registry_id(root)
    if registry_id is None:
        return requested_name
    if requested_name != DEFAULT_ENV_NAME:
        return requested_name
    return anyio.run(_fetch_environment_name, registry_id)


def run(
    *,
    seed: int,
    per_tier: int,
    tiers: list[Tier],
    env_name: str,
    output: Path,
    reward_config: dict[str, object],
    task_format: str,
    image: str | None,
) -> None:
    tasks = generate_tasks(seed=seed, per_tier=per_tier, tiers=tiers)
    root = Path(__file__).resolve().parents[1]
    resolved_image = _resolve_local_image(output=output, explicit_image=image) if task_format == LOCAL_EVAL_FORMAT else None
    resolved_env_name = (
        _resolve_remote_env_name(requested_name=env_name, root=root)
        if task_format == V5_HUB_FORMAT
        else env_name
    )
    lines = []
    for task in tasks:
        payload = task.model_dump(mode="json")
        if task_format == LOCAL_EVAL_FORMAT:
            record = _local_eval_task_record(
                server_name=env_name,
                image=resolved_image,
                task_payload=payload,
                reward_config=reward_config,
            )
        else:
            record = _task_record(env_name=resolved_env_name, task_payload=payload, reward_config=reward_config)
        lines.append(json.dumps(record))
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    summary = {
        "output": str(output),
        "seed": seed,
        "per_tier": per_tier,
        "tier_counts": tier_counts(tasks),
        "total": len(tasks),
        "reward_config": reward_config,
        "task_format": task_format,
        "env_name": resolved_env_name,
    }
    if resolved_image is not None:
        summary["image"] = resolved_image
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic HUD task JSONL from BrowseCraft tiers.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-tier", type=int, default=100)
    parser.add_argument("--tiers", default=None, help="Comma-separated list of tiers. Default: all tiers.")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--output", default="remote_tasks.jsonl")
    parser.add_argument("--format", choices=(V5_HUB_FORMAT, LOCAL_EVAL_FORMAT), default=V5_HUB_FORMAT)
    parser.add_argument("--image", default=None, help="Docker image for local-eval tasks.")
    parser.add_argument("--reward-config-file", default=None)
    parser.add_argument("--format-mode", choices=("gate", "weighted"), default=None)
    parser.add_argument("--weight-correctness", type=float, default=None)
    parser.add_argument("--weight-efficiency", type=float, default=None)
    parser.add_argument("--weight-structural", type=float, default=None)
    parser.add_argument("--weight-format", type=float, default=None)
    parser.add_argument("--efficiency-min-correctness", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tiers = _parse_tiers(args.tiers)
    overrides = {
        key: value
        for key, value in {
            "format_mode": args.format_mode,
            "weight_correctness": args.weight_correctness,
            "weight_efficiency": args.weight_efficiency,
            "weight_structural": args.weight_structural,
            "weight_format": args.weight_format,
            "efficiency_min_correctness": args.efficiency_min_correctness,
        }.items()
        if value is not None
    }
    reward_config = load_reward_config(path=args.reward_config_file, overrides=overrides).model_dump(mode="json")
    run(
        seed=args.seed,
        per_tier=args.per_tier,
        tiers=tiers,
        env_name=str(args.env_name),
        output=Path(args.output),
        reward_config=reward_config,
        task_format=str(args.format),
        image=args.image,
    )


if __name__ == "__main__":
    main()
