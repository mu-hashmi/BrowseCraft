from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Any

from browser_use_sdk import AsyncBrowserUse


_SKILL_TITLE = "BrowseCraft PlanetMinecraft Schematic Finder"
_SKILL_DESCRIPTION = "Find downloadable Minecraft Java schematic files on Planet Minecraft."
_SKILL_GOAL = (
    "Search Planet Minecraft for downloadable Minecraft Java schematic files and "
    "return only candidates that have direct downloadable .schem/.litematic/.schematic files."
)
_AGENT_PROMPT = """Use Planet Minecraft only.

Input parameters:
- query: search query string
- mc_version: Minecraft version string
- allowed_exts: list of allowed extensions (for example [".schem", ".litematic", ".schematic"])

Workflow:
1. Search Planet Minecraft with query and mc_version.
2. Open project pages and follow the Schematic download path.
3. Skip world save, map download, and zip-only projects.
4. Return candidates backed by downloadable files whose extension is in allowed_exts.

Output:
Return JSON object:
{
  "candidates": [
    {
      "canonical_url": "<project_url>",
      "filename": "<download_filename>",
      "title": "<project_title>",
      "score": <float between 0 and 1>,
      "download_url": "<direct_download_url>"
    }
  ]
}
"""


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


async def _find_skill_by_title(client: AsyncBrowserUse, title: str) -> Any | None:
    page = 1
    while True:
        response = await client.skills.list(page_size=100, page_number=page, query=title)
        for item in response.items:
            if item.title.casefold() == title.casefold():
                return item
        if len(response.items) < response.page_size:
            return None
        page += 1


async def _wait_until_ready(client: AsyncBrowserUse, skill_id: str, wait_seconds: int) -> Any:
    deadline = time.monotonic() + wait_seconds
    terminal = {"finished", "failed", "cancelled", "timed_out"}
    while True:
        skill = await client.skills.get(skill_id)
        status = skill.status.value if hasattr(skill.status, "value") else str(skill.status)
        if status in terminal:
            return skill
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Skill {skill_id} did not finish within {wait_seconds} seconds")
        await asyncio.sleep(2)


async def _run(args: argparse.Namespace) -> None:
    api_key = _required_env("BROWSER_USE_API_KEY")
    client = AsyncBrowserUse(api_key=api_key)
    try:
        existing = await _find_skill_by_title(client, args.title) if args.reuse_existing else None
        if existing is not None:
            skill_id = str(existing.id)
            await client.skills.update(
                skill_id,
                title=args.title,
                description=args.description,
                domains=["planetminecraft.com", "www.planetminecraft.com"],
                categories=["search"],
                is_enabled=True,
            )
        else:
            created = await client.skills.create(
                title=args.title,
                description=args.description,
                goal=_SKILL_GOAL,
                agent_prompt=_AGENT_PROMPT,
            )
            skill_id = str(created.id)
            await client.skills.update(
                skill_id,
                domains=["planetminecraft.com", "www.planetminecraft.com"],
                categories=["search"],
                is_enabled=True,
            )

        skill = await _wait_until_ready(client, skill_id, args.wait_seconds)
    finally:
        await client.close()

    status = skill.status.value if hasattr(skill.status, "value") else str(skill.status)
    if status != "finished":
        raise RuntimeError(f"Skill {skill_id} finished with status={status}")

    print(f"Skill ready: {skill_id}")
    print(f"BROWSER_USE_PLANET_MINECRAFT_SKILL_ID={skill_id}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or update the Planet Minecraft Browser Use skill.")
    parser.add_argument("--title", default=_SKILL_TITLE)
    parser.add_argument("--description", default=_SKILL_DESCRIPTION)
    parser.add_argument("--wait-seconds", type=int, default=240)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
