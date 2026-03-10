from __future__ import annotations

import argparse
import asyncio
import os

from browser_use_sdk import AsyncBrowserUse


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


async def _find_profile_by_name(client: AsyncBrowserUse, profile_name: str):
    page = 1
    while True:
        response = await client.profiles.list(page_size=100, page_number=page)
        for item in response.items:
            if item.name and item.name.casefold() == profile_name.casefold():
                return item
        if len(response.items) < response.page_size:
            return None
        page += 1


async def _run(args: argparse.Namespace) -> None:
    api_key = _required_env("BROWSER_USE_API_KEY")
    client = AsyncBrowserUse(api_key=api_key)
    try:
        profile = await _find_profile_by_name(client, args.name) if args.reuse_existing else None
        if profile is None:
            profile = await client.profiles.create(name=args.name)
    finally:
        await client.close()

    profile_id = str(profile.id)
    print(f"Profile ready: {profile_id}")
    print(f"BROWSER_USE_PROFILE_ID={profile_id}")
    print("Sync local cookies/profile state with:")
    print(
        "curl -fsSL https://browser-use.com/profile.sh | "
        f"BROWSER_USE_API_KEY=... PROFILE_ID={profile_id} sh"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or reuse a Browser Use profile for BrowseCraft.")
    parser.add_argument("--name", default="BrowseCraft PlanetMinecraft")
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
