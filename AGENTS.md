# Instructions For Coding Agents Working With This Repo

This is a hackathon project. Production-readiness is unnecessary.

## Run Tests

```bash
cd ~/BrowseCraft/backend && uv run pytest -q
cd ~/BrowseCraft/mod && gradle test
cd ~/BrowseCraft/mod && gradle build
```

## Verify Outputs

The closest thing to a real user test is the in-game `/build-test` command. If you can run `gradle runClientGametest`, check:

```bash
cat ~/BrowseCraft/mod/build/run/clientGameTest/browsecraft/build-test/ghost-state.json | jq '{validation, render_status, render_status_observed}'
find ~/BrowseCraft/mod/build/run/clientGameTest/screenshots -type f | sort | tail -1
```

The screenshot must show ghost block wireframe outlines in front of the player.

## Required Env Vars

Backend reads env vars from `backend/.env`:

- `BROWSER_USE_API_KEY`
- `GITHUB_TOKEN`
- `CURSEFORGE_API_KEY`
- `BROWSER_USE_PRIMARY_LLM`
- `BROWSER_USE_FALLBACK_LLM`
- `BROWSER_USE_PLANET_MINECRAFT_SKILL_ID`
- `BROWSER_USE_PROFILE_ID`
- `BROWSER_USE_TASK_TIMEOUT_SECONDS`
- `MAX_PLAN_BLOCKS`
- `APP_HOST`
- `APP_PORT`
- `LAMINAR_API_KEY`
- `CONVEX_URL`
- `CONVEX_ACCESS_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_CHAT_MODEL`
- `ANTHROPIC_CHAT_ESCALATION_MODEL`
- `ANTHROPIC_VISION_MODEL`
- `SUPERMEMORY_API_KEY`
- `IMAGINE_USE_GEMINI_TEXT_PLAN`
