# Demo Runbook

## Goal

Record a backup end-to-end demo after API keys are configured.

## Preflight

1. Fill `backend/.env` with valid keys:
   - `ANTHROPIC_API_KEY`
   - `LAMINAR_API_KEY` (optional but recommended)
   - `CONVEX_URL`, `CONVEX_ACCESS_KEY` (optional)
   - `SUPERMEMORY_API_KEY` (optional)
2. Run test gates:
   - `cd ~/BrowseCraft/backend && uv run pytest -q`
   - `cd ~/BrowseCraft/mod && gradle test`
3. Start backend:
   - `cd ~/BrowseCraft/backend && uv run browsecraft-backend`
4. Build/install mod jar:
   - `cd ~/BrowseCraft/mod && gradle build`
   - Copy `mod/build/libs/browsecraft-0.1.0.jar` into Minecraft `mods/`.

## Suggested Demo Flow

1. `/build-test` fallback first.
2. `/chat build a small starter house near me`.
3. `/chat add windows and a centered doorway`.
4. `/chat replace oak with birch`.
5. `/materials`.
6. `/blueprints save demo-1`.
7. `/session new`, then `/chat what did we build?`.

## Recording Notes

- Run the full flow 2-3 times before final recording.
- Keep backend logs visible for credibility during judging.
- If the model is slow, continue with `/build-test` and narrate expected `/chat` behavior.
