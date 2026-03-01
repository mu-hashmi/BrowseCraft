# Instructions For Coding Agents Working With This Repo

This is a hackathon project. Production-readiness is unnecessary.

## Preferred Codex Test Workflow

For `/build` debugging, use a real end-to-end loop that is fully programmatic. Do not stop at unit tests or curl-only checks.

### Non-negotiables

- Do not use GUI automation.
- Do not rely on manual clicking/typing to validate `/build`.
- Validate Minecraft ingestion/rendering, not just backend acceptance.
- `/build` source is Browser Use only. Treat non-`browser_use` source as failure.

### Required env keys

Set in `backend/.env`:

| Key                                | Required for                     |
| ---------------------------------- | -------------------------------- |
| `BROWSER_USE_API_KEY`              | `/build` discovery/download      |
| `CONVEX_URL` + `CONVEX_ACCESS_KEY` | session persistence (if used)    |
| `ANTHROPIC_API_KEY`                | `/chat` and image-plan paths     |
| `GOOGLE_API_KEY`                   | `/imagine` image generation only |
| `SUPERMEMORY_API_KEY`              | optional chat memory persistence |

Optional: `LAMINAR_API_KEY`.

### 1) Fast checks before E2E

```bash
cd ~/BrowseCraft/backend && uv run pytest -q
cd ~/BrowseCraft/mod && gradle test
cd ~/BrowseCraft/mod && gradle build
```

### 2) Start backend cleanly on 8080

Use `tmux` and ensure port `8080` is not occupied by stale processes.

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN || true
cd ~/BrowseCraft/backend
tmux new-session -d -s browsecraft-backend 'uv run uvicorn browsecraft_backend.app:app --host 127.0.0.1 --port 8080 --log-level info 2>&1 | tee /tmp/browsecraft-backend.log'
curl -s http://127.0.0.1:8080/health
```

### 3) Run programmatic in-game E2E

Use client gametest (programmatic, no GUI input):

```bash
cd ~/BrowseCraft/mod && gradle deleteGameTestRunDir runClientGameTest
```

The gametest must submit a real build query through the same backend path used by `/build`, then capture artifacts after `job.ready`.

### 4) Verify required artifacts

```bash
cat ~/BrowseCraft/mod/build/run/clientGameTest/browsecraft/build-test/ghost-state.json | jq '{validation, backend, render_status, render_status_observed}'
LATEST_SHOT=$(find ~/BrowseCraft/mod/build/run/clientGameTest/screenshots -type f | sort | tail -1)
echo "$LATEST_SHOT"
ls -lh "$LATEST_SHOT"
file "$LATEST_SHOT"
```

Pass criteria:

- `validation.passed == true`
- `backend.latest_ready_source_type == "browser_use"`
- `render_status_observed == true`
- screenshot file exists, is non-empty, and visibly shows ghost wireframe outlines
- latest backend run has `POST /v1/jobs ... 200` (no `422`/`5xx`)
- IMPORTANT: the only REAL source of truth is if the screenshot visibly shows ghost wireframe outlines

### 5) Iterate

After each code change:

1. restart backend
2. rerun `runClientGameTest`
3. re-check `ghost-state.json`, screenshot, and backend log

## Manual Live Run (Optional)

If the user wants manual confirmation in a real client session:

```bash
cd ~/BrowseCraft/mod && gradle runClient
```

Then user runs `/build ...` manually while agent inspects backend and artifact logs from terminal.

## Common Failure Signals

- `build submit failed: ... 422 ... Field required`: malformed/empty POST body from mod client.
- `runOnClient called when no client is running`: client closed during gametest run.
- `render_status_observed=true` but empty screenshot: screenshot pipeline bug; treat as failure.
- `ClassNotFoundException: dev.browsecraft.mod.BrowseCraftClientGameTests`: wrong class/package location; keep it under `mod/src/client/java/dev/browsecraft/mod/`.
