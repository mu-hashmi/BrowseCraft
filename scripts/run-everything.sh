#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
MOD_DIR="$ROOT_DIR/mod"
LOG_DIR="$ROOT_DIR/.logs"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}/health"
BACKEND_LOG="$LOG_DIR/backend-run.log"

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID"
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

require_command uv
require_command gradle
require_command curl
require_command lsof

if [[ ! -f "$BACKEND_DIR/.env" ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "Missing backend/.env and ANTHROPIC_API_KEY is not set in the shell." >&2
  echo "Create backend/.env from backend/.env.example or export ANTHROPIC_API_KEY first." >&2
  exit 1
fi

if lsof -iTCP:"$BACKEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $BACKEND_PORT is already in use. Stop the existing process or launch the client separately." >&2
  lsof -nP -iTCP:"$BACKEND_PORT" -sTCP:LISTEN >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
rm -f "$BACKEND_LOG"

echo "Starting backend on ${BACKEND_HOST}:${BACKEND_PORT}..."
(
  cd "$BACKEND_DIR"
  uv run browsecraft-backend
) >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

for _ in $(seq 1 120); do
  if curl -fsS "$BACKEND_URL" >/dev/null 2>&1; then
    echo "Backend is up."
    echo "Backend log: $BACKEND_LOG"
    echo "Launching Minecraft client..."
    cd "$MOD_DIR"
    exec gradle runClient
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend exited before becoming healthy. Last log lines:" >&2
    tail -n 40 "$BACKEND_LOG" >&2 || true
    exit 1
  fi
  sleep 0.25
done

echo "Backend did not become healthy within 30 seconds. Last log lines:" >&2
tail -n 40 "$BACKEND_LOG" >&2 || true
exit 1
