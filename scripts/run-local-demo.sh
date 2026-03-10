#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
MOD_DIR="$ROOT_DIR/mod"
BACKEND_URL="http://127.0.0.1:8080/health"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID"
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

cd "$BACKEND_DIR"
uv run browsecraft-backend &
BACKEND_PID=$!

until curl -fsS "$BACKEND_URL" >/dev/null; do
  sleep 0.25
done

echo "Backend is up at http://127.0.0.1:8080"
echo "Launching Minecraft client..."

cd "$MOD_DIR"
gradle runClient
