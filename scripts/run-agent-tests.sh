#!/usr/bin/env bash
set -euo pipefail
cd ~/BrowseCraft/backend && uv run pytest -q
cd ~/BrowseCraft/mod && gradle test
