#!/usr/bin/env bash
set -euo pipefail

export STT_HOST="${STT_HOST:-0.0.0.0}"
export STT_PORT="${STT_PORT:-8000}"
export CANARY_MODEL_NAME="${CANARY_MODEL_NAME:-nvidia/canary-1b-v2}"

python -m uvicorn server.canary_ws_server:app --host "$STT_HOST" --port "$STT_PORT"
