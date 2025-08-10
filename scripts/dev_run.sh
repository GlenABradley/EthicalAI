#!/usr/bin/env bash
set -euo pipefail

# Simple dev runner for the API
export PYTHONPATH=src:${PYTHONPATH:-}
exec uvicorn coherence.api.main:app --reload --host 0.0.0.0 --port 8080
