#!/bin/bash
# ─── TX Prediction Engine – Backend Launcher ────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/.python/bin/python3"

echo "🚀  Starting FastAPI backend on http://localhost:8000"
echo "📖  API Docs: http://localhost:8000/docs"
echo ""

cd "$SCRIPT_DIR"
exec "$PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
