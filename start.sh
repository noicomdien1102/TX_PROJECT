#!/bin/bash
# ─── TX Prediction Engine – Full Stack Launcher ─────────────────────────────
# Starts both backend and frontend in the background, shows both logs.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════╗"
echo "║   TX Prediction Engine  –  Full Stack    ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Backend  →  http://localhost:8000       ║"
echo "║  Frontend →  http://localhost:5173       ║"
echo "║  API Docs →  http://localhost:8000/docs  ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Kill both on Ctrl-C
cleanup() {
  echo ""
  echo "⏹  Stopping all servers..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Backend ─────────────────────────────────────────────────────────────────
PY="$REPO/backend/.python/bin/python3"
cd "$REPO/backend"
"$PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | sed 's/^/[API] /' &
BACKEND_PID=$!

# ── Frontend ────────────────────────────────────────────────────────────────
export PATH="/tmp/node-v22.14.0-darwin-arm64/bin:$PATH"
cd "$REPO/frontend"
npm run dev 2>&1 | sed 's/^/[WEB] /' &
FRONTEND_PID=$!

echo "✅  Both servers started. Press Ctrl-C to stop."
wait
