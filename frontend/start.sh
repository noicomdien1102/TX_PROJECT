#!/bin/bash
# ─── TX Prediction Engine – Frontend Launcher ───────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/tmp/node-v22.14.0-darwin-arm64/bin:$PATH"

echo "🎨  Starting React + Vite frontend on http://localhost:5173"
echo ""

cd "$SCRIPT_DIR"
exec npm run dev
