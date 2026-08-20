#!/usr/bin/env bash
# BatteryIQ — start backend (FastAPI) and frontend (React) together.
# Usage: ./run.sh          (from the BatteryIQ repo root)
#        Ctrl+C stops both.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/app/backend"
FRONTEND_DIR="$ROOT_DIR/app/frontend"

# Make sure both child processes are killed when this script exits
# (Ctrl+C, error, or normal exit).
cleanup() {
  echo ""
  echo "Stopping BatteryIQ..."
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI backend on :8000 ..."
(cd "$BACKEND_DIR" && python -m uvicorn main:app --reload --port 8000) &
BACKEND_PID=$!

echo "Starting React frontend on :3000 ..."
(cd "$FRONTEND_DIR" && npm start) &
FRONTEND_PID=$!

echo ""
echo "BatteryIQ is starting:"
echo "  Backend:  http://localhost:8000  (docs at /docs)"
echo "  Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop both."
echo ""

wait
