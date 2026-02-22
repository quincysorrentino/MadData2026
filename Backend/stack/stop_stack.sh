#!/usr/bin/env bash
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$BACKEND_DIR/stack/.runtime/backend.pid" ]]; then
  kill "$(cat "$BACKEND_DIR/stack/.runtime/backend.pid")" 2>/dev/null
  rm -f "$BACKEND_DIR/stack/.runtime/backend.pid"
fi

if [[ -f "$BACKEND_DIR/stack/.runtime/inference.pid" ]]; then
  kill "$(cat "$BACKEND_DIR/stack/.runtime/inference.pid")" 2>/dev/null
  rm -f "$BACKEND_DIR/stack/.runtime/inference.pid"
fi

pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8000" 2>/dev/null || true
pkill -f "uvicorn main:app --host 127.0.0.1 --port 8080" 2>/dev/null || true

echo "Stopped backend stack (if running)."
