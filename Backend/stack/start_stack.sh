#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_ORIGIN="${1:-http://localhost:3000}"

mkdir -p "$BACKEND_DIR/stack/.runtime" "$BACKEND_DIR/stack/.logs"

is_port_in_use() {
  local port="$1"
  ss -ltn | awk '{print $4}' | grep -q ":${port}$"
}

cd "$BACKEND_DIR/Inference"
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env

if is_port_in_use 8000; then
  echo "Inference service already bound to port 8000"
elif [[ -f "$BACKEND_DIR/stack/.runtime/inference.pid" ]] && kill -0 "$(cat "$BACKEND_DIR/stack/.runtime/inference.pid")" 2>/dev/null; then
  echo "Inference service already running"
else
  nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > "$BACKEND_DIR/stack/.logs/inference.log" 2>&1 &
  echo $! > "$BACKEND_DIR/stack/.runtime/inference.pid"
fi

cd "$BACKEND_DIR"
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

export CLASSIFIER_API_URL="http://127.0.0.1:8000"
export ALLOWED_ORIGINS="$FRONTEND_ORIGIN"
export LLM_STUB_MODE="true"

if is_port_in_use 8080; then
  echo "Backend API already bound to port 8080"
elif [[ -f "$BACKEND_DIR/stack/.runtime/backend.pid" ]] && kill -0 "$(cat "$BACKEND_DIR/stack/.runtime/backend.pid")" 2>/dev/null; then
  echo "Backend API already running"
else
  nohup uvicorn main:app --host 127.0.0.1 --port 8080 > "$BACKEND_DIR/stack/.logs/backend.log" 2>&1 &
  echo $! > "$BACKEND_DIR/stack/.runtime/backend.pid"
fi

echo "Started stack:"
echo "  Inference: http://127.0.0.1:8000"
echo "  Backend:   http://127.0.0.1:8080"
