#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing virtual environment at .venv. Create it first:" >&2
  echo "  python3 -m venv .venv" >&2
  exit 1
fi

source .venv/bin/activate

if ! command -v qai-hub >/dev/null 2>&1; then
  echo "qai-hub CLI not found. Install with:" >&2
  echo "  pip install 'qai-hub[torch]'" >&2
  exit 1
fi

api_token="${QAIHUB_API_TOKEN:-}"
if [[ -z "$api_token" ]]; then
  read -r -s -p "Enter QAIHUB_API_TOKEN: " api_token
  echo
fi

if [[ -z "$api_token" ]]; then
  echo "No API token provided. Aborting." >&2
  exit 1
fi

qai-hub configure --api_token "$api_token"

python - <<'PY'
import qai_hub as hub

devices = hub.get_devices(limit=1)
print(f"AI Hub authentication OK. Found {len(devices)} device result(s).")
PY

echo "Authentication configured successfully."
