param(
    [string]$FrontendOrigin = "http://localhost:3000"
)

$ErrorActionPreference = "Stop"

$backendDirWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendDirWsl = (wsl -e wslpath -a "$backendDirWin").Trim()

$commandTemplate = @'
set -e
mkdir -p '{0}/stack/.runtime' '{0}/stack/.logs'

is_port_in_use() {
  local port="$1"
  ss -ltn | awk '{print $4}' | grep -q ":${port}$"
}

cd '{0}/Inference'
python3 -m venv .venv || true
. .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env

if is_port_in_use 8000; then
  echo 'Inference service already bound to port 8000'
elif [ -f '{0}/stack/.runtime/inference.pid' ] && kill -0 $(cat '{0}/stack/.runtime/inference.pid') 2>/dev/null; then
  echo 'Inference service already running'
else
  nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > '{0}/stack/.logs/inference.log' 2>&1 &
  echo $! > '{0}/stack/.runtime/inference.pid'
fi

cd '{0}'
python3 -m venv .venv || true
. .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

export CLASSIFIER_API_URL='http://127.0.0.1:8000'
export ALLOWED_ORIGINS='{1}'
export LLM_STUB_MODE='true'

if is_port_in_use 8080; then
  echo 'Backend API already bound to port 8080'
elif [ -f '{0}/stack/.runtime/backend.pid' ] && kill -0 $(cat '{0}/stack/.runtime/backend.pid') 2>/dev/null; then
  echo 'Backend API already running'
else
  nohup uvicorn main:app --host 127.0.0.1 --port 8080 > '{0}/stack/.logs/backend.log' 2>&1 &
  echo $! > '{0}/stack/.runtime/backend.pid'
fi

echo 'Started stack:'
echo '  Inference: http://127.0.0.1:8000'
echo '  Backend:   http://127.0.0.1:8080'
'@

$command = [string]::Format($commandTemplate, $backendDirWsl, $FrontendOrigin)
$command = $command.Replace("`r", "")

wsl -e bash -lc "$command"
