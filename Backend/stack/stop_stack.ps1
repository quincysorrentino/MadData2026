$ErrorActionPreference = "Stop"

$backendDirWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendDirWsl = (wsl -e wslpath -a "$backendDirWin").Trim()

$commandTemplate = @'
set +e

if [ -f '{0}/stack/.runtime/backend.pid' ]; then
  kill $(cat '{0}/stack/.runtime/backend.pid') 2>/dev/null
  rm -f '{0}/stack/.runtime/backend.pid'
fi

if [ -f '{0}/stack/.runtime/inference.pid' ]; then
  kill $(cat '{0}/stack/.runtime/inference.pid') 2>/dev/null
  rm -f '{0}/stack/.runtime/inference.pid'
fi

pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8000" 2>/dev/null || true
pkill -f "uvicorn main:app --host 127.0.0.1 --port 8080" 2>/dev/null || true

echo 'Stopped backend stack (if running).'
'@

$command = [string]::Format($commandTemplate, $backendDirWsl)
$command = $command.Replace("`r", "")

wsl -e bash -lc "$command"
