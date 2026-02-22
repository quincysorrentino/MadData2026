param(
    [string]$RepoRoot = "C:\Users\Hackathon User\MadData2026",
    [string]$ModelName = "qwen2.5:3b"
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "[start-all] $Message" -ForegroundColor Cyan
}

function Resolve-RequiredPath([string]$Path, [string]$Label) {
    if (-not (Test-Path $Path)) {
        throw "$Label not found: $Path"
    }
    return (Resolve-Path $Path).Path
}

function Start-ServiceWindow([string]$Title, [string]$Command) {
    Start-Process powershell -ArgumentList @(
        "-NoExit",
        "-Command",
        "$Host.UI.RawUI.WindowTitle = '$Title'; $Command"
    ) | Out-Null
}

$ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$inferencePython = Join-Path $RepoRoot "Backend\Inference\.venv-qnn\Scripts\python.exe"
$frontendDir = Join-Path $RepoRoot "Frontend"
$inferenceDir = Join-Path $RepoRoot "Backend\Inference"
$backendDir = Join-Path $RepoRoot "Backend\src"
$localLlmDir = Join-Path $RepoRoot "Backend\Local_llm"

Write-Step "Validating required paths"
$RepoRoot = Resolve-RequiredPath $RepoRoot "Repository root"
$inferencePython = Resolve-RequiredPath $inferencePython "Inference Python"
$frontendDir = Resolve-RequiredPath $frontendDir "Frontend directory"
$inferenceDir = Resolve-RequiredPath $inferenceDir "Inference directory"
$backendDir = Resolve-RequiredPath $backendDir "Backend/src directory"
$localLlmDir = Resolve-RequiredPath $localLlmDir "Local_llm directory"

if (-not (Test-Path $ollamaExe)) {
    throw "Ollama not installed at expected path: $ollamaExe`nInstall with: winget install -e --id Ollama.Ollama --source winget"
}

Write-Step "Checking Ollama listener on 127.0.0.1:11434"
$ollamaListening = $false
try {
    $conn = Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort 11434 -State Listen -ErrorAction Stop
    if ($conn) {
        $ollamaListening = $true
    }
} catch {
    $ollamaListening = $false
}

if (-not $ollamaListening) {
    Write-Step "Starting Ollama"
        Start-ServiceWindow "Ollama" "`$env:OLLAMA_NUM_PARALLEL = '1'; `$env:OLLAMA_MAX_LOADED_MODELS = '1'; `$env:OLLAMA_KEEP_ALIVE = '5m'; `$env:OLLAMA_HOST = '127.0.0.1:11434'; & '$ollamaExe' serve"
    Start-Sleep -Seconds 2
} else {
    Write-Step "Ollama already running on 127.0.0.1:11434 (reusing existing daemon)"
}

Write-Step "Ensuring model '$ModelName' is available in Ollama"
try {
    & $ollamaExe pull $ModelName
} catch {
    Write-Warning "Model pull failed. You can retry manually: `"$ollamaExe`" pull $ModelName"
}

Write-Step "Starting Local LLM API (9000)"
Start-ServiceWindow "Local LLM :9000" "Set-Location '$localLlmDir'; `$env:MODEL_NAME = '$ModelName'; `$env:OLLAMA_TIMEOUT_SECONDS = '90'; `$env:OLLAMA_MAX_TOKENS = '512'; `$env:OLLAMA_NUM_CTX = '1024'; `$env:OLLAMA_NUM_THREAD = '4'; `$env:OLLAMA_CONCURRENCY = '1'; & '$inferencePython' main.py"

Write-Step "Starting Inference API (8000)"
Start-ServiceWindow "Inference API :8000" "Set-Location '$inferenceDir'; & '$inferencePython' -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

Write-Step "Starting Orchestrator API (8080)"
Start-ServiceWindow "Backend API :8080" "Set-Location '$backendDir'; `$env:LLM_MODEL = '$ModelName'; `$env:LLM_TIMEOUT_SECONDS = '90'; `$env:LLM_MAX_TOKENS = '512'; `$env:LLM_CONNECT_TIMEOUT_SECONDS = '5'; `$env:LLM_POOL_TIMEOUT_SECONDS = '5'; `$env:LLM_MAX_CONCURRENCY = '1'; & '$inferencePython' -m uvicorn main:app --host 127.0.0.1 --port 8080"

Write-Step "Starting Frontend (3000)"
Start-ServiceWindow "Frontend :3000" "Set-Location '$frontendDir'; `$env:Path = 'C:\Program Files\nodejs;' + `$env:Path; npm.cmd run dev -- --host 127.0.0.1 --port 3000"

Write-Step "All launch commands dispatched"
Write-Host "Frontend: http://127.0.0.1:3000" -ForegroundColor Green