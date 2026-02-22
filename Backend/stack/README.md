# Backend Stack Bundle

This folder is a **single backend bundle** for running both backend services independently of your frontend repo.

It starts:

- `Inference API` at `http://127.0.0.1:8000`
- `Backend API` at `http://127.0.0.1:8080`

The backend is preconfigured to point at the local inference service and runs with `LLM_STUB_MODE=true` so frontend integration works even without a live LLM server.

## Files in this bundle

- `start_stack.ps1` / `stop_stack.ps1` (Windows + WSL)
- `start_stack.sh` / `stop_stack.sh` (Linux/macOS/WSL)
- `frontend.env.example` (frontend API base URL)

Runtime output is written to:

- `stack/.logs/`
- `stack/.runtime/`

## Start (Windows)

From repo root:

```powershell
./Backend/stack/start_stack.ps1
```

Optional frontend origin for CORS:

```powershell
./Backend/stack/start_stack.ps1 -FrontendOrigin "http://localhost:5173"
```

## Stop (Windows)

```powershell
./Backend/stack/stop_stack.ps1
```

## Start (Linux/macOS/WSL)

```bash
bash Backend/stack/start_stack.sh
```

Optional frontend origin for CORS:

```bash
bash Backend/stack/start_stack.sh http://localhost:5173
```

## Stop (Linux/macOS/WSL)

```bash
bash Backend/stack/stop_stack.sh
```

## Frontend integration

In your frontend project, set:

```env
VITE_API_BASE_URL=http://127.0.0.1:8080/api
```

Then use:

- `POST /body-part`
- `POST /diagnose`
- `POST /chat`

Base URL example:

`http://127.0.0.1:8080/api/diagnose`
