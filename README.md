# MadData2026

Hackathon project scaffold for offline medical triage on Snapdragon devices.

## Backend inference quickstart

See [Backend/Inference/Inference_README.md](Backend/Inference/Inference_README.md) for the full Qualcomm + ONNX pipeline.

```bash
cd Backend/Inference

# Debian/Ubuntu only: install venv support if missing
sudo apt update && sudo apt install -y python3-venv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `POST /v1/infer` (multipart form field: `file`)
- `POST /infer` (alias)