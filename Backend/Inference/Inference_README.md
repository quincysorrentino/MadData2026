# Backend Inference (Snapdragon Native-First)

This guide is intentionally **native-first** for Snapdragon devices (simpler debugging than containers).

## 1) Prerequisites (on Snapdragon device)

- OS: Linux (Ubuntu/Debian recommended)
- Python: `3.10+`
- System packages: `python3-venv`, `python3-pip`

Install system deps:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

## 2) Clone and set up environment

```bash
git clone https://github.com/quincysorrentino/MadData2026.git
cd MadData2026/Backend/Inference

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 3) Download models (native)

From inside `Backend/Inference`:

```bash
source .venv/bin/activate
python download_hf_models.py
```

Expected output location:

- `Backend/Inference/models/classifier.onnx`
- (optional) segmentation model files if enabled in your config

## 4) Start API locally

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -sS http://127.0.0.1:8000/health
```

Test inference:

```bash
curl -sS -X POST "http://127.0.0.1:8000/infer" -F "file=@test.jpg"
```

## 5) Replicate on any other Snapdragon device

Use this exact sequence on each device:

1. Install Python + `venv` system packages.
2. Clone same repo branch/tag.
3. Create venv and install `requirements.txt`.
4. Copy `.env.example` to `.env` and keep settings consistent.
5. Run `python download_hf_models.py` on-device.
6. Start with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
7. Validate with `/health` then `/infer`.

### Fast replication option (recommended)

After a known-good setup on one Snapdragon device:

- Save package list:

```bash
source .venv/bin/activate
pip freeze > requirements.lock.txt
```

- On the next device, install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock.txt
```

This helps keep behavior consistent across Snapdragon machines.

## 6) Troubleshooting (native)

- `ModuleNotFoundError`: confirm venv is active and dependencies are installed.
- Model not found errors: re-run `python download_hf_models.py` and verify `models/` contents.
- Slow startup on first run: expected while model files load.
- Port conflict on `8000`: use another port, e.g. `--port 8001`.
