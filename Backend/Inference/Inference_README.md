# Backend Inference (Snapdragon Native-First)

This guide is intentionally **native-first** for Snapdragon devices (simpler debugging than containers).

## Windows Snapdragon X Elite (local NPU)

If you are running on a Windows laptop with Snapdragon X Elite, use this local flow.

Important distinction:

- `ai_hub_workbench.py` uses **cloud-hosted** Qualcomm devices (AI Hub Workbench).
- Local laptop NPU requires ONNX Runtime with QNN execution provider plus an ONNX model.

### 1) Create venv and install dependencies

```powershell
cd Backend/Inference
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install torch onnx
```

Install a QNN-enabled ONNX Runtime build for Windows ARM64 (for example `onnxruntime-qnn`, or vendor-provided wheel).

Run this in the same Windows venv to verify providers:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You must see `QNNExecutionProvider` in the output for local NPU execution.

### 2) Download models (automatic defaults)

```powershell
python download_hf_models.py --out-dir models
```

### 3) Export classifier snapshot to ONNX and update `.env`

```powershell
python export_hf_to_onnx.py --require-npu
```

This updates `.env` to use:

- `MODEL_PATH=models/classifier.onnx`
- `MODEL_HF_DIR=`
- `REQUIRE_EXTERNAL_MODEL=true`
- `REQUIRE_NPU=true`
- `ORT_ENABLE_CPU_FALLBACK=false`

### 4) Start API and verify provider

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On startup, verify log contains:

```text
active=['QNNExecutionProvider', ...]
```

If startup fails with provider error, your ONNX Runtime build does not expose QNN EP.

Note: This must be executed from native Windows Python on your Snapdragon X Elite machine, not from WSL.

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

`download_hf_models.py` now automatically downloads:

- Segmenter: `Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation`
- Classifier: `Anwarkh1/Skin_Cancer-Image_Classification`

It also updates `Backend/Inference/.env` automatically:

- `SEGMENTER_MODEL_PATH` (to downloaded segmenter weights)
- `MODEL_HF_DIR` (to downloaded classifier snapshot, when no `.onnx` is present)
- or `MODEL_PATH` (if classifier repo contains `.onnx`)

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

## 4.1) Enable NPU (QNN Execution Provider)

NPU execution is used only when you run an **ONNX model** (`MODEL_PATH=...onnx`) and ONNX Runtime can load Qualcomm QNN EP.

If `MODEL_PATH` is empty, the service uses a placeholder classifier (`placeholder-v1`) that is CPU-only and will never use NPU.

1. In `Backend/Inference/.env`, set:

```dotenv
MODEL_PATH=/absolute/path/to/your/model.onnx
ORT_EXECUTION_PROVIDERS=QNNExecutionProvider,CPUExecutionProvider
ORT_ENABLE_CPU_FALLBACK=true
REQUIRE_NPU=true
```

2. If your environment requires an explicit backend library path, also set:

```dotenv
QNN_BACKEND_PATH=/path/to/libQnnHtp.so
```

3. Start the API and check startup logs. You should see a line like:

```text
[Inference] ONNX Runtime provider selection: requested=[...], available=[...], active=['QNNExecutionProvider', ...]
```

If `active` starts with `CPUExecutionProvider`, then NPU is not active and the runtime fell back to CPU.

4. To fail fast when NPU is not available (no CPU fallback):

```dotenv
ORT_ENABLE_CPU_FALLBACK=false
```

The service will error on startup if QNN cannot be selected.

5. To fail fast when no external model is configured:

```dotenv
REQUIRE_EXTERNAL_MODEL=true
```

This prevents accidental fallback to the placeholder CPU model.

## 4.2) AI Hub Workbench flow (recommended for NPU validation)

To validate NPU usage the same way as Qualcomm's Workbench guide (compile → inference → profile):

1. Install Workbench-specific dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.workbench.txt
```

2. Ensure your Workbench account token is set:

```bash
export QAIHUB_API_TOKEN=<your_token>
```

Or run the helper script (prompts for token securely if env var is not set):

```bash
cd Backend/Inference
bash setup_qaihub_auth.sh
```

Equivalent manual configure command:

```bash
qai-hub configure --api_token <your_token>
```

3. Run the workflow script:

```bash
python ai_hub_workbench.py \
	--device "Samsung Galaxy S25 (Family)" \
	--runtime tflite
```

This script now follows the full Workbench path:

- Load classifier from `MODEL_HF_DIR` (defaults to `models/Anwarkh1__Skin_Cancer-Image_Classification`)
- Trace model with TorchScript
- Compile for target runtime (e.g., TFLite)
- Submit on-device inference job
- Submit on-device profile job

Use the profile job URL to confirm the active compute unit is NPU.

Optional quantization flow:

```bash
python ai_hub_workbench.py \
	--device "Samsung Galaxy S25 (Family)" \
	--runtime tflite \
	--quantize \
	--calibration-dir ./imagenette_samples/images \
	--calibration-samples 100
```

That path compiles to ONNX, quantizes (INT8), recompiles to target runtime, then runs inference + profile again.

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
