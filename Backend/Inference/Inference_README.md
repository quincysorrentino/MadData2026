# Inference Backend (Hackathon Starter)

This backend now supports a two-stage pipeline:
1) optional lesion segmentation to get bounding boxes
2) classification on the lesion crop (or full image if no box)

> ⚠️ This starter is for hackathon prototyping only, not medical diagnosis.

## 1) Setup

```bash
cd Backend/Inference
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 2) Run API

```bash
uvicorn app.main:app --reload --port 8000
```

Endpoints:
- `GET /health`
- `POST /v1/infer`
- `POST /infer` (alias)
- `POST /v1/infer/debug` (includes fusion decision diagnostics)

Request: multipart field name `file`.

## 3) Response format (with bounding boxes)

```json
{
  "label": "melanoma",
  "confidence": 0.91,
  "boxes": [
    {
      "x_min": 42,
      "y_min": 55,
      "width": 131,
      "height": 146,
      "score": 0.67,
      "label": "lesion"
    }
  ],
  "model_version": "external:classifier.onnx+segmenter:segmenter.onnx"
}
```

## 4) Use your pulled models

You can download from Hugging Face with:

```bash
python download_hf_models.py \
  --segmenter Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation \
  --classifier <your-classifier-repo-id>
```

If your classifier is not decided yet, run only segmenter:

```bash
python download_hf_models.py
```

Set `.env` values:

```bash
# Classifier (multi-class)
MODEL_PATH=/absolute/path/to/classifier.onnx
MODEL_HF_DIR=
MODEL_LABELS=melanoma,basal_cell_carcinoma,squamous_cell_carcinoma,nevus,keratosis,dermatofibroma,vascular_lesion
MODEL_INPUT_SIZE=224

# Segmenter for lesion localization
SEGMENTER_MODEL_PATH=/absolute/path/to/segmenter.onnx
SEGMENTER_INPUT_SIZE=128
SEGMENTER_THRESHOLD=0.5
SEGMENTER_MIN_AREA_RATIO=0.01
SEGMENTER_MAX_AREA_RATIO=0.85

# Prediction fusion (full image vs lesion crop)
FUSION_MARGIN=0.05
FUSION_MIN_CROP_CONFIDENCE=0.55
FUSION_MIN_BOX_SCORE=0.08

# Localization strategy
LOCALIZATION_MODE=classifier_first
SALIENCY_PERCENTILE=93
SALIENCY_MIN_AREA_RATIO=0.01
SALIENCY_MAX_AREA_RATIO=0.85
```

Supported formats:
- classifier: `.onnx`, `.pt`, `.jit`, `.torchscript`
- segmenter: `.onnx`, `.pt`, `.jit`, `.torchscript`

Hugging Face classifier directory support:
- set `MODEL_HF_DIR=/absolute/path/to/hf-model-dir`
- this is useful for repos that provide `model.safetensors` + `config.json`
- when `MODEL_HF_DIR` is set, `MODEL_PATH` is ignored

Install model-runtime dependencies (as needed):

```bash
pip install onnxruntime
# OR
pip install torch
# If using Hugging Face safetensors classifier
pip install transformers safetensors
```

Quick test:

```bash
curl -X POST "http://127.0.0.1:8000/infer" -F "file=@/path/to/test.jpg"
```

Inference behavior notes:
- runs full-image classification always
- gets box from classifier saliency first (`LOCALIZATION_MODE=classifier_first`), then segmenter fallback
- if a valid lesion box exists, also runs crop classification
- switches to crop prediction only if crop confidence exceeds full-image confidence by `FUSION_MARGIN`
- requires lesion box score >= `FUSION_MIN_BOX_SCORE` before crop override is allowed
- rejects implausible boxes using `SEGMENTER_MIN_AREA_RATIO` and `SEGMENTER_MAX_AREA_RATIO`

Debug endpoint example:

```bash
curl -X POST "http://127.0.0.1:8000/infer/debug" -F "file=@/path/to/test.jpg"
```

## 5) Qualcomm AI Hub Workbench scripts

- `ai_hub_workbench.py`: classification compile/infer/profile flow
- `ai_hub_workbench_segmentation.py`: segmentation compile/infer/profile flow

Run either script:

```bash
export QAIHUB_API_TOKEN=your_token_here
python ai_hub_workbench.py
python ai_hub_workbench_segmentation.py
```

Optional env vars:
- `QAIHUB_DEVICE` (default: `Samsung Galaxy S25 (Family)`)
- `QAIHUB_TARGET_RUNTIME` (default: `tflite`)

## 6) Run tests

```bash
pytest -q
```

## 7) Frontend handoff quickstart

Start backend for the frontend engineer:

```bash
cd Backend/Inference
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Useful request examples:

```bash
curl -X GET "http://127.0.0.1:8000/health"
curl -X POST "http://127.0.0.1:8000/infer" -F "file=@/path/to/image.jpg"
curl -X POST "http://127.0.0.1:8000/infer/debug" -F "file=@/path/to/image.jpg"
```

## 8) Accuracy evaluation workflow

Generate labeled evaluation images from HAM10000 and evaluate:

```bash
cd Backend/Inference
source .venv/bin/activate

# Create balanced labeled sample set
python pull_labeled_test_images.py --per-class 50 --out-dir test_samples/ham10000_50x2

# Evaluate live API endpoint
python evaluate_pipeline_accuracy.py \
  --manifest test_samples/ham10000_50x2/manifest.csv \
  --endpoint http://127.0.0.1:8000/infer \
  --report-out test_samples/ham10000_50x2/eval_report.json
```

Notes:
- `test_samples/` is intentionally git-ignored (generated data).
- Re-run these commands anytime to produce fresh benchmark reports.

