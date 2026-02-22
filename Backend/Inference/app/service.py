from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

INFERENCE_ROOT = Path(__file__).resolve().parents[1]


class BoundingBoxData(TypedDict):
    x_min: int
    y_min: int
    width: int
    height: int
    score: float
    label: str


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_scores = np.exp(shifted)
    return exp_scores / np.clip(np.sum(exp_scores), 1e-12, None)


def _requested_ort_providers() -> list[str]:
    raw = os.getenv("ORT_EXECUTION_PROVIDERS", "QNNExecutionProvider,CPUExecutionProvider")
    providers = [item.strip() for item in raw.split(",") if item.strip()]
    return providers or ["CPUExecutionProvider"]


def _provider_options(provider_name: str) -> dict[str, str]:
    if provider_name != "QNNExecutionProvider":
        return {}

    options: dict[str, str] = {}
    backend_path = os.getenv("QNN_BACKEND_PATH", "").strip()
    if backend_path:
        options["backend_path"] = backend_path

    htp_mode = os.getenv("QNN_HTP_PERFORMANCE_MODE", "").strip()
    if htp_mode:
        options["htp_performance_mode"] = htp_mode

    profiling_level = os.getenv("QNN_PROFILING_LEVEL", "").strip()
    if profiling_level:
        options["profiling_level"] = profiling_level

    return options


def _build_onnx_session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required. Install onnxruntime (or a Snapdragon QNN-enabled wheel such as onnxruntime-qnn)."
        ) from exc

    available_providers = ort.get_available_providers()
    requested_providers = _requested_ort_providers()
    enable_cpu_fallback = _parse_bool(os.getenv("ORT_ENABLE_CPU_FALLBACK"), default=True)
    require_npu = _parse_bool(os.getenv("REQUIRE_NPU"), default=False)
    strict_npu_only = require_npu and not enable_cpu_fallback

    selected_providers = [p for p in requested_providers if p in available_providers]
    if not selected_providers:
        raise RuntimeError(
            "None of the requested ONNX Runtime providers are available. "
            f"requested={requested_providers}, available={available_providers}"
        )

    if not enable_cpu_fallback:
        selected_providers = [p for p in selected_providers if p != "CPUExecutionProvider"]
        if not selected_providers:
            raise RuntimeError(
                "ORT_ENABLE_CPU_FALLBACK is false and no non-CPU provider is available. "
                f"requested={requested_providers}, available={available_providers}"
            )

    providers_arg: list[str | tuple[str, dict[str, str]]] = []
    for provider_name in selected_providers:
        options = _provider_options(provider_name)
        providers_arg.append((provider_name, options) if options else provider_name)

    session_options = ort.SessionOptions()
    if strict_npu_only:
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers_arg)
    active_providers = session.get_providers()

    print(
        "[Inference] ONNX Runtime providers "
        f"requested={requested_providers} available={available_providers} active={active_providers}"
    )

    if require_npu and "QNNExecutionProvider" not in active_providers:
        raise RuntimeError(
            "REQUIRE_NPU=true but QNNExecutionProvider is not active. "
            "Install a QNN-enabled ONNX Runtime wheel on Snapdragon and verify provider configuration."
        )

    if strict_npu_only and "CPUExecutionProvider" in requested_providers:
        raise RuntimeError(
            "Strict NPU mode requires ORT_EXECUTION_PROVIDERS to exclude CPUExecutionProvider. "
            "Set ORT_EXECUTION_PROVIDERS=QNNExecutionProvider."
        )

    return session


def _load_labels() -> list[str]:
    labels_csv = os.getenv("MODEL_LABELS", "").strip()
    if labels_csv:
        return [item.strip() for item in labels_csv.split(",") if item.strip()]

    model_root = INFERENCE_ROOT / "models"
    config_candidates = [
        model_root / "Anwarkh1__Skin_Cancer-Image_Classification" / "config.json",
        model_root / "config.json",
    ]
    for config_path in config_candidates:
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        id2label = payload.get("id2label")
        if isinstance(id2label, dict) and id2label:
            indexed = sorted((int(key), str(value)) for key, value in id2label.items())
            return [value for _, value in indexed]

    return ["non_suspicious", "suspicious"]


def _resolve_model_path(raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (INFERENCE_ROOT / candidate).resolve()


def _discover_classifier_model() -> Path:
    configured = os.getenv("CLASSIFIER_MODEL_PATH", "").strip() or os.getenv("MODEL_PATH", "").strip()
    if configured:
        return _resolve_model_path(configured)

    models_root = INFERENCE_ROOT / "models"
    preferred = models_root / "classifier.onnx"
    if preferred.exists():
        return preferred

    candidates = sorted(path for path in models_root.rglob("*.onnx") if "segment" not in path.name.lower())
    if not candidates:
        raise RuntimeError("Classifier ONNX model not found. Set CLASSIFIER_MODEL_PATH or MODEL_PATH.")
    return candidates[0]


def _discover_segmenter_model() -> Path:
    configured = os.getenv("SEGMENTER_MODEL_PATH", "").strip()
    if configured:
        return _resolve_model_path(configured)

    models_root = INFERENCE_ROOT / "models"
    preferred = models_root / "Ahmed-Selem__Shifaa-Skin-Cancer-UNet-Segmentation" / "segmenter.onnx"
    if preferred.exists():
        return preferred

    candidates = sorted(path for path in models_root.rglob("*.onnx") if "segment" in path.name.lower())
    if not candidates:
        raise RuntimeError("Segmentation ONNX model not found. Set SEGMENTER_MODEL_PATH.")
    return candidates[0]


class OnnxModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.session = _build_onnx_session(str(model_path))
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_name = self.session.get_outputs()[0].name

    def _infer_layout_and_size(self) -> tuple[str, int, int]:
        shape = list(self.input_meta.shape)
        if len(shape) != 4:
            return "nchw", 224, 224

        d1, d2, d3 = shape[1], shape[2], shape[3]
        c_first = isinstance(d1, int) and d1 in (1, 3)
        c_last = isinstance(d3, int) and d3 in (1, 3)

        if c_last and not c_first:
            height = int(d1) if isinstance(d1, int) and d1 > 0 else 224
            width = int(d2) if isinstance(d2, int) and d2 > 0 else 224
            return "nhwc", height, width

        height = int(d2) if isinstance(d2, int) and d2 > 0 else 224
        width = int(d3) if isinstance(d3, int) and d3 > 0 else 224
        return "nchw", height, width

    def run(self, tensor: np.ndarray) -> np.ndarray:
        result = self.session.run([self.output_name], {self.input_name: tensor})
        return np.asarray(result[0])


class Segmenter:
    def __init__(self, model: OnnxModel, threshold: float = 0.45) -> None:
        self.model = model
        self.threshold = threshold
        self.min_area_ratio = float(os.getenv("SEGMENTER_MIN_AREA_RATIO", "0.005"))
        self.max_area_ratio = float(os.getenv("SEGMENTER_MAX_AREA_RATIO", "0.90"))

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        layout, height, width = self.model._infer_layout_and_size()
        resized = np.asarray(
            Image.fromarray(image_rgb).resize((width, height), resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

        if layout == "nhwc":
            tensor = resized.astype(np.float32)[None, ...]
        else:
            channels = self.model.input_meta.shape[1]
            if isinstance(channels, int) and channels == 1:
                gray = (
                    0.299 * resized[:, :, 0]
                    + 0.587 * resized[:, :, 1]
                    + 0.114 * resized[:, :, 2]
                ).astype(np.float32)
                tensor = gray[None, None, ...]
            else:
                tensor = np.transpose(resized.astype(np.float32), (2, 0, 1))[None, ...]

        if float(np.max(tensor)) > 1.5:
            tensor = tensor / 255.0
        return tensor.astype(np.float32), (height, width)

    def _probability_map(self, image_rgb: np.ndarray) -> np.ndarray:
        input_tensor, _ = self._preprocess(image_rgb)
        output = self.model.run(input_tensor)
        squeezed = np.squeeze(output).astype(np.float32)

        if squeezed.ndim != 2:
            squeezed = np.squeeze(squeezed)
            if squeezed.ndim != 2:
                raise RuntimeError(f"Unexpected segmenter output shape: {output.shape}")

        if float(np.max(squeezed)) > 1.0 or float(np.min(squeezed)) < 0.0:
            clipped = np.clip(squeezed, -50.0, 50.0)
            squeezed = 1.0 / (1.0 + np.exp(-clipped))

        return np.clip(squeezed, 0.0, 1.0)

    def detect_boxes(self, image_rgb: np.ndarray) -> list[BoundingBoxData]:
        original_h, original_w = image_rgb.shape[:2]
        probability = self._probability_map(image_rgb)

        mask = (probability >= self.threshold).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            return []

        x = int(xs.min())
        y = int(ys.min())
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)

        mask_h, mask_w = probability.shape
        scale_x = original_w / float(mask_w)
        scale_y = original_h / float(mask_h)

        final_x = max(0, int(round(x * scale_x)))
        final_y = max(0, int(round(y * scale_y)))
        final_w = max(1, int(round(width * scale_x)))
        final_h = max(1, int(round(height * scale_y)))

        area_ratio = (final_w * final_h) / max(1.0, float(original_w * original_h))
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return []

        score = float(np.clip(probability[mask > 0].mean(), 0.0, 1.0))

        return [
            {
                "x_min": final_x,
                "y_min": final_y,
                "width": final_w,
                "height": final_h,
                "score": score,
                "label": "lesion",
            }
        ]


class Classifier:
    def __init__(self, model: OnnxModel, labels: list[str]) -> None:
        self.model = model
        self.labels = labels

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        layout, height, width = self.model._infer_layout_and_size()
        resized = np.asarray(
            Image.fromarray(image_rgb).resize((width, height), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        resized = resized / 255.0

        if layout == "nhwc":
            tensor = resized[None, ...]
        else:
            tensor = np.transpose(resized, (2, 0, 1))[None, ...]

        return tensor.astype(np.float32)

    def predict_from_array(self, image_rgb: np.ndarray) -> tuple[str, float]:
        tensor = self._preprocess(image_rgb)
        output = self.model.run(tensor)
        logits = np.asarray(output).astype(np.float32).reshape(-1)
        probabilities = _softmax(logits)
        class_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[class_idx])
        label = self.labels[class_idx] if class_idx < len(self.labels) else f"class_{class_idx}"
        return label, confidence


class InferencePipeline:
    def __init__(self) -> None:
        classifier_path = _discover_classifier_model()
        segmenter_path = _discover_segmenter_model()

        classifier_model = OnnxModel(classifier_path)
        segmenter_model = OnnxModel(segmenter_path)

        labels = _load_labels()
        threshold = float(os.getenv("SEGMENTER_THRESHOLD", "0.45"))

        self.classifier = Classifier(classifier_model, labels)
        self.segmenter = Segmenter(segmenter_model, threshold=threshold)

        classifier_provider = classifier_model.session.get_providers()[0]
        segmenter_provider = segmenter_model.session.get_providers()[0]
        self.model_version = (
            f"classifier:{classifier_path.name}@{classifier_provider}+"
            f"segmenter:{segmenter_path.name}@{segmenter_provider}"
        )

    @staticmethod
    def _load_rgb(image_bytes: bytes) -> np.ndarray:
        pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.asarray(pil, dtype=np.uint8)

    @staticmethod
    def _crop(image_rgb: np.ndarray, box: BoundingBoxData) -> np.ndarray:
        x_min = max(0, box["x_min"])
        y_min = max(0, box["y_min"])
        x_max = min(image_rgb.shape[1], x_min + max(1, box["width"]))
        y_max = min(image_rgb.shape[0], y_min + max(1, box["height"]))
        crop = image_rgb[y_min:y_max, x_min:x_max]
        return crop if crop.size else image_rgb

    def predict(self, image_bytes: bytes) -> tuple[str, float, str, list[BoundingBoxData]]:
        payload = self.predict_detailed(image_bytes)
        return payload["label"], payload["confidence"], payload["model_version"], payload["boxes"]

    def predict_detailed(self, image_bytes: bytes) -> dict[str, Any]:
        image_rgb = self._load_rgb(image_bytes)
        boxes = self.segmenter.detect_boxes(image_rgb)

        if boxes:
            target_image = self._crop(image_rgb, boxes[0])
            decision = "segmentation_crop"
        else:
            target_image = image_rgb
            decision = "full_image_fallback"

        label, confidence = self.classifier.predict_from_array(target_image)

        return {
            "label": label,
            "confidence": float(confidence),
            "boxes": boxes,
            "model_version": self.model_version,
            "debug": {
                "decision": decision,
                "has_box": bool(boxes),
                "box_count": len(boxes),
                "providers": {
                    "classifier": self.classifier.model.session.get_providers(),
                    "segmenter": self.segmenter.model.session.get_providers(),
                },
            },
        }


model = InferencePipeline()
