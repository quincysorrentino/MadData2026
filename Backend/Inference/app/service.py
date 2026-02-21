from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class BoundingBoxData(TypedDict):
    x_min: int
    y_min: int
    width: int
    height: int
    score: float
    label: str


class PlaceholderSkinCancerModel:
    """Deterministic placeholder classifier for API integration testing.

    This is not a medical model. It classifies based on image brightness only.
    """

    def __init__(self, model_version: str = "placeholder-v1") -> None:
        self.model_version = model_version

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0
        return image_array

    def predict_from_array(self, image_array: np.ndarray) -> tuple[str, float]:
        brightness = float(np.mean(image_array))

        if brightness < 0.45:
            label = "suspicious"
            confidence = min(0.95, 0.5 + (0.45 - brightness))
        else:
            label = "non_suspicious"
            confidence = min(0.95, 0.5 + (brightness - 0.45))

        return label, float(confidence)

    def predict(self, image_bytes: bytes) -> tuple[str, float, str]:
        image_array = self._preprocess(image_bytes)
        label, confidence = self.predict_from_array(image_array)
        return label, confidence, self.model_version


class ExternalModel:
    def __init__(self, model_path: str, labels: list[str], input_size: int = 224) -> None:
        self.model_path = model_path
        self.labels = labels
        self.input_size = input_size
        self.model_version = f"external:{Path(model_path).name}"

        suffix = Path(model_path).suffix.lower()
        if suffix == ".onnx":
            import onnxruntime as ort

            self.backend = "onnx"
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        elif suffix in {".pt", ".jit", ".torchscript"}:
            import torch

            self.backend = "torchscript"
            self.torch = torch
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
        else:
            raise ValueError(
                "Unsupported MODEL_PATH extension. Use .onnx, .pt, .jit, or .torchscript"
            )

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((self.input_size, self.input_size))
        image_array = np.array(image, dtype=np.float32) / 255.0
        chw = np.transpose(image_array, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def predict_from_array(self, image_array: np.ndarray) -> tuple[str, float]:
        resized = np.array(
            Image.fromarray((image_array * 255).astype(np.uint8)).resize((self.input_size, self.input_size))
        ).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0).astype(np.float32)

        if self.backend == "onnx":
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = np.array(outputs[0][0], dtype=np.float32)
        else:
            with self.torch.no_grad():
                logits_tensor = self.model(self.torch.from_numpy(input_tensor))
                logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32)

        probabilities = self._softmax(logits)
        class_idx = int(np.argmax(probabilities))
        label = self.labels[class_idx] if class_idx < len(self.labels) else f"class_{class_idx}"
        confidence = float(probabilities[class_idx])
        return label, confidence

    def predict(self, image_bytes: bytes) -> tuple[str, float, str]:
        image_array = self._preprocess(image_bytes)[0].transpose((1, 2, 0))
        label, confidence = self.predict_from_array(image_array)
        return label, confidence, self.model_version


class HuggingFaceClassifierModel:
    def __init__(self, model_dir: str, input_size: int = 224) -> None:
        self.model_dir = model_dir
        self.input_size = input_size
        self.model_version = f"hf:{Path(model_dir).name}"

        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError as exc:
            raise ValueError(
                "Hugging Face model support requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch. "
                f"Original import error: {exc}"
            ) from exc

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            self.labels = [id2label[idx] for idx in sorted(id2label.keys())]
        else:
            self.labels = []

    def predict_from_array(self, image_array: np.ndarray) -> tuple[str, float]:
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).resize(
            (self.input_size, self.input_size)
        )
        inputs = self.processor(images=pil_image, return_tensors="pt")

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)

        shifted = logits - np.max(logits)
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / np.sum(exp_scores)

        class_idx = int(np.argmax(probabilities))
        if class_idx < len(self.labels):
            label = self.labels[class_idx]
        else:
            label = f"class_{class_idx}"
        confidence = float(probabilities[class_idx])
        return label, confidence

    def predict(self, image_bytes: bytes) -> tuple[str, float, str]:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image, dtype=np.float32) / 255.0
        label, confidence = self.predict_from_array(image_array)
        return label, confidence, self.model_version

    def localize_box_from_array(self, image_array: np.ndarray) -> list[BoundingBoxData]:
        orig_h, orig_w = image_array.shape[:2]

        pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).resize(
            (self.input_size, self.input_size)
        )
        inputs = self.processor(images=pil_image, return_tensors="pt")
        if "pixel_values" not in inputs:
            return []

        pixel_values = inputs["pixel_values"].clone().detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits[0]
        target_idx = int(self.torch.argmax(logits).item())
        logits[target_idx].backward()

        if pixel_values.grad is None:
            return []

        gradients = pixel_values.grad[0].detach().cpu().numpy()
        saliency = np.abs(gradients).mean(axis=0)

        saliency_min = float(saliency.min())
        saliency_max = float(saliency.max())
        if saliency_max <= saliency_min:
            return []

        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

        percentile = float(os.getenv("SALIENCY_PERCENTILE", "93"))
        threshold = float(np.percentile(saliency, percentile))

        binary = (saliency >= threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        contour_candidates: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 5.0:
                continue
            contour_mask = np.zeros_like(binary, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, color=1, thickness=-1)
            mean_saliency = float(saliency[contour_mask == 1].mean())
            contour_candidates.append((area * mean_saliency, contour))

        if not contour_candidates:
            return []

        best_contour = max(contour_candidates, key=lambda item: item[0])[1]
        x, y, width, height = cv2.boundingRect(best_contour)

        scale_x = orig_w / float(self.input_size)
        scale_y = orig_h / float(self.input_size)

        final_x = max(0, int(round(x * scale_x)))
        final_y = max(0, int(round(y * scale_y)))
        final_w = max(1, int(round(width * scale_x)))
        final_h = max(1, int(round(height * scale_y)))

        area_ratio = (final_w * final_h) / float(max(1, orig_w * orig_h))
        min_area = float(os.getenv("SALIENCY_MIN_AREA_RATIO", "0.01"))
        max_area = float(os.getenv("SALIENCY_MAX_AREA_RATIO", "0.85"))
        if area_ratio < min_area or area_ratio > max_area:
            return []

        contour_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(contour_mask, [best_contour], -1, color=1, thickness=-1)
        score = float(min(1.0, max(0.0, saliency[contour_mask == 1].mean())))

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


class SegmenterModel:
    def __init__(self, model_path: str, input_size: int = 128, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.model_version = f"segmenter:{Path(model_path).name}"
        self.min_area_ratio = float(os.getenv("SEGMENTER_MIN_AREA_RATIO", "0.01"))
        self.max_area_ratio = float(os.getenv("SEGMENTER_MAX_AREA_RATIO", "0.85"))

        suffix = Path(model_path).suffix.lower()
        if suffix == ".onnx":
            import onnxruntime as ort

            self.backend = "onnx"
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        elif suffix in {".pt", ".jit", ".torchscript"}:
            import torch

            self.backend = "torchscript"
            self.torch = torch
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
        elif suffix == ".pth":
            import torch
            import torch.nn as nn

            def _block(in_channels: int, out_channels: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(),
                )

            class _ShifaaUNet(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.enc1 = _block(1, 16)
                    self.enc2 = _block(16, 32)
                    self.enc3 = _block(32, 64)
                    self.enc4 = _block(64, 128)
                    self.pool = nn.MaxPool2d(2)

                    self.bottleneck = _block(128, 256)

                    self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                    self.dec4 = _block(256, 128)
                    self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                    self.dec3 = _block(128, 64)
                    self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                    self.dec2 = _block(64, 32)
                    self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
                    self.dec1 = _block(32, 16)
                    self.final = nn.Conv2d(16, 1, kernel_size=1)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    e3 = self.enc3(self.pool(e2))
                    e4 = self.enc4(self.pool(e3))

                    b = self.bottleneck(self.pool(e4))

                    d4 = self.up4(b)
                    d4 = self.dec4(torch.cat([d4, e4], dim=1))
                    d3 = self.up3(d4)
                    d3 = self.dec3(torch.cat([d3, e3], dim=1))
                    d2 = self.up2(d3)
                    d2 = self.dec2(torch.cat([d2, e2], dim=1))
                    d1 = self.up1(d2)
                    d1 = self.dec1(torch.cat([d1, e1], dim=1))

                    return self.final(d1)

            self.backend = "pytorch_state_dict"
            self.torch = torch
            self.model = _ShifaaUNet()
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            raise ValueError(
                "Unsupported SEGMENTER_MODEL_PATH extension. Use .onnx, .pt, .jit, .torchscript, or .pth"
            )

    def _preprocess(self, rgb_image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray_image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized[None, None, :, :]

    def _infer_probability_map(self, rgb_image: np.ndarray) -> np.ndarray:
        input_tensor = self._preprocess(rgb_image)
        if self.backend == "onnx":
            outputs = self.session.run(None, {self.input_name: input_tensor.astype(np.float32)})
            logits = np.array(outputs[0], dtype=np.float32)
            if logits.ndim == 4:
                logits = logits[0, 0]
            else:
                logits = logits.squeeze()
        else:
            with self.torch.no_grad():
                logits_tensor = self.model(self.torch.from_numpy(input_tensor.astype(np.float32)))
                logits = logits_tensor.detach().cpu().numpy().squeeze().astype(np.float32)

        clipped_logits = np.clip(logits, -50.0, 50.0)
        probabilities = 1.0 / (1.0 + np.exp(-clipped_logits))
        return probabilities

    def detect_boxes(self, rgb_image: np.ndarray) -> list[BoundingBoxData]:
        orig_h, orig_w = rgb_image.shape[:2]
        probability_map = self._infer_probability_map(rgb_image)
        binary_mask = (probability_map > self.threshold).astype(np.uint8) * 255
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            dynamic_threshold = max(0.15, float(np.percentile(probability_map, 92)))
            binary_mask = (probability_map > dynamic_threshold).astype(np.uint8) * 255
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        contour_candidates: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 5.0:
                continue

            contour_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, color=1, thickness=-1)
            mean_probability = float(probability_map[contour_mask == 1].mean())
            weighted_score = area * mean_probability
            contour_candidates.append((weighted_score, contour))

        if not contour_candidates:
            return []

        best_contour = max(contour_candidates, key=lambda item: item[0])[1]
        x, y, width, height = cv2.boundingRect(best_contour)

        scale_x = orig_w / float(self.input_size)
        scale_y = orig_h / float(self.input_size)

        final_x = max(0, int(round(x * scale_x)))
        final_y = max(0, int(round(y * scale_y)))
        final_w = max(0, int(round(width * scale_x)))
        final_h = max(0, int(round(height * scale_y)))

        if final_w == 0 or final_h == 0:
            return []

        contour_area = float(cv2.contourArea(best_contour))
        image_area = float(self.input_size * self.input_size)
        score = float(min(1.0, max(0.0, contour_area / max(1.0, image_area))))

        full_area_ratio = (final_w * final_h) / float(max(1, orig_w * orig_h))
        if full_area_ratio < self.min_area_ratio or full_area_ratio > self.max_area_ratio:
            return []

        touches_left = final_x <= 1
        touches_top = final_y <= 1
        touches_right = final_x + final_w >= orig_w - 1
        touches_bottom = final_y + final_h >= orig_h - 1
        boundary_touches = sum([touches_left, touches_top, touches_right, touches_bottom])
        if boundary_touches >= 3 and full_area_ratio > 0.35:
            return []

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


class InferencePipeline:
    def __init__(
        self,
        classifier: PlaceholderSkinCancerModel | ExternalModel | HuggingFaceClassifierModel,
        segmenter: SegmenterModel | None = None,
    ) -> None:
        self.classifier = classifier
        self.segmenter = segmenter
        if segmenter is None:
            self.model_version = classifier.model_version
        else:
            self.model_version = f"{classifier.model_version}+{segmenter.model_version}"

    @staticmethod
    def _load_rgb(image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return np.array(image, dtype=np.float32) / 255.0

    @staticmethod
    def _crop_array(image_array: np.ndarray, box: BoundingBoxData) -> np.ndarray:
        x_min = max(0, box["x_min"])
        y_min = max(0, box["y_min"])
        x_max = min(image_array.shape[1], x_min + max(1, box["width"]))
        y_max = min(image_array.shape[0], y_min + max(1, box["height"]))
        crop = image_array[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return image_array
        return crop

    def predict_detailed(self, image_bytes: bytes) -> dict[str, Any]:
        image_array = self._load_rgb(image_bytes)
        boxes: list[BoundingBoxData] = []

        full_label, full_confidence = self.classifier.predict_from_array(image_array)
        chosen_label = full_label
        chosen_confidence = full_confidence

        fusion_margin = float(os.getenv("FUSION_MARGIN", "0.05"))
        min_crop_confidence = float(os.getenv("FUSION_MIN_CROP_CONFIDENCE", "0.55"))
        min_box_score = float(os.getenv("FUSION_MIN_BOX_SCORE", "0.08"))

        used_crop = False
        decision_reason = "full_image_only"
        box_source = "none"
        crop_label: str | None = None
        crop_confidence: float | None = None

        localization_mode = os.getenv("LOCALIZATION_MODE", "classifier_first")

        if localization_mode in {"classifier_first", "classifier_only"} and hasattr(
            self.classifier, "localize_box_from_array"
        ):
            classifier_boxes = self.classifier.localize_box_from_array(image_array)
            if classifier_boxes:
                boxes = classifier_boxes
                box_source = "classifier_saliency"

        if not boxes and self.segmenter is not None and localization_mode != "classifier_only":
            boxes = self.segmenter.detect_boxes(image_array)
            if boxes:
                box_source = "segmenter"
        elif localization_mode == "segmenter_only" and self.segmenter is not None:
            boxes = self.segmenter.detect_boxes(image_array)
            if boxes:
                box_source = "segmenter"

        if boxes:
            box_score = float(boxes[0]["score"])
            if box_score < min_box_score:
                decision_reason = "box_score_too_low"
            else:
                crop_input = self._crop_array(image_array, boxes[0])
                crop_label, crop_confidence = self.classifier.predict_from_array(crop_input)
                if crop_confidence < min_crop_confidence:
                    decision_reason = "crop_confidence_too_low"
                elif crop_confidence < full_confidence + fusion_margin:
                    decision_reason = "crop_no_confidence_gain"
                else:
                    chosen_label = crop_label
                    chosen_confidence = crop_confidence
                    used_crop = True
                    decision_reason = "crop_override"
        else:
            decision_reason = "no_valid_box"

        return {
            "label": chosen_label,
            "confidence": float(chosen_confidence),
            "model_version": self.model_version,
            "boxes": boxes,
            "debug": {
                "used_crop": used_crop,
                "decision_reason": decision_reason,
                "full_label": full_label,
                "full_confidence": float(full_confidence),
                "crop_label": crop_label,
                "crop_confidence": None if crop_confidence is None else float(crop_confidence),
                "fusion_margin": fusion_margin,
                "min_crop_confidence": min_crop_confidence,
                "min_box_score": min_box_score,
                "localization_mode": localization_mode,
                "box_source": box_source,
            },
        }

    def predict(self, image_bytes: bytes) -> tuple[str, float, str, list[BoundingBoxData]]:
        detailed = self.predict_detailed(image_bytes)
        return (
            detailed["label"],
            float(detailed["confidence"]),
            detailed["model_version"],
            detailed["boxes"],
        )


def build_classifier() -> PlaceholderSkinCancerModel | ExternalModel | HuggingFaceClassifierModel:
    hf_model_dir = os.getenv("MODEL_HF_DIR", "").strip()
    if hf_model_dir:
        input_size = int(os.getenv("MODEL_INPUT_SIZE", "224"))
        return HuggingFaceClassifierModel(model_dir=hf_model_dir, input_size=input_size)

    model_path = os.getenv("MODEL_PATH", "").strip()
    if not model_path:
        return PlaceholderSkinCancerModel()

    labels_csv = os.getenv("MODEL_LABELS", "non_suspicious,suspicious")
    labels = [item.strip() for item in labels_csv.split(",") if item.strip()]
    input_size = int(os.getenv("MODEL_INPUT_SIZE", "224"))
    return ExternalModel(model_path=model_path, labels=labels, input_size=input_size)


def build_segmenter() -> SegmenterModel | None:
    model_path = os.getenv("SEGMENTER_MODEL_PATH", "").strip()
    if not model_path:
        return None

    input_size = int(os.getenv("SEGMENTER_INPUT_SIZE", "128"))
    threshold = float(os.getenv("SEGMENTER_THRESHOLD", "0.5"))
    return SegmenterModel(model_path=model_path, input_size=input_size, threshold=threshold)


model = InferencePipeline(classifier=build_classifier(), segmenter=build_segmenter())
