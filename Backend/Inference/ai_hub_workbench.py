rom __future__ import annotations

"""AI Hub Workbench workflow for classifier optimization on Snapdragon devices.

This follows the canonical Workbench flow:
1) Load + trace model
2) Compile for target runtime
3) Submit inference job
4) Submit profile job
5) (Optional) Quantize from compiled ONNX and re-compile

Default classifier source is the downloaded Hugging Face snapshot:
models/Anwarkh1__Skin_Cancer-Image_Classification
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModelForImageClassification

try:
    import qai_hub as hub
except ImportError as exc:
    raise SystemExit("qai_hub is not installed. Run: pip install 'qai-hub[torch]'") from exc


load_dotenv(Path(__file__).resolve().parent / ".env")


def validate_ai_hub_authentication() -> None:
    token_present = bool(os.getenv("QAIHUB_API_TOKEN", "").strip())
    client_ini = Path.home() / ".qai_hub" / "client.ini"

    try:
        hub.get_devices()
    except Exception as exc:
        token_hint = (
            "QAIHUB_API_TOKEN is set in environment."
            if token_present
            else "QAIHUB_API_TOKEN is not set in environment."
        )
        ini_hint = (
            f"Found config file: {client_ini}"
            if client_ini.exists()
            else f"Missing config file: {client_ini}"
        )
        raise SystemExit(
            "AI Hub authentication is not configured.\n"
            f"- {token_hint}\n"
            f"- {ini_hint}\n"
            "Configure AI Hub first: https://aihub.qualcomm.com/get-started\n"
            "Then run: qai-hub configure --api_token <your_token>\n"
            f"Original error: {exc}"
        )


class HFClassifierWrapper(nn.Module):
    def __init__(self, base_model: AutoModelForImageClassification) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(pixel_values=pixel_values)
        return outputs.logits


def resolve_model_dir(cli_model_dir: str | None) -> Path:
    if cli_model_dir:
        return Path(cli_model_dir).expanduser().resolve()

    env_model_hf_dir = os.getenv("MODEL_HF_DIR", "").strip()
    if env_model_hf_dir:
        return (Path(__file__).resolve().parent / env_model_hf_dir).resolve()

    return (Path(__file__).resolve().parent / "models" / "Anwarkh1__Skin_Cancer-Image_Classification").resolve()


def build_traced_model(model_dir: Path, input_shape: tuple[int, int, int, int]) -> tuple[torch.jit.ScriptModule, AutoImageProcessor, dict[int, str]]:
    base_model = AutoModelForImageClassification.from_pretrained(str(model_dir)).eval()
    processor = AutoImageProcessor.from_pretrained(str(model_dir))

    id2label_raw = getattr(base_model.config, "id2label", {})
    id2label: dict[int, str] = {}
    if isinstance(id2label_raw, dict):
        for key, value in id2label_raw.items():
            try:
                id2label[int(key)] = str(value)
            except (ValueError, TypeError):
                continue

    wrapper = HFClassifierWrapper(base_model).eval()
    example_input = torch.rand(input_shape)
    traced_torch_model = torch.jit.trace(wrapper, example_input)
    return traced_torch_model, processor, id2label


def prepare_input(image_url: str, input_size: int, processor: AutoImageProcessor) -> np.ndarray:
    response = requests.get(image_url, stream=True, timeout=30)
    response.raise_for_status()
    response.raw.decode_content = True

    image = Image.open(response.raw).convert("RGB").resize((input_size, input_size))
    image_array = np.array(image, dtype=np.float32) / 255.0
    chw = np.transpose(image_array, (2, 0, 1))

    mean = np.array(getattr(processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape((3, 1, 1))
    std = np.array(getattr(processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape((3, 1, 1))

    normalized = (chw - mean) / std
    return np.expand_dims(normalized.astype(np.float32), axis=0)


def topk_predictions(logits: np.ndarray, id2label: dict[int, str], top_k: int = 5) -> list[tuple[int, str, float]]:
    probabilities = np.exp(logits - np.max(logits))
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

    top_indices = np.argsort(probabilities[0], axis=0)[-top_k:]
    results: list[tuple[int, str, float]] = []
    for index in reversed(top_indices):
        label = id2label.get(int(index), f"class_{index}")
        confidence = float(probabilities[0][index])
        results.append((int(index), label, confidence))
    return results


def collect_calibration_inputs(
    images_dir: Path,
    input_size: int,
    processor: AutoImageProcessor,
    max_samples: int,
) -> dict[str, list[np.ndarray]]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"Calibration directory does not exist: {images_dir}")

    mean = np.array(getattr(processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape((3, 1, 1))
    std = np.array(getattr(processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape((3, 1, 1))

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    sample_inputs: list[np.ndarray] = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in valid_suffixes:
            continue

        image = Image.open(image_path).convert("RGB").resize((input_size, input_size))
        image_array = np.array(image, dtype=np.float32) / 255.0
        chw = np.transpose(image_array, (2, 0, 1))
        normalized = ((chw - mean) / std).astype(np.float32)
        sample_inputs.append(np.expand_dims(normalized, axis=0))

        if len(sample_inputs) >= max_samples:
            break

    if not sample_inputs:
        raise ValueError(f"No calibration images found in {images_dir}")

    return {"image": sample_inputs}


def compile_model_for_runtime(
    model: hub.Model,
    device: hub.Device,
    input_shape: tuple[int, int, int, int],
    runtime: str,
) -> tuple[hub.CompileJob, hub.Model]:
    compile_job = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs={"image": input_shape},
        options=f"--target_runtime {runtime}",
    )

    print(f"Submitted compile job ({runtime}): {compile_job.url}")
    print("Waiting for compile job to finish...")

    last_status_line = ""
    while True:
        status = compile_job.get_status()
        status_line = f"[{status.symbol}] {status.code}: {status.message}"
        if status_line != last_status_line:
            print(status_line)
            last_status_line = status_line

        if status.finished:
            if status.success:
                break
            raise RuntimeError(f"Compile job failed: {status_line}")

        time.sleep(10)

    target_model = compile_job.get_target_model()
    return compile_job, target_model


def run_inference(
    target_model: hub.Model,
    device: hub.Device,
    input_array: np.ndarray,
) -> tuple[hub.InferenceJob, dict[str, np.ndarray]]:
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs={"image": [input_array]},
    )
    on_device_output = inference_job.download_output_data()
    return inference_job, on_device_output


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Hub Workbench compile/inference/profile flow")
    parser.add_argument("--model-dir", default=None, help="Path to Hugging Face classifier directory")
    parser.add_argument("--device", default=os.getenv("QAIHUB_DEVICE", "Samsung Galaxy S25 (Family)"), help="QAI Hub target device")
    parser.add_argument("--runtime", default=os.getenv("QAIHUB_TARGET_RUNTIME", "tflite"), help="Target runtime for compile (e.g., tflite, onnx)")
    parser.add_argument(
        "--sample-image-url",
        default="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg",
        help="Sample image URL for inference",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Input image size")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K predictions to print")
    parser.add_argument("--download-name", default="", help="Optional filename for compiled model download")

    parser.add_argument("--quantize", action="store_true", help="Run optional quantize flow")
    parser.add_argument("--calibration-dir", default="", help="Directory containing calibration images")
    parser.add_argument("--calibration-samples", type=int, default=100, help="Max calibration image count")
    parser.add_argument(
        "--quantized-download-name",
        default="",
        help="Optional filename for quantized compiled model download",
    )

    args = parser.parse_args()

    validate_ai_hub_authentication()

    model_dir = resolve_model_dir(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    input_shape = (1, 3, args.input_size, args.input_size)
    traced_torch_model, processor, id2label = build_traced_model(model_dir, input_shape)

    print(f"Using model directory: {model_dir}")
    print(f"Target device: {args.device}")
    print(f"Target runtime: {args.runtime}")

    device = hub.Device(args.device)

    compile_job, target_model = compile_model_for_runtime(
        model=traced_torch_model,
        device=device,
        input_shape=input_shape,
        runtime=args.runtime,
    )

    input_array = prepare_input(args.sample_image_url, args.input_size, processor)
    inference_job, on_device_output = run_inference(
        target_model=target_model,
        device=device,
        input_array=input_array,
    )

    output_name = list(on_device_output.keys())[0]
    logits = np.array(on_device_output[output_name][0], dtype=np.float32)
    if logits.ndim == 1:
        logits = np.expand_dims(logits, axis=0)

    print("Top predictions:")
    for index, label, confidence in topk_predictions(logits, id2label, top_k=args.top_k):
        print(f"  {index:4d}  {label:30s}  {confidence:>7.2%}")

    profile_job = hub.submit_profile_job(model=target_model, device=device)

    print("Compile job:", compile_job.url)
    print("Inference job:", inference_job.url)
    print("Profile job:", profile_job.url)

    if args.download_name:
        target_model.download(args.download_name)
        print(f"Downloaded compiled model to: {args.download_name}")

    if not args.quantize:
        return

    calibration_dir = Path(args.calibration_dir).expanduser().resolve() if args.calibration_dir else None
    if calibration_dir is None:
        raise SystemExit("--quantize requires --calibration-dir")

    onnx_compile_job, unquantized_onnx_model = compile_model_for_runtime(
        model=traced_torch_model,
        device=device,
        input_shape=input_shape,
        runtime="onnx",
    )

    calibration_data = collect_calibration_inputs(
        images_dir=calibration_dir,
        input_size=args.input_size,
        processor=processor,
        max_samples=args.calibration_samples,
    )

    quantize_job = hub.submit_quantize_job(
        model=unquantized_onnx_model,
        calibration_data=calibration_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    quantized_onnx_model = quantize_job.get_target_model()

    quantized_compile_job, quantized_target_model = compile_model_for_runtime(
        model=quantized_onnx_model,
        device=device,
        input_shape=input_shape,
        runtime=args.runtime,
    )

    quantized_inference_job, quantized_output = run_inference(
        target_model=quantized_target_model,
        device=device,
        input_array=input_array,
    )

    quantized_output_name = list(quantized_output.keys())[0]
    quantized_logits = np.array(quantized_output[quantized_output_name][0], dtype=np.float32)
    if quantized_logits.ndim == 1:
        quantized_logits = np.expand_dims(quantized_logits, axis=0)

    print("Top predictions (quantized):")
    for index, label, confidence in topk_predictions(quantized_logits, id2label, top_k=args.top_k):
        print(f"  {index:4d}  {label:30s}  {confidence:>7.2%}")

    quantized_profile_job = hub.submit_profile_job(model=quantized_target_model, device=device)

    print("ONNX compile job (pre-quantize):", onnx_compile_job.url)
    print("Quantize job:", quantize_job.url)
    print("Quantized compile job:", quantized_compile_job.url)
    print("Quantized inference job:", quantized_inference_job.url)
    print("Quantized profile job:", quantized_profile_job.url)

    if args.quantized_download_name:
        quantized_target_model.download(args.quantized_download_name)
        print(f"Downloaded quantized compiled model to: {args.quantized_download_name}")


if __name__ == "__main__":
    main()
