from __future__ import annotations

"""Qualcomm AI Hub Workbench flow for segmentation models.

This script demonstrates compile + inference + profile for a lesion segmenter.
Input tensor is expected as NCHW grayscale: (1, 1, 128, 128).

Usage:
1) pip install 'qai-hub[torch]' torch pillow numpy requests
2) export QAIHUB_API_TOKEN=...
3) python ai_hub_workbench_segmentation.py

Optional env vars:
- QAIHUB_DEVICE (default: Samsung Galaxy S25 (Family))
- QAIHUB_TARGET_RUNTIME (default: tflite)
- SEGMENTER_INPUT_SIZE (default: 128)
"""

import os

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

try:
    import qai_hub as hub
except ImportError as exc:
    raise SystemExit("qai_hub is not installed. Run: pip install 'qai-hub[torch]'") from exc


class TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


def build_placeholder_segmenter(input_shape: tuple[int, int, int, int]) -> torch.jit.ScriptModule:
    model = TinySegmenter().eval()
    example_input = torch.rand(input_shape)
    traced = torch.jit.trace(model, example_input)
    return traced


def prepare_grayscale_input(image_url: str, input_size: int) -> np.ndarray:
    response = requests.get(image_url, stream=True, timeout=30)
    response.raise_for_status()
    response.raw.decode_content = True

    image = Image.open(response.raw).convert("L").resize((input_size, input_size))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(np.expand_dims(image_array, axis=0), axis=0)


def main() -> None:
    input_size = int(os.getenv("SEGMENTER_INPUT_SIZE", "128"))
    input_shape = (1, 1, input_size, input_size)
    device_name = os.getenv("QAIHUB_DEVICE", "Samsung Galaxy S25 (Family)")
    target_runtime = os.getenv("QAIHUB_TARGET_RUNTIME", "tflite")

    traced_segmenter = build_placeholder_segmenter(input_shape)
    device = hub.Device(device_name)

    compile_job = hub.submit_compile_job(
        model=traced_segmenter,
        device=device,
        input_specs=dict(image=input_shape),
        options=f"--target_runtime {target_runtime}",
    )
    target_model = compile_job.get_target_model()

    sample_image_url = (
        "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
    )
    input_array = prepare_grayscale_input(sample_image_url, input_size)

    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs=dict(image=[input_array]),
    )
    on_device_output = inference_job.download_output_data()

    output_name = list(on_device_output.keys())[0]
    logits = on_device_output[output_name][0]

    print("On-device segmentation logits shape:", np.array(logits).shape)

    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print("Compile job:", compile_job.url)
    print("Inference job:", inference_job.url)
    print("Profile job:", profile_job.url)


if __name__ == "__main__":
    main()
