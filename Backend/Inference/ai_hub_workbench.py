from __future__ import annotations

"""Qualcomm AI Hub Workbench helper flow.

Usage:
1) Install deps: pip install qai-hub torch torchvision pillow numpy requests
2) Export token: export QAIHUB_API_TOKEN=...
3) Run: python ai_hub_workbench.py

This script uses a placeholder traced model for hackathon plumbing.
Replace `build_placeholder_torch_model` with your trained model when ready.
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
    raise SystemExit(
        "qai_hub is not installed. Run: pip install 'qai-hub[torch]'"
    ) from exc


class PlaceholderTorchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = features.flatten(1)
        return self.head(features)


def build_placeholder_torch_model() -> torch.jit.ScriptModule:
    torch_model = PlaceholderTorchModel().eval()
    input_shape = (1, 3, 224, 224)
    example_input = torch.rand(input_shape)
    traced_torch_model = torch.jit.trace(torch_model, example_input)
    return traced_torch_model


def prepare_input(image_url: str) -> np.ndarray:
    response = requests.get(image_url, stream=True, timeout=30)
    response.raise_for_status()
    response.raw.decode_content = True
    image = Image.open(response.raw).convert("RGB").resize((224, 224))
    input_array = np.expand_dims(
        np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
    )
    return input_array


def main() -> None:
    input_shape = (1, 3, 224, 224)
    device_name = os.getenv("QAIHUB_DEVICE", "Samsung Galaxy S25 (Family)")
    target_runtime = os.getenv("QAIHUB_TARGET_RUNTIME", "tflite")

    model = build_placeholder_torch_model()
    device = hub.Device(device_name)

    compile_job = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=dict(image=input_shape),
        options=f"--target_runtime {target_runtime}",
    )
    target_model = compile_job.get_target_model()

    sample_image_url = (
        "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
    )
    input_array = prepare_input(sample_image_url)

    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs=dict(image=[input_array]),
    )
    on_device_output = inference_job.download_output_data()

    output_name = list(on_device_output.keys())[0]
    out = on_device_output[output_name][0]
    probabilities = np.exp(out) / np.sum(np.exp(out), axis=1)

    print("On-device output shape:", out.shape)
    print("Class probabilities:", probabilities)

    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print("Compile job:", compile_job.url)
    print("Inference job:", inference_job.url)
    print("Profile job:", profile_job.url)


if __name__ == "__main__":
    main()
