from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class ShifaaUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = block(1, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.enc4 = block(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = block(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = block(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = block(32, 16)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Shifaa segmenter .pth to ONNX")
    parser.add_argument(
        "--input",
        default="models/Ahmed-Selem__Shifaa-Skin-Cancer-UNet-Segmentation/Shifaa-Skin-Cancer-UNet-Segmentation.pth",
        help="Path to .pth segmenter weights",
    )
    parser.add_argument(
        "--output",
        default="models/Ahmed-Selem__Shifaa-Skin-Cancer-UNet-Segmentation/segmenter.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="Input size for dummy tensor",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    model_path = (project_root / args.input).resolve()
    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Segmenter model not found: {model_path}")

    model = ShifaaUNet()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 1, args.input_size, args.input_size)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"Exported segmenter ONNX: {output_path}")


if __name__ == "__main__":
    main()
