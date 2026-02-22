from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForImageClassification


def upsert_env_value(env_path: Path, key: str, value: str) -> None:
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated = False
    output: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            output.append(f"{key}={value}")
            updated = True
        else:
            output.append(line)

    if not updated:
        output.append(f"{key}={value}")

    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def sanitize_label(label: str) -> str:
    return label.replace(",", " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HF classifier snapshot to ONNX and update .env")
    parser.add_argument(
        "--model-dir",
        default="models/Anwarkh1__Skin_Cancer-Image_Classification",
        help="Path to local Hugging Face classifier snapshot directory",
    )
    parser.add_argument(
        "--onnx-out",
        default="models/classifier.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size used for export",
    )
    parser.add_argument(
        "--require-npu",
        action="store_true",
        help="Set REQUIRE_NPU=true in .env after export",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    load_dotenv(env_path)

    model_dir = (project_root / args.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    onnx_out = (project_root / args.onnx_out).resolve()
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading classifier from: {model_dir}")
    model = AutoModelForImageClassification.from_pretrained(str(model_dir)).eval()

    input_shape = (1, 3, args.input_size, args.input_size)
    dummy_input = torch.randn(input_shape, dtype=torch.float32)

    print(f"Exporting ONNX to: {onnx_out}")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_out),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )

    id2label = getattr(model.config, "id2label", {})
    labels: list[str] = []
    if isinstance(id2label, dict) and id2label:
        for key in sorted(id2label.keys(), key=lambda item: int(item)):
            labels.append(sanitize_label(str(id2label[key])))

    relative_onnx = onnx_out.relative_to(project_root).as_posix()
    upsert_env_value(env_path, "MODEL_PATH", relative_onnx)
    upsert_env_value(env_path, "MODEL_HF_DIR", "")
    upsert_env_value(env_path, "REQUIRE_EXTERNAL_MODEL", "true")

    if args.require_npu:
        upsert_env_value(env_path, "REQUIRE_NPU", "true")
        upsert_env_value(env_path, "ORT_ENABLE_CPU_FALLBACK", "false")

    if labels:
        upsert_env_value(env_path, "MODEL_LABELS", ",".join(labels))

    print(f"Updated .env MODEL_PATH={relative_onnx}")
    print("Updated .env MODEL_HF_DIR=")
    print("Updated .env REQUIRE_EXTERNAL_MODEL=true")
    if args.require_npu:
        print("Updated .env REQUIRE_NPU=true")
        print("Updated .env ORT_ENABLE_CPU_FALLBACK=false")
    if labels:
        print(f"Updated .env MODEL_LABELS with {len(labels)} labels")


if __name__ == "__main__":
    main()
