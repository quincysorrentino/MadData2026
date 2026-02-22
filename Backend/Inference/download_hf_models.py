from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_repo(repo_id: str, output_dir: Path) -> Path:
    target_dir = output_dir / repo_id.replace("/", "__")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def upsert_env_value(env_path: Path, key: str, value: str) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated = False
    result_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            result_lines.append(f"{key}={value}")
            updated = True
        else:
            result_lines.append(line)

    if not updated:
        result_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")


def find_first_file(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    candidates = sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]
    )
    if not candidates:
        return None
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download classifier and segmenter models from Hugging Face")
    parser.add_argument(
        "--segmenter",
        default="Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation",
        help="Hugging Face repo id for segmentation model",
    )
    parser.add_argument(
        "--classifier",
        default="Anwarkh1/Skin_Cancer-Image_Classification",
        help="Hugging Face repo id for classifier model (defaults to Anwarkh1/Skin_Cancer-Image_Classification)",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Directory to store downloaded model snapshots",
    )

    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"

    print(f"Downloading segmenter repo: {args.segmenter}")
    segmenter_path = download_repo(args.segmenter, output_dir)
    print(f"Segmenter saved at: {segmenter_path}")

    segmenter_weight = find_first_file(segmenter_path, (".onnx", ".pth", ".pt", ".jit", ".torchscript"))
    if segmenter_weight is not None:
        relative_segmenter_weight = segmenter_weight.relative_to(project_root)
        upsert_env_value(env_path, "SEGMENTER_MODEL_PATH", str(relative_segmenter_weight).replace("\\", "/"))
        print(f"Updated .env SEGMENTER_MODEL_PATH={relative_segmenter_weight}")

    if args.classifier:
        print(f"Downloading classifier repo: {args.classifier}")
        classifier_path = download_repo(args.classifier, output_dir)
        print(f"Classifier saved at: {classifier_path}")

        classifier_onnx = find_first_file(classifier_path, (".onnx",))
        if classifier_onnx is not None:
            relative_classifier_onnx = classifier_onnx.relative_to(project_root)
            upsert_env_value(env_path, "MODEL_PATH", str(relative_classifier_onnx).replace("\\", "/"))
            upsert_env_value(env_path, "MODEL_HF_DIR", "")
            print(f"Updated .env MODEL_PATH={relative_classifier_onnx}")
            print("Cleared .env MODEL_HF_DIR (using MODEL_PATH)")
        else:
            relative_classifier_dir = classifier_path.relative_to(project_root)
            upsert_env_value(env_path, "MODEL_HF_DIR", str(relative_classifier_dir).replace("\\", "/"))
            upsert_env_value(env_path, "MODEL_PATH", "")
            print(f"Updated .env MODEL_HF_DIR={relative_classifier_dir}")
            print("Cleared .env MODEL_PATH (no classifier .onnx found)")
    else:
        print("Classifier explicitly disabled; skipping classifier download.")

    print("\nNext: inspect downloaded files and set .env MODEL_PATH / SEGMENTER_MODEL_PATH")


if __name__ == "__main__":
    main()
