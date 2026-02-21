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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download classifier and segmenter models from Hugging Face")
    parser.add_argument(
        "--segmenter",
        default="Ahmed-Selem/Shifaa-Skin-Cancer-UNet-Segmentation",
        help="Hugging Face repo id for segmentation model",
    )
    parser.add_argument(
        "--classifier",
        default="",
        help="Hugging Face repo id for classifier model (optional)",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Directory to store downloaded model snapshots",
    )

    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading segmenter repo: {args.segmenter}")
    segmenter_path = download_repo(args.segmenter, output_dir)
    print(f"Segmenter saved at: {segmenter_path}")

    if args.classifier:
        print(f"Downloading classifier repo: {args.classifier}")
        classifier_path = download_repo(args.classifier, output_dir)
        print(f"Classifier saved at: {classifier_path}")
    else:
        print("Classifier not provided; skipping classifier download.")

    print("\nNext: inspect downloaded files and set .env MODEL_PATH / SEGMENTER_MODEL_PATH")


if __name__ == "__main__":
    main()
