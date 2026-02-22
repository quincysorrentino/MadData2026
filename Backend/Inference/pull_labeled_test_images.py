from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

DX_TO_NAME = {
    "akiec": "actinic_keratoses",
    "bcc": "basal_cell_carcinoma",
    "bkl": "benign_keratosis_like_lesions",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic_nevi",
    "vasc": "vascular_lesions",
}

LABEL_TO_NAME = {
    0: "melanoma",
    1: "nevus",
}


def extract_image(image_value) -> Image.Image:
    if hasattr(image_value, "save"):
        return image_value
    if isinstance(image_value, dict):
        if "bytes" in image_value and image_value["bytes"] is not None:
            import io

            return Image.open(io.BytesIO(image_value["bytes"]))
        if "path" in image_value and image_value["path"]:
            return Image.open(image_value["path"])
    raise ValueError("Unsupported image field format in dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull labeled HAM10000 test images")
    parser.add_argument("--per-class", type=int, default=2, help="Images per class to export")
    parser.add_argument(
        "--out-dir",
        default="test_samples/ham10000",
        help="Output folder for exported images and manifest",
    )
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(
        repo_id="Nagabu/HAM10000",
        repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
    )
    df = pd.read_parquet(parquet_path)
    manifest = []

    if {"image", "dx", "image_id"}.issubset(df.columns):
        for dx_code, friendly_name in DX_TO_NAME.items():
            class_rows = df[df["dx"] == dx_code].head(args.per_class)
            class_dir = output_dir / dx_code
            class_dir.mkdir(parents=True, exist_ok=True)

            for _, row in class_rows.iterrows():
                image = extract_image(row["image"]).convert("RGB")
                image_id = str(row["image_id"])
                filename = f"{image_id}.jpg"
                path = class_dir / filename
                image.save(path, format="JPEG", quality=95)

                manifest.append(
                    {
                        "image_id": image_id,
                        "filename": str(path),
                        "dx": dx_code,
                        "label": friendly_name,
                    }
                )
    elif {"image", "label"}.issubset(df.columns):
        labels = sorted(df["label"].dropna().astype(int).unique().tolist())
        for label_id in labels:
            class_rows = df[df["label"] == label_id].head(args.per_class)
            class_name = LABEL_TO_NAME.get(label_id, f"class_{label_id}")
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for idx, row in class_rows.iterrows():
                image = extract_image(row["image"]).convert("RGB")
                image_id = f"label{label_id}_{idx}"
                filename = f"{image_id}.jpg"
                path = class_dir / filename
                image.save(path, format="JPEG", quality=95)

                manifest.append(
                    {
                        "image_id": image_id,
                        "filename": str(path),
                        "dx": f"label_{label_id}",
                        "label": class_name,
                    }
                )
    else:
        raise ValueError(
            "Unsupported dataset schema. Expected either columns "
            "['image', 'dx', 'image_id'] or ['image', 'label']."
        )

    manifest_csv = output_dir / "manifest.csv"
    manifest_json = output_dir / "manifest.json"

    with manifest_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "filename", "dx", "label"])
        writer.writeheader()
        writer.writerows(manifest)

    manifest_json.write_text(json.dumps(manifest, indent=2))

    print(f"Exported {len(manifest)} labeled images to {output_dir}")
    print(f"CSV manifest: {manifest_csv}")
    print(f"JSON manifest: {manifest_json}")


if __name__ == "__main__":
    main()
