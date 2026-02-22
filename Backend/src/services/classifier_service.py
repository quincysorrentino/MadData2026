import httpx
from fastapi import HTTPException
from config import CLASSIFIER_API_URL


def parse_classifier_response(raw: dict) -> tuple[str, dict]:
    """
    Extract classification label and bounding box from the Inference API response.

    Inference API response shape:
    {
        "label": "suspicious",
        "confidence": 0.85,
        "model_version": "placeholder-v1",
        "boxes": [
            {"x_min": 50, "y_min": 30, "width": 120, "height": 90,
             "score": 0.78, "label": "lesion"}
        ]
    }

    Returns:
        (classification: str, bounding_box: dict)
    """
    label = raw.get("label", "unknown")
    confidence = raw.get("confidence", 0.0)
    classification = f"{label} (confidence: {round(confidence * 100)}%)"

    boxes = raw.get("boxes", [])
    if boxes:
        first = boxes[0]
        bounding_box = {
            "x": first.get("x_min", 0),
            "y": first.get("y_min", 0),
            "w": first.get("width", 0),
            "h": first.get("height", 0),
        }
    else:
        bounding_box = {"x": 0, "y": 0, "w": 0, "h": 0}

    return classification, bounding_box


async def classify_image(image_bytes: bytes, filename: str) -> tuple[str, dict]:
    """
    Send an image to the Inference classifier API and return (classification, bounding_box).

    Raises:
        HTTPException 503 if the Inference server is unreachable.
        HTTPException 502 if the Inference server returns an error response.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CLASSIFIER_API_URL}/infer",
                files={"file": (filename, image_bytes)},
                timeout=60.0,
            )
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to the classifier server at {CLASSIFIER_API_URL}.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Classifier server returned an error: {e.response.status_code} {e.response.text}",
        )

    raw_output = response.json()
    print(f"[classifier-api raw] {raw_output}")

    return parse_classifier_response(raw_output)
