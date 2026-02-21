import httpx
from config import CLASSIFIER_API_URL


def parse_classifier_response(raw: dict) -> tuple[str, dict]:
    """
    Extract classification label and bounding box from the classifier API response.

    TODO: Update this function once the classifier API response format is finalised.
          The current implementation assumes this nested structure:
          {
            "results": {
              "classification": "Melanoma",
              "bounding_box": {
                "x": 120,
                "y": 85,
                "w": 60,
                "h": 45
              }
            }
          }
          Adjust the key paths below to match the actual response schema.

    Returns:
        (classification: str, bounding_box: dict)
    """
    results = raw.get("results", {})
    classification: str = results.get("classification", "Unknown")
    bounding_box: dict = results.get("bounding_box", {"x": 0, "y": 0, "w": 0, "h": 0})
    return classification, bounding_box


async def classify_image(image_bytes: bytes, filename: str) -> tuple[str, dict]:
    """
    Send an image to the classifier API and return (classification, bounding_box).

    When the classifier API is ready:
      1. Set CLASSIFIER_API_URL in config.py
      2. Remove the STUB block and uncomment the real HTTP call below
    """

    # --- STUB: remove this block when the classifier API is ready ---
    mock_response = {
        "results": {
            "classification": "Melanoma (stub)",
            "bounding_box": {"x": 100, "y": 80, "w": 60, "h": 50}
        }
    }
    return parse_classifier_response(mock_response)
    # ----------------------------------------------------------------

    # Real HTTP call (uncomment when classifier API is ready):
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         f"{CLASSIFIER_API_URL}/classify",
    #         files={"image": (filename, image_bytes)},
    #         timeout=30.0,
    #     )
    #     response.raise_for_status()
    #     return parse_classifier_response(response.json())
