from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .schemas import HealthResponse, InferenceResponse
from .service import model

app = FastAPI(title="Skin Cancer Inference API", version="0.1.0")


async def _read_image_upload(
    file: UploadFile | None,
    image: UploadFile | None,
) -> bytes:
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="Missing upload. Use multipart field 'file' or 'image'.")

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        with Image.open(BytesIO(image_bytes)) as parsed_image:
            parsed_image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File must be a valid image")

    return image_bytes


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/v1/infer", response_model=InferenceResponse)
@app.post("/infer", response_model=InferenceResponse)
async def infer(
    file: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
) -> InferenceResponse:
    image_bytes = await _read_image_upload(file, image)

    label, confidence, model_version, boxes = model.predict(image_bytes)
    return InferenceResponse(
        label=label,
        confidence=confidence,
        boxes=boxes,
        model_version=model_version,
    )


@app.post("/v1/infer/debug")
@app.post("/infer/debug")
async def infer_debug(
    file: UploadFile | None = File(default=None),
    image: UploadFile | None = File(default=None),
) -> dict:
    image_bytes = await _read_image_upload(file, image)

    return model.predict_detailed(image_bytes)
