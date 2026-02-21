from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from .schemas import HealthResponse, InferenceResponse
from .service import model

app = FastAPI(title="Skin Cancer Inference API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/v1/infer", response_model=InferenceResponse)
@app.post("/infer", response_model=InferenceResponse)
async def infer(file: UploadFile = File(...)) -> InferenceResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    label, confidence, model_version, boxes = model.predict(image_bytes)
    return InferenceResponse(
        label=label,
        confidence=confidence,
        boxes=boxes,
        model_version=model_version,
    )


@app.post("/v1/infer/debug")
@app.post("/infer/debug")
async def infer_debug(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    return model.predict_detailed(image_bytes)
