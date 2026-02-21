from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x_min: int = Field(ge=0)
    y_min: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)
    label: str = Field(default="lesion")


class InferenceResponse(BaseModel):
    label: str = Field(description="Predicted class label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    boxes: list[BoundingBox] = Field(default_factory=list, description="Detected lesion boxes")
    model_version: str = Field(description="Model or placeholder version used for inference")


class HealthResponse(BaseModel):
    status: str
