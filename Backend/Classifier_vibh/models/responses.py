from pydantic import BaseModel


class BodyPartResponse(BaseModel):
    success: bool
    body_part_name: str


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DiagnosisResponse(BaseModel):
    diagnosis: str
    bounding_box: BoundingBox


class ChatResponse(BaseModel):
    response: str
