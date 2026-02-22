from pydantic import BaseModel, Field


class BodyPartRequest(BaseModel):
    body_part_id: int = Field(..., description="ID of the selected body part (see system_prompts.json)")


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's follow-up message after receiving a diagnosis",
    )
