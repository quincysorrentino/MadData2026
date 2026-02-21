from pydantic import BaseModel, Field


class BodyPartRequest(BaseModel):
    body_part_id: int = Field(..., description="ID of the selected body part (see system_prompts.json)")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's follow-up message after receiving a diagnosis")
