from fastapi import APIRouter, HTTPException
from models.requests import BodyPartRequest
from models.responses import BodyPartResponse
import services.prompt_service as prompt_service
import state

router = APIRouter()


@router.post("/body-part", response_model=BodyPartResponse)
async def select_body_part(request: BodyPartRequest) -> BodyPartResponse:
    """
    Receive the body part selected by the user on the frontend.

    Loads the corresponding system prompt, resets all prior session state
    (clears conversation history, bounding box, etc.), and saves the
    body part name and base prompt for use by the /diagnose endpoint.
    """
    try:
        name, prompt = prompt_service.get_prompt_for_body_part(request.body_part_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    state.reset_state()
    state.session["body_part_name"] = name
    state.session["base_prompt"] = prompt

    return BodyPartResponse(success=True, body_part_name=name)
