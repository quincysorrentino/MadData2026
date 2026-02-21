from fastapi import APIRouter, HTTPException, UploadFile, File
from models.responses import DiagnosisResponse, BoundingBox
import services.classifier_service as classifier_service
import services.prompt_service as prompt_service
import services.llm_service as llm_service
import state

router = APIRouter()


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(image: UploadFile = File(...)) -> DiagnosisResponse:
    """
    Receive an uploaded skin image, classify it, generate an LLM diagnosis,
    and return the diagnosis text along with the bounding box of the lesion.

    Requires /api/body-part to have been called first to set the system prompt.
    """
    if state.session["base_prompt"] is None:
        raise HTTPException(
            status_code=400,
            detail="No body part selected. Call /api/body-part before uploading an image.",
        )

    image_bytes = await image.read()
    classification, bounding_box_dict = await classifier_service.classify_image(
        image_bytes, image.filename or "image"
    )

    state.session["bounding_box"] = bounding_box_dict

    full_prompt = prompt_service.build_full_prompt(
        state.session["base_prompt"], classification
    )
    state.session["full_prompt"] = full_prompt
    state.session["conversation"] = []

    diagnosis = await llm_service.call_llm(full_prompt, state.session["conversation"])

    state.session["conversation"].append({"role": "assistant", "content": diagnosis})

    return DiagnosisResponse(
        diagnosis=diagnosis,
        bounding_box=BoundingBox(**bounding_box_dict),
    )
