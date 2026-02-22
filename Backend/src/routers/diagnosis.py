from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from models.responses import DiagnosisResponse, BoundingBox
import services.classifier_service as classifier_service
import services.prompt_service as prompt_service
import services.llm_service as llm_service
import state
from config import DIAGNOSE_MAX_IMAGE_BYTES, DIAGNOSE_MAX_IMAGE_MB

router = APIRouter()


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    image: UploadFile = File(...),
    body_part_name: str | None = Form(default=None),
) -> DiagnosisResponse:
    """
    Receive an uploaded skin image, classify it, generate an LLM diagnosis,
    and return the diagnosis text along with the bounding box of the lesion.

    If /api/body-part has not been called, an optional body_part_name form field
    can initialize session context for the LLM prompt.
    """
    if state.session["base_prompt"] is None:
        selected_body_part_name = body_part_name or state.session["body_part_name"] or "skin area"
        state.session["body_part_name"] = selected_body_part_name
        state.session["base_prompt"] = (
            "You are an expert dermatology assistant. Provide careful, non-diagnostic guidance, "
            "highlight safety red flags, and recommend in-person dermatology follow-up when needed. "
            f"The affected body area is: {selected_body_part_name}."
        )

    image_bytes = await image.read()
    if len(image_bytes) > DIAGNOSE_MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is too large. Maximum allowed size is {DIAGNOSE_MAX_IMAGE_MB} MB."
            ),
        )

    classification, bounding_box_dict = await classifier_service.classify_image(
        image_bytes, image.filename or "image"
    )

    print(f"[classifier] classification={classification!r}  bounding_box={bounding_box_dict}")

    state.session["bounding_box"] = bounding_box_dict

    full_prompt = prompt_service.build_full_prompt(state.session["base_prompt"], classification)
    state.session["full_prompt"] = full_prompt
    state.session["conversation"] = []

    classifier_output = {
        "classification": classification,
        "bounding_box": bounding_box_dict,
    }
    print(f"[diagnose->llm] classifier_output={classifier_output}")

    diagnosis = await llm_service.call_llm(full_prompt, state.session["conversation"])

    state.session["conversation"].append({"role": "assistant", "content": diagnosis})

    return DiagnosisResponse(
        diagnosis=diagnosis,
        bounding_box=BoundingBox(**bounding_box_dict),
    )
