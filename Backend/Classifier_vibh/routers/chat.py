from fastapi import APIRouter, HTTPException
from models.requests import ChatRequest
from models.responses import ChatResponse
import services.llm_service as llm_service
import state

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle a user follow-up message after a diagnosis has been generated.

    Appends the user message to the conversation history, calls the LLM
    with the full history (system prompt + all prior turns), and returns
    the assistant's response.

    Requires /api/diagnose to have been called first.
    """
    if state.session["full_prompt"] is None:
        raise HTTPException(
            status_code=400,
            detail="No active diagnosis session. Call /api/body-part and /api/diagnose first.",
        )

    state.session["conversation"].append({"role": "user", "content": request.message})

    response = await llm_service.call_llm(
        state.session["full_prompt"],
        state.session["conversation"],
    )

    state.session["conversation"].append({"role": "assistant", "content": response})

    return ChatResponse(response=response)
