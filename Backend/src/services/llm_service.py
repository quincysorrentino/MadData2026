import httpx
import os
from fastapi import HTTPException
from config import LLM_API_URL, LLM_MODEL


async def call_llm(system_prompt: str, conversation: list[dict]) -> str:
    """
    Call the locally-running Mistral LLM via Nexa AI's OpenAI-compatible API.

    Prepends the system prompt as the first message on every call — this is
    required because the REST API is stateless. From the caller's perspective
    the system prompt is a one-time setting; this function handles the detail.

    Args:
        system_prompt: The full assembled system prompt for this session.
        conversation:  List of {"role": "user"|"assistant", "content": str} dicts.
                       Must NOT include a system message — this function adds it.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        HTTPException 503 if the Nexa AI server is unreachable.
        HTTPException 502 if the server returns an unexpected response.
    """
    if os.getenv("LLM_STUB_MODE", "false").lower() in {"1", "true", "yes", "on"}:
        latest_user_message = ""
        for message in reversed(conversation):
            if message.get("role") == "user":
                latest_user_message = message.get("content", "")
                break

        if latest_user_message:
            return (
                "Stub response (LLM_STUB_MODE enabled): "
                f"received your message '{latest_user_message}'."
            )
        return "Stub response (LLM_STUB_MODE enabled): diagnosis generated successfully."

    messages = [{"role": "system", "content": system_prompt}] + conversation

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LLM_API_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                },
                timeout=60.0,
            )
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to the LLM server at {LLM_API_URL}. "
                   "Ensure the Nexa AI / Qualcomm AI Server is running.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM server returned an error: {e.response.status_code} {e.response.text}",
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected response format from LLM server: {data}",
        )
