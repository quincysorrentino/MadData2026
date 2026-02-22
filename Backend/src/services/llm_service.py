import httpx
import os
import asyncio
from fastapi import HTTPException
from config import (
    LLM_API_URL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_TOKENS,
    LLM_CONNECT_TIMEOUT_SECONDS,
    LLM_POOL_TIMEOUT_SECONDS,
    LLM_CONVERSATION_MAX_TURNS,
    LLM_MAX_MESSAGE_CHARS,
    LLM_MAX_CONCURRENCY,
)


_timeout = httpx.Timeout(
    connect=float(LLM_CONNECT_TIMEOUT_SECONDS),
    read=float(LLM_TIMEOUT_SECONDS),
    write=float(LLM_TIMEOUT_SECONDS),
    pool=float(LLM_POOL_TIMEOUT_SECONDS),
)

_client = httpx.AsyncClient(
    timeout=_timeout,
    limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
)

_llm_semaphore = asyncio.Semaphore(max(1, int(LLM_MAX_CONCURRENCY)))


def _trim_content(value: str) -> str:
    if len(value) <= LLM_MAX_MESSAGE_CHARS:
        return value
    return value[:LLM_MAX_MESSAGE_CHARS]


def _build_messages(system_prompt: str, conversation: list[dict]) -> list[dict[str, str]]:
    trimmed_prompt = _trim_content(system_prompt)
    bounded_history = conversation[-(LLM_CONVERSATION_MAX_TURNS * 2):]

    sanitized_history: list[dict[str, str]] = []
    for message in bounded_history:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"user", "assistant"}:
            continue
        sanitized_history.append({"role": role, "content": _trim_content(str(content))})

    return [{"role": "system", "content": trimmed_prompt}, *sanitized_history]


async def call_llm(system_prompt: str, conversation: list[dict]) -> str:
    """
    Call the locally-running LLM via Ollama's OpenAI-compatible API.

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

    messages = _build_messages(system_prompt, conversation)

    try:
        async with _llm_semaphore:
            response = await _client.post(
                f"{LLM_API_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": LLM_MAX_TOKENS,
                },
            )
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to the LLM server at {LLM_API_URL}. "
                   "Ensure the Ollama server is running (ollama serve).",
        )
    except httpx.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail="LLM server timed out generating a response. The model may still be loading — try again in a moment.",
        )
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        if status_code in {503, 504}:
            raise HTTPException(
                status_code=status_code,
                detail=f"LLM server returned {status_code}: {e.response.text}",
            )
        raise HTTPException(
            status_code=502,
            detail=f"LLM server returned an error: {status_code} {e.response.text}",
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected response format from LLM server: {data}",
        )
