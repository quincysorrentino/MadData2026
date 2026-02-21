import json
import os
from pathlib import Path

_PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "system_prompts.json"

# Populated at startup by load_system_prompts()
_body_parts: dict[int, dict] = {}


def load_system_prompts() -> None:
    """Load system_prompts.json into memory. Called once at app startup."""
    global _body_parts
    with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _body_parts = {entry["id"]: entry for entry in data["body_parts"]}


def get_prompt_for_body_part(body_part_id: int) -> tuple[str, str]:
    """
    Return (name, prompt) for the given body_part_id.
    Raises ValueError if the id is not found.
    """
    entry = _body_parts.get(body_part_id)
    if entry is None:
        valid_ids = sorted(_body_parts.keys())
        raise ValueError(f"Unknown body_part_id {body_part_id}. Valid ids: {valid_ids}")
    return entry["name"], entry["prompt"]


def build_full_prompt(base_prompt: str, classification: str) -> str:
    """
    Append the classifier output to the base system prompt so the LLM
    knows what condition was detected before generating the diagnosis.
    """
    return (
        f"{base_prompt}\n\n"
        f"The AI skin cancer classifier has identified the following condition "
        f"in the uploaded image: {classification}. "
        f"Use this classification as the basis for your detailed assessment."
    )
