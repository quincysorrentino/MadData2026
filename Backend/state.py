"""
In-memory session state shared across all routers.

State resets automatically when:
  - The server restarts
  - reset_state() is called (triggered by a new /api/body-part request)

conversation holds only user/assistant messages. The system prompt (full_prompt)
is stored separately and prepended by llm_service.call_llm() on every LLM call.
"""

session: dict = {
    "body_part_name": None,  # e.g. "Arm/Hand"
    "base_prompt":    None,  # system prompt before classification is appended
    "full_prompt":    None,  # base_prompt + classification (set after /api/diagnose)
    "bounding_box":   None,  # {"x": int, "y": int, "w": int, "h": int}
    "conversation":   [],    # list of {"role": "user"|"assistant", "content": str}
}


def reset_state() -> None:
    """Clear all session state back to defaults."""
    session["body_part_name"] = None
    session["base_prompt"]    = None
    session["full_prompt"]    = None
    session["bounding_box"]   = None
    session["conversation"]   = []
