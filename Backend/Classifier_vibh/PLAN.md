# Backend FastAPI Service — Skin Cancer Classifier

## Overview

Python FastAPI service that bridges the frontend, a skin cancer classifier API, and a locally-running Mistral LLM (via Nexa AI / Qualcomm AI Server). Three endpoints manage: body part selection, image-based diagnosis, and follow-up chat.

---

## Folder Structure

```
Backend/
├── main.py                    # FastAPI app, CORS config, router registration
├── config.py                  # All configurable values (API URLs, ports)
├── state.py                   # Shared in-memory session state (single dict)
├── routers/
│   ├── body_part.py           # POST /api/body-part
│   ├── diagnosis.py           # POST /api/diagnose
│   └── chat.py                # POST /api/chat
├── services/
│   ├── classifier_service.py  # Calls classifier API (STUB — swap when ready)
│   ├── llm_service.py         # Calls Nexa AI / Mistral API
│   └── prompt_service.py      # Loads system prompts, assembles full prompt
├── models/
│   ├── requests.py            # Pydantic request schemas
│   └── responses.py           # Pydantic response schemas
├── prompts/
│   └── system_prompts.json    # Body part id → system prompt mapping (team to refine)
├── requirements.txt
└── PLAN.md                    # This file
```

---

## API Endpoints

| Method | Path              | Body / Upload               | Response                                                  |
|--------|-------------------|-----------------------------|-----------------------------------------------------------|
| POST   | `/api/body-part`  | `{"body_part_id": 2}`       | `{"success": true, "body_part_name": "Neck"}`             |
| POST   | `/api/diagnose`   | multipart image file        | `{"diagnosis": "...", "bounding_box": {"x":0,"y":0,...}}` |
| POST   | `/api/chat`       | `{"message": "..."}`        | `{"response": "..."}`                                     |

---

## Endpoint Flow

### `POST /api/body-part`
1. Receive `body_part_id` (int) from frontend
2. Look up the corresponding system prompt from `system_prompts.json`
3. Reset all session state (clears prior conversation, bounding box, etc.)
4. Save the body part name and base system prompt to session
5. Return the body part name

### `POST /api/diagnose`
1. Guard: requires body part to have been selected first
2. Receive uploaded image from frontend
3. Forward image to **classifier API** → receive `classification` string + `bounding_box`
4. Save `bounding_box` to session (will be sent to frontend)
5. Append `classification` to the base system prompt → `full_prompt`
6. Save `full_prompt` to session
7. Call **LLM API** with `full_prompt` and empty conversation → receive `diagnosis`
8. Save diagnosis as first assistant message in conversation history
9. Return `diagnosis` + `bounding_box` to frontend

### `POST /api/chat`
1. Guard: requires a diagnosis to have been run first
2. Receive user message from frontend
3. Append user message to conversation history
4. Call **LLM API** with `full_prompt` (system) + full conversation history → receive `response`
5. Append response to conversation history
6. Return `response` to frontend

---

## Session State (In-Memory)

State resets on server restart or when a new body part is selected. There is no persistence.

```python
session = {
    "body_part_name": None,   # e.g. "Arm/Hand"
    "base_prompt":    None,   # system prompt before classification is appended
    "full_prompt":    None,   # base_prompt + classification (set after /diagnose)
    "bounding_box":   None,   # {"x": int, "y": int, "w": int, "h": int}
    "conversation":   [],     # only user/assistant messages (no system message)
}
```

The system prompt (`full_prompt`) is **not** stored in `conversation`. Instead, `llm_service.call_llm()` prepends it as a system message on every LLM call internally. This is necessary because the LLM REST API is stateless — every request must include the full context — but from the caller's perspective the system prompt behaves as a one-time setting.

---

## Integration Points (Stubs)

### Classifier API (`services/classifier_service.py`)
- **To integrate:** Set `CLASSIFIER_API_URL` in `config.py` and replace the mock return in `classify_image()` with the real HTTP call
- Expected request: `multipart/form-data` with image file
- Expected response:
  ```json
  {
    "classification": "Melanoma",
    "bounding_box": { "x": 120, "y": 85, "w": 60, "h": 45 }
  }
  ```

### LLM API (`services/llm_service.py`)
- **To integrate:** Set `LLM_API_URL` and `LLM_MODEL` in `config.py`
- Assumes OpenAI-compatible `/v1/chat/completions` endpoint (standard for Nexa AI)
- If the Nexa AI server uses a different format, only `llm_service.py` needs to change

---

## Configuration (`config.py`)

```python
CLASSIFIER_API_URL = "http://localhost:8001"   # classifier team's API
LLM_API_URL        = "http://localhost:8000"   # Nexa AI server
LLM_MODEL          = "mistral"
ALLOWED_ORIGINS    = ["http://localhost:3000"] # frontend dev server
```

---

## Running Locally

```bash
cd Backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

Visit `http://localhost:8080/docs` for the interactive Swagger UI — useful for testing all three endpoints before the frontend exists.

**Test sequence:**
1. `POST /api/body-part` with `{"body_part_id": 1}`
2. `POST /api/diagnose` with a multipart image upload
3. `POST /api/chat` with `{"message": "Can you explain the diagnosis?"}`

---

## Notes for Team Members

- **System prompt content** (`prompts/system_prompts.json`): Placeholder text is included. The prompts team should replace these with clinically appropriate text per body part.
- **Classifier integration**: Only `services/classifier_service.py` needs updating once the classifier API is ready.
- **LLM runtime**: Set `LLM_API_URL` and `LLM_MODEL` in `config.py` once the Nexa AI / Qualcomm AI Server details are confirmed.
- **CORS**: `ALLOWED_ORIGINS` in `config.py` controls which frontend URLs are allowed. Update for production.
