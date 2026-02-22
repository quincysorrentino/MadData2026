"""
Local LLM Server using Ollama + Qwen 3 8B

Purpose:
--------
This service exposes a REST API that wraps a locally hosted Ollama model
(Qwen 3 8B) for medical diagnosis assistance and conversational follow-ups.

Architecture:
-------------
Client → FastAPI → Ollama (localhost:11434) → Qwen 3 8B → Response

The server dynamically constructs system prompts for structured
medical guidance based on classification results from an upstream model.
"""

import os
import requests
import json
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging


# -------------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------------
# Centralized logging enables visibility into:
# - Startup validation
# - Model availability
# - Runtime errors
# - External API failures
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------------
# These values allow flexible deployment across:
# - Local development
# - Docker containers
# - Cloud instances
#
# OLLAMA_BASE_URL:
#     URL where Ollama server is running
#
# MODEL_NAME:
#     Target model served by Ollama
#
# LLM_PORT:
#     Port where this FastAPI server runs
# -------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "qwen3:8b"  # Qwen 3 8B model served by Ollama
LLM_PORT = int(os.getenv("LLM_PORT", 9000))


# -------------------------------------------------------------------------
# FastAPI Application Initialization
# -------------------------------------------------------------------------
# Provides:
# - Automatic OpenAPI documentation
# - Request validation via Pydantic
# - Structured error handling
# -------------------------------------------------------------------------
app = FastAPI(
    title="Local LLM Diagnosis Server",
    description="Ollama-powered Qwen 3 8B for medical diagnosis with dynamic prompts",
    version="1.0.0"
)


# -------------------------------------------------------------------------
# Data Models (Request / Response Schemas)
# -------------------------------------------------------------------------

class DiagnosisRequest(BaseModel):
    """
    Input payload for diagnosis endpoint.

    Fields:
    -------
    classification : str
        Predicted condition (e.g., "Melanoma")

    confidence : float
        Model confidence score (0-1)

    body_part : str
        Location of condition

    duration : Optional[str]
        How long condition has existed

    description : Optional[str]
        Additional patient-provided details

    user_question : Optional[str]
        Optional follow-up question
    """
    classification: str
    confidence: float
    body_part: str
    duration: Optional[str] = None
    description: Optional[str] = None
    user_question: Optional[str] = None


class ChatMessage(BaseModel):
    """
    Represents a single chat message.

    role:
        "system" | "user" | "assistant"

    content:
        Natural language message content
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """
    Chat endpoint payload.

    messages:
        Ordered list of conversation messages

    temperature:
        Controls randomness of generation (0-1)

    max_tokens:
        Maximum response length
    """
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 500


class DiagnosisResponse(BaseModel):
    """
    Structured response returned to client.

    diagnosis:
        Core medical explanation

    recommendations:
        Suggested next actions

    follow_up_questions:
        Clarifying questions to ask patient
    """
    diagnosis: str
    recommendations: str
    follow_up_questions: List[str]


# -------------------------------------------------------------------------
# Prompt Construction Logic
# -------------------------------------------------------------------------
# This function dynamically builds a structured system prompt
# tailored to the classification result.
#
# The LLM is instructed to:
# - Provide assessment
# - Offer recommendations
# - Highlight risks
# - Suggest when urgent care is needed
#
# The design ensures consistent formatting for downstream parsing.
# -------------------------------------------------------------------------
def build_diagnosis_prompt(
    classification: str,
    confidence: float,
    body_part: str,
    duration: Optional[str] = None,
    description: Optional[str] = None
) -> str:

    confidence_pct = f"{confidence * 100:.1f}%"

    prompt = f"""You are an expert dermatology assistant. Analyze the following skin condition classification and provide professional medical guidance.

CLASSIFICATION RESULT:
- Condition: {classification}
- Confidence: {confidence_pct}
- Location: {body_part}
"""

    if duration:
        prompt += f"- Duration: {duration}\n"

    if description:
        prompt += f"- Patient Description: {description}\n"

    prompt += """
PROVIDE:
1. Professional assessment of the condition
2. Recommended next steps (consultation, monitoring, treatment options)
3. Risk factors to watch for
4. When to seek immediate medical attention

Keep response concise and professional. Always recommend consulting a dermatologist for definitive diagnosis and treatment."""

    return prompt


# -------------------------------------------------------------------------
# Startup Event
# -------------------------------------------------------------------------
# On server start:
# - Verify Ollama is reachable
# - Confirm target model is available
# - Log warnings if model is missing
# -------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)

        if response.status_code == 200:
            logger.info("Ollama connected")

            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]

            if any(MODEL_NAME in name for name in model_names):
                logger.info(f"Model {MODEL_NAME} available")
            else:
                logger.warning(f"Model {MODEL_NAME} not found. Run: ollama pull {MODEL_NAME}")
        else:
            logger.error("Ollama not responding")

    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")


# -------------------------------------------------------------------------
# Health Endpoint
# -------------------------------------------------------------------------
# Simple service verification endpoint.
# Used for:
# - Load balancers
# - Monitoring systems
# - Dev environment checks
# -------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_BASE_URL
    }


# -------------------------------------------------------------------------
# Diagnosis Endpoint
# -------------------------------------------------------------------------
# Workflow:
# 1. Construct dynamic system prompt
# 2. Build user message
# 3. Send request to Ollama /api/generate
# 4. Parse structured sections
# 5. Return normalized response
# -------------------------------------------------------------------------
@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):

    try:
        system_prompt = build_diagnosis_prompt(
            request.classification,
            request.confidence,
            request.body_part,
            request.duration,
            request.description
        )

        user_message = (
            f"Please provide a professional assessment of "
            f"this {request.classification} diagnosis found on the {request.body_part}."
        )

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": f"{system_prompt}\n\nUser: {user_message}",
                "stream": False,
                "temperature": 0.7
            },
            timeout=120
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

        result = response.json()
        diagnosis_text = result.get("response", "").strip()

        sections = parse_diagnosis_response(diagnosis_text)

        return DiagnosisResponse(
            diagnosis=sections.get("diagnosis", diagnosis_text),
            recommendations=sections.get("recommendations", ""),
            follow_up_questions=sections.get("questions", [])
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not running. Start with: ollama serve"
        )

    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# Chat Endpoint
# -------------------------------------------------------------------------
# Converts structured message history into a flat prompt
# using role prefixes.
#
# This is a lightweight chat abstraction over Ollama's generate API.
# -------------------------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):

    try:
        prompt = ""

        for msg in request.messages:
            prompt += f"{msg.role.capitalize()}: {msg.content}\n"

        prompt += "Assistant:"

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": request.temperature
            },
            timeout=120
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

        result = response.json()

        return {
            "role": "assistant",
            "content": result.get("response", "").strip()
        }

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not running. Start with: ollama serve"
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# Response Parsing
# -------------------------------------------------------------------------
# Attempts heuristic extraction of:
# - Diagnosis section
# - Recommendations
# - Follow-up questions
#
# This is text-based parsing and assumes the LLM follows structure.
# For production-grade systems, consider structured prompting with JSON.
# -------------------------------------------------------------------------
def parse_diagnosis_response(text: str) -> Dict:

    sections = {
        "diagnosis": text,
        "recommendations": "",
        "questions": []
    }

    lines = text.split("\n")
    current_section = "diagnosis"

    for line in lines:
        line_lower = line.lower()

        if "recommend" in line_lower or "next step" in line_lower:
            current_section = "recommendations"

        elif "question" in line_lower or "ask" in line_lower:
            current_section = "questions"

        elif current_section == "recommendations" and line.strip():
            sections["recommendations"] += line + "\n"

        elif (
            current_section == "questions"
            and line.strip()
            and line.startswith(("-", "•", "*", "1", "2"))
        ):
            sections["questions"].append(line.strip())

    return sections


# -------------------------------------------------------------------------
# Model Management Endpoints
# -------------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """Return models currently available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)

        if response.status_code == 200:
            return response.json()

        raise HTTPException(status_code=500, detail="Cannot reach Ollama")

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not available: {e}")


@app.post("/pull-model")
async def pull_model(model: str = MODEL_NAME):
    """
    Pull model from Ollama registry.

    Uses extended timeout because model downloads can be large.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model},
            timeout=600
        )

        if response.status_code == 200:
            return {"status": "success", "message": f"Model {model} pulled successfully"}

        raise HTTPException(status_code=500, detail=f"Failed to pull model: {response.text}")

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error pulling model: {e}")


