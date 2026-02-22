"""
Local LLM Server using Ollama + Qwen 3 8B
Dynamic system prompts for medical diagnosis context
"""

import os
import requests
import httpx
import json
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "qwen3:8b"  # Qwen 3 8B
LLM_PORT = int(os.getenv("LLM_PORT", 9000))

# FastAPI app
app = FastAPI(
    title="Local LLM Diagnosis Server",
    description="Ollama-powered Qwen 3 8B for medical diagnosis with dynamic prompts",
    version="1.0.0"
)


class DiagnosisRequest(BaseModel):
    """Request for diagnosis"""
    classification: str  # e.g., "Melanoma"
    confidence: float  # 0-1
    body_part: str  # e.g., "arm"
    duration: Optional[str] = None  # e.g., "2 weeks"
    description: Optional[str] = None  # User description
    user_question: Optional[str] = None  # Follow-up question


class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    """Chat request"""
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 500


class DiagnosisResponse(BaseModel):
    """Diagnosis response"""
    diagnosis: str
    recommendations: str
    follow_up_questions: List[str]


# --- OpenAI-compatible /v1/chat/completions models ---

class OpenAIChatMessage(BaseModel):
    role: str      # "system", "user", or "assistant"
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[OpenAIChatMessage]
    temperature: float = 0.6
    max_tokens: Optional[int] = None


def build_diagnosis_prompt(
    classification: str,
    confidence: float,
    body_part: str,
    duration: Optional[str] = None,
    description: Optional[str] = None
) -> str:
    """Build dynamic system prompt for diagnosis"""
    
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


@app.on_event("startup")
async def startup():
    """Check Ollama connection and model on startup"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama connected")
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            if any(MODEL_NAME in name for name in model_names):
                logger.info(f"✅ Model {MODEL_NAME} available")
            else:
                logger.warning(f"⚠️ Model {MODEL_NAME} not found. Pull it with: ollama pull {MODEL_NAME}")
        else:
            logger.error("❌ Ollama not responding")
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama: {e}")


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_BASE_URL
    }


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """
    Get diagnosis based on classification
    
    Args:
        classification: Skin condition class (e.g., "Melanoma")
        confidence: Confidence score 0-1
        body_part: Location on body
        duration: How long patient has had it
        description: Patient's own description
    
    Returns:
        Diagnosis, recommendations, and follow-up questions
    """
    
    try:
        # Build system prompt
        system_prompt = build_diagnosis_prompt(
            request.classification,
            request.confidence,
            request.body_part,
            request.duration,
            request.description
        )
        
        # Create user message
        user_message = f"Please provide a professional assessment of this {request.classification} diagnosis found on the {request.body_part}."
        
        # Call Ollama
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
        
        # Parse response into sections
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


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint for follow-up questions
    
    Args:
        messages: List of chat messages (role + content)
        temperature: Response creativity (0-1)
        max_tokens: Max response length
    
    Returns:
        Assistant response
    """
    
    try:
        # Build prompt from message history
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        
        prompt += "Assistant:"
        
        # Call Ollama
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


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    """
    OpenAI-compatible endpoint consumed by src/services/llm_service.py.
    Translates to Ollama /api/chat and wraps response into OpenAI choices shape.
    """
    payload = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": False,
        "options": {"temperature": request.temperature},
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not running. Start with: ollama serve",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {exc.response.status_code} {exc.response.text}",
        )

    data = response.json()
    try:
        content = data["message"]["content"]
    except (KeyError, TypeError):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Ollama response format: {data}",
        )

    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}}
        ]
    }


def parse_diagnosis_response(text: str) -> Dict:
    """
    Parse diagnosis response into structured sections
    
    Args:
        text: Raw diagnosis response text
    
    Returns:
        Dict with diagnosis, recommendations, questions
    """
    
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
        elif current_section == "questions" and line.strip() and line.startswith(("-", "•", "*", "1", "2")):
            sections["questions"].append(line.strip())
    
    return sections


@app.get("/models")
async def list_models():
    """List available models in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail="Cannot reach Ollama")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama not available: {e}")


@app.post("/pull-model")
async def pull_model(model: str = MODEL_NAME):
    """Pull a model from Ollama registry"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model},
            timeout=600  # 10 min timeout for download
        )
        
        if response.status_code == 200:
            return {"status": "success", "message": f"Model {model} pulled successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to pull model: {response.text}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error pulling model: {e}")


if __name__ == "__main__":
    logger.info(f"🚀 Starting Local LLM Server on port {LLM_PORT}")
    logger.info(f"📦 Using model: {MODEL_NAME}")
    logger.info(f"🔗 Ollama URL: {OLLAMA_BASE_URL}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=LLM_PORT,
        log_level="info"
    )
