# Local LLM Server - Ollama + Qwen 3 8B

FastAPI server for medical diagnosis with dynamic system prompts. Runs entirely locally using Ollama.

## Features

- 🤖 **Qwen 3 8B via Ollama** - Running locally, no API calls
- 🏥 **Medical Diagnosis** - Dynamic system prompts based on skin condition classification
- 💬 **Multi-turn Chat** - Follow-up questions and conversation history
- 🔒 **Privacy** - Everything runs on your machine
- ⚡ **Fast** - Optimized for local inference

## Quick Start

### 1. Install Ollama

Download from: https://ollama.ai

### 2. Setup

```bash
cd Local_llm

# Run setup (pulls model, installs dependencies)
chmod +x setup.sh
./setup.sh

# This will:
# - Install Ollama if needed
# - Pull qwen:8b model (~5GB)
# - Install Python dependencies
```

### 3. Start Services

**Terminal 1: Ollama Server**
```bash
ollama serve
```

Expected output:
```
2026/02/21 12:00:00 listening on 127.0.0.1:11434
```

**Terminal 2: LLM API Server**
```bash
cd Local_llm
python main.py
```

Expected output:
```
🚀 Starting Local LLM Server on port 8001
📦 Using model: qwen:8b
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 4. Test

**Terminal 3: Run Client**
```bash
cd Local_llm

# Get diagnosis
python client.py

# Multi-turn chat
python client.py chat
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "model": "qwen:8b",
  "ollama_url": "http://localhost:11434"
}
```

### Get Diagnosis

```bash
curl -X POST http://localhost:8001/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "classification": "Melanoma",
    "confidence": 0.92,
    "body_part": "arm",
    "duration": "2 weeks",
    "description": "Dark, irregularly shaped mole"
  }'
```

Response:
```json
{
  "diagnosis": "Based on the classification results...",
  "recommendations": "Recommended next steps...",
  "follow_up_questions": [
    "How long have you noticed any bleeding or oozing?",
    "On a scale of 1-10, how much does it itch?"
  ]
}
```

### Chat

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a dermatology assistant."
      },
      {
        "role": "user",
        "content": "What should I do about this melanoma?"
      }
    ],
    "temperature": 0.7
  }'
```

Response:
```json
{
  "role": "assistant",
  "content": "For a melanoma diagnosis, immediate action is important..."
}
```

## Python Client

Use the built-in client for easy integration:

```python
from client import LLMClient

client = LLMClient("http://localhost:8001")

# Get diagnosis
result = client.diagnose(
    classification="Melanoma",
    confidence=0.92,
    body_part="arm",
    duration="2 weeks"
)

print(result["diagnosis"])
print(result["recommendations"])

# Chat
messages = [
    {"role": "system", "content": "You are a dermatology expert."},
    {"role": "user", "content": "What are treatment options?"}
]

response = client.chat(messages)
print(response["content"])
```

## Dynamic System Prompts

The system automatically builds prompts based on input:

```python
classification: "Melanoma"
confidence: 0.92
body_part: "arm"
duration: "2 weeks"
description: "Dark, irregular mole"

# System Prompt Generated:
"""
You are an expert dermatology assistant. Analyze the following skin condition...
- Condition: Melanoma
- Confidence: 92.0%
- Location: arm
- Duration: 2 weeks
- Patient Description: Dark, irregular mole
"""
```

## Configuration

Set environment variables:

```bash
# Ollama server URL
export OLLAMA_BASE_URL="http://localhost:11434"

# LLM port
export LLM_PORT="8001"
```

Or edit in `main.py`:
```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "qwen:8b"  # Change model here
LLM_PORT = int(os.getenv("LLM_PORT", 8001))
```

## Troubleshooting

### Ollama not responding
```bash
# Start Ollama
ollama serve

# Verify connection
curl http://localhost:11434/api/tags
```

### Model not found
```bash
# Pull the model
ollama pull qwen:8b

# Verify
ollama list
```

### LLM server won't start
```bash
# Check Python environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with debug
python -u main.py
```

### Slow responses
- First request loads model into memory (~30s)
- Subsequent requests are faster
- Check system RAM: `free -h`
- Lower `max_tokens` in requests for faster responses

## Model Options

Change the model in `main.py`:

```python
# Lightweight
MODEL_NAME = "qwen:0.5b"  # 512MB

# Balanced
MODEL_NAME = "qwen:8b"    # 5GB (default)

# Powerful
MODEL_NAME = "qwen:14b"   # 9GB

# Other models
MODEL_NAME = "llama2:7b"
MODEL_NAME = "neural-chat"
```

Then pull and use:
```bash
ollama pull qwen:14b
export LLM_PORT=8001
python main.py
```

## Integration with Classifier

Combine with skin cancer classifier:

```python
# 1. Classify image
from Backend.classifier.client import SkinCancerAPIClient
classifier = SkinCancerAPIClient()
result = classifier.classify_image("image.jpg")

# 2. Get diagnosis
from Local_llm.client import LLMClient
llm = LLMClient()
diagnosis = llm.diagnose(
    classification=result["top_prediction"],
    confidence=result["confidence"],
    body_part="arm"
)

print(diagnosis["diagnosis"])
```

## Performance

Typical response times (Qwen 3 8B):

| Operation | Time |
|-----------|------|
| First diagnosis | 15-30s (model loading) |
| Subsequent diagnosis | 5-10s |
| Chat response | 3-8s |
| Long diagnosis | 15-20s |

Optimize:
```python
# Reduce context
messages = messages[-3:]  # Keep last 3 messages

# Lower max_tokens
{"max_tokens": 200}

# Increase temperature for creativity
{"temperature": 0.9}
```

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `requests` - HTTP client
- `pydantic` - Data validation
- `ollama` - CLI tool (installed separately)

## Files

- `main.py` - FastAPI server
- `client.py` - Python client + examples
- `requirements.txt` - Python dependencies
- `setup.sh` - Automated setup script

## Development

Add custom endpoints:

```python
@app.post("/custom")
async def custom_endpoint(request: dict):
    # Your logic here
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": "...",
            "stream": False
        }
    )
    return response.json()
```

## License

MIT

## Support

For issues:
1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Check model is installed: `ollama list`
3. View server logs: Check terminal where `python main.py` runs
4. Check client logs: Add `print()` or logging to `client.py`
