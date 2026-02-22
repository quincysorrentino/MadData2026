#!/bin/bash
# Setup and run Ollama + Local LLM Server

set -e

MODEL="qwen2.5:3b"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
LLM_PORT="${LLM_PORT:-8001}"

echo "=================================================="
echo "🚀 Local LLM Setup (Ollama + Qwen2.5 3B)"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not installed"
    echo "Install from: https://ollama.ai"
    exit 1
fi

echo "✅ Ollama found"

# Pull model
echo ""
echo "📥 Pulling $MODEL..."
ollama pull $MODEL

# Start Ollama server (background)
echo ""
echo "🔧 Starting Ollama server..."
if pgrep -f "ollama serve" > /dev/null; then
    echo "✅ Ollama already running on port $OLLAMA_PORT"
else
    ollama serve &
    OLLAMA_PID=$!
    echo "✅ Ollama started (PID: $OLLAMA_PID)"
    sleep 2
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "🎉 Setup complete!"
echo "=================================================="
echo ""
echo "Usage:"
echo "  Server: python main.py"
echo "  Client: python client.py"
echo "  Chat:   python client.py chat"
echo ""
echo "Endpoints:"
echo "  Health:      GET http://localhost:$LLM_PORT/health"
echo "  Diagnose:    POST http://localhost:$LLM_PORT/diagnose"
echo "  Chat:        POST http://localhost:$LLM_PORT/chat"
echo "  Models:      GET http://localhost:$LLM_PORT/models"
