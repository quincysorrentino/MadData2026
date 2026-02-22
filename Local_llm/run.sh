#!/bin/bash
# Start Local LLM Server

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "🚀 Local LLM Server (Ollama + Qwen 3 8B)"
echo "=================================================="

# Check if Ollama is running
echo ""
echo "🔍 Checking Ollama..."
if curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama not found. Starting ollama serve..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    echo "✅ Ollama started (PID: $OLLAMA_PID)"
    sleep 2
fi

# Check model
echo ""
echo "📦 Checking Qwen model..."
if ollama list | grep -q "qwen:8b"; then
    echo "✅ Qwen 3 8B found"
else
    echo "⚠️  Qwen not found. Pulling..."
    ollama pull qwen:8b
    echo "✅ Qwen pulled"
fi

# Activate venv if exists
if [ -f "../venv/bin/activate" ]; then
    echo ""
    echo "🐍 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Start server
echo ""
echo "🌐 Starting LLM API server..."
python main.py

# Cleanup on exit
trap "kill $(jobs -p) 2>/dev/null || true" EXIT
