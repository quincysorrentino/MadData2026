#!/usr/bin/env python3
"""
Interactive CLI chat with local Qwen 8B model via Ollama
"""

import requests
import json
import sys
from typing import List, Dict

BASE_URL = "http://localhost:9000"

def check_server():
    """Verify server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Connected to Local LLM Server")
            data = response.json()
            print(f"📦 Model: {data.get('model', 'unknown')}")
            print(f"🔗 Ollama: {data.get('ollama_url', 'unknown')}\n")
            return True
    except Exception as e:
        print(f"❌ Server not running: {e}")
        print(f"   Start it with: python main.py\n")
        return False

def chat(messages: List[Dict], temperature: float = 0.7) -> str:
    """Send chat message and get response"""
    try:
        payload = {
            "messages": messages,
            "temperature": temperature
        }
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=120  # Long timeout for model thinking
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("content", data.get("response", "No response"))
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "⏱️ Response timeout - model is thinking..."
    except Exception as e:
        return f"Error: {e}"

def main():
    """Interactive chat loop"""
    print("=" * 60)
    print("🤖 Qwen 8B Local Chat")
    print("=" * 60)
    
    if not check_server():
        sys.exit(1)
    
    print("Type 'exit' to quit, 'clear' to reset conversation\n")
    print("-" * 60 + "\n")
    
    messages: List[Dict] = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "exit":
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == "clear":
                messages = []
                print("🔄 Conversation cleared\n")
                continue
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            print("\n🤔 Thinking...\n")
            
            # Get response
            response = chat(messages)
            
            # Add assistant response to history
            messages.append({
                "role": "assistant",
                "content": response
            })
            
            print(f"Qwen: {response}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
