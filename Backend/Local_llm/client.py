"""
Client for Local LLM Diagnosis Server
"""

import requests
import json
from typing import List, Dict, Optional


class LLMClient:
    """Client for interacting with local LLM server"""
    
    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url
    
    def health_check(self) -> bool:
        """Check if LLM server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def diagnose(
        self,
        classification: str,
        confidence: float,
        body_part: str,
        duration: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict:
        """
        Get diagnosis for a skin condition
        
        Args:
            classification: Condition name (e.g., "Melanoma")
            confidence: Confidence 0-1
            body_part: Location (e.g., "arm")
            duration: How long present
            description: User description
        
        Returns:
            Dict with diagnosis, recommendations, follow-up questions
        """
        
        response = requests.post(
            f"{self.base_url}/diagnose",
            json={
                "classification": classification,
                "confidence": confidence,
                "body_part": body_part,
                "duration": duration,
                "description": description
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")
        
        return response.json()
    
    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7
    ) -> Dict:
        """
        Chat for follow-up questions
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Response creativity
        
        Returns:
            Dict with assistant response
        """
        
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "messages": messages,
                "temperature": temperature
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")
        
        return response.json()


def example_diagnosis():
    """Example: Get diagnosis for a melanoma finding"""
    
    client = LLMClient()
    
    # Check if server is running
    if not client.health_check():
        print("❌ LLM server not running. Start with: python main.py")
        return
    
    print("🔍 Getting diagnosis...")
    
    result = client.diagnose(
        classification="Melanoma",
        confidence=0.92,
        body_part="arm",
        duration="2 weeks",
        description="Dark, irregularly shaped mole with color variation"
    )
    
    print("\n" + "="*60)
    print("📋 DIAGNOSIS")
    print("="*60)
    print(result.get("diagnosis", ""))
    
    print("\n📌 RECOMMENDATIONS")
    print("="*60)
    print(result.get("recommendations", ""))
    
    if result.get("follow_up_questions"):
        print("\n❓ FOLLOW-UP QUESTIONS")
        print("="*60)
        for q in result["follow_up_questions"]:
            print(f"• {q}")


def example_chat():
    """Example: Multi-turn conversation"""
    
    client = LLMClient()
    
    # Check if server is running
    if not client.health_check():
        print("❌ LLM server not running. Start with: python main.py")
        return
    
    print("💬 Starting chat conversation...")
    
    messages = [
        {
            "role": "system",
            "content": "You are a dermatology assistant. Be professional and helpful."
        },
        {
            "role": "user",
            "content": "I have a melanoma diagnosis with 92% confidence. What should I do?"
        }
    ]
    
    # First response
    response = client.chat(messages)
    assistant_reply = response["content"]
    print(f"\nAssistant: {assistant_reply}")
    
    # Add to conversation
    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    # Follow-up question
    messages.append({
        "role": "user",
        "content": "What are the treatment options?"
    })
    
    response = client.chat(messages)
    print(f"\nAssistant: {response['content']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        example_chat()
    else:
        example_diagnosis()
