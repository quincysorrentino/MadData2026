#!/usr/bin/env python3
"""
Example: Complete diagnosis workflow
"""

import time
from client import LLMClient


def example_complete_workflow():
    """Full diagnosis workflow: classification → diagnosis → follow-up chat"""
    
    client = LLMClient("http://localhost:8001")
    
    # Check server
    print("🔍 Checking LLM server...")
    if not client.health_check():
        print("❌ LLM server not running. Start with: python main.py")
        return
    print("✅ Server healthy\n")
    
    # Example classification result (from skin cancer classifier)
    print("="*60)
    print("📸 SKIN CANCER CLASSIFICATION RESULT")
    print("="*60)
    
    classification = {
        "condition": "Melanoma",
        "confidence": 0.92,
        "body_part": "arm",
        "duration": "2 weeks",
        "description": "Dark, irregularly shaped mole with color variation"
    }
    
    print(f"Condition:  {classification['condition']}")
    print(f"Confidence: {classification['confidence']:.1%}")
    print(f"Location:   {classification['body_part']}")
    print(f"Duration:   {classification['duration']}")
    print(f"Description: {classification['description']}")
    print()
    
    # Get diagnosis
    print("🏥 Getting medical diagnosis...")
    print("(This may take 10-30 seconds on first run)")
    print()
    
    start = time.time()
    
    result = client.diagnose(
        classification=classification["condition"],
        confidence=classification["confidence"],
        body_part=classification["body_part"],
        duration=classification["duration"],
        description=classification["description"]
    )
    
    elapsed = time.time() - start
    
    print("="*60)
    print("📋 DIAGNOSIS")
    print("="*60)
    print(result.get("diagnosis", ""))
    print(f"\n⏱️  Generated in {elapsed:.1f}s\n")
    
    print("="*60)
    print("📌 RECOMMENDATIONS")
    print("="*60)
    print(result.get("recommendations", ""))
    print()
    
    if result.get("follow_up_questions"):
        print("="*60)
        print("❓ SUGGESTED FOLLOW-UP QUESTIONS")
        print("="*60)
        for i, q in enumerate(result["follow_up_questions"], 1):
            print(f"{i}. {q}")
        print()
    
    # Interactive follow-up
    print("="*60)
    print("💬 MULTI-TURN CONVERSATION")
    print("="*60)
    
    conversation_history = [
        {
            "role": "system",
            "content": "You are an expert dermatology assistant providing professional medical guidance."
        },
        {
            "role": "user",
            "content": f"I was diagnosed with {classification['condition']} on my {classification['body_part']} with {classification['confidence']:.0%} confidence. {classification['description']}"
        },
        {
            "role": "assistant",
            "content": result.get("diagnosis", "")
        }
    ]
    
    # First follow-up
    print("\nUser: What are the treatment options for this condition?")
    print()
    
    conversation_history.append({
        "role": "user",
        "content": "What are the treatment options for this condition?"
    })
    
    response = client.chat(conversation_history)
    assistant_response = response["content"]
    
    print("Assistant: ", end="")
    print(assistant_response)
    print()
    
    conversation_history.append({
        "role": "assistant",
        "content": assistant_response
    })
    
    # Second follow-up
    print("User: When should I see a dermatologist?")
    print()
    
    conversation_history.append({
        "role": "user",
        "content": "When should I see a dermatologist?"
    })
    
    response = client.chat(conversation_history)
    assistant_response = response["content"]
    
    print("Assistant: ", end="")
    print(assistant_response)
    print()
    
    # Summary
    print("="*60)
    print("✅ WORKFLOW COMPLETE")
    print("="*60)
    print("Summary:")
    print(f"  • Classification: {classification['condition']} ({classification['confidence']:.0%})")
    print(f"  • Location: {classification['body_part']}")
    print(f"  • Diagnosis generated: ✓")
    print(f"  • Follow-up questions: {len(result.get('follow_up_questions', []))}")
    print(f"  • Chat turns completed: 2")
    print()
    print("Next steps:")
    print("  1. Share diagnosis with dermatologist")
    print("  2. Schedule immediate appointment")
    print("  3. Monitor for changes per recommendations")


def example_quick_diagnosis():
    """Quick diagnosis without conversation"""
    
    client = LLMClient("http://localhost:8001")
    
    if not client.health_check():
        print("❌ LLM server not running")
        return
    
    print("🏥 Quick Diagnosis Example\n")
    
    # Start
    classifications = [
        {
            "name": "Melanoma",
            "confidence": 0.92,
            "body_part": "arm"
        },
        {
            "name": "Basal Cell Carcinoma",
            "confidence": 0.88,
            "body_part": "face"
        },
        {
            "name": "Nevus",
            "confidence": 0.95,
            "body_part": "back"
        }
    ]
    
    for item in classifications:
        print(f"📋 {item['name']} ({item['confidence']:.0%}) on {item['body_part']}")
        
        result = client.diagnose(
            classification=item["name"],
            confidence=item["confidence"],
            body_part=item["body_part"]
        )
        
        print(f"   → {result['diagnosis'][:100]}...")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        example_quick_diagnosis()
    else:
        example_complete_workflow()
