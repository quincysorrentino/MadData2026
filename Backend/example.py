"""
Example script demonstrating how to use the Skin Cancer Classification API
"""

import requests
import json
from pathlib import Path

# API base URL (update if running on different host/port)
API_URL = "http://localhost:8000"


def health_check():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        print("✓ API Health Check:")
        print(json.dumps(response.json(), indent=2))
        return True
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running on", API_URL)
        return False


def get_model_info():
    """Get information about the model"""
    response = requests.get(f"{API_URL}/info")
    print("\n✓ Model Information:")
    print(json.dumps(response.json(), indent=2))


def classify_single_image(image_path: str):
    """Classify a single image"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"\n✓ Classifying: {image_path.name}")
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/classify", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Status: {result['status']}")
        print(f"  Top Prediction: {result['top_prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  All Predictions:")
        for pred in result['predictions']:
            print(f"    - {pred['class']}: {pred['percentage']:.2f}%")
    else:
        print(f"  Error: {response.json()}")


def classify_batch(image_paths: list):
    """Classify multiple images at once"""
    valid_paths = []
    
    for path in image_paths:
        image_path = Path(path)
        if image_path.exists():
            valid_paths.append(image_path)
        else:
            print(f"✗ Image not found: {path}")
    
    if not valid_paths:
        print("No valid images to classify")
        return
    
    print(f"\n✓ Batch Classifying: {len(valid_paths)} images")
    
    files = [("files", open(path, "rb")) for path in valid_paths]
    response = requests.post(f"{API_URL}/classify-batch", files=files)
    
    # Close file handles
    for _, f in files:
        f.close()
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Total Files: {result['total_files']}")
        print(f"  Successful: {result['successful']}")
        print(f"  Failed: {result['failed']}")
        print(f"\n  Results:")
        
        for i, res in enumerate(result['results'], 1):
            print(f"    {i}. {res['image_name']}")
            if res['status'] == 'success':
                print(f"       Top Prediction: {res['top_prediction']}")
                print(f"       Confidence: {res['confidence']:.2%}")
            else:
                print(f"       Error: {res.get('error', 'Unknown error')}")
    else:
        print(f"  Error: {response.json()}")


def main():
    """Main example runner"""
    print("=" * 60)
    print("Skin Cancer Classification API - Example Usage")
    print("=" * 60)
    
    # Check if API is running
    if not health_check():
        print("\nPlease start the API with: python main.py")
        return
    
    # Get model information
    get_model_info()
    
    # Example: Classify a single image
    print("\n" + "=" * 60)
    print("Example Single Image Classification:")
    print("=" * 60)
    print("To classify an image, call:")
    print("  classify_single_image('path/to/image.jpg')")
    
    # Example: Batch classification
    print("\n" + "=" * 60)
    print("Example Batch Classification:")
    print("=" * 60)
    print("To classify multiple images, call:")
    print("  classify_batch(['image1.jpg', 'image2.jpg', 'image3.jpg'])")
    
    print("\n" + "=" * 60)
    print("Usage in Python:")
    print("=" * 60)
    print("""
from example import classify_single_image, classify_batch

# Single image
classify_single_image('skin_sample.jpg')

# Multiple images
classify_batch(['image1.jpg', 'image2.jpg', 'image3.jpg'])
    """)


if __name__ == "__main__":
    main()
