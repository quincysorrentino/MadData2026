"""
Client library for communicating with the Skin Cancer Classification API
"""

import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class SkinCancerAPIClient:
    """
    Client for interacting with the Skin Cancer Classification API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """Verify that the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to API at {self.base_url}: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if API is running and healthy
        
        Returns:
            Health status information
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the classification model
        
        Returns:
            Model information including classes and training metrics
        """
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a single skin cancer image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Classification result with predictions and confidence scores
            
        Raises:
            FileNotFoundError: If image doesn't exist
            requests.HTTPError: If API error occurs
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{self.base_url}/classify", files=files)
        
        response.raise_for_status()
        return response.json()
    
    def classify_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Classify multiple skin cancer images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Batch classification results
            
        Raises:
            FileNotFoundError: If any image doesn't exist
            requests.HTTPError: If API error occurs
        """
        files = []
        valid_paths = []
        
        for path in image_paths:
            image_path = Path(path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            valid_paths.append(image_path)
        
        try:
            files = [("files", open(path, "rb")) for path in valid_paths]
            response = requests.post(f"{self.base_url}/classify-batch", files=files)
            response.raise_for_status()
            return response.json()
        finally:
            # Close all file handles
            for _, f in files:
                f.close()
    
    def print_classification_result(self, result: Dict[str, Any]) -> None:
        """
        Pretty print a classification result
        
        Args:
            result: Classification result dictionary
        """
        print(f"\n📊 Classification Results: {result.get('image_name', 'Unknown')}")
        print("-" * 50)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Top Prediction: {result.get('top_prediction', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        
        if result.get('predictions'):
            print(f"\nAll Predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['percentage']:.2f}%")
    
    def print_batch_results(self, results: Dict[str, Any]) -> None:
        """
        Pretty print batch classification results
        
        Args:
            results: Batch classification results dictionary
        """
        print(f"\n📦 Batch Classification Results")
        print("=" * 50)
        print(f"Total Files: {results.get('total_files', 0)}")
        print(f"Successful: {results.get('successful', 0)}")
        print(f"Failed: {results.get('failed', 0)}")
        
        print(f"\nDetails:")
        for i, res in enumerate(results.get('results', []), 1):
            print(f"\n  {i}. {res.get('image_name', 'Unknown')}")
            if res.get('status') == 'success':
                print(f"     ✓ {res.get('top_prediction', 'N/A')} ({res.get('confidence', 0):.2%})")
            else:
                print(f"     ✗ Error: {res.get('error', 'Unknown error')}")


# Convenience functions for quick usage
def classify_image(image_path: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Quick function to classify a single image
    
    Args:
        image_path: Path to image file
        api_url: API base URL
        
    Returns:
        Classification result
    """
    client = SkinCancerAPIClient(api_url)
    return client.classify_image(image_path)


def classify_batch(image_paths: List[str], api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Quick function to classify multiple images
    
    Args:
        image_paths: List of image file paths
        api_url: API base URL
        
    Returns:
        Batch classification results
    """
    client = SkinCancerAPIClient(api_url)
    return client.classify_batch(image_paths)
