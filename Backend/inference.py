"""
Skin Cancer Image Classification Model Inference Module
Uses Hugging Face Inference API to classify skin cancer images
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
from huggingface_hub import InferenceClient
import logging

logger = logging.getLogger(__name__)

# Class labels for skin cancer classification
CLASS_LABELS = {
    0: "Benign keratosis-like lesions",
    1: "Basal cell carcinoma",
    2: "Actinic keratoses",
    3: "Vascular lesions",
    4: "Melanocytic nevi",
    5: "Melanoma",
    6: "Dermatofibroma"
}

MODEL_ID = "Anwarkh1/Skin_Cancer-Image_Classification"


class SkinCancerClassifier:
    """
    Classifier for skin cancer images using Vision Transformer
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the classifier with Hugging Face Inference API
        
        Args:
            api_key: Hugging Face API key (uses HF_TOKEN env var if not provided)
        """
        if api_key is None:
            api_key = os.environ.get("HF_TOKEN")
            if not api_key:
                raise ValueError("No Hugging Face API key provided. Set HF_TOKEN environment variable.")
        
        self.client = InferenceClient(token=api_key)
        self.model_id = MODEL_ID
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a skin cancer image from a local file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing classification results with scores
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Verify it's an image file
        valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        if image_path.suffix.lower() not in valid_formats:
            raise ValueError(f"Invalid image format. Supported: {valid_formats}")
        
        try:
            # Call Hugging Face Inference API for image classification
            results = self.client.image_classification(
                str(image_path),
                model=self.model_id
            )
            
            # Parse and format results
            classification_results = self._format_results(results)
            
            logger.info(f"Successfully classified image: {image_path}")
            return classification_results
            
        except Exception as e:
            logger.error(f"Error classifying image: {str(e)}")
            raise
    
    def classify_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple skin cancer images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of classification results for each image
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.classify_image(image_path)
                result['image_path'] = image_path
                result['status'] = 'success'
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                })
                logger.error(f"Failed to classify {image_path}: {str(e)}")
        
        return results
    
    def _format_results(self, raw_results: List[Dict]) -> Dict[str, Any]:
        """
        Format raw inference results to a user-friendly format
        
        Args:
            raw_results: Results from Hugging Face Inference API
            
        Returns:
            Formatted classification results
        """
        if not raw_results:
            return {"error": "No results returned"}
        
        # Assuming raw_results is list of dicts with 'label' and 'score' keys
        formatted = {
            "predictions": [],
            "top_prediction": None,
            "confidence": 0
        }
        
        for idx, result in enumerate(raw_results):
            label = result.get('label', f'Class_{idx}')
            score = result.get('score', 0)
            
            # Convert label to human-readable format if it's a number
            try:
                class_idx = int(label)
                class_name = CLASS_LABELS.get(class_idx, label)
            except (ValueError, TypeError):
                class_name = label
            
            formatted["predictions"].append({
                "class": class_name,
                "confidence": round(float(score), 4),
                "percentage": round(float(score) * 100, 2)
            })
        
        # Set top prediction
        if formatted["predictions"]:
            top_pred = formatted["predictions"][0]
            formatted["top_prediction"] = top_pred["class"]
            formatted["confidence"] = top_pred["confidence"]
        
        return formatted


def get_classifier(api_key: str = None) -> SkinCancerClassifier:
    """
    Factory function to create a classifier instance
    
    Args:
        api_key: Optional Hugging Face API key
        
    Returns:
        SkinCancerClassifier instance
    """
    return SkinCancerClassifier(api_key)
