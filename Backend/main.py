"""
FastAPI Backend for Skin Cancer Image Classification
Provides REST API endpoints for image classification
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
from inference import get_classifier, SkinCancerClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Skin Cancer Image Classification API",
    description="API for classifying skin cancer images using Vision Transformer",
    version="1.0.0"
)

# Add CORS middleware to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier: Optional[SkinCancerClassifier] = None


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    api_version: str
    model_name: str


class ClassificationResponse(BaseModel):
    """Classification response model"""
    status: str
    image_name: str
    predictions: List[dict]
    top_prediction: str
    confidence: float


@app.on_event("startup")
async def startup_event():
    """Initialize classifier on startup"""
    global classifier
    try:
        classifier = get_classifier()
        logger.info("Classifier initialized successfully")
    except ValueError as e:
        logger.error(f"Failed to initialize classifier: {str(e)}")
        raise


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running
    
    Returns:
        Health status and API information
    """
    return HealthResponse(
        status="healthy",
        api_version="1.0.0",
        model_name="Anwarkh1/Skin_Cancer-Image_Classification"
    )


@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_image(file: UploadFile = File(...)):
    """
    Classify a single skin cancer image
    
    Args:
        file: Image file (jpg, png, bmp, etc.)
        
    Returns:
        Classification results with predictions and confidence scores
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier not initialized"
        )
    
    # Validate file type
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in valid_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Supported: {valid_formats}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Classify the image
        results = classifier.classify_image(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return ClassificationResponse(
            status="success",
            image_name=file.filename,
            predictions=results.get("predictions", []),
            top_prediction=results.get("top_prediction", "Unknown"),
            confidence=results.get("confidence", 0)
        )
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/classify-batch", tags=["Classification"])
async def classify_batch(files: List[UploadFile] = File(...)):
    """
    Classify multiple skin cancer images in batch
    
    Args:
        files: List of image files
        
    Returns:
        List of classification results
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier not initialized"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    results = []
    temp_files = []
    
    try:
        # Save all files temporarily
        valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for file in files:
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension not in valid_formats:
                results.append({
                    "status": "error",
                    "image_name": file.filename,
                    "error": f"Invalid file format. Supported: {valid_formats}"
                })
                continue
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                contents = await file.read()
                tmp_file.write(contents)
                temp_files.append((tmp_file.name, file.filename))
        
        # Classify each image
        for tmp_path, original_filename in temp_files:
            try:
                result = classifier.classify_image(tmp_path)
                results.append({
                    "status": "success",
                    "image_name": original_filename,
                    "predictions": result.get("predictions", []),
                    "top_prediction": result.get("top_prediction", "Unknown"),
                    "confidence": result.get("confidence", 0)
                })
            except Exception as e:
                logger.error(f"Error classifying {original_filename}: {str(e)}")
                results.append({
                    "status": "error",
                    "image_name": original_filename,
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "total_files": len(files),
            "successful": sum(1 for r in results if r['status'] == 'success'),
            "failed": sum(1 for r in results if r['status'] == 'error'),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch classification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing batch: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        for tmp_path, _ in temp_files:
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.get("/info", tags=["Info"])
async def model_info():
    """
    Get information about the model and supported classes
    
    Returns:
        Model information and classification classes
    """
    return {
        "model_name": "Vision Transformer (ViT)",
        "model_id": "Anwarkh1/Skin_Cancer-Image_Classification",
        "architecture": "Google ViT with 16x16 patch size (ImageNet21k pre-trained)",
        "classes": [
            "Benign keratosis-like lesions",
            "Basal cell carcinoma",
            "Actinic keratoses",
            "Vascular lesions",
            "Melanocytic nevi",
            "Melanoma",
            "Dermatofibroma"
        ],
        "training_metrics": {
            "train_loss": 0.1208,
            "train_accuracy": 0.9614,
            "val_loss": 0.1000,
            "val_accuracy": 0.9695
        },
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "gif", "webp"]
    }


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information
    
    Returns:
        API documentation and available endpoints
    """
    return {
        "message": "Skin Cancer Image Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classify_single": "/classify",
            "classify_batch": "/classify-batch",
            "model_info": "/info",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment or use defaults
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)
