# Skin Cancer Image Classification API

A FastAPI-based REST API for classifying skin cancer images using a Vision Transformer model fine-tuned on the Marmal88 Skin Cancer Dataset.

## Model Details

- **Architecture**: Vision Transformer (ViT)
- **Pre-trained Model**: Google's ViT with 16x16 patch size (ImageNet21k)
- **Model ID**: `Anwarkh1/Skin_Cancer-Image_Classification`
- **Final Validation Accuracy**: 96.95%

### Classification Classes

1. Benign keratosis-like lesions
2. Basal cell carcinoma
3. Actinic keratoses
4. Vascular lesions
5. Melanocytic nevi
6. Melanoma
7. Dermatofibroma

## Setup

### Prerequisites

- Python 3.8+
- Hugging Face API Token (Get from [huggingface.co](https://huggingface.co))

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   ```bash
   export HF_TOKEN="your_huggingface_api_token"
   ```

   Or create a `.env` file:
   ```
   HF_TOKEN=your_huggingface_api_token
   API_HOST=0.0.0.0
   API_PORT=8000
   ```

### Running the API

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Check if the API is running and model is loaded
- **Response**:
  ```json
  {
    "status": "healthy",
    "api_version": "1.0.0",
    "model_name": "Anwarkh1/Skin_Cancer-Image_Classification"
  }
  ```

### 2. Classify Single Image
- **Endpoint**: `POST /classify`
- **Description**: Classify a single skin cancer image
- **Request**:
  - multipart/form-data
  - `file`: Image file (jpg, png, bmp, gif, webp)
- **Response**:
  ```json
  {
    "status": "success",
    "image_name": "skin_sample.jpg",
    "predictions": [
      {
        "class": "Melanoma",
        "confidence": 0.9523,
        "percentage": 95.23
      },
      {
        "class": "Benign keratosis-like lesions",
        "confidence": 0.0412,
        "percentage": 4.12
      }
    ],
    "top_prediction": "Melanoma",
    "confidence": 0.9523
  }
  ```

### 3. Classify Multiple Images (Batch)
- **Endpoint**: `POST /classify-batch`
- **Description**: Classify multiple images at once
- **Request**:
  - multipart/form-data
  - `files`: Multiple image files
- **Response**:
  ```json
  {
    "status": "completed",
    "total_files": 3,
    "successful": 3,
    "failed": 0,
    "results": [
      {
        "status": "success",
        "image_name": "image1.jpg",
        "predictions": [...],
        "top_prediction": "Melanoma",
        "confidence": 0.9523
      },
      ...
    ]
  }
  ```

### 4. Get Model Information
- **Endpoint**: `GET /info`
- **Description**: Get details about the model and supported classes
- **Response**:
  ```json
  {
    "model_name": "Vision Transformer (ViT)",
    "model_id": "Anwarkh1/Skin_Cancer-Image_Classification",
    "architecture": "Google ViT with 16x16 patch size (ImageNet21k pre-trained)",
    "classes": [...],
    "training_metrics": {
      "train_loss": 0.1208,
      "train_accuracy": 0.9614,
      "val_loss": 0.1000,
      "val_accuracy": 0.9695
    },
    "supported_formats": ["jpg", "jpeg", "png", "bmp", "gif", "webp"]
  }
  ```

### 5. API Documentation
- **Endpoint**: `GET /docs`
- **Description**: Interactive API documentation (Swagger UI)

- **Endpoint**: `GET /redoc`
- **Description**: Alternative API documentation (ReDoc)

## Usage Examples

### Python

```python
import requests

# Single image classification
with open("skin_sample.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/classify", files=files)
    print(response.json())

# Batch classification
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb"))
]
response = requests.post("http://localhost:8000/classify-batch", files=files)
print(response.json())
```

### cURL

```bash
# Single image
curl -X POST -F "file=@skin_sample.jpg" http://localhost:8000/classify

# Batch images
curl -X POST \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  http://localhost:8000/classify-batch
```

### JavaScript/Fetch

```javascript
// Single image
const formData = new FormData();
formData.append("file", document.getElementById("fileInput").files[0]);

fetch("http://localhost:8000/classify", {
  method: "POST",
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

## Model Performance

Trained on the Marmal88 Skin Cancer Dataset:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|----------|---------|---------|
| 1     | 0.7168    | 75.86%   | 0.4994  | 83.55%  |
| 2     | 0.4550    | 84.66%   | 0.3237  | 89.73%  |
| 3     | 0.2959    | 90.28%   | 0.1790  | 95.30%  |
| 4     | 0.1595    | 94.82%   | 0.1498  | 95.55%  |
| 5     | 0.1208    | 96.14%   | 0.1000  | 96.95%  |

## Project Structure

```
Backend/
├── main.py              # FastAPI application
├── inference.py         # Model inference logic
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid file format or missing files
- **503 Service Unavailable**: Classifier not initialized (missing API token)
- **500 Internal Server Error**: Unexpected server errors

## Troubleshooting

### "No Hugging Face API key provided"
- Ensure `HF_TOKEN` environment variable is set
- Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### "Classifier not initialized"
- API started before environment variables were loaded
- Restart the API after setting `HF_TOKEN`

### Slow Classification
- First request may be slow due to model loading
- Subsequent requests should be faster
- Consider using batch endpoints for multiple images

## Future Improvements

- Add image preprocessing/validation
- Implement caching for repeated classifications
- Add confidence threshold settings
- Support for batch prediction with progress tracking
- Add model versioning support
- Implement request rate limiting
- Add comprehensive logging and monitoring

## License

This project uses the Hugging Face Inference API. See LICENSE file for details.

## References

- [Hugging Face Model Card](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)
- [Marmal88 Dataset](https://huggingface.co/datasets/marmal88/skin_cancer)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
