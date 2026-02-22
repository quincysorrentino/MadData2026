# Quick Start Guide - Skin Cancer Classification API

## ⚡ Quick Setup (5 minutes)

### Step 1: Get Hugging Face API Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read permission is enough)
3. Copy the token

### Step 2: Set Environment Variable

```bash
# Option A: Linux/Mac
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Option B: Windows (Command Prompt)
set HF_TOKEN=hf_YOUR_TOKEN_HERE

# Option C: Create .env file
echo "HF_TOKEN=hf_YOUR_TOKEN_HERE" > .env
```

### Step 3: Install Dependencies

```bash
cd Backend
pip install -r requirements.txt
```

### Step 4: Start the API

```bash
python main.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 5: Test the API

Open in your browser or use curl:

```bash
# Health check
curl http://localhost:8000/health

# API documentation
# Open browser: http://localhost:8000/docs

# Classify an image
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/classify
```

## 📚 Common Tasks

### Classify a Single Image (Python)

```python
from client import classify_image

result = classify_image("skin_sample.jpg")
print(f"Prediction: {result['top_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Classify Multiple Images (Python)

```python
from client import classify_batch

results = classify_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
print(f"Success: {results['successful']}/{results['total_files']}")
```

### Using the API Client Class

```python
from client import SkinCancerAPIClient

client = SkinCancerAPIClient("http://localhost:8000")

# Get model info
info = client.get_model_info()
print(f"Model Classes: {info['classes']}")

# Single classification
result = client.classify_image("skin_sample.jpg")
client.print_classification_result(result)

# Batch classification
batch_results = client.classify_batch(["img1.jpg", "img2.jpg"])
client.print_batch_results(batch_results)
```

## 🔍 API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check if API is running |
| `/classify` | POST | Classify single image |
| `/classify-batch` | POST | Classify multiple images |
| `/info` | GET | Get model information |
| `/docs` | GET | Interactive API documentation |
| `/redoc` | GET | Alternative API documentation |

## 🐛 Troubleshooting

### "No Hugging Face API key provided"
```bash
export HF_TOKEN="your_token_here"
python main.py
```

### "Cannot connect to API"
- Ensure API is running: `python main.py`
- Check URL: default is `http://localhost:8000`
- Check firewall settings

### "Invalid file format"
- Supported formats: jpg, jpeg, png, bmp, gif, webp
- Ensure file extension matches actual format

### API is slow
- First request may load model (slower)
- Subsequent requests should be faster
- Use batch endpoint for multiple images

## 📝 Model Information

- **Architecture**: Vision Transformer (ViT)
- **Accuracy**: 96.95% on validation set
- **Classes**: 7 skin cancer types
- **Image Size**: 224x224 pixels
- **Input Formats**: jpg, png, bmp, gif, webp

## 🎯 Example Workflow

```python
from client import SkinCancerAPIClient

# 1. Create client
client = SkinCancerAPIClient()

# 2. Check API is running
print(client.health_check())

# 3. Get model details
model_info = client.get_model_info()
print(f"Classes: {model_info['classes']}")

# 4. Classify images
result = client.classify_image("patient_sample.jpg")
client.print_classification_result(result)

# 5. If positive, batch classify similar samples
if result['confidence'] > 0.8:
    batch = client.classify_batch([
        "similar_sample_1.jpg",
        "similar_sample_2.jpg"
    ])
    client.print_batch_results(batch)
```

## 🚀 Production Deployment

For production use:

```bash
# Use gunicorn with multiple workers
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 main:app

# Or use Docker (create Dockerfile first)
docker build -t skin-cancer-api .
docker run -p 8000:8000 -e HF_TOKEN=$HF_TOKEN skin-cancer-api
```

## 📖 Additional Resources

- [API Documentation](/docs)
- [Hugging Face Model](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)
- [Dataset Source](https://huggingface.co/datasets/marmal88/skin_cancer)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

## ✅ Verification Checklist

- [ ] Hugging Face token acquired
- [ ] HF_TOKEN environment variable set
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] API started (`python main.py`)
- [ ] Health check passes (`curl http://localhost:8000/health`)
- [ ] API documentation accessible (`http://localhost:8000/docs`)
- [ ] Can classify test image
- [ ] Batch classification works

---

Happy classifying! 🏥
