import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import time

from model import ChestXrayModel, load_model, get_transforms, LABEL_NAMES
from grad_cam import GradCAM, create_heatmap_overlay
from severity import estimate_severity

app = FastAPI(
    title="Chest X-Ray AI Screening API",
    description="Multi-label Pneumonia & TB detection with Grad-CAM visualization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DEVICE = 'cpu'
CHECKPOINT_PATH = os.environ.get('MODEL_CHECKPOINT', 'checkpoints/best_model.pth')

model = None
grad_cam_engine = None
transform = None

@app.on_event("startup")
async def startup():
    global model, grad_cam_engine, transform
    model = load_model(CHECKPOINT_PATH, DEVICE)
    # Target the last DenseBlock's last DenseLayer
    target_layer = model.densenet.features.denseblock4.denselayer16.conv2
    grad_cam_engine = GradCAM(model, target_layer)
    transform = get_transforms()
    print("Model loaded and ready for inference.")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "labels": LABEL_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Enable gradients for Grad-CAM
    input_tensor.requires_grad_(True)
    
    # Run Grad-CAM (also gets model output)
    # Find the class with highest activation for heatmap
    with torch.enable_grad():
        cam, output = grad_cam_engine.generate(input_tensor)
    
    # Get probabilities
    probs = torch.sigmoid(output).detach().cpu().numpy()[0]
    probabilities = {name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs)}
    
    # Generate heatmap overlay
    heatmap_base64 = create_heatmap_overlay(image, cam)
    
    # Estimate severity
    severity = estimate_severity(probabilities)
    
    # Determine primary diagnosis
    max_idx = int(np.argmax(probs))
    primary_diagnosis = LABEL_NAMES[max_idx]
    
    inference_time = round(time.time() - start_time, 3)
    
    return JSONResponse(content={
        "probabilities": probabilities,
        "primary_diagnosis": primary_diagnosis,
        "severity": severity,
        "heatmap_base64": heatmap_base64,
        "inference_time_seconds": inference_time,
        "model_info": {
            "backbone": "DenseNet-121",
            "labels": LABEL_NAMES,
            "device": DEVICE
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
