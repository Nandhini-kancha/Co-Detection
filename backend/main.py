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


def validate_chest_xray(image: Image.Image) -> tuple:
    """
    Validate whether an uploaded image looks like a chest X-ray.
    Returns (is_valid, reason) tuple.
    
    Checks:
    1. Image is predominantly grayscale (X-rays are grayscale)
    2. Image dimensions are reasonable (not tiny icons or extreme aspect ratios)
    3. Pixel intensity distribution matches X-ray characteristics
    """
    width, height = image.size
    
    # Check 1: Minimum dimensions - X-rays should be at least 100x100
    if width < 100 or height < 100:
        return False, "Image is too small. Chest X-rays should be at least 100x100 pixels."
    
    # Check 2: Aspect ratio - chest X-rays are roughly square or portrait (0.5 to 2.0)
    aspect_ratio = width / height
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return False, "Invalid aspect ratio. This does not appear to be a chest X-ray image."
    
    # Check 3: Grayscale check - X-rays are predominantly grayscale
    # Convert to RGB and check if color channels are similar
    rgb_image = image.convert('RGB')
    img_array = np.array(rgb_image)
    
    # Sample center region (avoid borders)
    h, w = img_array.shape[:2]
    margin_h, margin_w = h // 4, w // 4
    center = img_array[margin_h:h-margin_h, margin_w:w-margin_w]
    
    r, g, b = center[:,:,0].astype(float), center[:,:,1].astype(float), center[:,:,2].astype(float)
    
    # Calculate mean absolute difference between channels
    rg_diff = np.mean(np.abs(r - g))
    rb_diff = np.mean(np.abs(r - b))
    gb_diff = np.mean(np.abs(g - b))
    avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3.0
    
    # X-rays have very low color difference (< 15), colorful images have high (> 30)
    if avg_color_diff > 35:
        return False, "This appears to be a color photograph, not a chest X-ray. Please upload a grayscale chest X-ray image."
    
    # Check 4: Intensity distribution - X-rays have a wide dynamic range
    gray = np.array(image.convert('L'))
    std_dev = np.std(gray)
    
    # Very low std means solid/uniform color image (not an X-ray)
    if std_dev < 10:
        return False, "Image has insufficient contrast. This does not appear to be a medical X-ray image."
    
    # Check 5: X-rays typically have dark regions (lungs) and bright regions (bones)
    # Check that there's meaningful content in both dark and bright ranges
    dark_fraction = np.mean(gray < 80)
    bright_fraction = np.mean(gray > 180)
    
    if dark_fraction < 0.05 and bright_fraction < 0.05:
        # Image is all mid-tones, unlikely to be an X-ray
        if avg_color_diff > 20:
            return False, "This does not appear to be a chest X-ray. Please upload a valid chest X-ray image."
    
    return True, "Valid"


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
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Validate that the image is a chest X-ray
    is_valid, reason = validate_chest_xray(image)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Enable gradients for Grad-CAM
    input_tensor.requires_grad_(True)
    
    # Run Grad-CAM (also gets model output)
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
