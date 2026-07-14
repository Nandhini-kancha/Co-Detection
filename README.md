# 🫁 ChestAI — Pneumonia & Tuberculosis Co-Detection System

A production-ready AI medical imaging pipeline that simultaneously detects **Pneumonia** and **Tuberculosis** from chest X-rays using multi-label classification, Grad-CAM heatmaps for abnormality localization, and severity estimation.

---

## 🏗️ Architecture

```
chest-xray-detector/
├── backend/
│   ├── main.py              # FastAPI server: POST /predict, GET /health
│   ├── model.py             # DenseNet-121 multi-label classifier (3 outputs)
│   ├── grad_cam.py          # Grad-CAM heatmap generation → base64 PNG
│   ├── severity.py          # Probability → Severity level mapping
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadPanel.jsx     # Drag & drop X-ray upload
│   │   │   ├── ResultCard.jsx      # Probability bars + severity badge
│   │   │   └── HeatmapViewer.jsx   # Grad-CAM side-by-side viewer
│   │   ├── App.jsx           # Main dashboard orchestrator
│   │   ├── index.css         # Global styles, glassmorphism
│   │   └── main.jsx          # React entry point
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── train.py                  # Unified training script (NIH + Shenzhen + RSNA)
└── README.md
```

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML/DL** | PyTorch, DenseNet-121 (torchvision) |
| **Backend** | FastAPI, Uvicorn |
| **Heatmaps** | Custom Grad-CAM (hooks on last conv layer) |
| **Frontend** | React 18, Vite 5, Tailwind CSS 3 |
| **Charts** | Recharts |
| **Icons** | Lucide React |

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API starts at `http://localhost:8000`. If no model checkpoint exists, it runs in **demo mode** with random weights.

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The dashboard opens at `http://localhost:5173` with API proxy to the backend.

### 3. Training (Optional)

```bash
# With real datasets
python train.py --data_dir ./data --epochs 30 --batch_size 32 --lr 1e-4

# Demo mode (generates synthetic data)
python train.py
```

## 📡 API Reference

### `GET /health`

Returns API status, model state, and supported labels.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "labels": ["Pneumonia", "Tuberculosis", "Normal"]
}
```

### `POST /predict`

Upload a chest X-ray image for analysis.

**Request**: `multipart/form-data` with `file` field (image)

**Response**:
```json
{
  "probabilities": {
    "Pneumonia": 0.7234,
    "Tuberculosis": 0.1456,
    "Normal": 0.2891
  },
  "primary_diagnosis": "Pneumonia",
  "severity": {
    "level": "Moderate",
    "score": 0.7234,
    "color": "#F97316",
    "description": "Moderate abnormality detected. Further investigation recommended."
  },
  "heatmap_base64": "iVBORw0KGgo...",
  "inference_time_seconds": 1.234,
  "model_info": {
    "backbone": "DenseNet-121",
    "labels": ["Pneumonia", "Tuberculosis", "Normal"],
    "device": "cpu"
  }
}
```

## 📊 Severity Thresholds

| Max Disease Probability | Level | Color |
|------------------------|-------|-------|
| < 0.3 | Normal | 🟢 Green |
| 0.3 – 0.6 | Mild | 🟡 Amber |
| 0.6 – 0.8 | Moderate | 🟠 Orange |
| ≥ 0.8 | Severe | 🔴 Red |

## 📁 Dataset Configuration

### Supported Datasets

1. **NIH ChestX-ray14** (112,120 images)
   ```
   data/nih/
   ├── Data_Entry_2017.csv
   └── images/
       ├── 00000001_000.png
       └── ...
   ```

2. **Shenzhen TB Dataset** (662 images)
   ```
   data/shenzhen/
   └── images/
       ├── CHNCXR_0001_0.png  (normal)
       ├── CHNCXR_0002_1.png  (TB positive)
       └── ...
   ```

3. **RSNA Pneumonia Detection** (Kaggle)
   ```
   data/rsna/
   ├── stage_2_train_labels.csv
   └── stage_2_train_images/
       ├── patient_id.dcm
       └── ...
   ```

The training script automatically discovers available datasets and merges them. If no datasets are found, it generates a synthetic demo dataset.

## 🧠 Model Details

- **Backbone**: DenseNet-121 (pretrained on ImageNet)
- **Classifier Head**: Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→3)
- **Loss**: BCEWithLogitsLoss with class-weight balancing
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine Annealing
- **Augmentation**: RandomCrop, HorizontalFlip, Rotation, ColorJitter

## ⚠️ Clinical Disclaimer

This tool is for **research and screening purposes only**. Results must be validated by a qualified radiologist. Not approved for clinical diagnosis.

## 📄 License

MIT License
