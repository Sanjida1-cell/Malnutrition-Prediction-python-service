# 🚀 Malnutrition Prediction API

A FastAPI-based machine learning service for predicting malnutrition using an ensemble model with 11 features.

## 📊 API Overview

This service provides REST API endpoints for malnutrition prediction using a trained ensemble model. It accepts 11 features related to child health metrics and returns malnutrition risk predictions.

## 🔧 Quick Start

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation
```bash
# Clone or download the project
# Navigate to the project directory
cd "c:\Users\ASUS\OneDrive\Desktop\sanjida project\Prediction_malnutrition_model_service"

# Install dependencies
pip install -r requirements.txt
```

### Start the Server

#### Option 1: Using the Start Script (Recommended)
```bash
# On Linux/Mac
./run_server.sh

# On Windows (if you have bash/git bash)
bash run_server.sh
```

#### Option 2: Direct Command
```bash
# Manual start
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

#### Option 3: With Virtual Environment (Manual)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies and run
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Access the API
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Documentation**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/health`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Make malnutrition predictions |
| `GET` | `/health` | Check API and model status |
| `GET` | `/docs` | Interactive API documentation |
| `GET` | `/` | API status information |

## 📋 Request Format

### Prediction Request
**POST** `/predict`

```json
{
  "month": 39.66,
  "weight": 11.0,
  "height": 80.0,
  "muac": 13.5,
  "waz": -2.529,
  "haz": -4.76,
  "whz": 0.468,
  "sex_f": 0.0,
  "sex_m": 1.0,
  "area_lakshmipur": 1.0,
  "area_noakhali": 0.0
}
```

### Field Specifications

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `month` | float | 0-120 | Age in months |
| `weight` | float | 0-50 | Weight in kg |
| `height` | float | 0-150 | Height in cm |
| `muac` | float | 0-30 | Mid-Upper Arm Circumference in cm |
| `waz` | float | -6 to 6 | Weight-for-Age Z-score |
| `haz` | float | -6 to 6 | Height-for-Age Z-score |
| `whz` | float | -6 to 6 | Weight-for-Height Z-score |
| `sex_f` | float | 0 or 1 | Sex Female (0 or 1) |
| `sex_m` | float | 0 or 1 | Sex Male (0 or 1) |
| `area_lakshmipur` | float | 0 or 1 | Area Lakshmipur (0 or 1) |
| `area_noakhali` | float | 0 or 1 | Area NoaKhali (0 or 1) |

## 📊 Response Format

### Successful Prediction Response
```json
{
  "prediction": [2.0],
  "model_loaded": true,
  "features_used": [39.66, 11.0, 80.0, 13.5, -2.529, -4.76, 0.468, 0.0, 1.0, 1.0, 0.0]
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Model loaded successfully"
}
```

## 🧪 Testing the API

### PowerShell Example
```powershell
# Health check
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method GET

# Prediction request
$body = @{
    month = 39.66; weight = 11.0; height = 80.0; muac = 13.5
    waz = -2.529; haz = -4.76; whz = 0.468
    sex_f = 0.0; sex_m = 1.0; area_lakshmipur = 1.0; area_noakhali = 0.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

### Python Example
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "month": 39.66, "weight": 11.0, "height": 80.0, "muac": 13.5,
        "waz": -2.529, "haz": -4.76, "whz": 0.468,
        "sex_f": 0.0, "sex_m": 1.0, "area_lakshmipur": 1.0, "area_noakhali": 0.0
    }
)
print(response.json())
```

## 📁 Project Structure

```
├── main.py                 # Main FastAPI application
├── ensemble_model.pkl      # Trained ML model
├── requirements.txt        # Python dependencies
├── run_server.sh          # Bash server start script
├── README.md              # This documentation
└── .gitignore             # Git ignore rules
```

## 🔒 Status Codes

| Code | Description | When it occurs |
|------|-------------|----------------|
| `200` | Success | Valid request with successful prediction |
| `422` | Validation Error | Invalid input data or missing fields |
| `500` | Internal Server Error | Model not loaded or prediction failure |

## 🎯 Prediction Values

- `0`: No malnutrition
- `1`: Moderate malnutrition
- `2`: Severe malnutrition

## 🚨 Troubleshooting

1. **Model not loading**: Ensure `ensemble_model.pkl` exists in the project directory
2. **Port already in use**: Change port with `--port 8001`
3. **Import errors**: Install dependencies with `pip install -r requirements.txt`

---

**🎉 Your malnutrition prediction API is ready for production use!**