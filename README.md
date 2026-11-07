# 🚀 Malnutrition Prediction Multi-Model API (v2.1)

A modular, production-grade FastAPI machine learning inference service for predicting acute child malnutrition using validated, non-circular machine learning models (**Predictor Set C**).

The service allows selecting on-the-fly from **7 trained algorithms** organized in a dedicated `models/` directory for comparative evaluation and clinical triage.

---

## 📊 Overview & Research Background

This API deploys models developed to correct critical data leakage identified in earlier literature:
- **Prior Research Issue (Circular Leakage)**: Existing literature predicted composite WHO Z-score outcomes using predictors ($WAZ, HAZ, WHZ$) derived from the exact same mathematical formulas, yielding artificial $>99\%$ accuracy.
- **Corrected Methodology**: The target was reformulated to an independent clinical outcome: **`muac_malnourished`** ($1$ if $\text{MUAC} < 12.5\text{ cm}$, $0$ otherwise).
- **Multi-Model Suite (Set C)**: All models receive the 8 non-circular Set C features. The primary recommended model is **Logistic Regression** ($AUC \approx 0.72$), alongside 6 other benchmarked algorithms stored under `models/`.

---

## 🤖 Available Models

| Model Key (`model_name`) | Model File Location | Algorithm Type | Recommended | Clinical Notes |
|:---|:---|:---|:---:|:---|
| **`logistic_regression`** *(default)* | `models/logistic_regression_set_c.pkl` | `Pipeline(StandardScaler, LogisticRegression)` | ⭐ **Yes** | Optimal model selected in paper. Balanced class weighting, robust generalization ($AUC \approx 0.72$), fully interpretable. |
| **`random_forest`** | `models/random_forest_set_c.pkl` | `RandomForestClassifier` | No | Bagged ensemble of decision trees with high non-linear interaction capacity. |
| **`gradient_boosting`** | `models/gradient_boosting_set_c.pkl` | `GradientBoostingClassifier` | No | Sequential boosting ensemble optimizing deviance loss. |
| **`stacking_ensemble`** | `models/stacking_ensemble_set_c.pkl` | `StackingClassifier` | No | Meta-ensemble combining multiple base models with a logistic meta-learner. |
| **`svm_rbf`** | `models/svm_rbf_set_c.pkl` | `Pipeline(StandardScaler, SVC)` | No | Support Vector Machine with Radial Basis Function kernel and probability calibration. |
| **`mlp`** | `models/mlp_set_c.pkl` | `Pipeline(StandardScaler, MLPClassifier)` | No | Feedforward artificial neural network trained via backpropagation. |
| **`decision_tree`** | `models/decision_tree_set_c.pkl` | `DecisionTreeClassifier` | No | Single tree classifier with axis-aligned splits. |

---

## 🔧 Quick Start

### Prerequisites
- Python 3.10+ (tested up to Python 3.13)
- Virtual environment (recommended)

### Installation
```bash
# Clone or navigate to the project directory
cd "Malnutrition-Prediction-python-service"

# Install dependencies
pip install -r requirements.txt
```

### Start the Server

#### Option 1: On Windows (PowerShell or Command Prompt)
```powershell
# Using PowerShell script:
.\run_server.ps1

# Or using batch file:
run_server.bat
```

#### Option 2: On Linux/Mac (or Git Bash)
```bash
chmod +x run_server.sh
./run_server.sh
```

#### Option 3: Manual Startup
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows: venv\Scripts\activate
# On Linux/Mac: source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run Uvicorn
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Access Points
- **API Base URL**: `http://127.0.0.1:8000`
- **Model Registry Catalog**: `http://127.0.0.1:8000/models`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Alternative Docs**: `http://127.0.0.1:8000/redoc`
- **Health Check**: `http://127.0.0.1:8000/health`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/` | API status, default model, and available endpoints |
| `GET` | `/models` | List all 7 models, loaded status, and algorithm descriptions |
| `GET` | `/health` | Health probe (verifies memory loading of all 7 models) |
| `POST` | `/predict` | Predict malnutrition risk using selected model |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/redoc` | OpenAPI ReDoc specification |

---

## 📋 Request & Feature Specifications

### POST `/predict`

Predicts acute malnutrition risk based on **8 features (Predictor Set C)** and an optional **`model_name`**:

```json
{
  "month": 39.66,
  "sex_male": 1,
  "area_noakhali": 0,
  "weight": 11.0,
  "height": 80.0,
  "waz": -2.529,
  "haz": -4.76,
  "whz": 0.468,
  "model_name": "logistic_regression"
}
```

### Field Definitions

| Field | Type | Expected Range | Description & Clinical Context |
|:---|:---:|:---:|:---|
| `month` | float | 0 – 120 | Age in months (cohort focused on 6 – 60 months) |
| `sex_male` | int | 0 or 1 | Child sex: `1` if Male, `0` if Female |
| `area_noakhali` | int | 0 or 1 | Geographic study district: `1` if Noakhali, `0` if Lakshmipur |
| `weight` | float | 0 – 50 | Weight in kilograms (kg) |
| `height` | float | 0 – 150 | Height/length in centimeters (cm) |
| `waz` | float | -6.0 to +6.0 | Weight-for-Age Z-score (indicates underweight) |
| `haz` | float | -6.0 to +6.0 | Height-for-Age Z-score (indicates stunting) |
| `whz` | float | -6.0 to +6.0 | Weight-for-Height Z-score (indicates wasting) |
| `model_name` | string | *See table above* | **(Optional)** Algorithm to use. Defaults to `"logistic_regression"`. Aliases accepted (e.g. `"rf"`, `"gb"`, `"svm"`, `"stacking"`, `"mlp"`). |

> ℹ️ You can also override the model using a query parameter:  
> `POST /predict?model=random_forest` (overrides payload `model_name`).

---

## 📊 Response Format

### Successful Prediction Response (`200 OK`)
```json
{
  "model_used": "logistic_regression",
  "model_file": "logistic_regression_set_c.pkl",
  "model_type": "Pipeline",
  "prediction": 0,
  "status": "Normal",
  "risk_probability": 0.3399,
  "risk_level": "Moderate Risk",
  "model_loaded": true,
  "features_used": [39.66, 1.0, 0.0, 11.0, 80.0, -2.529, -4.76, 0.468],
  "feature_dict": {
    "month": 39.66,
    "sex_male": 1.0,
    "area_noakhali": 0.0,
    "weight": 11.0,
    "height": 80.0,
    "waz": -2.529,
    "haz": -4.76,
    "whz": 0.468
  }
}
```

### Health Check Response (`GET /health`)
```json
{
  "status": "healthy",
  "total_configured": 7,
  "loaded_count": 7,
  "loaded_models": [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "stacking_ensemble",
    "svm_rbf",
    "mlp",
    "decision_tree"
  ],
  "unloaded_models": [],
  "message": "All 7 models are loaded and ready for inference"
}
```

---

## 🧪 Testing Multi-Model Inference

### Running the Automated Test Suite
```bash
python tests/run_tests.py
```

### PowerShell Example: Comparing Two Models
```powershell
# Define child clinical metrics
$child = @{
    month = 24.0
    sex_male = 1
    area_noakhali = 1
    weight = 7.5
    height = 85.0
    waz = -3.5
    haz = -1.5
    whz = -3.8
}

# 1. Predict with Logistic Regression (Recommended)
$bodyLR = $child.Clone()
$bodyLR["model_name"] = "logistic_regression"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Body ($bodyLR | ConvertTo-Json) -ContentType "application/json"

# 2. Predict with Stacking Ensemble
$bodyStack = $child.Clone()
$bodyStack["model_name"] = "stacking_ensemble"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -Body ($bodyStack | ConvertTo-Json) -ContentType "application/json"
```

### cURL Example
```bash
# Using model_name in JSON body:
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "month": 24.0,
       "sex_male": 0,
       "area_noakhali": 1,
       "weight": 9.5,
       "height": 78.0,
       "waz": -1.8,
       "haz": -2.1,
       "whz": -0.9,
       "model_name": "random_forest"
     }'
```

---

## 📁 Repository Architecture

The project is structured according to clean architecture principles:

```
Malnutrition-Prediction-python-service/
├── app/
│   ├── __init__.py
│   ├── config.py             # Global constants, paths, thresholds, and model configs
│   ├── main.py               # FastAPI application factory and middleware configuration
│   ├── routes/
│   │   ├── __init__.py
│   │   └── predict.py        # Route handlers for /, /health, /models, and /predict
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── prediction.py     # Pydantic schemas (InputData, PredictionResponse, ModelChoice)
│   │   └── health.py         # Pydantic schemas (HealthResponse, ModelsListResponse)
│   └── services/
│       ├── __init__.py
│       └── model_service.py  # Model lifecycle manager, feature assembler, and inference engine
├── models/                   # Dedicated directory for serialized machine learning models
│   ├── decision_tree_set_c.pkl
│   ├── gradient_boosting_set_c.pkl
│   ├── logistic_regression_set_c.pkl
│   ├── mlp_set_c.pkl
│   ├── random_forest_set_c.pkl
│   ├── stacking_ensemble_set_c.pkl
│   └── svm_rbf_set_c.pkl
├── tests/
│   ├── __init__.py
│   ├── run_tests.py          # Automated test suite executable with python tests/run_tests.py
│   └── test_api.py           # Pytest test cases
├── main.py                   # Root entrypoint exposing 'app' for uvicorn main:app
├── requirements.txt          # Python package dependencies
├── run_server.sh             # Linux/macOS/Git Bash startup script
├── run_server.ps1            # Windows PowerShell startup script
├── run_server.bat            # Windows Batch startup script
├── README.md                 # Project documentation
└── .gitignore                # Git ignore configuration
```
