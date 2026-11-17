# Malnutrition Prediction Multi-Model API (v2.1-revision)

A modular FastAPI inference service for predicting acute child malnutrition using non-circular machine learning models (internally validated only), **Predictor Set C**. This is a **research prototype** that accompanies a published analysis. It has not been externally validated and must not be used to guide clinical decisions about individual children.

The service allows selecting from **7 trained algorithms** in a dedicated `models/` directory for comparative evaluation.

---

## 📊 Overview & Research Background

This API deploys models developed to correct critical data leakage identified in earlier literature:
- **Prior Research Issue (Circular Leakage)**: Existing literature predicted composite WHO Z-score outcomes using predictors ($WAZ, HAZ, WHZ$) derived from the exact same mathematical formulas, yielding artificial >99% accuracy.
- **Corrected Methodology**: The target was reformulated to a separately measured outcome: **`muac_malnourished`** ($1$ if $\text{MUAC} < 12.5\text{ cm}$, $0$ otherwise).
- **Multi-Model Suite (Set C)**: All models receive the 8 non-circular Set C features. The primary model in the paper is **Logistic Regression** ($\text{AUC} = 0.71$), alongside 6 other benchmarked algorithms stored under `models/`.

> [!IMPORTANT]
> **Intended Use**: This is a research prototype. A positive result means the model predicts MUAC below 12.5 cm; this must be confirmed by measuring MUAC.

---

## 🤖 Available Models

| Model Key (`model_name`) | Model File | Algorithm Type | Primary Model | Notes |
|:---|:---|:---|:---:|:---|
| **`logistic_regression`** *(default)* | `model_set_c_logistic_regression.pkl` | `Pipeline(StandardScaler, LogisticRegression)` | ⭐ **Yes** | Primary model in the paper. Balanced class weighting. Threshold: **0.662**. |
| **`random_forest`** | `model_set_c_random_forest.pkl` | `RandomForestClassifier(n_estimators=300, max_depth=6)` | No | Ensemble of 300 decision trees with max depth 6, min samples leaf 3. |
| **`gradient_boosting`** | `model_set_c_gradient_boosting.pkl` | `GradientBoostingClassifier()` | No | Sequential boosting ensemble optimizing deviance loss. |
| **`stacking_ensemble`** | `model_set_c_stacking_ensemble.pkl` | `StackingClassifier(LR + DT + RF)` | No | Meta-ensemble combining Logistic Regression, Decision Tree, and Random Forest. |
| **`svm_rbf`** | `model_set_c_svm_rbf.pkl` | `Pipeline(StandardScaler, SVC)` | No | Support Vector Classifier with RBF kernel and probability estimates. |
| **`mlp`** | `model_set_c_mlp.pkl` | `Pipeline(StandardScaler, MLPClassifier)` | No | Feedforward neural network with ReLU activation, hidden layer size (32,). |
| **`decision_tree`** | `model_set_c_decision_tree.pkl` | `DecisionTreeClassifier(max_depth=4)` | No | Single decision tree with maximum depth 4, min samples leaf 10. |

---

## 🔧 Quick Start

### Prerequisites
- Python 3.10+ (tested up to Python 3.13)
- Virtual environment (recommended)

### Installation
```bash
cd "Malnutrition-Prediction-python-service"
pip install -r requirements.txt
```

### Start the Server

#### On Windows (PowerShell or Command Prompt)
```powershell
.\run_server.ps1
# or:
run_server.bat
```

#### On Linux/Mac (or Git Bash)
```bash
chmod +x run_server.sh
./run_server.sh
```

#### Manual Startup
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Access Points
- **API Base URL**: `http://127.0.0.1:8000`
- **Model Registry Catalog**: `http://127.0.0.1:8000/models`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/health`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/` | API status and available endpoints |
| `GET` | `/models` | List all 7 models, loaded status, and algorithm descriptions |
| `GET` | `/health` | Health probe (verifies loading of all models) |
| `POST` | `/predict` | Predict malnutrition using selected model |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `GET` | `/redoc` | OpenAPI ReDoc specification |

---

## 📋 Request & Feature Specifications

### POST `/predict`

Predicts whether MUAC < 12.5 cm based on **8 features (Predictor Set C)**:

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

| Field | Type | Valid Range | Description |
|:---|:---:|:---:|:---|
| `month` | float | 0 – 120 | Age in months. **Cohort: 6–60 months. Predictions outside 6 to 60 months are extrapolations.** |
| `sex_male` | int | 0 or 1 | Child sex: `1` if Male, `0` if Female |
| `area_noakhali` | int | 0 or 1 | Study district: `1` if Noakhali, `0` if Lakshmipur |
| `weight` | float | 0 – 50 | Weight in kilograms (kg) |
| `height` | float | 0 – 150 | Height/length in centimeters (cm) |
| `waz` | float | -6.0 to +6.0 | Weight-for-Age Z-score (indicates underweight) |
| `haz` | float | -6.0 to +6.0 | Height-for-Age Z-score (indicates stunting) |
| `whz` | float | -6.0 to +6.0 | Weight-for-Height Z-score (indicates wasting) |
| `model_name` | string | *See table above* | **(Optional)** Algorithm to use. Defaults to `"logistic_regression"`. |

---

## 📊 Response Format

### Successful Prediction Response (`200 OK`)
```json
{
  "feature_set": "c",
  "model_used": "logistic_regression",
  "model_file": "model_set_c_logistic_regression.pkl",
  "model_type": "Pipeline",
  "prediction": 0,
  "status": "Predicted MUAC >= 12.5 cm",
  "risk_probability": 0.3399,
  "threshold": 0.662,
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

**Key response fields:**
- `prediction`: `1` = Predicted MUAC < 12.5 cm; `0` = Predicted MUAC ≥ 12.5 cm
- `status`: Human-readable prediction outcome
- `risk_probability`: Raw model probability (class-weighted, not a calibrated prevalence estimate)
- `threshold`: The classification threshold chosen by inner cross-validation in the paper (e.g. `0.662` for logistic regression), **not 0.5**

> [!NOTE]
> Each model classifies with the threshold chosen by inner cross-validation in the paper (0.662 for logistic regression), not 0.5. The probabilities shown are class-weighted and have not been calibrated to population prevalence.

---

## 📦 Model Files & Training

The model files in this directory were produced by `export_models.py` in the primary analysis repository:
**[https://github.com/imak2363/child-malnutrition](https://github.com/imak2363/child-malnutrition)**

For the full training procedure, hyperparameter settings, and analysis notebooks, see `models/README.md`.

---

## 🧪 Testing Multi-Model Inference

### Running the Automated Test Suite
```bash
python tests/run_tests.py
```

### cURL Example
```bash
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
       "model_name": "logistic_regression"
     }'
```

---

## 📁 Repository Architecture

```
Malnutrition-Prediction-python-service/
├── app/
│   ├── config.py             # Model registry, thresholds, and feature set definitions
│   ├── main.py               # FastAPI application factory and middleware
│   ├── routes/
│   │   └── predict.py        # Route handlers for /, /health, /models, and /predict
│   ├── schemas/
│   │   ├── prediction.py     # Pydantic schemas (InputData, PredictionResponse)
│   │   └── health.py         # Pydantic schemas (HealthResponse, ModelsListResponse)
│   └── services/
│       └── model_service.py  # Model loader, feature assembler, and inference engine
├── data/
│   └── Malnutrition_Analysis_Dataset_FINAL.csv  # 465-child cohort dataset
├── models/                   # Serialized scikit-learn models (Set C)
│   ├── README.md             # Training procedure and threshold documentation
│   ├── thresholds.json       # Decision thresholds for each model
│   └── model_set_c_*.pkl     # Serialized model files
├── export_models.py          # Reproduces .pkl files from the dataset (see models/README.md)
├── tests/
│   └── test_api.py           # Pytest test cases
├── requirements.txt
├── run_server.sh / .ps1 / .bat
└── README.md
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](../Predicting_malnutrition/LICENSE) file for details.
