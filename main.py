from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import List, Optional

# Load the trained ensemble model with error handling
try:
    if not os.path.exists("ensemble_model.pkl"):
        raise FileNotFoundError("Model file 'ensemble_model.pkl' not found")
    model = joblib.load("ensemble_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Define the input data format with 11 features matching CSV columns
class InputData(BaseModel):
    month: float = Field(..., ge=0, le=120, description="Age in months")
    weight: float = Field(..., ge=0, le=50, description="Weight in kg")
    height: float = Field(..., ge=0, le=150, description="Height in cm")
    muac: float = Field(..., ge=0, le=30, description="Mid-Upper Arm Circumference in cm")
    waz: float = Field(..., ge=-6, le=6, description="Weight-for-Age Z-score")
    haz: float = Field(..., ge=-6, le=6, description="Height-for-Age Z-score")
    whz: float = Field(..., ge=-6, le=6, description="Weight-for-Height Z-score")
    sex_f: float = Field(..., ge=0, le=1, description="Sex Female (0 or 1)")
    sex_m: float = Field(..., ge=0, le=1, description="Sex Male (0 or 1)")
    area_lakshmipur: float = Field(..., ge=0, le=1, description="Area Lakshmipur (0 or 1)")
    area_noakhali: float = Field(..., ge=0, le=1, description="Area NoaKhali (0 or 1)")

class PredictionResponse(BaseModel):
    prediction: List[float] = Field(..., description="Model prediction results")
    model_loaded: bool = Field(..., description="Whether the model was successfully loaded")
    features_used: List[float] = Field(..., description="Input features used for prediction")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    message: str = Field(..., description="Status message")

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Malnutrition Prediction API",
    description="Machine Learning API for predicting malnutrition using ensemble model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/", response_model=dict)
def home():
    """Root endpoint - API status check"""
    return {"message": "Malnutrition Prediction API is running", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        message="Model loaded successfully" if model is not None else "Model not loaded"
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    """
    Make prediction using the loaded ensemble model
    
    - **month**: Age in months (0-120)
    - **weight**: Weight in kg (0-50)
    - **height**: Height in cm (0-150)
    - **muac**: Mid-Upper Arm Circumference in cm (0-30)
    - **waz**: Weight-for-Age Z-score (-6 to 6)
    - **haz**: Height-for-Age Z-score (-6 to 6)
    - **whz**: Weight-for-Height Z-score (-6 to 6)
    - **sex_f**: Sex Female (0 or 1)
    - **sex_m**: Sex Male (0 or 1)
    - **area_lakshmipur**: Area Lakshmipur (0 or 1)
    - **area_noakhali**: Area NoaKhali (0 or 1)
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs and ensure ensemble_model.pkl exists"
        )
    
    try:
        # Prepare features array with all 11 features matching CSV columns
        features = np.array([[
            data.month, data.weight, data.height, data.muac, data.waz,
            data.haz, data.whz, data.sex_f, data.sex_m, 
            data.area_lakshmipur, data.area_noakhali
        ]])
        
        # Make prediction
        prediction = model.predict(features)
        
        return PredictionResponse(
            prediction=prediction.tolist(),
            model_loaded=True,
            features_used=features[0].tolist()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )
