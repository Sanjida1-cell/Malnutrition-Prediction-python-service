from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained ensemble model
model = joblib.load("ensemble_model.pkl")

# Define the input data format
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add other features here

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Model API is running."}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.feature1, data.feature2, data.feature3]])  # Adjust if more features
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
