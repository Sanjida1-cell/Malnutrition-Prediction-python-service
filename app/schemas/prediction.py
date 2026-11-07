"""Prediction request and response schemas."""

from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    """Available Set C machine learning models."""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    STACKING_ENSEMBLE = "stacking_ensemble"
    SVM_RBF = "svm_rbf"
    MLP = "mlp"
    DECISION_TREE = "decision_tree"


class InputData(BaseModel):
    """
    Child anthropometric and demographic input feature vector (Predictor Set C)
    with model selector.
    """
    month: float = Field(
        ..., ge=0, le=120, description="Child age in months (cohort focused on 6-60 months)"
    )
    sex_male: int = Field(
        ..., ge=0, le=1, description="Child sex: 1 if Male, 0 if Female"
    )
    area_noakhali: int = Field(
        ..., ge=0, le=1, description="Study district: 1 if Noakhali, 0 if Lakshmipur"
    )
    weight: float = Field(
        ..., ge=0, le=50, description="Child weight in kilograms (kg)"
    )
    height: float = Field(
        ..., ge=0, le=150, description="Child height/length in centimeters (cm)"
    )
    waz: float = Field(
        ..., ge=-6.0, le=6.0, description="Weight-for-Age Z-score (-6.0 to +6.0)"
    )
    haz: float = Field(
        ..., ge=-6.0, le=6.0, description="Height-for-Age Z-score (-6.0 to +6.0)"
    )
    whz: float = Field(
        ..., ge=-6.0, le=6.0, description="Weight-for-Height Z-score (-6.0 to +6.0)"
    )
    model_name: Optional[ModelChoice] = Field(
        default=ModelChoice.LOGISTIC_REGRESSION,
        description=(
            "Model algorithm to execute prediction on. Options: 'logistic_regression' (default, recommended), "
            "'random_forest', 'gradient_boosting', 'stacking_ensemble', 'svm_rbf', 'mlp', 'decision_tree'."
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Clinical malnutrition prediction result and risk assessment."""
    model_used: str = Field(..., description="Key of the model selected for inference")
    model_file: str = Field(..., description="File name of the model")
    model_type: str = Field(..., description="Underlying ML algorithm class name")
    prediction: int = Field(..., description="Binary classification (0 = Normal, 1 = Malnourished)")
    status: str = Field(..., description="Clinical status interpretation ('Normal' or 'Malnourished')")
    risk_probability: float = Field(..., description="Predicted probability of acute malnutrition (0.0 to 1.0)")
    risk_level: str = Field(..., description="Clinical risk tier: 'Low Risk', 'Moderate Risk', or 'High Risk'")
    model_loaded: bool = Field(..., description="Whether the model was loaded successfully")
    features_used: List[float] = Field(..., description="Ordered feature values used for prediction")
    feature_dict: Dict[str, float] = Field(..., description="Named feature values dictionary")
