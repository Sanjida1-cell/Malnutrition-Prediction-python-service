"""Prediction request and response schemas supporting Sets A, B, and C."""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class ModelChoice(str, Enum):
    """Available machine learning algorithm keys."""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    STACKING_ENSEMBLE = "stacking_ensemble"
    SVM_RBF = "svm_rbf"
    MLP = "mlp"
    DECISION_TREE = "decision_tree"


class FeatureSetChoice(str, Enum):
    """Available predictor feature sets."""
    SET_A = "a"
    SET_B = "b"
    SET_C = "c"


class InputData(BaseModel):
    """
    Child anthropometric, demographic and clinical input feature vector.
    Features dynamically adapt based on the selected `feature_set`:
      - Set A: month, sex_male, area_noakhali (3 features)
      - Set B: month, sex_male, area_noakhali, weight, height (5 features)
      - Set C: month, sex_male, area_noakhali, weight, height, waz, haz, whz (8 features)
    """
    feature_set: Optional[str] = Field(
        default="c",
        description="Predictor set to use: 'a' (demographics), 'b' (demographics + body size), or 'c' (full predictors + WHO Z-scores). Defaults to 'c'."
    )
    month: float = Field(
        ..., ge=0, le=120, description="Child age in months (cohort focused on 6 to 60 months; predictions outside 6 to 60 months are extrapolations)"
    )
    sex_male: int = Field(
        ..., ge=0, le=1, description="Child sex: 1 if Male, 0 if Female"
    )
    area_noakhali: int = Field(
        ..., ge=0, le=1, description="Study district: 1 if Noakhali, 0 if Lakshmipur"
    )
    weight: Optional[float] = Field(
        default=None, ge=0, le=50, description="Child weight in kilograms (kg) - required for Set B & Set C"
    )
    height: Optional[float] = Field(
        default=None, ge=0, le=150, description="Child height/length in centimeters (cm) - required for Set B & Set C"
    )
    waz: Optional[float] = Field(
        default=None, ge=-6.0, le=6.0, description="Weight-for-Age Z-score (-6.0 to +6.0) - required for Set C"
    )
    haz: Optional[float] = Field(
        default=None, ge=-6.0, le=6.0, description="Height-for-Age Z-score (-6.0 to +6.0) - required for Set C"
    )
    whz: Optional[float] = Field(
        default=None, ge=-6.0, le=6.0, description="Weight-for-Height Z-score (-6.0 to +6.0) - required for Set C"
    )
    model_name: Optional[str] = Field(
        default="logistic_regression",
        description=(
            "Model algorithm to execute inference on. Options: 'logistic_regression' (default), "
            "'random_forest', 'gradient_boosting', 'stacking_ensemble', 'svm_rbf', 'mlp', 'decision_tree'."
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "feature_set": "c",
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
    """Research prototype prediction result."""
    feature_set: str = Field(..., description="Feature set used for prediction ('a', 'b', or 'c')")
    model_used: str = Field(..., description="Key of the model algorithm selected")
    model_file: str = Field(..., description="File name of the model")
    model_type: str = Field(..., description="Underlying ML algorithm class name")
    prediction: int = Field(..., description="Binary classification (0 = Normal/MUAC>=12.5cm, 1 = Predicted MUAC<12.5cm)")
    status: str = Field(..., description="Status interpretation ('Predicted MUAC >= 12.5 cm' or 'Predicted MUAC < 12.5 cm')")
    risk_probability: float = Field(..., description="Predicted probability of acute malnutrition (0.0 to 1.0)")
    threshold: float = Field(..., description="Model classification decision threshold chosen by inner cross-validation")
    model_loaded: bool = Field(..., description="Whether the model was loaded successfully")
    features_used: List[float] = Field(..., description="Ordered feature values used for prediction")
    feature_dict: Dict[str, float] = Field(..., description="Named feature values dictionary")
