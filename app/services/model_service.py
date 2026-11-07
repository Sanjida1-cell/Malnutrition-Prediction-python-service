"""Model service managing loading, caching, normalization, and predictions."""

import os
import sys
import pickle
import warnings
from typing import Dict, Any, Optional
import numpy as np
from fastapi import HTTPException

from app.config import (
    MODELS_DIR,
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    FEATURE_NAMES,
    RISK_THRESHOLD_MODERATE,
    RISK_THRESHOLD_HIGH,
)
from app.schemas.prediction import InputData, PredictionResponse, ModelChoice

# Suppress minor unpickling version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure Cython _loss module mapping exists if needed for GradientBoostingClassifier
try:
    import sklearn._loss._loss as _loss_mod
    sys.modules["_loss"] = _loss_mod
except Exception:
    pass


class ModelService:
    """Manages model lifecycle, resolution, and inference execution."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.load_models()

    def load_models(self) -> None:
        """Load all registered Set C models from the models directory into memory."""
        self.models.clear()
        
        for key, config in MODEL_CONFIGS.items():
            file_path = MODELS_DIR / config["file"]
            if not file_path.exists():
                print(f"[WARN] Model file '{file_path}' for '{key}' not found.")
                continue
            try:
                with open(file_path, "rb") as f:
                    self.models[key] = pickle.load(f)
                print(f"[SUCCESS] Loaded model '{key}' from {file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model '{key}' from {file_path}: {e}")

    def resolve_model_key(self, model_identifier: Optional[Any]) -> str:
        """
        Normalize and resolve a model identifier to a canonical key.
        Accepts ModelChoice Enum, canonical key, file name, or common shorthand aliases.
        """
        if model_identifier is None:
            return DEFAULT_MODEL

        # Handle ModelChoice enum
        if isinstance(model_identifier, ModelChoice):
            return model_identifier.value

        clean_id = str(model_identifier).strip().lower()
        if not clean_id:
            return DEFAULT_MODEL

        # Direct canonical key match
        if clean_id in MODEL_CONFIGS:
            return clean_id

        # Match on filename (with or without .pkl)
        for key, config in MODEL_CONFIGS.items():
            if clean_id == config["file"].lower() or clean_id == config["file"].replace(".pkl", "").lower():
                return key

        # Flexible alias mapping
        alias_map = {
            "lr": "logistic_regression",
            "logistic": "logistic_regression",
            "rf": "random_forest",
            "forest": "random_forest",
            "gb": "gradient_boosting",
            "gbm": "gradient_boosting",
            "boosting": "gradient_boosting",
            "dt": "decision_tree",
            "tree": "decision_tree",
            "nn": "mlp",
            "neural": "mlp",
            "neural_network": "mlp",
            "stacking": "stacking_ensemble",
            "stack": "stacking_ensemble",
            "ensemble": "stacking_ensemble",
            "svm": "svm_rbf",
            "svc": "svm_rbf",
            "rbf": "svm_rbf",
        }
        if clean_id in alias_map:
            return alias_map[clean_id]

        valid_options = list(MODEL_CONFIGS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_identifier}'. Available options are: {valid_options}",
        )

    def predict(
        self,
        data: InputData,
        model_override: Optional[str] = None
    ) -> PredictionResponse:
        """
        Execute prediction on child data using the resolved model.
        """
        # Determine model key (override parameter takes precedence if supplied as non-empty string)
        chosen_identifier = (
            model_override
            if isinstance(model_override, str) and model_override.strip()
            else data.model_name
        )
        model_key = self.resolve_model_key(chosen_identifier)

        active_model = self.models.get(model_key)
        if active_model is None:
            file_name = MODEL_CONFIGS[model_key]["file"]
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Model '{model_key}' ({file_name}) is not loaded. "
                    "Please verify that the file exists in the models directory."
                ),
            )

        try:
            # Build feature dictionary
            feature_dict = {
                "month": float(data.month),
                "sex_male": float(data.sex_male),
                "area_noakhali": float(data.area_noakhali),
                "weight": float(data.weight),
                "height": float(data.height),
                "waz": float(data.waz),
                "haz": float(data.haz),
                "whz": float(data.whz),
            }

            # Assemble NumPy 2D array in exact training order
            features = np.array(
                [[feature_dict[name] for name in FEATURE_NAMES]],
                dtype=np.float64,
            )

            # Binary prediction: 0 = Normal, 1 = Malnourished
            raw_prediction = int(active_model.predict(features)[0])

            # Class 1 (Malnourished) probability
            if hasattr(active_model, "predict_proba"):
                probabilities = active_model.predict_proba(features)[0]
                risk_probability = round(float(probabilities[1]), 4)
            else:
                risk_probability = 1.0 if raw_prediction == 1 else 0.0

            # Determine risk tier
            if risk_probability >= RISK_THRESHOLD_HIGH:
                risk_level = "High Risk"
            elif risk_probability >= RISK_THRESHOLD_MODERATE:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"

            status = "Malnourished" if raw_prediction == 1 else "Normal"
            model_config = MODEL_CONFIGS[model_key]

            return PredictionResponse(
                model_used=model_key,
                model_file=model_config["file"],
                model_type=type(active_model).__name__,
                prediction=raw_prediction,
                status=status,
                risk_probability=risk_probability,
                risk_level=risk_level,
                model_loaded=True,
                features_used=features[0].tolist(),
                feature_dict=feature_dict,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error using model '{model_key}': {str(e)}",
            )


# Global singleton service instance
model_service = ModelService()
