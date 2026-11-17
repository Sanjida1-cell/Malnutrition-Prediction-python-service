"""Model service managing loading, caching, normalization, and predictions across Sets A, B, and C."""

import os
import sys
import pickle
import warnings
from typing import Dict, Any, Optional, Tuple
import numpy as np
from fastapi import HTTPException

# Suppress minor unpickling version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure Cython _loss module mapping exists for GradientBoostingClassifier unpickling
try:
    import sklearn._loss._loss as _loss_mod
    sys.modules["_loss"] = _loss_mod
except Exception:
    pass

from app.config import (
    MODELS_DIR,
    FEATURE_SETS,
    ALGORITHMS,
    MODEL_CONFIGS,
    DEFAULT_SET,
    DEFAULT_MODEL,
    MODEL_THRESHOLDS,
)
from app.schemas.prediction import InputData, PredictionResponse, ModelChoice, FeatureSetChoice


class ModelService:
    """Manages multi-set model lifecycle, resolution, and inference execution."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.load_models()

    def load_models(self) -> None:
        """Load all registered models across Sets A, B, and C from the models directory."""
        self.models.clear()

        # Ensure _loss is in sys.modules prior to unpickling
        try:
            import sklearn._loss._loss as _loss_mod
            sys.modules["_loss"] = _loss_mod
        except Exception:
            pass

        for composite_key, config in MODEL_CONFIGS.items():
            file_path = MODELS_DIR / config["file"]
            if not file_path.exists():
                print(f"[WARN] Model file '{file_path}' for '{composite_key}' not found.")
                continue
            try:
                with open(file_path, "rb") as f:
                    self.models[composite_key] = pickle.load(f)
                print(f"[SUCCESS] Loaded model '{composite_key}' from {config['file']}")
            except Exception as e:
                print(f"[ERROR] Failed to load model '{composite_key}' from {file_path}: {e}")

    def resolve_feature_set(self, set_identifier: Optional[Any]) -> str:
        """Normalize feature set identifier to 'a', 'b', or 'c'."""
        if set_identifier is None:
            return DEFAULT_SET

        if isinstance(set_identifier, FeatureSetChoice):
            return set_identifier.value

        clean_id = str(set_identifier).strip().lower()
        if not clean_id:
            return DEFAULT_SET

        if clean_id in ("a", "set_a", "set-a", "set a", "demographics"):
            return "a"
        if clean_id in ("b", "set_b", "set-b", "set b", "measurements"):
            return "b"
        if clean_id in ("c", "set_c", "set-c", "set c", "full", "recommended", "who"):
            return "c"

        raise HTTPException(
            status_code=400,
            detail=f"Unknown feature set '{set_identifier}'. Valid options are: 'a', 'b', 'c'.",
        )

    def resolve_algorithm_key(self, model_identifier: Optional[Any]) -> str:
        """Normalize and resolve an algorithm identifier to its canonical key."""
        if model_identifier is None:
            return DEFAULT_MODEL

        if isinstance(model_identifier, ModelChoice):
            return model_identifier.value

        clean_id = str(model_identifier).strip().lower()
        if not clean_id:
            return DEFAULT_MODEL

        # Direct canonical key match
        if clean_id in ALGORITHMS:
            return clean_id

        # Flexible alias mapping
        alias_map = {
            "lr": "logistic_regression",
            "logistic": "logistic_regression",
            "logreg": "logistic_regression",
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

        valid_options = list(ALGORITHMS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm '{model_identifier}'. Available options are: {valid_options}",
        )

    def resolve_model_key(
        self,
        set_identifier: Optional[Any],
        model_identifier: Optional[Any],
    ) -> Tuple[str, str, str]:
        """
        Resolves (set_key, algorithm_key, composite_key).
        Accepts composite keys like 'set_a_logistic_regression' or individual set & model identifiers.
        """
        raw_model = str(model_identifier or "").strip().lower()

        # If model_identifier contains set prefix (e.g. "set_a_logistic_regression" or "a_logistic_regression")
        for s in ("a", "b", "c"):
            prefix_match1 = f"model_set_{s}_"
            prefix_match2 = f"set_{s}_"
            prefix_match3 = f"{s}_"
            if raw_model.startswith(prefix_match1):
                clean_algo = raw_model[len(prefix_match1):].replace(".pkl", "")
                set_key = s
                algo_key = self.resolve_algorithm_key(clean_algo)
                return set_key, algo_key, f"{set_key}_{algo_key}"
            if raw_model.startswith(prefix_match2):
                clean_algo = raw_model[len(prefix_match2):]
                set_key = s
                algo_key = self.resolve_algorithm_key(clean_algo)
                return set_key, algo_key, f"{set_key}_{algo_key}"
            if raw_model.startswith(prefix_match3) and raw_model != s:
                clean_algo = raw_model[len(prefix_match3):]
                if clean_algo in ALGORITHMS or clean_algo in ["lr", "rf", "gb", "dt", "svm", "mlp", "stacking"]:
                    set_key = s
                    algo_key = self.resolve_algorithm_key(clean_algo)
                    return set_key, algo_key, f"{set_key}_{algo_key}"

        set_key = self.resolve_feature_set(set_identifier)
        algo_key = self.resolve_algorithm_key(model_identifier)
        composite_key = f"{set_key}_{algo_key}"
        return set_key, algo_key, composite_key

    def predict(
        self,
        data: InputData,
        model_override: Optional[str] = None,
        set_override: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Execute prediction on child data using the resolved set and model.
        """
        # Determine set and model key
        chosen_set = set_override if (isinstance(set_override, str) and set_override.strip()) else data.feature_set
        chosen_model = model_override if (isinstance(model_override, str) and model_override.strip()) else data.model_name

        set_key, algo_key, composite_key = self.resolve_model_key(chosen_set, chosen_model)

        active_model = self.models.get(composite_key)
        if active_model is None:
            config = MODEL_CONFIGS.get(composite_key, {})
            file_name = config.get("file", f"model_set_{set_key}_{algo_key}.pkl")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Model '{composite_key}' ({file_name}) is not loaded. "
                    "Please verify that the file exists in the models directory."
                ),
            )

        # Validate inputs for the chosen set
        feature_dict: Dict[str, float] = {
            "month": float(data.month),
            "sex_male": float(data.sex_male),
            "area_noakhali": float(data.area_noakhali),
        }

        if set_key in ("b", "c"):
            if data.weight is None:
                raise HTTPException(status_code=422, detail="Weight (kg) is required for Set B and Set C models.")
            if data.height is None:
                raise HTTPException(status_code=422, detail="Height (cm) is required for Set B and Set C models.")
            feature_dict["weight"] = float(data.weight)
            feature_dict["height"] = float(data.height)

        if set_key == "c":
            if data.waz is None or data.haz is None or data.whz is None:
                raise HTTPException(
                    status_code=422,
                    detail="WHO Z-scores (waz, haz, whz) are required for Set C models.",
                )
            feature_dict["waz"] = float(data.waz)
            feature_dict["haz"] = float(data.haz)
            feature_dict["whz"] = float(data.whz)

        # Construct ordered feature array matching model's expected features
        feature_names = FEATURE_SETS[set_key]["features"]
        feature_values = [feature_dict[name] for name in feature_names]
        features = np.array([feature_values], dtype=np.float64)

        try:
            # Class 1 probability
            if hasattr(active_model, "predict_proba"):
                probabilities = active_model.predict_proba(features)[0]
                risk_probability = round(float(probabilities[1]), 4)
            else:
                risk_probability = -1.0  # fallback: no proba available

            # Classification via paper threshold (not 0.5)
            decision_threshold = MODEL_THRESHOLDS.get(algo_key, 0.5)
            if risk_probability >= 0:
                raw_prediction = 1 if risk_probability >= decision_threshold else 0
            else:
                raw_prediction = int(active_model.predict(features)[0])
                risk_probability = 1.0 if raw_prediction == 1 else 0.0

            status = (
                "Predicted MUAC < 12.5 cm" if raw_prediction == 1
                else "Predicted MUAC >= 12.5 cm"
            )
            model_config = MODEL_CONFIGS[composite_key]

            return PredictionResponse(
                feature_set=set_key,
                model_used=algo_key,
                model_file=model_config["file"],
                model_type=type(active_model).__name__,
                prediction=raw_prediction,
                status=status,
                risk_probability=risk_probability,
                threshold=decision_threshold,
                model_loaded=True,
                features_used=feature_values,
                feature_dict=feature_dict,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error using model '{composite_key}': {str(e)}",
            )


# Global singleton service instance
model_service = ModelService()
