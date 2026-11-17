"""Application configuration and multi-set model registry definitions."""

from pathlib import Path
import os
import json
from typing import Dict, Any, List

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

# Feature definitions per set
FEATURE_SETS: Dict[str, Dict[str, Any]] = {
    "a": {
        "name": "Predictor Set A",
        "description": "Demographic indicators (Age, Sex, District)",
        "features": ["month", "sex_male", "area_noakhali"],
        "recommended": False,
    },
    "b": {
        "name": "Predictor Set B",
        "description": "Demographics + Anthropometric measurements (+ Weight, Height)",
        "features": ["month", "sex_male", "area_noakhali", "weight", "height"],
        "recommended": False,
    },
    "c": {
        "name": "Predictor Set C",
        "description": "Full Predictor Set: Demographics + Measurements + WHO Growth Standards (+ WAZ, HAZ, WHZ)",
        "features": ["month", "sex_male", "area_noakhali", "weight", "height", "waz", "haz", "whz"],
        "recommended": True,
    },
}

DEFAULT_SET = "c"
DEFAULT_MODEL = "logistic_regression"

# Model classification thresholds chosen by inner cross-validation in the paper
MODEL_THRESHOLDS: Dict[str, float] = {
    "logistic_regression": 0.662,
    "decision_tree": 0.783,
    "svm_rbf": 0.099,
    "mlp": 0.131,
    "random_forest": 0.576,
    "gradient_boosting": 0.106,
    "stacking_ensemble": 0.599,
}

# Load thresholds from thresholds.json if present
thresholds_path = MODELS_DIR / "thresholds.json"
if thresholds_path.exists():
    try:
        with open(thresholds_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            for k, v in t_data.items():
                if isinstance(v, dict) and "threshold" in v:
                    MODEL_THRESHOLDS[k] = float(v["threshold"])
                elif isinstance(v, (int, float)):
                    MODEL_THRESHOLDS[k] = float(v)
    except Exception:
        pass

# Available algorithms adhering to Table S4 specifications
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "algorithm": "Pipeline(StandardScaler, LogisticRegression)",
        "recommended": True,
        "threshold": MODEL_THRESHOLDS.get("logistic_regression", 0.662),
        "description": "Primary model in the paper. Balanced class weighting, threshold 0.662.",
    },
    "random_forest": {
        "name": "Random Forest Classifier",
        "algorithm": "RandomForestClassifier(n_estimators=300, max_depth=6)",
        "recommended": False,
        "threshold": MODEL_THRESHOLDS.get("random_forest", 0.576),
        "description": "Ensemble of 300 decision trees with max depth 6 and min samples leaf 3.",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting Classifier",
        "algorithm": "GradientBoostingClassifier()",
        "recommended": False,
        "threshold": MODEL_THRESHOLDS.get("gradient_boosting", 0.106),
        "description": "Sequential boosting ensemble optimizing deviance loss across predictors.",
    },
    "stacking_ensemble": {
        "name": "Stacking Ensemble Classifier",
        "algorithm": "StackingClassifier(LR + DT + RF)",
        "recommended": False,
        "threshold": MODEL_THRESHOLDS.get("stacking_ensemble", 0.599),
        "description": "Meta-ensemble combining Logistic Regression, Decision Tree, and Random Forest base models.",
    },
    "svm_rbf": {
        "name": "Support Vector Machine (RBF Kernel)",
        "algorithm": "Pipeline(StandardScaler, SVC)",
        "recommended": False,
        "threshold": MODEL_THRESHOLDS.get("svm_rbf", 0.099),
        "description": "Support Vector Classifier with Radial Basis Function kernel and probability estimates.",
    },
    "mlp": {
        "name": "Multi-Layer Perceptron (Neural Network)",
        "algorithm": "Pipeline(StandardScaler, MLPClassifier)",
        "recommended": False,
        "threshold": MODEL_THRESHOLDS.get("mlp", 0.131),
        "description": "Feedforward neural network with ReLU activation and hidden layer size (32,).",
    },
    "decision_tree": {
        "name": "Decision Tree Classifier",
        "algorithm": "DecisionTreeClassifier(max_depth=4)",
        "recommended": False,
        "threshold": MODEL_THRESHOLDS.get("decision_tree", 0.783),
        "description": "Single decision tree with maximum depth 4 and min samples leaf 10.",
    },
}

# Registry metadata for all 21 models (7 per set)
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {}

for set_key, set_info in FEATURE_SETS.items():
    for algo_key, algo_info in ALGORITHMS.items():
        composite_key = f"{set_key}_{algo_key}"
        file_name = f"model_set_{set_key}_{algo_key}.pkl"
        MODEL_CONFIGS[composite_key] = {
            "key": composite_key,
            "set": set_key,
            "algorithm_key": algo_key,
            "file": file_name,
            "name": f"{algo_info['name']} (Set {set_key.upper()})",
            "algorithm": algo_info["algorithm"],
            "threshold": algo_info["threshold"],
            "recommended": (algo_key == "logistic_regression"),
            "description": f"Set {set_key.upper()} - {algo_info['description']}",
        }

# Backwards compatibility: legacy Set C feature names list
FEATURE_NAMES = FEATURE_SETS["c"]["features"]
