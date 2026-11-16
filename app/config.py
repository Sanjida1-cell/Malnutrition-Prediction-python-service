"""Application configuration and multi-set model registry definitions."""

from pathlib import Path
import os
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
        "description": "Full Clinical: Demographics + Measurements + WHO Growth Standards (+ WAZ, HAZ, WHZ)",
        "features": ["month", "sex_male", "area_noakhali", "weight", "height", "waz", "haz", "whz"],
        "recommended": True,
    },
}

DEFAULT_SET = "c"
DEFAULT_MODEL = "logistic_regression"

# Available algorithms
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "algorithm": "Pipeline(StandardScaler, LogisticRegression)",
        "recommended": True,
        "description": "Optimal model selected in clinical paper. Balanced class weighting, interpretable, robust calibration.",
    },
    "random_forest": {
        "name": "Random Forest Classifier",
        "algorithm": "RandomForestClassifier",
        "recommended": False,
        "description": "Ensemble of bagged decision trees with high non-linear feature interaction capacity.",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting Classifier",
        "algorithm": "GradientBoostingClassifier",
        "recommended": False,
        "description": "Sequential boosting ensemble optimizing deviance loss across predictors.",
    },
    "stacking_ensemble": {
        "name": "Stacking Ensemble Classifier",
        "algorithm": "StackingClassifier",
        "recommended": False,
        "description": "Heterogeneous meta-ensemble stacking multiple base models with a logistic meta-learner.",
    },
    "svm_rbf": {
        "name": "Support Vector Machine (RBF Kernel)",
        "algorithm": "Pipeline(StandardScaler, SVC)",
        "recommended": False,
        "description": "Support Vector Classifier with Radial Basis Function kernel and calibrated probability estimates.",
    },
    "mlp": {
        "name": "Multi-Layer Perceptron (Neural Network)",
        "algorithm": "Pipeline(StandardScaler, MLPClassifier)",
        "recommended": False,
        "description": "Feedforward artificial neural network trained via backpropagation with tanh activation.",
    },
    "decision_tree": {
        "name": "Decision Tree Classifier",
        "algorithm": "DecisionTreeClassifier",
        "recommended": False,
        "description": "Single decision tree with axis-aligned splits and maximum depth constraint.",
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
            "recommended": (algo_key == "logistic_regression"),
            "description": f"Set {set_key.upper()} - {algo_info['description']}",
        }

# Backwards compatibility: legacy Set C feature names list
FEATURE_NAMES = FEATURE_SETS["c"]["features"]

# Risk probability thresholds
RISK_THRESHOLD_MODERATE = 0.30
RISK_THRESHOLD_HIGH = 0.50
