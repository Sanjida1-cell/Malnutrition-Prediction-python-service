"""Application configuration and model registry definitions."""

from pathlib import Path
import os
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

# Feature vector ordering required by all Set C models
FEATURE_NAMES = [
    "month",
    "sex_male",
    "area_noakhali",
    "weight",
    "height",
    "waz",
    "haz",
    "whz",
]

# Risk probability thresholds
RISK_THRESHOLD_MODERATE = 0.30
RISK_THRESHOLD_HIGH = 0.50

# Registry metadata for all 7 Set C models
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "file": "logistic_regression_set_c.pkl",
        "name": "Logistic Regression (Set C Pipeline)",
        "algorithm": "Pipeline(StandardScaler, LogisticRegression)",
        "recommended": True,
        "description": "Selected optimal model from research paper. Balanced class weighting, highly interpretable, robust AUC (~0.72).",
    },
    "random_forest": {
        "file": "random_forest_set_c.pkl",
        "name": "Random Forest Classifier",
        "algorithm": "RandomForestClassifier",
        "recommended": False,
        "description": "Ensemble of bagged decision trees with high non-linear feature interaction capacity.",
    },
    "gradient_boosting": {
        "file": "gradient_boosting_set_c.pkl",
        "name": "Gradient Boosting Classifier",
        "algorithm": "GradientBoostingClassifier",
        "recommended": False,
        "description": "Sequential boosting ensemble optimizing deviance loss across Set C predictors.",
    },
    "stacking_ensemble": {
        "file": "stacking_ensemble_set_c.pkl",
        "name": "Stacking Ensemble Classifier",
        "algorithm": "StackingClassifier",
        "recommended": False,
        "description": "Heterogeneous meta-ensemble stacking multiple base models with a logistic meta-learner.",
    },
    "svm_rbf": {
        "file": "svm_rbf_set_c.pkl",
        "name": "Support Vector Machine (RBF Kernel)",
        "algorithm": "Pipeline(StandardScaler, SVC)",
        "recommended": False,
        "description": "Support Vector Classifier with Radial Basis Function kernel and calibrated probability estimates.",
    },
    "mlp": {
        "file": "mlp_set_c.pkl",
        "name": "Multi-Layer Perceptron (Neural Network)",
        "algorithm": "Pipeline(StandardScaler, MLPClassifier)",
        "recommended": False,
        "description": "Feedforward artificial neural network trained via backpropagation with tanh activation.",
    },
    "decision_tree": {
        "file": "decision_tree_set_c.pkl",
        "name": "Decision Tree Classifier",
        "algorithm": "DecisionTreeClassifier",
        "recommended": False,
        "description": "Single decision tree with axis-aligned splits and maximum depth constraint.",
    },
}

DEFAULT_MODEL = "logistic_regression"
