"""Export trained Set C models and decision thresholds for child malnutrition prediction.

Reproduces the exact training procedure and settings reported in Table S4 and Table S5
of the manuscript, fitting 7 benchmarked machine learning models on Predictor Set C
using the 465-child cohort dataset (stratified 80/20 train/test split, random_state=42).

Reference repository: https://github.com/imak2363/child-malnutrition
"""

import os
import csv
import json
import pickle
import warnings
from pathlib import Path
import numpy as np

# Suppress sklearn deprecation/future warnings for clean output
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, accuracy_score, balanced_accuracy_score, roc_auc_score

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Malnutrition_Analysis_Dataset_FINAL.csv"
MODELS_DIR = BASE_DIR / "models"


def load_dataset():
    """Load the 465-row dataset and extract Set C features and target."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Feature Set C: 8 parameters
    # month, sex_male, area_noakhali, weight, height, waz, haz, whz
    def extract_features(r):
        return [
            float(r["MONTH"]),
            1.0 if r["SEX"].strip().lower() == "male" else 0.0,
            1.0 if r["Area"].strip().lower() == "noakhali" else 0.0,
            float(r["WEIGHT"]),
            float(r["HEIGHT"]),
            float(r["WAZ"]),
            float(r["HAZ"]),
            float(r["WHZ"]),
        ]

    X = np.array([extract_features(r) for r in rows], dtype=np.float64)
    y = np.array([int(r["muac_malnourished"]) for r in rows], dtype=int)
    return X, y, rows


# Model constructor functions adhering to Table S4 specifications
def make_lr():
    return Pipeline([
        ("s", StandardScaler()),
        ("m", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def make_dt():
    return DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
    )


def make_svm():
    return Pipeline([
        ("s", StandardScaler()),
        ("m", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def make_mlp():
    return Pipeline([
        ("s", StandardScaler()),
        ("m", MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000, random_state=RANDOM_STATE)),
    ])


def make_gb():
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
    )


def make_stack():
    return StackingClassifier(
        estimators=[("lr", make_lr()), ("dt", make_dt()), ("rf", make_rf())],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        cv=5,
        n_jobs=1,
    )


MODEL_PROTOS = {
    "logistic_regression": make_lr,
    "decision_tree": make_dt,
    "svm_rbf": make_svm,
    "mlp": make_mlp,
    "random_forest": make_rf,
    "gradient_boosting": make_gb,
    "stacking_ensemble": make_stack,
}


def youden_threshold(y_true, proba):
    """Compute Youden-optimal threshold maximizing (sensitivity + specificity - 1)."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    return float(thr[np.argmax(tpr - fpr)])


def inner_threshold(model_proto, X_train, y_train):
    """Compute Youden-optimal threshold from 5-fold inner cross-validation out-of-fold probabilities."""
    p_inner = cross_val_predict(
        clone(model_proto()),
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
    )[:, 1]
    return youden_threshold(y_train, p_inner)


def export_all():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, _ = load_dataset()
    print(f"Loaded cohort: {len(y)} children ({int(y.sum())} malnourished, {len(y) - int(y.sum())} normal)")

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    print(f"Training split: n={len(ytr)} (malnourished {int(ytr.sum())})")
    print(f"Held-out test split: n={len(yte)} (malnourished {int(yte.sum())})")
    print("-" * 75)

    thresholds_dict = {}

    for name, proto in MODEL_PROTOS.items():
        print(f"Training and evaluating: {name}...")
        model = proto().fit(Xtr, ytr)
        thr = inner_threshold(proto, Xtr, ytr)

        # Save model pickle
        out_file = MODELS_DIR / f"model_set_c_{name}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(model, f)

        # Evaluate on held-out test set
        pr_te = model.predict_proba(Xte)[:, 1]
        pred = (pr_te >= thr).astype(int)
        acc = accuracy_score(yte, pred)
        bacc = balanced_accuracy_score(yte, pred)
        auc = roc_auc_score(yte, pr_te)
        tp = int(((pred == 1) & (yte == 1)).sum())

        thresholds_dict[name] = {
            "threshold": round(thr, 4),
            "held_out_accuracy": round(float(acc), 4),
            "held_out_bacc": round(float(bacc), 4),
            "held_out_auc": round(float(auc), 4),
            "tp_identified": tp,
            "total_positives": int(yte.sum()),
        }

        print(f"  -> Saved {out_file.name}")
        print(f"     Threshold: {thr:.4f} | Held-out Acc: {acc*100:.1f}% | AUC: {auc:.3f} | TP: {tp}/{int(yte.sum())}")

    # Save thresholds.json
    thresholds_file = MODELS_DIR / "thresholds.json"
    with open(thresholds_file, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)
    print("-" * 75)
    print(f"Successfully saved thresholds to {thresholds_file.name}")
    print("Export complete.")


if __name__ == "__main__":
    export_all()
