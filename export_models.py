"""Export trained models and decision thresholds across Predictor Sets A, B, and C.

Reproduces the exact training procedure and settings reported in Table S4 and Table S5
of the manuscript, fitting 7 benchmarked machine learning models across all three predictor sets
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def feat_a(r):
        return [
            float(r["MONTH"]),
            1.0 if r["SEX"].strip().lower() == "male" else 0.0,
            1.0 if r["Area"].strip().lower() == "noakhali" else 0.0,
        ]

    def feat_b(r):
        return feat_a(r) + [
            float(r["WEIGHT"]),
            float(r["HEIGHT"]),
        ]

    def feat_c(r):
        return feat_b(r) + [
            float(r["WAZ"]),
            float(r["HAZ"]),
            float(r["WHZ"]),
        ]

    y = np.array([int(r["muac_malnourished"]) for r in rows], dtype=int)
    sets = {
        "a": np.array([feat_a(r) for r in rows], dtype=np.float64),
        "b": np.array([feat_b(r) for r in rows], dtype=np.float64),
        "c": np.array([feat_c(r) for r in rows], dtype=np.float64),
    }
    return sets, y


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
    fpr, tpr, thr = roc_curve(y_true, proba)
    return float(thr[np.argmax(tpr - fpr)])


def inner_threshold(model_proto, X_train, y_train):
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
    sets, y = load_dataset()
    print(f"Loaded cohort: {len(y)} children ({int(y.sum())} malnourished, {len(y) - int(y.sum())} normal)")

    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    ytr, yte = y[tr_idx], y[te_idx]

    thresholds_dict = {}

    for set_key in ["a", "b", "c"]:
        X = sets[set_key]
        Xtr, Xte = X[tr_idx], X[te_idx]
        print(f"\n=== Training Predictor Set {set_key.upper()} ({X.shape[1]} features) ===")

        for name, proto in MODEL_PROTOS.items():
            model = proto().fit(Xtr, ytr)
            thr = inner_threshold(proto, Xtr, ytr)

            out_file = MODELS_DIR / f"model_set_{set_key}_{name}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(model, f)

            pr_te = model.predict_proba(Xte)[:, 1]
            pred = (pr_te >= thr).astype(int)
            acc = accuracy_score(yte, pred)
            bacc = balanced_accuracy_score(yte, pred)
            auc = roc_auc_score(yte, pr_te)
            tp = int(((pred == 1) & (yte == 1)).sum())

            key_name = name if set_key == "c" else f"set_{set_key}_{name}"
            thresholds_dict[key_name] = {
                "set": set_key.upper(),
                "threshold": round(thr, 4),
                "held_out_accuracy": round(float(acc), 4),
                "held_out_bacc": round(float(bacc), 4),
                "held_out_auc": round(float(auc), 4),
                "tp_identified": tp,
                "total_positives": int(yte.sum()),
            }

            print(f"  [{set_key.upper()}] {name:<20} | Thr: {thr:.4f} | Acc: {acc*100:.1f}% | AUC: {auc:.3f} | TP: {tp}/{int(yte.sum())}")

    thresholds_file = MODELS_DIR / "thresholds.json"
    with open(thresholds_file, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)
    print("\n" + "=" * 75)
    print(f"Successfully trained and saved all 21 models + thresholds.json!")


if __name__ == "__main__":
    export_all()