"""Automated test suite for Malnutrition Prediction API supporting Sets A, B, and C."""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import MODEL_CONFIGS, FEATURE_SETS

client = TestClient(app)

SAMPLE_SET_A_CHILD = {
    "feature_set": "a",
    "month": 24.0,
    "sex_male": 1,
    "area_noakhali": 1,
}

SAMPLE_SET_B_CHILD = {
    "feature_set": "b",
    "month": 24.0,
    "sex_male": 1,
    "area_noakhali": 1,
    "weight": 7.5,
    "height": 85.0,
}

SAMPLE_HEALTHY_SET_C_CHILD = {
    "feature_set": "c",
    "month": 39.66,
    "sex_male": 1,
    "area_noakhali": 0,
    "weight": 11.0,
    "height": 80.0,
    "waz": -2.529,
    "haz": -4.76,
    "whz": 0.468,
}

SAMPLE_MALNOURISHED_SET_C_CHILD = {
    "feature_set": "c",
    "month": 24.0,
    "sex_male": 1,
    "area_noakhali": 1,
    "weight": 7.5,
    "height": 85.0,
    "waz": -3.5,
    "haz": -1.5,
    "whz": -3.8,
}


def test_root_endpoint():
    """Verify / returns valid status and metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["default_model"] == "logistic_regression"
    assert data["total_configured_models"] == len(MODEL_CONFIGS)


def test_models_catalog_endpoint():
    """Verify /models enumerates all 21 registered models and supports set filtering."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["total_models"] == 21

    # Filter Set A
    resp_a = client.get("/models?feature_set=a")
    assert resp_a.status_code == 200
    assert resp_a.json()["total_models"] == 7

    # Filter Set B
    resp_b = client.get("/models?feature_set=b")
    assert resp_b.status_code == 200
    assert resp_b.json()["total_models"] == 7

    # Filter Set C
    resp_c = client.get("/models?feature_set=c")
    assert resp_c.status_code == 200
    assert resp_c.json()["total_models"] == 7


def test_health_endpoint():
    """Verify /health reports healthy status and all 21 models loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_configured"] == 21
    assert data["loaded_count"] == 21
    assert len(data["unloaded_models"]) == 0


def test_predict_set_a():
    """Verify Set A prediction using only demographic metrics."""
    response = client.post("/predict", json=SAMPLE_SET_A_CHILD)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_set"] == "a"
    assert data["model_used"] == "logistic_regression"
    assert data["model_file"] == "model_set_a_logistic_regression.pkl"
    assert len(data["features_used"]) == 3
    assert data["prediction"] in (0, 1)


def test_predict_set_b():
    """Verify Set B prediction using demographics and weight/height."""
    response = client.post("/predict", json=SAMPLE_SET_B_CHILD)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_set"] == "b"
    assert data["model_used"] == "logistic_regression"
    assert data["model_file"] == "model_set_b_logistic_regression.pkl"
    assert len(data["features_used"]) == 5
    assert data["prediction"] in (0, 1)


def test_predict_default_model_set_c():
    """Verify Set C prediction defaults to logistic_regression when model_name is omitted."""
    response = client.post("/predict", json=SAMPLE_HEALTHY_SET_C_CHILD)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_set"] == "c"
    assert data["model_used"] == "logistic_regression"
    assert data["model_file"] == "model_set_c_logistic_regression.pkl"
    assert len(data["features_used"]) == 8
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["risk_probability"] <= 1.0


@pytest.mark.parametrize("model_key", list(MODEL_CONFIGS.keys()))
def test_predict_all_models_explicitly(model_key):
    """Verify each of the 21 models across Sets A, B, and C executes prediction successfully."""
    config = MODEL_CONFIGS[model_key]
    set_key = config["set"]
    algo_key = config["algorithm_key"]

    if set_key == "a":
        payload = {**SAMPLE_SET_A_CHILD, "model_name": algo_key}
    elif set_key == "b":
        payload = {**SAMPLE_SET_B_CHILD, "model_name": algo_key}
    else:
        payload = {**SAMPLE_HEALTHY_SET_C_CHILD, "model_name": algo_key}

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_set"] == set_key
    assert data["model_used"] == algo_key
    assert data["model_file"] == config["file"]
    assert data["prediction"] in (0, 1)


def test_predict_query_override():
    """Verify query parameter ?model=... and ?feature_set=... override payload."""
    payload = {**SAMPLE_SET_B_CHILD, "model_name": "logistic_regression"}
    response = client.post("/predict?model=random_forest&feature_set=b", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_set"] == "b"
    assert data["model_used"] == "random_forest"


def test_predict_malnourished_case():
    """Verify acute malnutrition case triggers Malnourished classification."""
    payload = {**SAMPLE_MALNOURISHED_SET_C_CHILD, "model_name": "logistic_regression"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["status"] == "Malnourished"


def test_predict_missing_set_b_features():
    """Verify Set B fails with 422 if weight or height is missing."""
    invalid = {"feature_set": "b", "month": 24.0, "sex_male": 1, "area_noakhali": 1}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_missing_set_c_features():
    """Verify Set C fails with 422 if Z-scores are missing."""
    invalid = {**SAMPLE_SET_B_CHILD, "feature_set": "c"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
