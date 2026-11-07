"""Automated test suite for Malnutrition Prediction API."""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import MODEL_CONFIGS

client = TestClient(app)

SAMPLE_HEALTHY_CHILD = {
    "month": 39.66,
    "sex_male": 1,
    "area_noakhali": 0,
    "weight": 11.0,
    "height": 80.0,
    "waz": -2.529,
    "haz": -4.76,
    "whz": 0.468,
}

SAMPLE_MALNOURISHED_CHILD = {
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
    assert data["total_available_models"] == len(MODEL_CONFIGS)


def test_models_catalog_endpoint():
    """Verify /models enumerates all 7 registered models."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["total_models"] == 7
    loaded_ids = [m["id"] for m in data["models"] if m["is_loaded"]]
    assert len(loaded_ids) == 7


def test_health_endpoint():
    """Verify /health reports healthy status and all 7 models loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_configured"] == 7
    assert data["loaded_count"] == 7
    assert len(data["unloaded_models"]) == 0


def test_predict_default_model():
    """Verify /predict defaults to logistic_regression when model_name is omitted."""
    response = client.post("/predict", json=SAMPLE_HEALTHY_CHILD)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "logistic_regression"
    assert data["model_file"] == "logistic_regression_set_c.pkl"
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["risk_level"] in ("Low Risk", "Moderate Risk", "High Risk")


@pytest.mark.parametrize("model_key", list(MODEL_CONFIGS.keys()))
def test_predict_all_models_explicitly(model_key):
    """Verify each of the 7 models executes prediction successfully."""
    payload = {**SAMPLE_HEALTHY_CHILD, "model_name": model_key}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == model_key
    assert data["model_file"] == MODEL_CONFIGS[model_key]["file"]
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["risk_probability"] <= 1.0


def test_predict_query_override():
    """Verify query parameter ?model=... overrides body model_name."""
    payload = {**SAMPLE_HEALTHY_CHILD, "model_name": "logistic_regression"}
    response = client.post("/predict?model=random_forest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "random_forest"


def test_predict_alias_resolution():
    """Verify shorthand model aliases are properly resolved."""
    payload = {**SAMPLE_HEALTHY_CHILD, "model_name": "rf"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["model_used"] == "random_forest"


def test_predict_malnourished_case():
    """Verify acute malnutrition case triggers Malnourished classification."""
    payload = {**SAMPLE_MALNOURISHED_CHILD, "model_name": "logistic_regression"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["status"] == "Malnourished"
    assert data["risk_level"] == "High Risk"
    assert data["risk_probability"] >= 0.50


def test_predict_unknown_model_error():
    """Verify 400 Bad Request when an unknown model name is requested."""
    payload = {**SAMPLE_HEALTHY_CHILD, "model_name": "non_existent_model"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


def test_predict_validation_error():
    """Verify 422 Unprocessable Entity when feature values violate bounds."""
    invalid_payload = {**SAMPLE_HEALTHY_CHILD, "sex_male": 5}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
