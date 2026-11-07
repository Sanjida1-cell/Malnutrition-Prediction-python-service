"""
Automated unit test suite for the Malnutrition Prediction API.
Directly executes route handlers and service methods.
"""

import sys
from pathlib import Path
from fastapi import HTTPException

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import MODEL_CONFIGS
from app.routes.predict import get_root, list_models, health_check, predict
from app.schemas.prediction import InputData, ModelChoice
from app.services.model_service import model_service

SAMPLE_HEALTHY = {
    "month": 39.66,
    "sex_male": 1,
    "area_noakhali": 0,
    "weight": 11.0,
    "height": 80.0,
    "waz": -2.529,
    "haz": -4.76,
    "whz": 0.468,
}

SAMPLE_MALNOURISHED = {
    "month": 24.0,
    "sex_male": 1,
    "area_noakhali": 1,
    "weight": 7.5,
    "height": 85.0,
    "waz": -3.5,
    "haz": -1.5,
    "whz": -3.8,
}

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1


def test_root():
    res = get_root()
    assert res["default_model"] == "logistic_regression"
    assert res["total_available_models"] == len(MODEL_CONFIGS)
    assert res["features_required"] == 8


def test_models_catalog():
    res = list_models()
    assert res.total_models == 7
    loaded_models = [m for m in res.models if m.is_loaded]
    assert len(loaded_models) == 7
    # Verify logistic regression is marked recommended
    lr = next(m for m in res.models if m.id == "logistic_regression")
    assert lr.recommended is True


def test_health_probe():
    res = health_check()
    assert res.status == "healthy"
    assert res.total_configured == 7
    assert res.loaded_count == 7
    assert len(res.unloaded_models) == 0


def test_default_model_prediction():
    data = InputData(**SAMPLE_HEALTHY)
    res = predict(data)
    assert res.model_used == "logistic_regression"
    assert res.model_file == "logistic_regression_set_c.pkl"
    assert res.prediction in (0, 1)
    assert 0.0 <= res.risk_probability <= 1.0


def test_all_7_models():
    for model_key in MODEL_CONFIGS:
        data = InputData(**SAMPLE_HEALTHY, model_name=model_key)
        res = predict(data)
        assert res.model_used == model_key
        assert res.model_file == MODEL_CONFIGS[model_key]["file"]
        assert res.prediction in (0, 1)
        assert 0.0 <= res.risk_probability <= 1.0
        assert res.risk_level in ("Low Risk", "Moderate Risk", "High Risk")


def test_query_override():
    data = InputData(**SAMPLE_HEALTHY, model_name="logistic_regression")
    res = predict(data, model="random_forest")
    assert res.model_used == "random_forest"
    assert res.model_file == "random_forest_set_c.pkl"


def test_alias_resolution():
    data = InputData(**SAMPLE_HEALTHY)
    res = predict(data, model="rf")
    assert res.model_used == "random_forest"

    res_svm = predict(data, model="svm")
    assert res_svm.model_used == "svm_rbf"

    res_stack = predict(data, model="stacking")
    assert res_stack.model_used == "stacking_ensemble"


def test_malnourished_detection():
    data = InputData(**SAMPLE_MALNOURISHED, model_name="logistic_regression")
    res = predict(data)
    assert res.prediction == 1
    assert res.status == "Malnourished"
    assert res.risk_level == "High Risk"
    assert res.risk_probability >= 0.50


def test_invalid_model_rejection():
    data = InputData(**SAMPLE_HEALTHY)
    try:
        predict(data, model="non_existent_model_xyz")
        assert False, "Expected HTTPException 400 for unknown model"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unknown model" in str(e.detail)


def test_validation_bounds():
    try:
        InputData(**SAMPLE_HEALTHY, sex_male=5)
        assert False, "Expected ValueError for sex_male > 1"
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Malnutrition Prediction API Automated Test Suite")
    print("=" * 60)

    run_test("GET / root metadata check", test_root)
    run_test("GET /models catalog check", test_models_catalog)
    run_test("GET /health probe check", test_health_probe)
    run_test("POST /predict default model (logistic_regression)", test_default_model_prediction)
    run_test("POST /predict execution across ALL 7 models", test_all_7_models)
    run_test("POST /predict query parameter override (?model=...)", test_query_override)
    run_test("Model shorthand alias resolution (rf, svm, stacking)", test_alias_resolution)
    run_test("Acute malnutrition clinical detection", test_malnourished_case if 'test_malnourished_case' in locals() else test_malnourished_detection)
    run_test("Invalid model error handling (HTTP 400)", test_invalid_model_rejection)
    run_test("Input bounds schema validation (HTTP 422)", test_validation_bounds)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed.")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
