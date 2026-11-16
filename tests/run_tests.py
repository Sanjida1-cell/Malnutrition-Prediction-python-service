"""
Automated unit test suite for the Malnutrition Prediction API.
Directly executes route handlers and service methods across Sets A, B, and C.
"""

import sys
from pathlib import Path
from fastapi import HTTPException

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import MODEL_CONFIGS, FEATURE_SETS, ALGORITHMS
from app.routes.predict import get_root, list_models, health_check, predict
from app.schemas.prediction import InputData
from app.services.model_service import model_service

SAMPLE_SET_A = {
    "feature_set": "a",
    "month": 24.0,
    "sex_male": 1,
    "area_noakhali": 1,
}

SAMPLE_SET_B = {
    "feature_set": "b",
    "month": 24.0,
    "sex_male": 1,
    "area_noakhali": 1,
    "weight": 7.5,
    "height": 85.0,
}

SAMPLE_HEALTHY_SET_C = {
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

SAMPLE_MALNOURISHED_SET_C = {
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
    assert res["total_available_models"] == 21
    assert res["total_configured_models"] == 21
    assert "feature_sets" in res


def test_models_catalog():
    res = list_models()
    assert res.total_models == 21
    loaded_models = [m for m in res.models if m.is_loaded]
    assert len(loaded_models) == 21

    # Check Set A filtering
    res_a = list_models(feature_set="a")
    assert res_a.total_models == 7

    # Check Set B filtering
    res_b = list_models(feature_set="b")
    assert res_b.total_models == 7

    # Check Set C filtering
    res_c = list_models(feature_set="c")
    assert res_c.total_models == 7


def test_health_probe():
    res = health_check()
    assert res.status == "healthy"
    assert res.total_configured == 21
    assert res.loaded_count == 21
    assert len(res.unloaded_models) == 0


def test_set_a_prediction():
    data = InputData(**SAMPLE_SET_A, model_name="logistic_regression")
    res = predict(data)
    assert res.feature_set == "a"
    assert res.model_used == "logistic_regression"
    assert res.model_file == "model_set_a_logistic_regression.pkl"
    assert len(res.features_used) == 3


def test_set_b_prediction():
    data = InputData(**SAMPLE_SET_B, model_name="logistic_regression")
    res = predict(data)
    assert res.feature_set == "b"
    assert res.model_used == "logistic_regression"
    assert res.model_file == "model_set_b_logistic_regression.pkl"
    assert len(res.features_used) == 5


def test_default_model_prediction():
    data = InputData(**SAMPLE_HEALTHY_SET_C)
    res = predict(data)
    assert res.feature_set == "c"
    assert res.model_used == "logistic_regression"
    assert res.model_file == "model_set_c_logistic_regression.pkl"
    assert len(res.features_used) == 8
    assert res.prediction in (0, 1)
    assert 0.0 <= res.risk_probability <= 1.0


def test_all_21_models():
    for composite_key, config in MODEL_CONFIGS.items():
        s = config["set"]
        algo = config["algorithm_key"]
        if s == "a":
            payload = {**SAMPLE_SET_A, "model_name": algo}
        elif s == "b":
            payload = {**SAMPLE_SET_B, "model_name": algo}
        else:
            payload = {**SAMPLE_HEALTHY_SET_C, "model_name": algo}

        data = InputData(**payload)
        res = predict(data)
        assert res.feature_set == s
        assert res.model_used == algo
        assert res.model_file == config["file"]
        assert res.prediction in (0, 1)
        assert 0.0 <= res.risk_probability <= 1.0
        assert res.risk_level in ("Low Risk", "Moderate Risk", "High Risk")


def test_query_override():
    data = InputData(**SAMPLE_SET_B, model_name="logistic_regression")
    res = predict(data, model="random_forest", feature_set="b")
    assert res.model_used == "random_forest"
    assert res.feature_set == "b"


def test_alias_resolution():
    for alias, canonical in [("rf", "random_forest"), ("svm", "svm_rbf"), ("stacking", "stacking_ensemble")]:
        data = InputData(**SAMPLE_HEALTHY_SET_C, model_name=alias)
        res = predict(data)
        assert res.model_used == canonical


def test_malnourished_detection():
    data = InputData(**SAMPLE_MALNOURISHED_SET_C, model_name="logistic_regression")
    res = predict(data)
    assert res.prediction == 1
    assert res.status == "Malnourished"


def test_invalid_model():
    try:
        data = InputData(**SAMPLE_HEALTHY_SET_C, model_name="invalid_model_123")
        predict(data)
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 400


def test_missing_features_validation():
    try:
        # Set C without Z-scores should fail
        payload = dict(SAMPLE_SET_B)
        payload["feature_set"] = "c"
        data = InputData(**payload)
        predict(data)
        assert False, "Should have failed missing Z-scores"
    except HTTPException as e:
        assert e.status_code == 422


if __name__ == "__main__":
    print("=" * 60)
    print("  Malnutrition Prediction API Automated Test Suite (Sets A, B, C)")
    print("=" * 60)

    run_test("GET / root metadata check", test_root)
    run_test("GET /models catalog check (all sets + filtering)", test_models_catalog)
    run_test("GET /health probe check (all 21 models loaded)", test_health_probe)
    run_test("POST /predict Set A (demographics only, 3 features)", test_set_a_prediction)
    run_test("POST /predict Set B (demographics + anthropometrics, 5 features)", test_set_b_prediction)
    run_test("POST /predict Set C (full clinical + WHO Z-scores, 8 features)", test_default_model_prediction)
    run_test("POST /predict execution across ALL 21 models", test_all_21_models)
    run_test("POST /predict query parameter override (?model=... & ?feature_set=...)", test_query_override)
    run_test("Model shorthand alias resolution (rf, svm, stacking)", test_alias_resolution)
    run_test("Acute malnutrition clinical detection", test_malnourished_detection)
    run_test("Invalid model error handling (HTTP 400)", test_invalid_model)
    run_test("Missing set features validation (HTTP 422)", test_missing_features_validation)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed.")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
