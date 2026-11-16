"""API route handlers for malnutrition prediction, health checks, and model metadata across Sets A, B, and C."""

from typing import Optional
from fastapi import APIRouter, Query

from app.config import MODEL_CONFIGS, FEATURE_SETS, ALGORITHMS, DEFAULT_SET, DEFAULT_MODEL
from app.schemas.prediction import InputData, PredictionResponse
from app.schemas.health import HealthResponse, ModelsListResponse, ModelInfo
from app.services.model_service import model_service

router = APIRouter()


@router.get("/", response_model=dict, tags=["Metadata"])
def get_root():
    """Root endpoint - API status and multi-set configuration metadata."""
    return {
        "message": "Malnutrition Prediction Multi-Model API (Sets A, B, C) is running",
        "version": "2.2.0",
        "default_set": DEFAULT_SET,
        "default_model": DEFAULT_MODEL,
        "available_models": list(ALGORITHMS.keys()),
        "total_available_models": len(model_service.models),
        "total_configured_models": len(MODEL_CONFIGS),
        "total_loaded_models": len(model_service.models),
        "feature_sets": {k: v["name"] for k, v in FEATURE_SETS.items()},
        "algorithms": list(ALGORITHMS.keys()),
        "target": "Acute Malnutrition (MUAC < 12.5 cm)",
        "features_required": 8,
        "endpoints": {
            "predict": "POST /predict",
            "models": "GET /models",
            "health": "GET /health",
            "documentation": "GET /docs",
        },
    }


@router.get("/models", response_model=ModelsListResponse, tags=["Metadata"])
def list_models(
    feature_set: Optional[str] = Query(
        None,
        description="Filter models by predictor set ('a', 'b', or 'c'). If omitted, all models across all sets are returned.",
    )
):
    """List available machine learning models, their loaded status, and descriptions."""
    target_set = None
    if isinstance(feature_set, str) and feature_set.strip():
        target_set = model_service.resolve_feature_set(feature_set)

    model_list = []
    for key, config in MODEL_CONFIGS.items():
        if target_set and config["set"] != target_set:
            continue
        model_list.append(
            ModelInfo(
                id=key,
                set=config["set"],
                algorithm_key=config["algorithm_key"],
                name=config["name"],
                file=config["file"],
                algorithm=config["algorithm"],
                is_loaded=key in model_service.models,
                recommended=config["recommended"],
                description=config["description"],
            )
        )

    return ModelsListResponse(
        total_models=len(model_list),
        default_model=DEFAULT_MODEL,
        default_set=DEFAULT_SET,
        models=model_list,
        sets=FEATURE_SETS,
    )


@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint to verify API operational state and loaded model count."""
    loaded_keys = list(model_service.models.keys())
    unloaded_keys = [k for k in MODEL_CONFIGS if k not in model_service.models]

    if len(loaded_keys) == len(MODEL_CONFIGS):
        overall_status = "healthy"
        msg = f"All {len(loaded_keys)} models across Sets A, B, and C are loaded and ready for inference"
    elif len(loaded_keys) > 0:
        overall_status = "degraded"
        msg = f"{len(loaded_keys)} of {len(MODEL_CONFIGS)} models loaded; some models unavailable"
    else:
        overall_status = "unhealthy"
        msg = "No models are currently loaded in memory"

    return HealthResponse(
        status=overall_status,
        total_configured=len(MODEL_CONFIGS),
        loaded_count=len(loaded_keys),
        loaded_models=loaded_keys,
        unloaded_models=unloaded_keys,
        message=msg,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(
    data: InputData,
    model: Optional[str] = Query(
        None,
        description="Optional query parameter to override model selection (?model=random_forest)",
    ),
    feature_set: Optional[str] = Query(
        None,
        description="Optional query parameter to override predictor set (?feature_set=a)",
    ),
):
    """
    Predict acute child malnutrition risk using a selected Set (A, B, or C) and Model.

    **Inputs**:
    - **feature_set**: 'a' (demographics), 'b' (+ weight/height), 'c' (+ WHO Z-scores). Defaults to 'c'.
    - **month**: Age in months (0-120 months)
    - **sex_male**: 1 if Male, 0 if Female
    - **area_noakhali**: 1 if Noakhali, 0 if Lakshmipur
    - **weight**: Weight in kg (required for Set B & Set C)
    - **height**: Height in cm (required for Set B & Set C)
    - **waz**: Weight-for-Age Z-score (required for Set C)
    - **haz**: Height-for-Age Z-score (required for Set C)
    - **whz**: Weight-for-Height Z-score (required for Set C)
    - **model_name**: Algorithm choice ('logistic_regression', 'random_forest', etc.)
    """
    override_model = model if isinstance(model, str) and model.strip() else None
    override_set = feature_set if isinstance(feature_set, str) and feature_set.strip() else None
    return model_service.predict(
        data=data,
        model_override=override_model,
        set_override=override_set,
    )
