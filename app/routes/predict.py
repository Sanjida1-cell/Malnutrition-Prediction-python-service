"""API route handlers for malnutrition prediction, health checks, and model metadata."""

from typing import Optional
from fastapi import APIRouter, Query

from app.config import MODEL_CONFIGS, DEFAULT_MODEL
from app.schemas.prediction import InputData, PredictionResponse
from app.schemas.health import HealthResponse, ModelsListResponse, ModelInfo
from app.services.model_service import model_service

router = APIRouter()


@router.get("/", response_model=dict, tags=["Metadata"])
def get_root():
    """Root endpoint - API status and configuration metadata."""
    return {
        "message": "Malnutrition Prediction Multi-Model API is running",
        "version": "2.1.0",
        "default_model": DEFAULT_MODEL,
        "available_models": list(MODEL_CONFIGS.keys()),
        "total_available_models": len(model_service.models),
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
def list_models():
    """List all available Set C machine learning models, their status, and descriptions."""
    model_list = [
        ModelInfo(
            id=key,
            name=config["name"],
            file=config["file"],
            algorithm=config["algorithm"],
            is_loaded=key in model_service.models,
            recommended=config["recommended"],
            description=config["description"],
        )
        for key, config in MODEL_CONFIGS.items()
    ]
    return ModelsListResponse(
        total_models=len(MODEL_CONFIGS),
        default_model=DEFAULT_MODEL,
        models=model_list,
    )


@router.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint to verify API operational state and loaded models."""
    loaded_keys = list(model_service.models.keys())
    unloaded_keys = [k for k in MODEL_CONFIGS if k not in model_service.models]

    if len(loaded_keys) == len(MODEL_CONFIGS):
        overall_status = "healthy"
        msg = f"All {len(loaded_keys)} models are loaded and ready for inference"
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
):
    """
    Predict acute child malnutrition risk using a selected Set C model.

    - **month**: Age in months (6-60 months)
    - **sex_male**: 1 if Male, 0 if Female
    - **area_noakhali**: 1 if Noakhali, 0 if Lakshmipur
    - **weight**: Weight in kg
    - **height**: Height in cm
    - **waz**: Weight-for-Age Z-score (-6 to +6)
    - **haz**: Height-for-Age Z-score (-6 to +6)
    - **whz**: Weight-for-Height Z-score (-6 to +6)
    - **model_name**: Model algorithm selection in body (e.g. 'logistic_regression', 'random_forest', 'stacking_ensemble', etc.)
    - **model**: Optional query param override (?model=stacking_ensemble)
    """
    override_val = model if isinstance(model, str) and model.strip() else None
    return model_service.predict(data=data, model_override=override_val)
