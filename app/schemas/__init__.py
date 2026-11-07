"""Schemas package."""

from app.schemas.prediction import ModelChoice, InputData, PredictionResponse
from app.schemas.health import ModelInfo, ModelsListResponse, HealthResponse

__all__ = [
    "ModelChoice",
    "InputData",
    "PredictionResponse",
    "ModelInfo",
    "ModelsListResponse",
    "HealthResponse",
]
