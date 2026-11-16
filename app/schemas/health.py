"""Health check and model catalog schemas supporting Sets A, B, and C."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata regarding an individual registered model."""
    id: str = Field(..., description="Unique model identifier key (e.g. 'c_logistic_regression')")
    set: str = Field(..., description="Feature set ('a', 'b', or 'c')")
    algorithm_key: str = Field(..., description="Algorithm identifier (e.g. 'logistic_regression')")
    name: str = Field(..., description="Human-readable model name")
    file: str = Field(..., description="Serialized pickle filename")
    algorithm: str = Field(..., description="Underlying algorithm or pipeline class")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded in memory")
    recommended: bool = Field(..., description="Whether this is the recommended model")
    description: str = Field(..., description="Methodological notes and description")


class ModelsListResponse(BaseModel):
    """List of available registered models across predictor sets."""
    total_models: int = Field(..., description="Total number of models configured")
    default_model: str = Field(..., description="Default model algorithm used when unspecified")
    default_set: str = Field(..., description="Default feature set used when unspecified")
    models: List[ModelInfo] = Field(..., description="List of model information objects")
    sets: Optional[Dict[str, Any]] = Field(None, description="Definitions and metadata for feature sets A, B, and C")


class HealthResponse(BaseModel):
    """Service and model pool health status."""
    status: str = Field(..., description="Overall service status ('healthy', 'degraded', 'unhealthy')")
    total_configured: int = Field(..., description="Total models configured")
    loaded_count: int = Field(..., description="Number of models currently in memory")
    loaded_models: List[str] = Field(..., description="Keys of successfully loaded models")
    unloaded_models: List[str] = Field(..., description="Keys of models unavailable or failed to load")
    message: str = Field(..., description="Status summary message")
