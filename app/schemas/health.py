"""Health check and model catalog schemas."""

from typing import List
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata regarding an individual registered model."""
    id: str = Field(..., description="Unique model identifier key")
    name: str = Field(..., description="Human-readable model name")
    file: str = Field(..., description="Serialized pickle filename")
    algorithm: str = Field(..., description="Underlying algorithm or pipeline class")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded in memory")
    recommended: bool = Field(..., description="Whether this is the recommended model from research")
    description: str = Field(..., description="Methodological notes and description")


class ModelsListResponse(BaseModel):
    """List of all available registered models."""
    total_models: int = Field(..., description="Total number of models configured")
    default_model: str = Field(..., description="Default model used when unspecified")
    models: List[ModelInfo] = Field(..., description="List of model information objects")


class HealthResponse(BaseModel):
    """Service and model pool health status."""
    status: str = Field(..., description="Overall service status ('healthy', 'degraded', 'unhealthy')")
    total_configured: int = Field(..., description="Total models configured")
    loaded_count: int = Field(..., description="Number of models currently in memory")
    loaded_models: List[str] = Field(..., description="Keys of successfully loaded models")
    unloaded_models: List[str] = Field(..., description="Keys of models unavailable or failed to load")
    message: str = Field(..., description="Status summary message")
