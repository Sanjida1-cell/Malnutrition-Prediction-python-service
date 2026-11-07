"""FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router
from app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring models are loaded on application start."""
    if not model_service.models:
        model_service.load_models()
    yield


app = FastAPI(
    title="Malnutrition Prediction Multi-Model API",
    description=(
        "Production-grade Machine Learning inference service for predicting acute child malnutrition "
        "(MUAC < 12.5 cm) using validated, non-circular Predictor Set C models. Supports dynamic "
        "selection across 7 trained algorithms (Logistic Regression, Random Forest, Gradient Boosting, "
        "Stacking Ensemble, SVM, MLP, Decision Tree)."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS) for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)
