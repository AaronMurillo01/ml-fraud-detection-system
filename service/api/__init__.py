"""FastAPI application package for fraud detection service."""

from .app import create_app
from .health import router as health_router
from .metrics import router as metrics_router
from .fraud import router as fraud_router

__all__ = [
    "create_app",
    "health_router",
    "metrics_router", 
    "fraud_router",
]