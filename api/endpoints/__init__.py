"""API endpoints package.

This package contains all the API endpoint modules for the fraud detection system:
- fraud_detection: Core fraud detection endpoints
- health: Health check and monitoring endpoints
- model_management: Model version and deployment management endpoints
"""

from .fraud_detection import router as fraud_detection_router
from .health import router as health_router
from .model_management import router as model_management_router

__all__ = [
    "fraud_detection_router",
    "health_router",
    "model_management_router"
]