"""API module for fraud detection system.

This module provides RESTful API endpoints for:
- Real-time fraud detection
- Batch transaction processing
- Model management and monitoring
- Health checks and system status
"""

from .main import app
from .endpoints import fraud_detection, health, model_management
from .middleware import setup_middleware
from .dependencies import get_ml_service, get_feature_pipeline

__all__ = [
    "app",
    "fraud_detection",
    "health", 
    "models",
    "setup_middleware",
    "get_ml_service",
    "get_feature_pipeline"
]