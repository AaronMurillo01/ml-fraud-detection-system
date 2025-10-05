"""Pydantic data models for the fraud detection system."""

from .base import BaseModel, TimestampMixin, UUIDMixin, EntityModel
from .transaction import (
    Transaction,
    EnrichedTransaction,
    TransactionRequest,
    BatchTransactionRequest,
    TransactionType,
    TransactionStatus,
    PaymentMethod,
)
from .score import (
    ModelScore,
    FeatureImportance,
    ModelPerformanceMetrics,
    ScoringResponse,
    BatchScoringResponse,
    RiskLevel,
    ModelVersion,
    ActionRecommendation,
)
from .prediction import (
    PredictionResult,
    BatchPredictionResult,
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelEvaluationResult,
    HealthCheckResponse,
    ServiceMetrics,
    TrainingStatus,
)

__all__ = [
    # Base models
    "BaseModel",
    "TimestampMixin", 
    "UUIDMixin",
    "EntityModel",
    
    # Transaction models
    "Transaction",
    "EnrichedTransaction",
    "TransactionRequest",
    "BatchTransactionRequest",
    
    # Transaction enums
    "TransactionType",
    "TransactionStatus",
    "PaymentMethod",
    
    # Scoring models
    "ModelScore",
    "FeatureImportance",
    "ModelPerformanceMetrics",
    "ScoringResponse",
    "BatchScoringResponse",
    
    # Scoring enums
    "RiskLevel",
    "ModelVersion",
    "ActionRecommendation",
    
    # Prediction models
    "PredictionResult",
    "BatchPredictionResult",
    "ModelTrainingRequest",
    "ModelTrainingResponse",
    "ModelEvaluationResult",
    "HealthCheckResponse",
    "ServiceMetrics",
    "TrainingStatus",
]