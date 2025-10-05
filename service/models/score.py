"""Model scoring and prediction data models."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID

from pydantic import Field, validator

from .base import BaseModel, EntityModel


class RiskLevel(str, Enum):
    """Risk level enumeration based on fraud probability."""
    LOW = "low"          # 0.0 - 0.3
    MEDIUM = "medium"    # 0.3 - 0.7
    HIGH = "high"        # 0.7 - 0.9
    CRITICAL = "critical" # 0.9 - 1.0


class ModelVersion(str, Enum):
    """Available model versions."""
    XGBOOST_V1 = "xgboost_v1"
    XGBOOST_V2 = "xgboost_v2"
    LIGHTGBM_V1 = "lightgbm_v1"
    ENSEMBLE_V1 = "ensemble_v1"


class ActionRecommendation(str, Enum):
    """Recommended actions based on fraud score."""
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"
    BLOCK_CARD = "block_card"
    REQUIRE_2FA = "require_2fa"


class FeatureImportance(BaseModel):
    """Individual feature importance for explainability."""
    
    feature_name: str = Field(
        ...,
        description="Name of the feature",
        json_schema_extra={"example": "amount_zscore_user_7d"}
    )
    
    importance_score: float = Field(
        ...,
        description="SHAP importance score (can be negative)",
        json_schema_extra={"example": 0.15}
    )
    
    feature_value: Optional[Any] = Field(
        None,
        description="Actual value of the feature for this transaction",
        json_schema_extra={"example": 2.3}
    )
    
    description: Optional[str] = Field(
        None,
        description="Human-readable description of the feature",
        json_schema_extra={"example": "Transaction amount is 2.3 standard deviations above user's 7-day average"}
    )


class ModelScore(EntityModel):
    """Fraud detection model score and prediction."""
    
    # Reference to the transaction
    transaction_id: str = Field(
        ...,
        description="Transaction identifier that was scored",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "txn_1234567890abcdef"}
    )
    
    # Model information
    model_version: ModelVersion = Field(
        ...,
        description="Version of the model used for scoring",
        json_schema_extra={"example": "xgboost_v2"}
    )
    
    model_name: str = Field(
        ...,
        description="Name of the specific model",
        json_schema_extra={"example": "fraud_detector_xgb_v2.1"}
    )
    
    # Prediction results
    fraud_probability: float = Field(
        ...,
        description="Probability that the transaction is fraudulent (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.15}
    )
    
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk level based on fraud probability",
        json_schema_extra={"example": "low"}
    )
    
    confidence_score: float = Field(
        ...,
        description="Model confidence in the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92}
    )
    
    # Recommendations
    recommended_action: ActionRecommendation = Field(
        ...,
        description="Recommended action based on the score",
        json_schema_extra={"example": "approve"}
    )
    
    # Timing information
    scored_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the scoring was performed",
        json_schema_extra={"example": "2024-01-15T14:30:01Z"}
    )
    
    inference_time_ms: float = Field(
        ...,
        description="Time taken for model inference in milliseconds",
        gt=0,
        json_schema_extra={"example": 12.5}
    )
    
    # Feature explanations (optional)
    feature_importances: Optional[List[FeatureImportance]] = Field(
        None,
        description="SHAP feature importances for explainability",
        max_items=50
    )
    
    # Threshold information
    decision_threshold: float = Field(
        ...,
        description="Threshold used for binary classification",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.5}
    )
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional scoring metadata",
        json_schema_extra={
            "example": {
                "feature_count": 45,
                "preprocessing_version": "v1.2",
                "ensemble_weights": {"xgb": 0.7, "lgb": 0.3}
            }
        }
    )
    
    @validator('risk_level', pre=False, always=True)
    def set_risk_level(cls, v, values):
        """Automatically set risk level based on fraud probability."""
        if 'fraud_probability' in values:
            prob = values['fraud_probability']
            if prob < 0.3:
                return RiskLevel.LOW
            elif prob < 0.7:
                return RiskLevel.MEDIUM
            elif prob < 0.9:
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
        return v
    
    @validator('recommended_action', pre=False, always=True)
    def set_recommended_action(cls, v, values):
        """Automatically set recommended action based on fraud probability."""
        if 'fraud_probability' in values:
            prob = values['fraud_probability']
            if prob < 0.1:
                return ActionRecommendation.APPROVE
            elif prob < 0.5:
                return ActionRecommendation.APPROVE  # Low risk, approve
            elif prob < 0.8:
                return ActionRecommendation.REVIEW   # Medium risk, manual review
            elif prob < 0.95:
                return ActionRecommendation.DECLINE  # High risk, decline
            else:
                return ActionRecommendation.BLOCK_CARD  # Critical risk, block card
        return v


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics for monitoring."""
    
    model_version: ModelVersion = Field(
        ...,
        description="Version of the model"
    )
    
    # Performance metrics
    accuracy: float = Field(
        ...,
        description="Model accuracy",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.95}
    )
    
    precision: float = Field(
        ...,
        description="Precision for fraud detection",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.88}
    )
    
    recall: float = Field(
        ...,
        description="Recall for fraud detection",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92}
    )
    
    f1_score: float = Field(
        ...,
        description="F1 score",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.90}
    )
    
    auc_roc: float = Field(
        ...,
        description="Area under ROC curve",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.97}
    )
    
    # Operational metrics
    avg_inference_time_ms: float = Field(
        ...,
        description="Average inference time in milliseconds",
        gt=0,
        json_schema_extra={"example": 15.2}
    )
    
    p95_inference_time_ms: float = Field(
        ...,
        description="95th percentile inference time in milliseconds",
        gt=0,
        json_schema_extra={"example": 28.5}
    )
    
    # Data drift metrics
    feature_drift_score: Optional[float] = Field(
        None,
        description="Feature drift score compared to training data",
        ge=0.0,
        json_schema_extra={"example": 0.05}
    )
    
    prediction_drift_score: Optional[float] = Field(
        None,
        description="Prediction drift score",
        ge=0.0,
        json_schema_extra={"example": 0.03}
    )
    
    # Evaluation period
    evaluation_start: datetime = Field(
        ...,
        description="Start of evaluation period"
    )
    
    evaluation_end: datetime = Field(
        ...,
        description="End of evaluation period"
    )
    
    sample_count: int = Field(
        ...,
        description="Number of samples in evaluation",
        gt=0,
        json_schema_extra={"example": 10000}
    )


class ScoringResponse(BaseModel):
    """Response model for transaction scoring API."""
    
    transaction_id: str = Field(
        ...,
        description="Transaction identifier",
        json_schema_extra={"example": "txn_1234567890abcdef"}
    )
    
    score: ModelScore = Field(
        ...,
        description="Fraud detection score and metadata"
    )
    
    enriched_transaction: Optional[Dict[str, Any]] = Field(
        None,
        description="Enriched transaction with computed features (if requested)"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Total processing time including feature computation",
        gt=0,
        json_schema_extra={"example": 25.8}
    )


class BatchScoringResponse(BaseModel):
    """Response model for batch transaction scoring."""
    
    results: List[ScoringResponse] = Field(
        ...,
        description="Scoring results for each transaction"
    )
    
    batch_id: str = Field(
        ...,
        description="Unique identifier for this batch",
        json_schema_extra={"example": "batch_20240115_143001"}
    )
    
    total_processing_time_ms: float = Field(
        ...,
        description="Total time to process the entire batch",
        gt=0,
        json_schema_extra={"example": 156.7}
    )
    
    successful_count: int = Field(
        ...,
        description="Number of successfully processed transactions",
        ge=0,
        json_schema_extra={"example": 98}
    )
    
    failed_count: int = Field(
        ...,
        description="Number of failed transactions",
        ge=0,
        json_schema_extra={"example": 2}
    )
    
    errors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Details of any processing errors"
    )