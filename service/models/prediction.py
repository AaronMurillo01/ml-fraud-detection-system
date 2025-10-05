"""Prediction result models for the fraud detection system."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import Field, validator
from enum import Enum

from .base import BaseModel
from shared.models import RiskLevel


class PredictionResult(BaseModel):
    """Individual fraud prediction result."""
    
    transaction_id: str = Field(
        ...,
        description="Transaction identifier that was predicted",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "txn_1234567890abcdef"}
    )
    
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
    
    decision: str = Field(
        ...,
        description="Decision made based on the prediction (APPROVE, DECLINE, REVIEW)",
        json_schema_extra={"example": "APPROVE"}
    )
    
    confidence_score: float = Field(
        default=0.0,
        description="Model confidence in the prediction (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.92}
    )
    
    feature_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature contributions to the prediction",
        json_schema_extra={
            "example": {
                "amount": 0.3,
                "merchant_risk": 0.25,
                "velocity": 0.2
            }
        }
    )
    
    model_version: str = Field(
        default="unknown",
        description="Version of the model used for prediction",
        json_schema_extra={"example": "v1.2.0"}
    )
    
    processing_time_ms: float = Field(
        default=0.0,
        description="Time taken for prediction in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 45.2}
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the prediction was made",
        json_schema_extra={"example": "2024-01-15T14:30:01Z"}
    )
    
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the prediction",
        json_schema_extra={"example": "High fraud probability due to unusual transaction pattern"}
    )
    
    @validator('fraud_probability')
    def validate_fraud_probability(cls, v):
        """Validate fraud probability is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Fraud probability must be between 0 and 1")
        return v
    
    @validator('processing_time_ms')
    def validate_processing_time(cls, v):
        """Validate processing time is non-negative."""
        if v < 0:
            raise ValueError("Processing time must be non-negative")
        return v


class TrainingStatus(str, Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelTrainingRequest(BaseModel):
    """Request model for training a new model."""
    
    model_name: str = Field(..., description="Name of the model to train")
    model_type: str = Field(..., description="Type of model (e.g., xgboost, sklearn, lightgbm)")
    training_data_path: str = Field(..., description="Path to training data")
    validation_data_path: Optional[str] = Field(None, description="Path to validation data")
    test_data_path: Optional[str] = Field(None, description="Path to test data")
    feature_columns: List[str] = Field(..., description="List of feature column names")
    target_column: str = Field(..., description="Target column name")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    cross_validation_folds: int = Field(5, description="Number of cross-validation folds")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    save_model_path: Optional[str] = Field(None, description="Path to save the trained model")
    
    @validator('feature_columns')
    def validate_feature_columns(cls, v):
        if not v:
            raise ValueError("Feature columns cannot be empty")
        return v
    
    @validator('cross_validation_folds')
    def validate_cv_folds(cls, v):
        if v < 2:
            raise ValueError("Cross validation folds must be at least 2")
        return v


class ModelTrainingResponse(BaseModel):
    """Response model for training request."""
    
    training_id: str = Field(..., description="Unique training job identifier")
    model_name: str = Field(..., description="Name of the model being trained")
    model_version: str = Field(..., description="Version of the trained model")
    status: TrainingStatus = Field(..., description="Current training status")
    started_at: datetime = Field(..., description="Training start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Training completion timestamp")
    training_duration_seconds: Optional[float] = Field(None, description="Training duration in seconds")
    model_path: Optional[str] = Field(None, description="Path to the saved model")
    training_metrics: Dict[str, float] = Field(default_factory=dict, description="Training performance metrics")
    validation_metrics: Dict[str, float] = Field(default_factory=dict, description="Validation performance metrics")
    test_metrics: Dict[str, float] = Field(default_factory=dict, description="Test performance metrics")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    hyperparameters_used: Dict[str, Any] = Field(default_factory=dict, description="Final hyperparameters used")
    error_message: Optional[str] = Field(None, description="Error message if training failed")
    
    class Config:
        use_enum_values = True


class ModelEvaluationResult(BaseModel):
    """Model evaluation result."""
    
    evaluation_id: str = Field(..., description="Unique evaluation identifier")
    model_name: str = Field(..., description="Name of the evaluated model")
    model_version: str = Field(..., description="Version of the evaluated model")
    evaluation_dataset: str = Field(..., description="Dataset used for evaluation")
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")
    
    # Performance metrics
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Model accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Model precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Model recall")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    auc_roc: float = Field(..., ge=0.0, le=1.0, description="AUC-ROC score")
    auc_pr: float = Field(..., ge=0.0, le=1.0, description="AUC-PR score")
    
    # Additional metrics
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    classification_report: Dict[str, Dict[str, float]] = Field(..., description="Detailed classification report")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    threshold_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Metrics at different thresholds")
    
    # Evaluation metadata
    sample_count: int = Field(..., ge=0, description="Number of samples evaluated")
    evaluation_duration_seconds: float = Field(..., ge=0.0, description="Evaluation duration in seconds")
    model_size_mb: Optional[float] = Field(None, ge=0.0, description="Model size in MB")
    
    @validator('confusion_matrix')
    def validate_confusion_matrix(cls, v):
        if not v or not all(isinstance(row, list) for row in v):
            raise ValueError("Confusion matrix must be a list of lists")
        return v


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: int = Field(..., ge=0, description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Status of service dependencies")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Service metrics")


class ServiceMetrics(BaseModel):
    """Service metrics model."""
    
    total_requests: int = Field(..., ge=0, description="Total number of requests")
    successful_requests: int = Field(..., ge=0, description="Number of successful requests")
    failed_requests: int = Field(..., ge=0, description="Number of failed requests")
    average_response_time_ms: float = Field(0.0, ge=0.0, description="Average response time in milliseconds")
    requests_per_minute: int = Field(0, ge=0, description="Requests per minute")
    error_rate: float = Field(0.0, ge=0.0, le=1.0, description="Error rate (0-1)")
    cache_hit_rate: float = Field(0.0, ge=0.0, le=1.0, description="Cache hit rate (0-1)")
    active_models: int = Field(0, ge=0, description="Number of active models")
    memory_usage_mb: float = Field(0.0, ge=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(0.0, ge=0.0, le=100.0, description="CPU usage percentage")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    
    @validator('total_requests')
    def validate_total_requests(cls, v):
        if v < 0:
            raise ValueError("Total requests must be non-negative")
        return v
    
    @validator('error_rate')
    def validate_error_rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Error rate must be between 0 and 1")
        return v


class BatchPredictionResult(BaseModel):
    """Batch fraud prediction result."""
    
    batch_id: str = Field(
        ...,
        description="Unique identifier for the batch",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "batch_1234567890abcdef"}
    )
    
    predictions: List[PredictionResult] = Field(
        ...,
        description="List of individual prediction results",
        json_schema_extra={"example": []}
    )
    
    total_count: int = Field(
        ...,
        description="Total number of transactions in the batch",
        ge=0,
        json_schema_extra={"example": 100}
    )
    
    success_count: int = Field(
        ...,
        description="Number of successful predictions",
        ge=0,
        json_schema_extra={"example": 98}
    )
    
    error_count: int = Field(
        ...,
        description="Number of failed predictions",
        ge=0,
        json_schema_extra={"example": 2}
    )
    
    processing_time_ms: float = Field(
        default=0.0,
        description="Total processing time for the batch in milliseconds",
        ge=0.0,
        json_schema_extra={"example": 1250.5}
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the batch prediction was completed",
        json_schema_extra={"example": "2024-01-15T14:30:01Z"}
    )
    
    model_version: str = Field(
        default="unknown",
        description="Version of the model used for predictions",
        json_schema_extra={"example": "v1.0.0"}
    )
    
    @validator('total_count', pre=False, always=True)
    def validate_counts(cls, v, values):
        """Validate that total count equals success count plus error count."""
        if 'success_count' in values and 'error_count' in values:
            success_count = values['success_count']
            error_count = values['error_count']
            if v != success_count + error_count:
                raise ValueError("Total count must equal success count plus error count")
        return v
    
    @validator('processing_time_ms')
    def validate_processing_time(cls, v):
        """Validate processing time is non-negative."""
        if v < 0:
            raise ValueError("Processing time must be non-negative")
        return v