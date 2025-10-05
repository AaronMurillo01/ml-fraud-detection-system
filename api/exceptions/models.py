"""Error response models for the fraud detection API."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Error code enumeration."""
    
    # General errors
    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    
    # Authentication/Authorization errors
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_FAILED = "authorization_failed"
    INVALID_TOKEN = "invalid_token"
    TOKEN_EXPIRED = "token_expired"
    INVALID_API_KEY = "invalid_api_key"
    
    # Rate limiting errors
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    
    # Model/ML errors
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    PREDICTION_FAILED = "prediction_failed"
    FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"
    
    # Database errors
    DATABASE_CONNECTION_FAILED = "database_connection_failed"
    DATABASE_QUERY_FAILED = "database_query_failed"
    DATABASE_TIMEOUT = "database_timeout"
    
    # Business logic errors
    INVALID_TRANSACTION = "invalid_transaction"
    BATCH_SIZE_EXCEEDED = "batch_size_exceeded"
    UNSUPPORTED_OPERATION = "unsupported_operation"


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    
    field: str = Field(..., description="Field name that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="The invalid value")
    
    class Config:
        json_schema_extra = {
            "example": {
                "field": "amount",
                "message": "Amount must be greater than 0",
                "invalid_value": -10.50
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error_code: ErrorCode = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Additional error details")
    validation_errors: Optional[List[ValidationErrorDetail]] = Field(
        None, description="Validation error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")
    path: Optional[str] = Field(None, description="API path where error occurred")
    method: Optional[str] = Field(None, description="HTTP method")
    
    # Additional context for debugging (only in development)
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        json_schema_extra = {
            "example": {
                "error_code": "validation_error",
                "message": "Request validation failed",
                "details": "One or more fields contain invalid values",
                "validation_errors": [
                    {
                        "field": "amount",
                        "message": "Amount must be greater than 0",
                        "invalid_value": -10.50
                    }
                ],
                "timestamp": "2024-01-15T14:30:01Z",
                "request_id": "req_123456789",
                "path": "/api/v1/fraud/analyze",
                "method": "POST"
            }
        }


class RateLimitErrorResponse(ErrorResponse):
    """Rate limit error response with additional rate limit information."""
    
    retry_after: int = Field(..., description="Seconds to wait before retrying")
    limit: int = Field(..., description="Rate limit threshold")
    window: int = Field(..., description="Rate limit window in seconds")
    remaining: int = Field(default=0, description="Remaining requests in current window")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "rate_limit_exceeded",
                "message": "Rate limit exceeded",
                "details": "Too many requests. Please try again later.",
                "timestamp": "2024-01-15T14:30:01Z",
                "retry_after": 60,
                "limit": 100,
                "window": 3600,
                "remaining": 0
            }
        }


class ModelErrorResponse(ErrorResponse):
    """Model-specific error response."""
    
    model_name: Optional[str] = Field(None, description="Model name")
    model_version: Optional[str] = Field(None, description="Model version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "model_load_failed",
                "message": "Failed to load ML model",
                "details": "Model file not found or corrupted",
                "timestamp": "2024-01-15T14:30:01Z",
                "model_name": "fraud_detector_xgb",
                "model_version": "v1.2.0"
            }
        }


class DatabaseErrorResponse(ErrorResponse):
    """Database-specific error response."""

    operation: Optional[str] = Field(None, description="Database operation that failed")
    table: Optional[str] = Field(None, description="Database table involved")

    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "database_query_failed",
                "message": "Database query failed",
                "details": "Connection timeout while executing query",
                "timestamp": "2024-01-15T14:30:01Z",
                "operation": "INSERT",
                "table": "fraud_scores"
            }
        }


class BusinessLogicErrorResponse(ErrorResponse):
    """Business logic error response."""

    transaction_id: Optional[str] = Field(None, description="Transaction ID related to error")
    user_id: Optional[str] = Field(None, description="User ID related to error")

    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "invalid_transaction",
                "message": "Transaction validation failed",
                "details": "Transaction amount exceeds daily limit",
                "timestamp": "2024-01-15T14:30:01Z",
                "transaction_id": "txn_123456789",
                "user_id": "user_987654321"
            }
        }
