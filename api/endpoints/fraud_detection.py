"""Fraud detection API endpoints.

This module provides endpoints for:
- Single transaction fraud detection
- Batch transaction processing
- Real-time fraud scoring
- Model explanations and insights
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from shared.models import Transaction, EnrichedTransaction
from service.models.score import ModelScore, RiskLevel
from service.ml_inference import MLInferenceService, get_inference_service
from features.feature_pipeline import FeaturePipeline, get_feature_pipeline
from api.dependencies import (
    get_database,
    get_rate_limiter,
    require_api_access,
    require_analyst_role
)
from api.auth import get_current_active_user, User
from api.exceptions import (
    FraudDetectionException,
    ModelException,
    DatabaseException,
    ValidationException,
    ErrorCode
)
from api.validation import validate_transaction, validate_batch_size, sanitize_transaction_data
from config.settings import get_settings
from service.models.prediction import (
    PredictionResult,
    BatchPredictionResult,
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelEvaluationResult,
    HealthCheckResponse,
    ServiceMetrics,
    TrainingStatus,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/fraud-detection", tags=["fraud-detection"])


# Request/Response Models
class FraudDetectionRequest(BaseModel):
    """Request model for fraud detection."""
    
    transaction: Transaction = Field(..., description="Transaction to analyze")
    include_explanation: bool = Field(default=False, description="Include model explanation")
    include_features: bool = Field(default=False, description="Include extracted features")
    model_version: Optional[str] = Field(default=None, description="Specific model version to use")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction": {
                    "transaction_id": "txn_123456789",
                    "user_id": "user_987654321",
                    "amount": 150.75,
                    "currency": "USD",
                    "merchant_id": "merchant_abc123",
                    "merchant_category": "grocery",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "location": {
                        "latitude": 40.7128,
                        "longitude": -74.0060,
                        "country": "US",
                        "city": "New York"
                    },
                    "payment_method": {
                        "type": "credit_card",
                        "last_four": "1234",
                        "issuer": "visa"
                    }
                },
                "include_explanation": True,
                "include_features": False
            }
        }


class BatchFraudDetectionRequest(BaseModel):
    """Request model for batch fraud detection."""
    
    transactions: List[Transaction] = Field(..., description="List of transactions to analyze")
    include_explanation: bool = Field(default=False, description="Include model explanations")
    include_features: bool = Field(default=False, description="Include extracted features")
    model_version: Optional[str] = Field(default=None, description="Specific model version to use")
    
    def model_post_init(self, __context) -> None:
        """Validate batch after model initialization."""
        validate_batch_size(len(self.transactions))
        for transaction in self.transactions:
            validate_transaction(transaction)


class FraudDetectionResponse(BaseModel):
    """Response model for fraud detection."""
    
    transaction_id: str = Field(..., description="Transaction ID")
    fraud_score: float = Field(..., description="Fraud probability score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    
    # Optional fields
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Model explanation")
    features: Optional[Dict[str, float]] = Field(default=None, description="Extracted features")
    enriched_transaction: Optional[EnrichedTransaction] = Field(default=None, description="Enriched transaction data")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_123456789",
                "fraud_score": 0.15,
                "risk_level": "low",
                "confidence": 0.92,
                "model_version": "xgboost_v1.2.0",
                "processing_time_ms": 45.2,
                "timestamp": "2024-01-15T10:30:01Z"
            }
        }


class BatchFraudDetectionResponse(BaseModel):
    """Response model for batch fraud detection."""
    
    results: List[FraudDetectionResponse] = Field(..., description="Detection results")
    batch_id: str = Field(..., description="Batch processing ID")
    total_processed: int = Field(..., description="Total transactions processed")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    average_processing_time_ms: float = Field(..., description="Average processing time per transaction")
    timestamp: datetime = Field(..., description="Batch processing timestamp")
    
    # Summary statistics
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    
    model_version: str = Field(..., description="Current model version")
    model_type: str = Field(..., description="Model type")
    is_loaded: bool = Field(..., description="Whether model is loaded")
    last_updated: datetime = Field(..., description="Last model update")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    feature_count: int = Field(..., description="Number of features")
    

# Endpoints
@router.post(
    "/analyze",
    response_model=FraudDetectionResponse,
    summary="Analyze single transaction for fraud",
    description="Analyze a single transaction and return fraud detection results with optional explanations."
)
async def analyze_transaction(
    request: FraudDetectionRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLInferenceService = Depends(get_inference_service),
    feature_pipeline: FeaturePipeline = Depends(get_feature_pipeline),
    db = Depends(get_database),
    rate_limiter = Depends(get_rate_limiter),
    current_user: User = Depends(get_current_active_user)
):
    """Analyze a single transaction for fraud.
    
    Args:
        request: Fraud detection request
        background_tasks: FastAPI background tasks
        ml_service: ML inference service
        feature_pipeline: Feature processing pipeline
        db: Database connection
        rate_limiter: Rate limiting service
        api_key: API key for authentication
        current_user: Current authenticated user
        
    Returns:
        Fraud detection response
        
    Raises:
        HTTPException: If analysis fails
    """
    start_time = datetime.now()

    try:
        # Validate and sanitize transaction data
        validate_transaction(request.transaction)
        sanitized_transaction = sanitize_transaction_data(request.transaction)

        logger.info(f"Starting fraud analysis for transaction {sanitized_transaction.transaction_id}")

        # Process transaction through feature pipeline
        try:
            enriched_transaction = await feature_pipeline.process_transaction(sanitized_transaction)
        except Exception as e:
            logger.error(f"Feature pipeline failed for transaction {sanitized_transaction.transaction_id}: {e}")
            raise FraudDetectionException(
                message="Feature extraction failed",
                error_code=ErrorCode.FEATURE_EXTRACTION_FAILED,
                details=str(e)
            )

        # Get fraud prediction
        try:
            prediction = await ml_service.predict_single(
                enriched_transaction,
                model_version=request.model_version,
                include_explanation=request.include_explanation
            )
        except Exception as e:
            logger.error(f"ML prediction failed for transaction {sanitized_transaction.transaction_id}: {e}")
            raise ModelException(
                message="Fraud prediction failed",
                error_code=ErrorCode.PREDICTION_FAILED,
                details=str(e),
                model_name="fraud_detector",
                model_version=request.model_version or "latest"
            )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Build response
        response = FraudDetectionResponse(
            transaction_id=request.transaction.transaction_id,
            fraud_score=prediction.fraud_probability,
            risk_level=prediction.risk_level,
            confidence=prediction.confidence,
            model_version=prediction.model_version,
            processing_time_ms=processing_time,
            timestamp=datetime.now()
        )
        
        # Add optional fields
        if request.include_explanation and prediction.explanation:
            response.explanation = prediction.explanation
        
        if request.include_features:
            response.features = enriched_transaction.features
            response.enriched_transaction = enriched_transaction
        
        # Log successful analysis
        logger.info(
            f"Fraud analysis completed for transaction {request.transaction.transaction_id} - "
            f"Score: {prediction.fraud_probability:.3f} - "
            f"Risk: {prediction.risk_level} - "
            f"Time: {processing_time:.1f}ms"
        )
        
        # Store result in background (async)
        try:
            background_tasks.add_task(
                store_fraud_analysis_result,
                db,
                sanitized_transaction,
                prediction,
                processing_time
            )
        except Exception as e:
            logger.warning(f"Failed to store fraud analysis result: {e}")
            # Don't fail the request if background storage fails

        return response

    except (ValidationException, FraudDetectionException, ModelException):
        # Re-raise our custom exceptions to be handled by the exception handler
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(
            f"Unexpected error in fraud analysis for transaction {request.transaction.transaction_id}: {e}",
            exc_info=True
        )

        raise FraudDetectionException(
            message="Fraud analysis failed due to unexpected error",
            error_code=ErrorCode.INTERNAL_ERROR,
            details=f"Processing time: {processing_time:.1f}ms"
        )


@router.post(
    "/analyze-batch",
    response_model=BatchFraudDetectionResponse,
    summary="Analyze batch of transactions for fraud",
    description="Analyze multiple transactions in batch and return fraud detection results."
)
async def analyze_batch_transactions(
    request: BatchFraudDetectionRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLInferenceService = Depends(get_inference_service),
    feature_pipeline: FeaturePipeline = Depends(get_feature_pipeline),
    db = Depends(get_database),
    rate_limiter = Depends(get_rate_limiter),
    current_user: User = Depends(get_current_active_user)
):
    """Analyze a batch of transactions for fraud.
    
    Args:
        request: Batch fraud detection request
        background_tasks: FastAPI background tasks
        ml_service: ML inference service
        feature_pipeline: Feature processing pipeline
        db: Database connection
        rate_limiter: Rate limiting service
        api_key: API key for authentication
        current_user: Current authenticated user
        
    Returns:
        Batch fraud detection response
        
    Raises:
        HTTPException: If batch analysis fails
    """
    start_time = datetime.now()
    batch_id = f"batch_{int(start_time.timestamp())}"
    
    try:
        logger.info(f"Starting batch fraud analysis - Batch ID: {batch_id} - Count: {len(request.transactions)}")
        
        # Process transactions through feature pipeline
        enriched_transactions = []
        for transaction in request.transactions:
            enriched = await feature_pipeline.process_transaction(transaction)
            enriched_transactions.append(enriched)
        
        # Get batch predictions
        predictions = await ml_service.predict_batch(
            enriched_transactions,
            model_version=request.model_version,
            include_explanation=request.include_explanation
        )
        
        # Calculate processing time
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        avg_processing_time = total_processing_time / len(request.transactions)
        
        # Build individual results
        results = []
        fraud_scores = []
        risk_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for i, (transaction, enriched, prediction) in enumerate(zip(
            request.transactions, enriched_transactions, predictions
        )):
            result = FraudDetectionResponse(
                transaction_id=transaction.transaction_id,
                fraud_score=prediction.fraud_probability,
                risk_level=prediction.risk_level,
                confidence=prediction.confidence,
                model_version=prediction.model_version,
                processing_time_ms=avg_processing_time,  # Approximate per transaction
                timestamp=datetime.now()
            )
            
            # Add optional fields
            if request.include_explanation and prediction.explanation:
                result.explanation = prediction.explanation
            
            if request.include_features:
                result.features = enriched.features
                result.enriched_transaction = enriched
            
            results.append(result)
            fraud_scores.append(prediction.fraud_probability)
            risk_levels[prediction.risk_level.value] += 1
        
        # Calculate summary statistics
        summary = {
            "total_transactions": len(request.transactions),
            "fraud_score_stats": {
                "min": min(fraud_scores),
                "max": max(fraud_scores),
                "mean": sum(fraud_scores) / len(fraud_scores),
                "median": sorted(fraud_scores)[len(fraud_scores) // 2]
            },
            "risk_level_distribution": risk_levels,
            "high_risk_count": risk_levels["high"] + risk_levels["critical"],
            "high_risk_percentage": (risk_levels["high"] + risk_levels["critical"]) / len(request.transactions) * 100
        }
        
        # Build batch response
        response = BatchFraudDetectionResponse(
            results=results,
            batch_id=batch_id,
            total_processed=len(request.transactions),
            total_processing_time_ms=total_processing_time,
            average_processing_time_ms=avg_processing_time,
            timestamp=datetime.now(),
            summary=summary
        )
        
        # Log successful batch analysis
        logger.info(
            f"Batch fraud analysis completed - Batch ID: {batch_id} - "
            f"Processed: {len(request.transactions)} - "
            f"High Risk: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%) - "
            f"Time: {total_processing_time:.1f}ms"
        )
        
        # Store batch results in background
        background_tasks.add_task(
            store_batch_fraud_analysis_results,
            db,
            batch_id,
            request.transactions,
            predictions,
            total_processing_time
        )
        
        return response
        
    except Exception as e:
        total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(
            f"Batch fraud analysis failed - Batch ID: {batch_id} - "
            f"Error: {str(e)} - "
            f"Time: {total_processing_time:.1f}ms",
            exc_info=True
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "batch_fraud_analysis_failed",
                "message": "Failed to analyze batch of transactions for fraud",
                "batch_id": batch_id,
                "total_processing_time_ms": total_processing_time
            }
        )


@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    summary="Get model status",
    description="Get current fraud detection model status and performance metrics."
)
async def get_model_status(
    ml_service: MLInferenceService = Depends(get_inference_service),
    current_user: User = Depends(get_current_active_user)
):
    """Get fraud detection model status.
    
    Args:
        ml_service: ML inference service
        api_key: API key for authentication
        
    Returns:
        Model status response
    """
    try:
        status = await ml_service.get_model_status()
        
        return ModelStatusResponse(
            model_version=status["version"],
            model_type=status["type"],
            is_loaded=status["loaded"],
            last_updated=datetime.fromisoformat(status["last_updated"]),
            performance_metrics=status["performance_metrics"],
            feature_count=status["feature_count"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_status_failed",
                "message": "Failed to retrieve model status"
            }
        )


@router.post(
    "/model/reload",
    summary="Reload fraud detection model",
    description="Reload the fraud detection model from storage."
)
async def reload_model(
    model_version: Optional[str] = Query(default=None, description="Specific model version to load"),
    ml_service: MLInferenceService = Depends(get_inference_service),
    current_user: User = Depends(require_analyst_role())
):
    """Reload fraud detection model.
    
    Args:
        model_version: Specific model version to load
        ml_service: ML inference service
        api_key: API key for authentication
        current_user: Current authenticated user
        
    Returns:
        Success response
    """
    try:
        logger.info(f"Reloading model - Version: {model_version or 'latest'} - User: {current_user}")
        
        await ml_service.reload_model(model_version)
        
        return JSONResponse(
            content={
                "message": "Model reloaded successfully",
                "model_version": model_version or "latest",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_reload_failed",
                "message": "Failed to reload model"
            }
        )


# Background tasks
async def store_fraud_analysis_result(
    db,
    transaction: Transaction,
    prediction,
    processing_time: float
):
    """Store fraud analysis result in database.
    
    Args:
        db: Database connection
        transaction: Original transaction
        prediction: ML prediction result
        processing_time: Processing time in milliseconds
    """
    try:
        # Store in fraud_scores table
        query = """
        INSERT INTO fraud_scores (
            transaction_id, user_id, fraud_probability, risk_level,
            confidence, model_version, processing_time_ms, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        await db.execute(
            query,
            transaction.transaction_id,
            transaction.user_id,
            prediction.fraud_probability,
            prediction.risk_level.value,
            prediction.confidence,
            prediction.model_version,
            processing_time,
            datetime.now()
        )
        
        logger.debug(f"Stored fraud analysis result for transaction {transaction.transaction_id}")
        
    except Exception as e:
        logger.error(f"Failed to store fraud analysis result: {e}", exc_info=True)


async def store_batch_fraud_analysis_results(
    db,
    batch_id: str,
    transactions: List[Transaction],
    predictions,
    total_processing_time: float
):
    """Store batch fraud analysis results in database.
    
    Args:
        db: Database connection
        batch_id: Batch processing ID
        transactions: List of transactions
        predictions: List of ML predictions
        total_processing_time: Total processing time in milliseconds
    """
    try:
        # Store individual results
        for transaction, prediction in zip(transactions, predictions):
            await store_fraud_analysis_result(
                db, transaction, prediction, 
                total_processing_time / len(transactions)
            )
        
        # Store batch summary
        batch_query = """
        INSERT INTO fraud_analysis_batches (
            batch_id, transaction_count, total_processing_time_ms,
            high_risk_count, created_at
        ) VALUES ($1, $2, $3, $4, $5)
        """
        
        high_risk_count = sum(
            1 for p in predictions 
            if p.risk_level.value in ["high", "critical"]
        )
        
        await db.execute(
            batch_query,
            batch_id,
            len(transactions),
            total_processing_time,
            high_risk_count,
            datetime.now()
        )
        
        logger.debug(f"Stored batch fraud analysis results for batch {batch_id}")
        
    except Exception as e:
        logger.error(f"Failed to store batch fraud analysis results: {e}", exc_info=True)