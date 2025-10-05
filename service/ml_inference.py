"""ML inference service for fraud detection.

This module provides the main inference service for real-time fraud detection
using XGBoost models with sub-50ms latency requirements.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from service.model_loader import (
    ModelLoader, ModelMetadata, ModelPrediction, ModelPreprocessor,
    get_model_loader
)
from service.models import EnrichedTransaction
from cache import get_ml_cache_service, MLCacheService

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Request for ML inference."""
    transaction: EnrichedTransaction
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    include_feature_importance: bool = True
    include_model_features: bool = True


class InferenceResponse(BaseModel):
    """Response from ML inference."""
    transaction_id: str
    prediction: ModelPrediction
    model_metadata: Dict[str, Any]
    inference_timestamp: datetime
    success: bool = True
    error_message: Optional[str] = None


class BatchInferenceRequest(BaseModel):
    """Request for batch ML inference."""
    transactions: List[EnrichedTransaction]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    include_feature_importance: bool = False
    include_model_features: bool = False


class BatchInferenceResponse(BaseModel):
    """Response from batch ML inference."""
    predictions: List[InferenceResponse]
    batch_size: int
    total_processing_time_ms: float
    average_processing_time_ms: float
    success_count: int
    error_count: int
    model_metadata: Dict[str, Any]
    inference_timestamp: datetime


class MLInferenceService:
    """Main ML inference service for fraud detection."""
    
    def __init__(self, 
                 model_loader: Optional[ModelLoader] = None,
                 default_model_name: str = "fraud_detector_v1",
                 default_model_version: str = "1.0.0"):
        """Initialize ML inference service.
        
        Args:
            model_loader: Model loader instance (uses global if None)
            default_model_name: Default model name to use
            default_model_version: Default model version to use
        """
        self.model_loader = model_loader or get_model_loader()
        self.default_model_name = default_model_name
        self.default_model_version = default_model_version

        # Cache for model metadata to avoid repeated database queries
        self._metadata_cache: Dict[str, ModelMetadata] = {}

        # ML cache service for predictions and features
        self._ml_cache: Optional[MLCacheService] = None

        logger.info(f"MLInferenceService initialized with default model: {default_model_name}:{default_model_version}")

    async def _ensure_cache(self):
        """Ensure ML cache service is available."""
        if self._ml_cache is None:
            self._ml_cache = await get_ml_cache_service()

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Perform fraud detection inference on a single transaction.
        
        Args:
            request: Inference request with transaction data
            
        Returns:
            Inference response with prediction results
        """
        start_time = time.perf_counter()

        try:
            # Get model metadata
            model_name = request.model_name or self.default_model_name
            model_version = request.model_version or self.default_model_version

            # Check cache first
            await self._ensure_cache()
            cached_prediction = await self._ml_cache.get_cached_prediction(
                request.transaction, model_name, model_version
            )

            if cached_prediction:
                logger.debug(f"Returning cached prediction for transaction {request.transaction.transaction_id}")
                return InferenceResponse(
                    transaction_id=request.transaction.transaction_id,
                    prediction=cached_prediction,
                    model_metadata={
                        "model_name": model_name,
                        "model_version": model_version,
                        "cached": True
                    },
                    inference_timestamp=datetime.utcnow()
                )

            metadata = await self._get_model_metadata(model_name, model_version)

            # Load model asynchronously for better performance
            model = await self.model_loader.load_model_async(metadata)
            
            # Prepare features
            preprocessor = ModelPreprocessor(metadata.feature_columns)
            transaction_dict = request.transaction.dict()
            features_df = preprocessor.prepare_features(transaction_dict)
            
            # Make prediction - core ML inference step
            prediction_start = time.perf_counter()
            # Get probability distribution for both classes (non-fraud, fraud)
            fraud_probabilities = model.predict_proba(features_df)
            # Extract fraud probability (class 1) - this is our main risk score
            fraud_score = float(fraud_probabilities[0][1])  # Probability of fraud class
            prediction_time = (time.perf_counter() - prediction_start) * 1000

            # Calculate feature importance (if requested and supported)
            # This helps explain which features contributed most to the prediction
            feature_importance = {}
            if request.include_feature_importance and hasattr(model, 'feature_importances_'):
                # Map feature names to their importance scores for interpretability
                feature_importance = dict(zip(
                    metadata.feature_columns,
                    model.feature_importances_.tolist()
                ))

            # Prepare model features (if requested)
            # Return the actual feature values used in prediction for debugging/auditing
            model_features = {}
            if request.include_model_features:
                model_features = {col: features_df.iloc[0][col] for col in metadata.feature_columns}

            # Determine risk level and decision based on business rules
            # Convert continuous probability to discrete risk categories and actions
            risk_level, decision, decision_reason = self._classify_risk(
                fraud_score, metadata.threshold_config
            )

            # Calculate confidence score - measure of prediction reliability
            # Higher confidence means the model is more certain about its prediction
            confidence_score = self._calculate_confidence(fraud_score, metadata)
            
            # Create prediction result
            prediction = ModelPrediction(
                fraud_score=fraud_score,
                risk_level=risk_level,
                confidence_score=confidence_score,
                feature_importance=feature_importance,
                model_features=model_features,
                processing_time_ms=prediction_time,
                threshold_used=metadata.threshold_config.get('high_risk', 0.8),
                decision=decision,
                decision_reason=decision_reason
            )
            
            total_time = (time.perf_counter() - start_time) * 1000

            # Cache the prediction for future requests
            try:
                await self._ml_cache.cache_prediction(
                    request.transaction, prediction, model_name, model_version
                )
                logger.debug(f"Cached prediction for transaction {request.transaction.transaction_id}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache prediction: {cache_error}")

            logger.debug(f"Inference completed for {request.transaction.transaction_id} in {total_time:.2f}ms")

            response = InferenceResponse(
                transaction_id=request.transaction.transaction_id,
                prediction=prediction,
                model_metadata={
                    "model_name": metadata.model_name,
                    "model_version": metadata.model_version,
                    "model_type": metadata.model_type,
                    "cached": False
                },
                inference_timestamp=datetime.utcnow()
            )

            return response
            
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Inference failed for {request.transaction.transaction_id}: {e}")
            
            return InferenceResponse(
                transaction_id=request.transaction.transaction_id,
                prediction=ModelPrediction(
                    fraud_score=0.5,  # Default neutral score
                    risk_level="UNKNOWN",
                    confidence_score=0.0,
                    feature_importance={},
                    model_features={},
                    processing_time_ms=total_time,
                    threshold_used=0.5,
                    decision="REVIEW",
                    decision_reason=f"Inference error: {str(e)}"
                ),
                model_metadata={},
                inference_timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    async def predict_batch(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """Perform batch fraud detection inference.
        
        Args:
            request: Batch inference request
            
        Returns:
            Batch inference response with all predictions
        """
        start_time = time.perf_counter()
        
        try:
            # Get model metadata
            model_name = request.model_name or self.default_model_name
            model_version = request.model_version or self.default_model_version
            
            metadata = await self._get_model_metadata(model_name, model_version)

            # Load model once for all predictions (async for better performance)
            model = await self.model_loader.load_model_async(metadata)
            preprocessor = ModelPreprocessor(metadata.feature_columns)
            
            predictions = []
            success_count = 0
            error_count = 0
            
            # Process each transaction
            for transaction in request.transactions:
                try:
                    # Create individual inference request
                    individual_request = InferenceRequest(
                        transaction=transaction,
                        model_name=model_name,
                        model_version=model_version,
                        include_feature_importance=request.include_feature_importance,
                        include_model_features=request.include_model_features
                    )
                    
                    # Perform prediction (reuse loaded model)
                    response = await self._predict_single_with_loaded_model(
                        individual_request, model, metadata, preprocessor
                    )
                    
                    predictions.append(response)
                    
                    if response.success:
                        success_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"Batch prediction failed for {transaction.transaction_id}: {e}")
                    error_count += 1
                    
                    # Add error response
                    error_response = InferenceResponse(
                        transaction_id=transaction.transaction_id,
                        prediction=ModelPrediction(
                            fraud_score=0.5,
                            risk_level="UNKNOWN",
                            confidence_score=0.0,
                            feature_importance={},
                            model_features={},
                            processing_time_ms=0.0,
                            threshold_used=0.5,
                            decision="REVIEW",
                            decision_reason=f"Batch inference error: {str(e)}"
                        ),
                        model_metadata={},
                        inference_timestamp=datetime.utcnow(),
                        success=False,
                        error_message=str(e)
                    )
                    predictions.append(error_response)
            
            total_time = (time.perf_counter() - start_time) * 1000
            avg_time = total_time / len(request.transactions) if request.transactions else 0
            
            logger.info(f"Batch inference completed: {len(request.transactions)} transactions in {total_time:.2f}ms")
            
            return BatchInferenceResponse(
                predictions=predictions,
                batch_size=len(request.transactions),
                total_processing_time_ms=total_time,
                average_processing_time_ms=avg_time,
                success_count=success_count,
                error_count=error_count,
                model_metadata={
                    "model_name": metadata.model_name,
                    "model_version": metadata.model_version,
                    "model_type": metadata.model_type
                },
                inference_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Batch inference failed: {e}")
            
            return BatchInferenceResponse(
                predictions=[],
                batch_size=len(request.transactions),
                total_processing_time_ms=total_time,
                average_processing_time_ms=0.0,
                success_count=0,
                error_count=len(request.transactions),
                model_metadata={},
                inference_timestamp=datetime.utcnow()
            )

    async def batch_predict_optimized(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """Optimized batch prediction with concurrent processing and caching.

        Args:
            request: Batch inference request

        Returns:
            Batch inference response with all predictions
        """
        start_time = time.perf_counter()

        try:
            # Get model metadata
            model_name = request.model_name or self.default_model_name
            model_version = request.model_version or self.default_model_version

            metadata = await self._get_model_metadata(model_name, model_version)

            # Load model once for all predictions
            model = await self.model_loader.load_model_async(metadata)

            # Process transactions concurrently with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent predictions

            async def predict_single_transaction(transaction):
                async with semaphore:
                    try:
                        # Check cache first
                        await self._ensure_cache()
                        cached_prediction = await self._ml_cache.get_cached_prediction(
                            transaction, model_name, model_version
                        )

                        if cached_prediction:
                            return InferenceResponse(
                                transaction_id=transaction.transaction_id,
                                prediction=cached_prediction,
                                model_metadata={
                                    "model_name": model_name,
                                    "model_version": model_version,
                                    "cached": True
                                },
                                inference_timestamp=datetime.utcnow()
                            )

                        # Create individual inference request
                        individual_request = InferenceRequest(
                            transaction=transaction,
                            model_name=model_name,
                            model_version=model_version,
                            include_feature_importance=request.include_feature_importance,
                            include_model_features=request.include_model_features
                        )

                        # Perform prediction with loaded model
                        response = await self._predict_single_with_loaded_model(
                            individual_request, model, metadata, ModelPreprocessor(metadata.feature_columns)
                        )

                        # Cache the prediction
                        try:
                            await self._ml_cache.cache_prediction(
                                transaction, response.prediction, model_name, model_version
                            )
                        except Exception as cache_error:
                            logger.warning(f"Failed to cache prediction: {cache_error}")

                        return response

                    except Exception as e:
                        logger.error(f"Prediction failed for transaction {transaction.transaction_id}: {e}")
                        return InferenceResponse(
                            transaction_id=transaction.transaction_id,
                            prediction=ModelPrediction(
                                fraud_score=0.5,
                                risk_level="UNKNOWN",
                                confidence_score=0.0,
                                feature_importance={},
                                model_features={},
                                processing_time_ms=0,
                                threshold_used=0.5,
                                decision="ERROR",
                                decision_reason=f"Prediction error: {str(e)}"
                            ),
                            model_metadata={
                                "model_name": model_name,
                                "model_version": model_version,
                                "error": str(e)
                            },
                            inference_timestamp=datetime.utcnow()
                        )

            # Execute all predictions concurrently
            prediction_tasks = [
                predict_single_transaction(transaction)
                for transaction in request.transactions
            ]

            predictions = await asyncio.gather(*prediction_tasks, return_exceptions=True)

            # Process results
            successful_predictions = []
            success_count = 0
            error_count = 0

            for prediction in predictions:
                if isinstance(prediction, Exception):
                    error_count += 1
                    logger.error(f"Batch prediction task failed: {prediction}")
                elif hasattr(prediction, 'prediction') and prediction.prediction.decision != "ERROR":
                    successful_predictions.append(prediction)
                    success_count += 1
                else:
                    successful_predictions.append(prediction)
                    error_count += 1

            total_time = (time.perf_counter() - start_time) * 1000
            avg_time = total_time / len(request.transactions) if request.transactions else 0

            logger.info(f"Optimized batch inference completed: {success_count} success, {error_count} errors in {total_time:.2f}ms")

            return BatchInferenceResponse(
                predictions=successful_predictions,
                batch_size=len(request.transactions),
                total_processing_time_ms=total_time,
                average_processing_time_ms=avg_time,
                success_count=success_count,
                error_count=error_count,
                model_metadata={
                    "model_name": metadata.model_name,
                    "model_version": metadata.model_version,
                    "model_type": metadata.model_type,
                    "optimized": True,
                    "concurrent_processing": True
                },
                inference_timestamp=datetime.utcnow()
            )

        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Optimized batch inference failed: {e}")

            return BatchInferenceResponse(
                predictions=[],
                batch_size=len(request.transactions),
                total_processing_time_ms=total_time,
                average_processing_time_ms=0.0,
                success_count=0,
                error_count=len(request.transactions),
                model_metadata={
                    "model_name": request.model_name or self.default_model_name,
                    "model_version": request.model_version or self.default_model_version,
                    "error": str(e),
                    "optimized": True
                },
                inference_timestamp=datetime.utcnow()
            )

    async def _predict_single_with_loaded_model(
        self, 
        request: InferenceRequest, 
        model: Any, 
        metadata: ModelMetadata, 
        preprocessor: ModelPreprocessor
    ) -> InferenceResponse:
        """Perform prediction with pre-loaded model (for batch processing)."""
        start_time = time.perf_counter()
        
        try:
            # Prepare features
            transaction_dict = request.transaction.dict()
            features_df = preprocessor.prepare_features(transaction_dict)
            
            # Make prediction
            prediction_start = time.perf_counter()
            fraud_probabilities = model.predict_proba(features_df)
            fraud_score = float(fraud_probabilities[0][1])
            prediction_time = (time.perf_counter() - prediction_start) * 1000
            
            # Calculate feature importance (if requested)
            feature_importance = {}
            if request.include_feature_importance and hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    metadata.feature_columns,
                    model.feature_importances_.tolist()
                ))
            
            # Prepare model features (if requested)
            model_features = {}
            if request.include_model_features:
                model_features = {col: features_df.iloc[0][col] for col in metadata.feature_columns}
            
            # Determine risk level and decision
            risk_level, decision, decision_reason = self._classify_risk(
                fraud_score, metadata.threshold_config
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(fraud_score, metadata)
            
            # Create prediction result
            prediction = ModelPrediction(
                fraud_score=fraud_score,
                risk_level=risk_level,
                confidence_score=confidence_score,
                feature_importance=feature_importance,
                model_features=model_features,
                processing_time_ms=prediction_time,
                threshold_used=metadata.threshold_config.get('high_risk', 0.8),
                decision=decision,
                decision_reason=decision_reason
            )
            
            return InferenceResponse(
                transaction_id=request.transaction.transaction_id,
                prediction=prediction,
                model_metadata={
                    "model_name": metadata.model_name,
                    "model_version": metadata.model_version,
                    "model_type": metadata.model_type
                },
                inference_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            
            return InferenceResponse(
                transaction_id=request.transaction.transaction_id,
                prediction=ModelPrediction(
                    fraud_score=0.5,
                    risk_level="UNKNOWN",
                    confidence_score=0.0,
                    feature_importance={},
                    model_features={},
                    processing_time_ms=total_time,
                    threshold_used=0.5,
                    decision="REVIEW",
                    decision_reason=f"Prediction error: {str(e)}"
                ),
                model_metadata={},
                inference_timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
    
    def _classify_risk(self, fraud_score: float, threshold_config: Dict[str, float]) -> tuple[str, str, str]:
        """Classify risk level and decision based on fraud score.
        
        Args:
            fraud_score: Fraud probability score (0-1)
            threshold_config: Threshold configuration
            
        Returns:
            Tuple of (risk_level, decision, decision_reason)
        """
        low_threshold = threshold_config.get('low_risk', 0.3)
        medium_threshold = threshold_config.get('medium_risk', 0.6)
        high_threshold = threshold_config.get('high_risk', 0.8)
        critical_threshold = threshold_config.get('critical_risk', 0.95)
        
        if fraud_score < low_threshold:
            return "LOW", "APPROVE", f"Low fraud score ({fraud_score:.3f}) below threshold ({low_threshold})"
        elif fraud_score < medium_threshold:
            return "LOW", "APPROVE", f"Low-medium fraud score ({fraud_score:.3f}) within acceptable range"
        elif fraud_score < high_threshold:
            return "MEDIUM", "REVIEW", f"Medium fraud score ({fraud_score:.3f}) requires manual review"
        elif fraud_score < critical_threshold:
            return "HIGH", "DECLINE", f"High fraud score ({fraud_score:.3f}) exceeds risk tolerance"
        else:
            return "CRITICAL", "DECLINE", f"Critical fraud score ({fraud_score:.3f}) indicates high fraud risk"
    
    def _calculate_confidence(self, fraud_score: float, metadata: ModelMetadata) -> float:
        """Calculate confidence score based on model performance and prediction.
        
        Args:
            fraud_score: Fraud probability score
            metadata: Model metadata with performance metrics
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from model's validation AUC
        base_confidence = metadata.validation_metrics.get('auc', 0.8)
        
        # Adjust confidence based on prediction certainty
        # More confident when prediction is closer to 0 or 1
        prediction_certainty = abs(fraud_score - 0.5) * 2
        
        # Combine base confidence with prediction certainty
        confidence = base_confidence * (0.7 + 0.3 * prediction_certainty)
        
        return min(confidence, 1.0)
    
    async def _get_model_metadata(self, model_name: str, model_version: str) -> ModelMetadata:
        """Get model metadata (with caching).
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            Model metadata
            
        Note:
            In a real implementation, this would query the database.
            For now, we'll return mock metadata.
        """
        cache_key = f"{model_name}_{model_version}"
        
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        # Mock metadata (in real implementation, query from database)
        metadata = ModelMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type="XGBOOST",
            model_path=f"/models/{model_name}_{model_version}.pkl",
            feature_columns=[
                'amount', 'merchant_category', 'hour_of_day', 'day_of_week',
                'user_avg_amount', 'merchant_risk_score', 'location_risk_score',
                'velocity_1h', 'velocity_24h'
            ],
            target_column="is_fraud",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            },
            threshold_config={
                "low_risk": 0.3,
                "medium_risk": 0.6,
                "high_risk": 0.8,
                "critical_risk": 0.95
            },
            training_metrics={"auc": 0.95, "precision": 0.89, "recall": 0.87},
            validation_metrics={"auc": 0.93, "precision": 0.86, "recall": 0.84},
            test_metrics={"auc": 0.94, "precision": 0.87, "recall": 0.85},
            created_at=datetime.utcnow()
        )
        
        self._metadata_cache[cache_key] = metadata
        return metadata
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "default_model": f"{self.default_model_name}:{self.default_model_version}",
            "metadata_cache_size": len(self._metadata_cache),
            "model_loader_stats": self.model_loader.get_cache_stats()
        }


# Global inference service instance
_inference_service: Optional[MLInferenceService] = None


def get_inference_service() -> MLInferenceService:
    """Get global ML inference service instance.
    
    Returns:
        MLInferenceService instance
    """
    global _inference_service
    
    if _inference_service is None:
        _inference_service = MLInferenceService()
    
    return _inference_service


def initialize_inference_service(
    model_loader: Optional[ModelLoader] = None,
    default_model_name: str = "fraud_detector_v1",
    default_model_version: str = "1.0.0"
) -> MLInferenceService:
    """Initialize global ML inference service.
    
    Args:
        model_loader: Model loader instance
        default_model_name: Default model name
        default_model_version: Default model version
        
    Returns:
        MLInferenceService instance
    """
    global _inference_service
    
    _inference_service = MLInferenceService(
        model_loader=model_loader,
        default_model_name=default_model_name,
        default_model_version=default_model_version
    )
    
    return _inference_service