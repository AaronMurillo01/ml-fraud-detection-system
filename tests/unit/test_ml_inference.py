"""Unit tests for ML inference service."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any
from decimal import Decimal
import numpy as np
import pandas as pd

from service.ml_inference import (
    MLInferenceService,
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    BatchInferenceResponse
)
from service.model_loader import ModelLoader, ModelMetadata, ModelPrediction
from service.models import EnrichedTransaction
from service.models.transaction import Transaction, PaymentMethod, TransactionStatus


class TestMLInferenceService:
    """Test cases for MLInferenceService."""
    
    @pytest.fixture
    def mock_model_loader(self):
        """Create a mock model loader."""
        loader = Mock(spec=ModelLoader)
        return loader
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock ML model."""
        model = Mock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% fraud probability
        model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        return model
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample model metadata."""
        return ModelMetadata(
            model_name="fraud_detector_v1",
            model_version="1.0.0",
            model_type="xgboost",
            model_path="/models/fraud_detector_v1.pkl",
            feature_columns=["amount", "hour", "merchant_risk", "user_velocity"],
            target_column="is_fraud",
            hyperparameters={"max_depth": 6, "n_estimators": 100},
            threshold_config={"low_risk": 0.3, "high_risk": 0.7},
            training_metrics={"auc": 0.95, "precision": 0.88},
            validation_metrics={"auc": 0.93, "precision": 0.85},
            test_metrics={"auc": 0.92, "precision": 0.84},
            created_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_transaction(self):
        """Create a sample enriched transaction."""
        base_transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_456",
            merchant_id="merchant_789",
            amount=100.50,
            currency="USD",
            transaction_time=datetime.utcnow(),
            payment_method="credit_card",
            transaction_type="purchase"
        )
        
        return EnrichedTransaction(
            **base_transaction.model_dump(),
            is_weekend=False,
            is_night_time=False,
            hour_of_day=14,
            day_of_week=2,
            user_transaction_count_1h=2,
            user_transaction_count_24h=15,
            user_amount_sum_1h=Decimal('250.00'),
            user_amount_sum_24h=Decimal('1500.00'),
            card_transaction_count_1h=1,
            card_transaction_count_24h=8,
            merchant_transaction_count_1h=45,
            distance_from_home=5.2,
            distance_from_last_transaction=2.3,
            is_new_device=False,
            is_new_ip=True,
            amount_zscore_user_7d=1.5,
            amount_zscore_merchant_7d=0.8
        )
    
    @pytest.fixture
    def inference_service(self, mock_model_loader):
        """Create ML inference service with mocked dependencies."""
        return MLInferenceService(
            model_loader=mock_model_loader,
            default_model_name="fraud_detector_v1",
            default_model_version="1.0.0"
        )
    
    def test_init(self, mock_model_loader):
        """Test MLInferenceService initialization."""
        service = MLInferenceService(
            model_loader=mock_model_loader,
            default_model_name="test_model",
            default_model_version="2.0.0"
        )
        
        assert service.model_loader == mock_model_loader
        assert service.default_model_name == "test_model"
        assert service.default_model_version == "2.0.0"
        assert service._metadata_cache == {}
    
    @pytest.mark.asyncio
    async def test_predict_success(self, inference_service, mock_model_loader, mock_model, 
                                 sample_metadata, sample_transaction):
        """Test successful prediction."""
        # Setup mocks
        mock_model_loader.load_model.return_value = mock_model
        inference_service._get_model_metadata = AsyncMock(return_value=sample_metadata)
        
        # Create request
        request = InferenceRequest(
            transaction=sample_transaction,
            include_feature_importance=True,
            include_model_features=True
        )
        
        # Mock preprocessor
        with patch('service.ml_inference.ModelPreprocessor') as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor.prepare_features.return_value = pd.DataFrame({
                'amount': [100.50],
                'hour': [14],
                'merchant_risk': [0.3],
                'user_velocity': [2]
            })
            mock_preprocessor_class.return_value = mock_preprocessor
            
            # Execute prediction
            response = await inference_service.predict(request)
        
        # Assertions
        assert response.success is True
        assert response.transaction_id == "txn_123"
        assert response.prediction.fraud_score == 0.7
        assert response.prediction.risk_level in ["LOW", "MEDIUM", "HIGH"]
        assert response.prediction.confidence_score >= 0.0
        assert response.prediction.confidence_score <= 1.0
        assert len(response.prediction.feature_importance) == 4
        assert len(response.prediction.model_features) == 4
        assert response.prediction.processing_time_ms > 0
        assert response.error_message is None
        
        # Verify model loader was called
        mock_model_loader.load_model.assert_called_once_with(sample_metadata)
    
    @pytest.mark.asyncio
    async def test_predict_with_error(self, inference_service, mock_model_loader, sample_transaction):
        """Test prediction with error handling."""
        # Setup mock to raise exception
        mock_model_loader.load_model.side_effect = Exception("Model loading failed")
        inference_service._get_model_metadata = AsyncMock(side_effect=Exception("Metadata error"))
        
        # Create request
        request = InferenceRequest(transaction=sample_transaction)
        
        # Execute prediction
        response = await inference_service.predict(request)
        
        # Assertions
        assert response.success is False
        assert response.transaction_id == "txn_123"
        assert response.prediction.fraud_score == 0.5  # Default neutral score
        assert response.prediction.risk_level == "UNKNOWN"
        assert response.prediction.confidence_score == 0.0
        assert response.prediction.decision == "REVIEW"
        assert "Metadata error" in response.error_message
    
    @pytest.mark.asyncio
    async def test_predict_without_feature_importance(self, inference_service, mock_model_loader, 
                                                    mock_model, sample_metadata, sample_transaction):
        """Test prediction without feature importance."""
        # Setup mocks
        mock_model_loader.load_model.return_value = mock_model
        inference_service._get_model_metadata = AsyncMock(return_value=sample_metadata)
        
        # Create request without feature importance
        request = InferenceRequest(
            transaction=sample_transaction,
            include_feature_importance=False,
            include_model_features=False
        )
        
        # Mock preprocessor
        with patch('service.ml_inference.ModelPreprocessor') as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor.prepare_features.return_value = pd.DataFrame({
                'amount': [100.50],
                'hour': [14],
                'merchant_risk': [0.3],
                'user_velocity': [2]
            })
            mock_preprocessor_class.return_value = mock_preprocessor
            
            # Execute prediction
            response = await inference_service.predict(request)
        
        # Assertions
        assert response.success is True
        assert response.prediction.feature_importance == {}
        assert response.prediction.model_features == {}
    
    @pytest.mark.asyncio
    async def test_predict_batch_success(self, inference_service, mock_model_loader, 
                                       mock_model, sample_metadata, sample_transaction):
        """Test successful batch prediction."""
        # Setup mocks
        mock_model_loader.load_model.return_value = mock_model
        inference_service._get_model_metadata = AsyncMock(return_value=sample_metadata)
        
        # Create batch request with multiple transactions
        transactions = [sample_transaction, sample_transaction]
        request = BatchInferenceRequest(
            transactions=transactions,
            include_feature_importance=False,
            include_model_features=False
        )
        
        # Mock preprocessor
        with patch('service.ml_inference.ModelPreprocessor') as mock_preprocessor_class:
            mock_preprocessor = Mock()
            mock_preprocessor.prepare_features.return_value = pd.DataFrame({
                'amount': [100.50],
                'hour': [14],
                'merchant_risk': [0.3],
                'user_velocity': [2]
            })
            mock_preprocessor_class.return_value = mock_preprocessor
            
            # Execute batch prediction
            response = await inference_service.predict_batch(request)
        
        # Assertions
        assert response.batch_size == 2
        assert len(response.predictions) == 2
        assert response.success_count == 2
        assert response.error_count == 0
        assert response.total_processing_time_ms > 0
        assert response.average_processing_time_ms > 0
        
        # Check individual predictions
        for prediction in response.predictions:
            assert prediction.success is True
            assert prediction.prediction.fraud_score == 0.7
    
    def test_classify_risk_low(self, inference_service, sample_metadata):
        """Test risk classification for low risk score."""
        risk_level, decision, reason = inference_service._classify_risk(
            fraud_score=0.2,
            threshold_config=sample_metadata.threshold_config
        )
        
        assert risk_level == "LOW"
        assert decision == "APPROVE"
        assert "low risk" in reason.lower()
    
    def test_classify_risk_medium(self, inference_service, sample_metadata):
        """Test risk classification for medium risk score."""
        risk_level, decision, reason = inference_service._classify_risk(
            fraud_score=0.5,
            threshold_config=sample_metadata.threshold_config
        )
        
        assert risk_level == "MEDIUM"
        assert decision == "REVIEW"
        assert "medium risk" in reason.lower()
    
    def test_classify_risk_high(self, inference_service, sample_metadata):
        """Test risk classification for high risk score."""
        risk_level, decision, reason = inference_service._classify_risk(
            fraud_score=0.8,
            threshold_config=sample_metadata.threshold_config
        )
        
        assert risk_level == "HIGH"
        assert decision == "DECLINE"
        assert "high risk" in reason.lower()
    
    def test_calculate_confidence(self, inference_service, sample_metadata):
        """Test confidence score calculation."""
        # Test high confidence (score near extremes)
        confidence_high = inference_service._calculate_confidence(0.9, sample_metadata)
        assert confidence_high > 0.7
        
        confidence_low = inference_service._calculate_confidence(0.1, sample_metadata)
        assert confidence_low > 0.7
        
        # Test low confidence (score near middle)
        confidence_medium = inference_service._calculate_confidence(0.5, sample_metadata)
        assert confidence_medium < 0.7
    
    @pytest.mark.asyncio
    async def test_get_model_metadata_cached(self, inference_service, sample_metadata):
        """Test model metadata caching."""
        # Add metadata to cache
        cache_key = "fraud_detector_v1_1.0.0"
        inference_service._metadata_cache[cache_key] = sample_metadata
        
        # Get metadata (should use cache)
        result = await inference_service._get_model_metadata("fraud_detector_v1", "1.0.0")
        
        assert result == sample_metadata
    
    def test_get_service_stats(self, inference_service):
        """Test service statistics retrieval."""
        # Add some metadata to cache
        inference_service._metadata_cache["test_model"] = Mock()
        
        # Mock model loader stats
        inference_service.model_loader.get_cache_stats.return_value = {
            "cache_size": 2,
            "cache_hits": 10
        }
        
        stats = inference_service.get_service_stats()
        
        assert "default_model" in stats
        assert "metadata_cache_size" in stats
        assert "model_loader_stats" in stats
        assert stats["metadata_cache_size"] == 1
        assert stats["default_model"] == "fraud_detector_v1:1.0.0"


class TestInferenceModels:
    """Test cases for inference request/response models."""
    
    def test_inference_request_validation(self, sample_transaction):
        """Test InferenceRequest validation."""
        # Valid request
        request = InferenceRequest(
            transaction=sample_transaction,
            model_name="test_model",
            model_version="1.0.0",
            include_feature_importance=True,
            include_model_features=False
        )
        
        assert request.transaction == sample_transaction
        assert request.model_name == "test_model"
        assert request.model_version == "1.0.0"
        assert request.include_feature_importance is True
        assert request.include_model_features is False
    
    def test_inference_request_defaults(self, sample_transaction):
        """Test InferenceRequest default values."""
        request = InferenceRequest(transaction=sample_transaction)
        
        assert request.model_name is None
        assert request.model_version is None
        assert request.include_feature_importance is True
        assert request.include_model_features is True
    
    def test_batch_inference_request_validation(self, sample_transaction):
        """Test BatchInferenceRequest validation."""
        transactions = [sample_transaction, sample_transaction]
        
        request = BatchInferenceRequest(
            transactions=transactions,
            model_name="batch_model",
            include_feature_importance=False
        )
        
        assert len(request.transactions) == 2
        assert request.model_name == "batch_model"
        assert request.include_feature_importance is False
        assert request.include_model_features is False  # Default


class TestGlobalFunctions:
    """Test cases for global functions."""
    
    def test_get_inference_service_singleton(self):
        """Test that get_inference_service returns singleton."""
        from service.ml_inference import get_inference_service, _inference_service
        
        # Clear global instance
        import service.ml_inference
        service.ml_inference._inference_service = None
        
        # Get service instances
        service1 = get_inference_service()
        service2 = get_inference_service()
        
        # Should be the same instance
        assert service1 is service2
        assert isinstance(service1, MLInferenceService)
    
    def test_initialize_inference_service(self):
        """Test inference service initialization."""
        from service.ml_inference import initialize_inference_service
        
        mock_loader = Mock(spec=ModelLoader)
        
        service = initialize_inference_service(
            model_loader=mock_loader,
            default_model_name="custom_model",
            default_model_version="2.0.0"
        )
        
        assert service.model_loader == mock_loader
        assert service.default_model_name == "custom_model"
        assert service.default_model_version == "2.0.0"