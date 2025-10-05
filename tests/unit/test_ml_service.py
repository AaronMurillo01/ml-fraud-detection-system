"""Unit tests for ML service components."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from service.models.transaction import Transaction, EnrichedTransaction
from service.model_loader import ModelPrediction, ModelMetadata
from service.models.prediction import PredictionResult, ModelTrainingResponse
from service.ml_inference import (
    MLInferenceService, InferenceRequest, InferenceResponse,
    BatchInferenceRequest, BatchInferenceResponse
)
from service.xgboost_model import XGBoostModelWrapper, XGBoostPrediction
from shared.models import TransactionStatus, PaymentMethod


class TestMLInferenceService:
    """Test cases for MLInferenceService."""
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample enriched transaction for testing."""
        from decimal import Decimal
        
        base_transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_456",
            merchant_id="merchant_789",
            amount=Decimal("100.50"),
            currency="USD",
            transaction_type="purchase",
            payment_method="credit_card",
            transaction_time=datetime.now(timezone.utc)
        )
        
        # Create enriched transaction
        enriched_transaction = EnrichedTransaction(
            **base_transaction.model_dump(),
            is_weekend=False,
            is_night_time=False,
            hour_of_day=14,
            day_of_week=2
        )
        
        return enriched_transaction
    
    def test_service_initialization(self):
        """Test ML inference service initialization."""
        service = MLInferenceService(
            default_model_name="test_model",
            default_model_version="1.0.0"
        )
        
        assert service.default_model_name == "test_model"
        assert service.default_model_version == "1.0.0"
        assert service.model_loader is not None
    
    @pytest.mark.asyncio
    async def test_inference_request_creation(self, sample_transaction):
        """Test creating an inference request."""
        request = InferenceRequest(
            transaction=sample_transaction,
            model_name="test_model",
            model_version="1.0.0",
            include_feature_importance=True
        )
        
        assert request.transaction.transaction_id == "txn_123"
        assert request.model_name == "test_model"
        assert request.model_version == "1.0.0"
        assert request.include_feature_importance is True
    
    @pytest.mark.asyncio
    async def test_inference_response_structure(self, sample_transaction):
        """Test inference response structure."""
        # Mock the model loader and prediction
        with patch('service.ml_inference.get_model_loader') as mock_loader:
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_loader.return_value.load_model.return_value = mock_model
            
            service = MLInferenceService()
            request = InferenceRequest(transaction=sample_transaction)
            
            # Mock the metadata method
            with patch.object(service, '_get_model_metadata') as mock_metadata:
                mock_metadata.return_value = Mock(
                    model_name="test_model",
                    model_version="1.0.0",
                    model_type="xgboost",
                    feature_columns=["amount", "is_weekend"],
                    threshold_config={"high_risk": 0.8}
                )
                
                response = await service.predict(request)
                
                assert isinstance(response, InferenceResponse)
                assert response.transaction_id == "txn_123"
                assert response.success is True
                assert response.prediction is not None


class TestXGBoostModelWrapper:
    """Test cases for XGBoostModelWrapper class."""
    
    def test_wrapper_initialization(self):
        """Test XGBoost model wrapper initialization."""
        from service.model_loader import ModelMetadata
        
        metadata = ModelMetadata(
            model_name="test_xgboost",
            model_version="1.0",
            model_type="xgboost",
            model_path="/path/to/model.pkl",
            feature_columns=["amount", "merchant_category", "hour_of_day"],
            target_column="is_fraud"
        )
        
        # Use a mock model path since we're testing initialization
        with patch('pathlib.Path.exists', return_value=False):
            try:
                wrapper = XGBoostModelWrapper("mock_path.pkl", metadata)
                # This should fail due to missing file, which is expected
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                # This is expected behavior
                pass
    
    def test_prediction_structure(self):
        """Test XGBoost prediction structure."""
        from service.model_loader import ModelMetadata
        
        metadata = ModelMetadata(
            model_name="test_xgboost",
            model_version="1.0",
            model_type="xgboost",
            model_path="/path/to/model.pkl",
            feature_columns=["amount", "merchant_category", "hour_of_day"],
            target_column="is_fraud"
        )
        
        # Mock a trained model
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pickle.load', return_value=mock_model):
            wrapper = XGBoostModelWrapper("mock_path.pkl", metadata)
        wrapper.model = mock_model
        wrapper.feature_names = ["amount", "is_weekend"]
        
        # Create sample features
        features = np.array([[150.0, 1]])
        
        prediction = wrapper.predict(features)
        
        assert isinstance(prediction, XGBoostPrediction)
        assert prediction.fraud_probability >= 0
        assert prediction.fraud_probability <= 1


class TestModelManager:
    """Test cases for ModelManager."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a model manager for testing."""
        from training.model_manager import ModelManager
        return ModelManager(registry_path="/tmp/test_models")
    
    def test_model_manager_initialization(self, model_manager):
        """Test model manager initialization."""
        assert model_manager.registry is not None
        assert model_manager.evaluator is not None
        assert model_manager.mlflow_client is not None
    
    @patch('training.model_manager.ModelRegistry')
    def test_deploy_model(self, mock_registry_class, model_manager):
        """Test deploying a model."""
        mock_registry = Mock()
        mock_registry.get_model.return_value = Mock()
        mock_registry.update_deployment_status.return_value = True
        model_manager.registry = mock_registry
        
        result = model_manager.deploy_model("test_model_123", "staging")
        
        assert result is True
        mock_registry.get_model.assert_called_once_with("test_model_123")
        mock_registry.update_deployment_status.assert_called_once_with("test_model_123", "staging")
    
    @patch('training.model_manager.ModelRegistry')
    def test_rollback_model(self, mock_registry_class, model_manager):
        """Test rolling back a model."""
        mock_registry = Mock()
        mock_registry.get_model.return_value = Mock()
        mock_registry.update_deployment_status.return_value = True
        model_manager.registry = mock_registry
        
        result = model_manager.rollback_model("current_model", "previous_model")
        
        assert result is True
        assert mock_registry.get_model.call_count == 2
        assert mock_registry.update_deployment_status.call_count == 2
    
    @patch('training.model_manager.ModelRegistry')
    def test_deploy_nonexistent_model(self, mock_registry_class, model_manager):
        """Test deploying a non-existent model."""
        mock_registry = Mock()
        mock_registry.get_model.return_value = None
        model_manager.registry = mock_registry
        
        result = model_manager.deploy_model("nonexistent_model", "staging")
        
        assert result is False
    
    @patch('training.model_manager.ModelEvaluator')
    def test_compare_models(self, mock_evaluator_class, model_manager):
        """Test comparing multiple models."""
        mock_evaluator = Mock()
        mock_evaluator.compare_models.return_value = Mock()
        model_manager.evaluator = mock_evaluator
        
        mock_registry = Mock()
        mock_registry.get_model.return_value = Mock()
        model_manager.registry = mock_registry
        
        import pandas as pd
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = pd.Series([0, 1, 0])
        
        result = model_manager.compare_models(["model1", "model2"], X_test, y_test)
        
        assert result is not None
        mock_evaluator.compare_models.assert_called_once()


# Note: FeatureEngineer and TransactionFeatures classes need to be implemented
# This section is reserved for future feature engineering tests


if __name__ == "__main__":
    pytest.main([__file__])