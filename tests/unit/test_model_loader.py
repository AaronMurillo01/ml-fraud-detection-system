"""Unit tests for model loader service."""

import pytest
import pickle
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

from service.model_loader import (
    ModelLoader,
    ModelMetadata,
    ModelPrediction,
    LoadedModel,
    ModelLoadError,
    ModelValidationError
)


class TestModelMetadata:
    """Test cases for ModelMetadata."""
    
    def test_model_metadata_creation(self):
        """Test ModelMetadata creation with all fields."""
        metadata = ModelMetadata(
            model_name="fraud_detector",
            model_version="1.0.0",
            model_type="xgboost",
            model_path="/models/fraud_detector.pkl",
            feature_columns=["amount", "merchant_risk"],
            target_column="is_fraud",
            hyperparameters={"max_depth": 6},
            threshold_config={"low_risk": 0.3, "high_risk": 0.7},
            training_metrics={"auc": 0.95},
            validation_metrics={"auc": 0.93},
            test_metrics={"auc": 0.92},
            created_at=datetime.utcnow()
        )
        
        assert metadata.model_name == "fraud_detector"
        assert metadata.model_version == "1.0.0"
        assert metadata.model_type == "xgboost"
        assert len(metadata.feature_columns) == 2
        assert metadata.target_column == "is_fraud"
        assert metadata.hyperparameters["max_depth"] == 6
        assert metadata.threshold_config["low_risk"] == 0.3
    
    def test_model_metadata_defaults(self):
        """Test ModelMetadata with default values."""
        metadata = ModelMetadata(
            model_name="test_model",
            model_version="1.0.0",
            model_type="sklearn",
            model_path="/test/path",
            feature_columns=["feature1"],
            target_column="target"
        )
        
        assert metadata.hyperparameters == {}
        assert metadata.threshold_config == {}
        assert metadata.training_metrics == {}
        assert metadata.validation_metrics == {}
        assert metadata.test_metrics == {}
        assert isinstance(metadata.created_at, datetime)
    
    def test_model_metadata_validation(self):
        """Test ModelMetadata validation."""
        # Test empty feature columns
        with pytest.raises(ValueError, match="Feature columns cannot be empty"):
            ModelMetadata(
                model_name="test",
                model_version="1.0.0",
                model_type="sklearn",
                model_path="/test",
                feature_columns=[],
                target_column="target"
            )
        
        # Test empty model name
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ModelMetadata(
                model_name="",
                model_version="1.0.0",
                model_type="sklearn",
                model_path="/test",
                feature_columns=["feature1"],
                target_column="target"
            )


class TestModelPrediction:
    """Test cases for ModelPrediction."""
    
    def test_model_prediction_creation(self):
        """Test ModelPrediction creation."""
        prediction = ModelPrediction(
            fraud_score=0.75,
            risk_level="HIGH",
            confidence_score=0.85,
            decision="DECLINE",
            reason="High fraud probability detected",
            feature_importance={"amount": 0.4, "merchant_risk": 0.6},
            model_features={"amount": 100.0, "merchant_risk": 0.8},
            processing_time_ms=25.5
        )
        
        assert prediction.fraud_score == 0.75
        assert prediction.risk_level == "HIGH"
        assert prediction.confidence_score == 0.85
        assert prediction.decision == "DECLINE"
        assert prediction.reason == "High fraud probability detected"
        assert len(prediction.feature_importance) == 2
        assert len(prediction.model_features) == 2
        assert prediction.processing_time_ms == 25.5
    
    def test_model_prediction_defaults(self):
        """Test ModelPrediction with default values."""
        prediction = ModelPrediction(
            fraud_score=0.5,
            risk_level="MEDIUM",
            confidence_score=0.6,
            decision="REVIEW"
        )
        
        assert prediction.reason == ""
        assert prediction.feature_importance == {}
        assert prediction.model_features == {}
        assert prediction.processing_time_ms == 0.0
    
    def test_model_prediction_validation(self):
        """Test ModelPrediction validation."""
        # Test fraud_score range
        with pytest.raises(ValueError, match="Fraud score must be between 0 and 1"):
            ModelPrediction(
                fraud_score=1.5,
                risk_level="HIGH",
                confidence_score=0.8,
                decision="DECLINE"
            )
        
        # Test confidence_score range
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            ModelPrediction(
                fraud_score=0.5,
                risk_level="MEDIUM",
                confidence_score=-0.1,
                decision="REVIEW"
            )


class TestLoadedModel:
    """Test cases for LoadedModel."""
    
    def test_loaded_model_creation(self):
        """Test LoadedModel creation."""
        mock_model = Mock()
        metadata = ModelMetadata(
            model_name="test_model",
            model_version="1.0.0",
            model_type="sklearn",
            model_path="/test/path",
            feature_columns=["feature1"],
            target_column="target"
        )
        
        loaded_model = LoadedModel(
            model=mock_model,
            metadata=metadata,
            loaded_at=datetime.utcnow()
        )
        
        assert loaded_model.model == mock_model
        assert loaded_model.metadata == metadata
        assert isinstance(loaded_model.loaded_at, datetime)
    
    def test_loaded_model_defaults(self):
        """Test LoadedModel with default loaded_at."""
        mock_model = Mock()
        metadata = ModelMetadata(
            model_name="test_model",
            model_version="1.0.0",
            model_type="sklearn",
            model_path="/test/path",
            feature_columns=["feature1"],
            target_column="target"
        )
        
        loaded_model = LoadedModel(model=mock_model, metadata=metadata)
        
        assert isinstance(loaded_model.loaded_at, datetime)


class TestModelLoader:
    """Test cases for ModelLoader."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample model metadata."""
        return ModelMetadata(
            model_name="fraud_detector",
            model_version="1.0.0",
            model_type="xgboost",
            model_path="/models/fraud_detector.pkl",
            feature_columns=["amount", "merchant_risk", "user_velocity"],
            target_column="is_fraud",
            hyperparameters={"max_depth": 6, "n_estimators": 100},
            threshold_config={"low_risk": 0.3, "high_risk": 0.7},
            training_metrics={"auc": 0.95, "precision": 0.88}
        )
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock ML model."""
        model = Mock()
        model.predict_proba.return_value = [[0.3, 0.7]]
        model.feature_importances_ = [0.2, 0.3, 0.5]
        return model
    
    @pytest.fixture
    def model_loader(self):
        """Create ModelLoader instance."""
        return ModelLoader(
            model_cache_size=10,
            cache_ttl_hours=24,
            model_base_path="/models"
        )
    
    def test_model_loader_init(self):
        """Test ModelLoader initialization."""
        loader = ModelLoader(
            model_cache_size=5,
            cache_ttl_hours=12,
            model_base_path="/custom/path"
        )
        
        assert loader.model_cache_size == 5
        assert loader.cache_ttl_hours == 12
        assert loader.model_base_path == "/custom/path"
        assert loader._model_cache == {}
        assert loader._cache_stats["cache_hits"] == 0
        assert loader._cache_stats["cache_misses"] == 0
    
    def test_model_loader_defaults(self):
        """Test ModelLoader with default values."""
        loader = ModelLoader()
        
        assert loader.model_cache_size == 100
        assert loader.cache_ttl_hours == 24
        assert loader.model_base_path == "./models"
    
    def test_load_model_success(self, model_loader, sample_metadata, mock_model):
        """Test successful model loading."""
        # Mock file operations
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pickle.load', return_value=mock_model) as mock_pickle:
                with patch('os.path.exists', return_value=True):
                    
                    loaded_model = model_loader.load_model(sample_metadata)
                    
                    assert loaded_model.model == mock_model
                    assert loaded_model.metadata == sample_metadata
                    assert isinstance(loaded_model.loaded_at, datetime)
                    
                    # Verify file operations
                    mock_file.assert_called_once_with(sample_metadata.model_path, 'rb')
                    mock_pickle.assert_called_once()
    
    def test_load_model_file_not_found(self, model_loader, sample_metadata):
        """Test model loading when file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(ModelLoadError, match="Model file not found"):
                model_loader.load_model(sample_metadata)
    
    def test_load_model_pickle_error(self, model_loader, sample_metadata):
        """Test model loading with pickle error."""
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', side_effect=pickle.PickleError("Corrupt file")):
                with patch('os.path.exists', return_value=True):
                    
                    with pytest.raises(ModelLoadError, match="Failed to load model"):
                        model_loader.load_model(sample_metadata)
    
    def test_load_model_validation_error(self, model_loader, sample_metadata):
        """Test model loading with validation error."""
        # Create a model without required attributes
        invalid_model = Mock()
        del invalid_model.predict_proba  # Remove required method
        
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=invalid_model):
                with patch('os.path.exists', return_value=True):
                    
                    with pytest.raises(ModelValidationError, match="Model must have predict_proba method"):
                        model_loader.load_model(sample_metadata)
    
    def test_load_model_caching(self, model_loader, sample_metadata, mock_model):
        """Test model caching functionality."""
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=mock_model):
                with patch('os.path.exists', return_value=True):
                    
                    # First load - should load from file
                    loaded_model1 = model_loader.load_model(sample_metadata)
                    
                    # Second load - should load from cache
                    loaded_model2 = model_loader.load_model(sample_metadata)
                    
                    # Should be the same cached instance
                    assert loaded_model1 is loaded_model2
                    
                    # Check cache stats
                    stats = model_loader.get_cache_stats()
                    assert stats["cache_hits"] == 1
                    assert stats["cache_misses"] == 1
                    assert stats["cache_size"] == 1
    
    def test_cache_expiry(self, model_loader, sample_metadata, mock_model):
        """Test cache expiry functionality."""
        # Set very short TTL for testing
        model_loader.cache_ttl_hours = 0.001  # ~3.6 seconds
        
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=mock_model):
                with patch('os.path.exists', return_value=True):
                    
                    # Load model
                    loaded_model1 = model_loader.load_model(sample_metadata)
                    
                    # Manually expire the cache entry
                    cache_key = model_loader._get_cache_key(sample_metadata)
                    if cache_key in model_loader._model_cache:
                        model_loader._model_cache[cache_key].loaded_at = (
                            datetime.utcnow() - timedelta(hours=1)
                        )
                    
                    # Load again - should reload from file due to expiry
                    loaded_model2 = model_loader.load_model(sample_metadata)
                    
                    # Should be different instances due to reload
                    assert loaded_model1 is not loaded_model2
    
    def test_cache_size_limit(self, sample_metadata, mock_model):
        """Test cache size limit enforcement."""
        # Create loader with small cache size
        loader = ModelLoader(model_cache_size=2)
        
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=mock_model):
                with patch('os.path.exists', return_value=True):
                    
                    # Create multiple metadata objects
                    metadata1 = sample_metadata
                    metadata2 = ModelMetadata(
                        model_name="model2",
                        model_version="1.0.0",
                        model_type="sklearn",
                        model_path="/models/model2.pkl",
                        feature_columns=["feature1"],
                        target_column="target"
                    )
                    metadata3 = ModelMetadata(
                        model_name="model3",
                        model_version="1.0.0",
                        model_type="sklearn",
                        model_path="/models/model3.pkl",
                        feature_columns=["feature1"],
                        target_column="target"
                    )
                    
                    # Load models to fill cache
                    loader.load_model(metadata1)
                    loader.load_model(metadata2)
                    
                    # Cache should be at limit
                    assert len(loader._model_cache) == 2
                    
                    # Load third model - should evict oldest
                    loader.load_model(metadata3)
                    
                    # Cache should still be at limit
                    assert len(loader._model_cache) == 2
                    
                    # First model should be evicted
                    key1 = loader._get_cache_key(metadata1)
                    assert key1 not in loader._model_cache
    
    def test_validate_model_success(self, model_loader, mock_model):
        """Test successful model validation."""
        # Mock model has predict_proba method
        model_loader._validate_model(mock_model)
        # Should not raise any exception
    
    def test_validate_model_missing_method(self, model_loader):
        """Test model validation with missing predict_proba method."""
        invalid_model = Mock()
        del invalid_model.predict_proba
        
        with pytest.raises(ModelValidationError, match="Model must have predict_proba method"):
            model_loader._validate_model(invalid_model)
    
    def test_validate_model_invalid_method(self, model_loader):
        """Test model validation with non-callable predict_proba."""
        invalid_model = Mock()
        invalid_model.predict_proba = "not_callable"
        
        with pytest.raises(ModelValidationError, match="predict_proba must be callable"):
            model_loader._validate_model(invalid_model)
    
    def test_get_cache_key(self, model_loader, sample_metadata):
        """Test cache key generation."""
        cache_key = model_loader._get_cache_key(sample_metadata)
        
        expected_key = f"{sample_metadata.model_name}_{sample_metadata.model_version}_{sample_metadata.model_path}"
        assert cache_key == expected_key
    
    def test_cleanup_expired_cache(self, model_loader, sample_metadata, mock_model):
        """Test expired cache cleanup."""
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=mock_model):
                with patch('os.path.exists', return_value=True):
                    
                    # Load model
                    model_loader.load_model(sample_metadata)
                    
                    # Manually expire the cache entry
                    cache_key = model_loader._get_cache_key(sample_metadata)
                    model_loader._model_cache[cache_key].loaded_at = (
                        datetime.utcnow() - timedelta(hours=25)  # Older than TTL
                    )
                    
                    # Trigger cleanup
                    model_loader._cleanup_expired_cache()
                    
                    # Cache should be empty
                    assert len(model_loader._model_cache) == 0
    
    def test_get_cache_stats(self, model_loader, sample_metadata, mock_model):
        """Test cache statistics retrieval."""
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=mock_model):
                with patch('os.path.exists', return_value=True):
                    
                    # Load model twice to generate stats
                    model_loader.load_model(sample_metadata)
                    model_loader.load_model(sample_metadata)  # Cache hit
                    
                    stats = model_loader.get_cache_stats()
                    
                    assert stats["cache_size"] == 1
                    assert stats["cache_hits"] == 1
                    assert stats["cache_misses"] == 1
                    assert stats["hit_rate"] == 0.5
                    assert "total_requests" in stats
    
    def test_clear_cache(self, model_loader, sample_metadata, mock_model):
        """Test cache clearing."""
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=mock_model):
                with patch('os.path.exists', return_value=True):
                    
                    # Load model to populate cache
                    model_loader.load_model(sample_metadata)
                    
                    assert len(model_loader._model_cache) == 1
                    
                    # Clear cache
                    model_loader.clear_cache()
                    
                    assert len(model_loader._model_cache) == 0
                    
                    # Stats should be reset
                    stats = model_loader.get_cache_stats()
                    assert stats["cache_hits"] == 0
                    assert stats["cache_misses"] == 0


class TestExceptions:
    """Test cases for custom exceptions."""
    
    def test_model_load_error(self):
        """Test ModelLoadError exception."""
        error = ModelLoadError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_model_validation_error(self):
        """Test ModelValidationError exception."""
        error = ModelValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, Exception)