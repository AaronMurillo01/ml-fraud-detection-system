"""Unit tests for feature pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

from features.feature_pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
    FeaturePipelineMode,
    FeatureValidationLevel,
    ProcessedFeatures,
    FeatureQualityMetrics
)
from service.models.transaction import Transaction
from service.models import EnrichedTransaction


class TestFeaturePipelineConfig:
    """Test cases for FeaturePipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FeaturePipelineConfig()
        
        assert config.mode == FeaturePipelineMode.INFERENCE
        assert config.validation_level == FeatureValidationLevel.BASIC
        assert config.enable_enrichment is True
        assert config.enable_extraction is True
        assert config.enable_user_behavior is True
        assert config.enable_risk_features is True
        assert config.enable_parallel_processing is True
        assert config.max_workers == 4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeaturePipelineConfig(
            mode=FeaturePipelineMode.TRAINING,
            validation_level=FeatureValidationLevel.STRICT,
            enable_enrichment=False,
            max_workers=8
        )
        
        assert config.mode == FeaturePipelineMode.TRAINING
        assert config.validation_level == FeatureValidationLevel.STRICT
        assert config.enable_enrichment is False
        assert config.max_workers == 8


class TestProcessedFeatures:
    """Test cases for ProcessedFeatures."""
    
    def test_processed_features_creation(self):
        """Test creating processed features."""
        features = ProcessedFeatures(
            transaction_id="txn_123",
            user_id="user_123",
            features={"amount_zscore": 1.5, "frequency_1h": 3},
            feature_names=["amount_zscore", "frequency_1h"],
            feature_vector=[1.5, 3.0]
        )
        
        assert features.transaction_id == "txn_123"
        assert features.user_id == "user_123"
        assert features.features["amount_zscore"] == 1.5
        assert features.feature_names == ["amount_zscore", "frequency_1h"]
        assert features.feature_vector == [1.5, 3.0]
        assert features.pipeline_version == "1.0.0"
    
    def test_processed_features_defaults(self):
        """Test default values for processed features."""
        features = ProcessedFeatures(
            transaction_id="txn_123",
            user_id="user_123"
        )
        
        assert features.features == {}
        assert features.feature_names == []
        assert features.feature_vector is None
        assert features.validation_passed is True
        assert features.validation_errors == []


class TestFeaturePipeline:
    """Test cases for FeaturePipeline."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic pipeline configuration."""
        return FeaturePipelineConfig(
            enable_parallel_processing=False  # Disable for simpler testing
        )
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction for testing."""
        return Transaction(
            transaction_id="txn_123",
            user_id="user_123",
            merchant_id="merchant_123",
            amount=Decimal("150.75"),
            currency="USD",
            transaction_time=datetime.now(timezone.utc),
            transaction_type="purchase",
            payment_method="credit_card",
            location_lat=40.7128,
            location_lon=-74.0060,
            country_code="US",
            metadata={
                "category": "grocery",
                "card_type": "visa"
            }
        )
    
    @patch('features.feature_pipeline.get_enricher')
    @patch('features.feature_pipeline.get_feature_extractor')
    @patch('features.feature_pipeline.get_behavior_analyzer')
    @patch('features.feature_pipeline.get_risk_calculator')
    def test_pipeline_initialization(self, mock_risk_calc, mock_behavior, 
                                   mock_extractor, mock_enricher, basic_config):
        """Test pipeline initialization."""
        mock_enricher.return_value = Mock()
        mock_extractor.return_value = Mock()
        mock_behavior.return_value = Mock()
        mock_risk_calc.return_value = Mock()
        
        pipeline = FeaturePipeline(basic_config)
        
        assert pipeline.config == basic_config
        assert pipeline.enricher is not None
        assert pipeline.extractor is not None
        assert pipeline.behavior_analyzer is not None
        assert pipeline.risk_calculator is not None
        mock_enricher.assert_called_once()
        mock_extractor.assert_called_once()
        mock_behavior.assert_called_once()
        mock_risk_calc.assert_called_once()
    
    @patch('features.feature_pipeline.get_enricher')
    @patch('features.feature_pipeline.get_feature_extractor')
    @patch('features.feature_pipeline.get_behavior_analyzer')
    @patch('features.feature_pipeline.get_risk_calculator')
    def test_pipeline_disabled_components(self, mock_risk_calc, mock_behavior, 
                                        mock_extractor, mock_enricher):
        """Test pipeline with disabled components."""
        config = FeaturePipelineConfig(
            enable_enrichment=False,
            enable_extraction=False,
            enable_user_behavior=False,
            enable_risk_features=False
        )
        
        pipeline = FeaturePipeline(config)
        
        assert pipeline.enricher is None
        assert pipeline.extractor is None
        assert pipeline.behavior_analyzer is None
        assert pipeline.risk_calculator is None
        mock_enricher.assert_not_called()
        mock_extractor.assert_not_called()
        mock_behavior.assert_not_called()
        mock_risk_calc.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('features.feature_pipeline.get_enricher')
    @patch('features.feature_pipeline.get_feature_extractor')
    @patch('features.feature_pipeline.get_behavior_analyzer')
    @patch('features.feature_pipeline.get_risk_calculator')
    async def test_process_transaction_success(self, mock_risk_calc, mock_behavior, 
                                       mock_extractor, mock_enricher, 
                                       basic_config, sample_transaction):
        """Test successful transaction processing."""
        # Mock components
        mock_enricher_instance = Mock()
        mock_extractor_instance = Mock()
        mock_behavior_instance = Mock()
        mock_risk_calc_instance = Mock()
        
        mock_enricher.return_value = mock_enricher_instance
        mock_extractor.return_value = mock_extractor_instance
        mock_behavior.return_value = mock_behavior_instance
        mock_risk_calc.return_value = mock_risk_calc_instance
        
        # Mock enriched transaction
        enriched_transaction = EnrichedTransaction(
            **sample_transaction.model_dump(),
            is_weekend=False,
            is_night_time=False,
            hour_of_day=14,
            day_of_week=2
        )
        mock_enricher_instance.enrich.return_value = enriched_transaction
        
        # Mock feature extraction results
        mock_extractor_instance.extract.return_value = {
            "amount_zscore": 1.5,
            "frequency_1h": 3
        }
        
        mock_behavior_instance.analyze.return_value = {
            "velocity_score": 0.2,
            "pattern_score": 0.8
        }
        
        mock_risk_calc_instance.calculate.return_value = {
            "location_risk": 0.1,
            "merchant_risk": 0.15
        }
        
        pipeline = FeaturePipeline(basic_config)
        result = await pipeline.process_transaction(sample_transaction)
        
        assert isinstance(result, ProcessedFeatures)
        assert result.transaction_id == "txn_123"
        assert result.user_id == "user_123"
        assert "amount_zscore" in result.features
        assert "velocity_score" in result.features
        assert "location_risk" in result.features
        assert result.validation_passed is True
        
        # Verify component calls
        mock_enricher_instance.enrich.assert_called_once_with(sample_transaction)
        mock_extractor_instance.extract.assert_called_once_with(enriched_transaction)
        mock_behavior_instance.analyze.assert_called_once_with(enriched_transaction)
        mock_risk_calc_instance.calculate.assert_called_once_with(enriched_transaction)
    
    @pytest.mark.asyncio
    @patch('features.feature_pipeline.get_enricher')
    async def test_process_transaction_enrichment_error(self, mock_enricher, 
                                                basic_config, sample_transaction):
        """Test transaction processing with enrichment error."""
        mock_enricher_instance = Mock()
        mock_enricher.return_value = mock_enricher_instance
        mock_enricher_instance.enrich.side_effect = Exception("Enrichment failed")
        
        pipeline = FeaturePipeline(basic_config)
        result = await pipeline.process_transaction(sample_transaction)
        
        assert isinstance(result, ProcessedFeatures)
        assert result.transaction_id == "txn_123"
        assert result.validation_passed is False
        assert len(result.validation_errors) > 0
        assert "Enrichment failed" in str(result.validation_errors[0])
    
    def test_validate_features_basic(self, basic_config):
        """Test basic feature validation."""
        pipeline = FeaturePipeline(basic_config)
        
        # Valid features
        valid_features = {
            "amount_zscore": 1.5,
            "frequency_1h": 3,
            "risk_score": 0.2
        }
        
        result = pipeline._validate_features(valid_features)
        assert result["passed"] is True
        assert len(result["errors"]) == 0
        
        # Invalid features (with None values)
        invalid_features = {
            "amount_zscore": None,
            "frequency_1h": float('inf'),
            "risk_score": "invalid"
        }
        
        result = pipeline._validate_features(invalid_features)
        assert result["passed"] is False
        assert len(result["errors"]) > 0
    
    def test_validate_features_strict(self, basic_config):
        """Test strict feature validation."""
        config = FeaturePipelineConfig(
            validation_level=FeatureValidationLevel.STRICT
        )
        pipeline = FeaturePipeline(config)
        
        # Features with extreme values
        extreme_features = {
            "amount_zscore": 100.0,  # Very high z-score
            "frequency_1h": -1,      # Negative frequency
            "risk_score": 1.5        # Risk score > 1
        }
        
        result = pipeline._validate_features(extreme_features)
        assert result["passed"] is False
        assert len(result["errors"]) > 0
    
    def test_preprocess_features(self, basic_config):
        """Test feature preprocessing."""
        config = FeaturePipelineConfig(
            enable_scaling=True,
            scaling_method="standard"
        )
        pipeline = FeaturePipeline(config)
        
        features = {
            "amount_zscore": 1.5,
            "frequency_1h": 3,
            "risk_score": 0.2,
            "category": "grocery"  # Categorical feature
        }
        
        processed = pipeline._preprocess_features(features)
        
        assert isinstance(processed, dict)
        assert "category" in processed  # Categorical features preserved
        assert "amount_zscore" in processed
        assert "frequency_1h" in processed
        assert "risk_score" in processed
    
    def test_calculate_quality_metrics(self, basic_config):
        """Test feature quality metrics calculation."""
        pipeline = FeaturePipeline(basic_config)
        
        features = {
            "amount_zscore": 1.5,
            "frequency_1h": 3,
            "risk_score": 0.2
        }
        
        metrics = pipeline._calculate_quality_metrics(features)
        
        assert isinstance(metrics, dict)
        for feature_name in features.keys():
            assert feature_name in metrics
            assert isinstance(metrics[feature_name], FeatureQualityMetrics)
    
    @pytest.mark.asyncio
    @patch('features.feature_pipeline.get_enricher')
    async def test_caching_functionality(self, mock_enricher, sample_transaction):
        """Test feature caching functionality."""
        config = FeaturePipelineConfig(
            enable_caching=True,
            cache_ttl_minutes=60
        )
        
        mock_enricher_instance = Mock()
        mock_enricher.return_value = mock_enricher_instance
        
        enriched_transaction = EnrichedTransaction(
            **sample_transaction.model_dump(),
            is_weekend=False,
            is_night_time=False,
            hour_of_day=14,
            day_of_week=2
        )
        mock_enricher_instance.enrich.return_value = enriched_transaction
        
        pipeline = FeaturePipeline(config)
        
        # First call should process and cache
        result1 = await pipeline.process_transaction(sample_transaction)
        
        # Second call should use cache
        result2 = await pipeline.process_transaction(sample_transaction)
        
        assert result1.transaction_id == result2.transaction_id
        # Enricher should only be called once due to caching
        mock_enricher_instance.enrich.assert_called_once()
    
    def test_get_feature_importance(self, basic_config):
        """Test getting feature importance scores."""
        pipeline = FeaturePipeline(basic_config)
        
        importance = pipeline.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        for feature, score in importance.items():
            assert isinstance(feature, str)
            assert isinstance(score, float)
            assert 0 <= score <= 1


class TestFeatureQualityMetrics:
    """Test cases for FeatureQualityMetrics."""
    
    def test_quality_metrics_creation(self):
        """Test creating feature quality metrics."""
        metrics = FeatureQualityMetrics(
            missing_rate=0.05,
            outlier_rate=0.02,
            variance=0.92,
            skewness=0.12,
            kurtosis=2.1,
            correlation_with_target=0.15,
            information_value=0.25
        )
        
        assert metrics.missing_rate == 0.05
        assert metrics.outlier_rate == 0.02
        assert metrics.variance == 0.92
        assert metrics.skewness == 0.12
        assert metrics.kurtosis == 2.1
        assert metrics.correlation_with_target == 0.15
        assert metrics.information_value == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])