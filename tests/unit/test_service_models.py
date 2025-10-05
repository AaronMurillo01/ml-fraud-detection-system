"""Unit tests for service models."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional
from pydantic import ValidationError

from service.models import (
    EnrichedTransaction,
    ModelPerformanceMetrics,
    FeatureImportance,
    PredictionResult,
    BatchPredictionResult,
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelEvaluationResult,
    HealthCheckResponse,
    ServiceMetrics
)
from shared.models import (
    Transaction,
    PaymentMethod,
    TransactionStatus,
    RiskLevel
)


class TestEnrichedTransaction:
    """Test cases for EnrichedTransaction model."""
    
    @pytest.fixture
    def base_transaction_data(self):
        """Create base transaction data."""
        from service.models.transaction import TransactionType, TransactionStatus, PaymentMethod
        
        return {
            "transaction_id": "txn_123456",
            "user_id": "user_789",
            "merchant_id": "merchant_456",
            "amount": 150.75,
            "transaction_time": datetime.utcnow(),  # Correct field name
            "payment_method": PaymentMethod.CREDIT_CARD,
            "transaction_type": TransactionType.PURCHASE,  # Use enum
            "status": TransactionStatus.APPROVED,
            "currency": "USD",
            "metadata": {"channel": "online", "category": "retail"},
            # Required fields for service EnrichedTransaction
            "is_weekend": False,
            "is_night_time": False,
            "hour_of_day": 14,
            "day_of_week": 2
        }
    

    
    def test_enriched_transaction_creation(self, base_transaction_data):
        """Test EnrichedTransaction creation with all fields."""
        # Add service-specific enrichment fields
        service_enrichment = {
            "user_transaction_count_1h": 3,
            "user_transaction_count_24h": 15,
            "user_amount_sum_1h": 450.25,
            "user_amount_sum_24h": 2150.75,
            "card_transaction_count_1h": 1,
            "card_transaction_count_24h": 8,
            "merchant_transaction_count_1h": 45,
            "distance_from_home": 15.5,
            "distance_from_last_transaction": 2.3,
            "is_new_device": False,
            "is_new_ip": True,
            "amount_zscore_user_7d": 1.5,
            "amount_zscore_merchant_7d": 0.8
        }
        
        enriched_txn = EnrichedTransaction(
            **base_transaction_data,
            **service_enrichment
        )
        
        # Test base transaction fields
        assert enriched_txn.transaction_id == "txn_123456"
        assert enriched_txn.user_id == "user_789"
        assert enriched_txn.merchant_id == "merchant_456"
        assert enriched_txn.amount == 150.75
        assert enriched_txn.payment_method == PaymentMethod.CREDIT_CARD
        assert enriched_txn.status == TransactionStatus.APPROVED
        
        # Test service-specific enrichment fields
        assert enriched_txn.user_transaction_count_1h == 3
        assert enriched_txn.user_transaction_count_24h == 15
        assert enriched_txn.distance_from_home == 15.5
        assert enriched_txn.is_new_ip == True
    
    def test_enriched_transaction_minimal(self, base_transaction_data):
        """Test EnrichedTransaction with minimal required fields."""
        enriched_txn = EnrichedTransaction(**base_transaction_data)
        
        # Test default values for service enrichment fields
        assert enriched_txn.user_transaction_count_1h == 0
        assert enriched_txn.user_transaction_count_24h == 0
        assert enriched_txn.user_amount_sum_1h == 0
        assert enriched_txn.user_amount_sum_24h == 0
        assert enriched_txn.card_transaction_count_1h == 0
        assert enriched_txn.card_transaction_count_24h == 0
        assert enriched_txn.merchant_transaction_count_1h == 0
        assert enriched_txn.is_new_device == False
        assert enriched_txn.is_new_ip == False
    
    def test_enriched_transaction_validation(self):
        """Test EnrichedTransaction validation."""
        from service.models.transaction import TransactionType, TransactionStatus, PaymentMethod
        
        # Create base data without hour_of_day and day_of_week for testing
        base_data = {
            "transaction_id": "txn_123456",
            "user_id": "user_789", 
            "merchant_id": "merchant_456",
            "amount": 150.75,
            "transaction_time": datetime.utcnow(),
            "payment_method": PaymentMethod.CREDIT_CARD,
            "transaction_type": TransactionType.PURCHASE,
            "status": TransactionStatus.APPROVED,
            "currency": "USD",
            "metadata": {"channel": "online"},
            "is_weekend": False,
            "is_night_time": False
        }
        
        # Test invalid hour_of_day (out of range)
        with pytest.raises(ValidationError):
            EnrichedTransaction(
                **base_data,
                hour_of_day=25,  # Invalid hour
                day_of_week=2
            )
        
        # Test invalid day_of_week (out of range)
        with pytest.raises(ValidationError):
            EnrichedTransaction(
                **base_data,
                hour_of_day=14,
                day_of_week=8  # Invalid day
            )
        
        # Test negative transaction counts
        with pytest.raises(ValidationError):
            EnrichedTransaction(
                **base_data,
                hour_of_day=14,
                day_of_week=2,
                user_transaction_count_1h=-1  # Negative count
            )
    
    def test_enriched_transaction_serialization(self, base_transaction_data):
        """Test EnrichedTransaction serialization."""
        enriched_txn = EnrichedTransaction(
            **base_transaction_data,
            user_transaction_count_1h=5,
            distance_from_home=10.0
        )
        
        # Test dict conversion
        txn_dict = enriched_txn.model_dump()
        assert "user_transaction_count_1h" in txn_dict
        assert "distance_from_home" in txn_dict
        assert "is_weekend" in txn_dict
        assert "is_night_time" in txn_dict
        
        # Test JSON serialization
        txn_json = enriched_txn.model_dump_json()
        assert isinstance(txn_json, str)
        assert "user_transaction_count_1h" in txn_json


class TestModelPerformanceMetrics:
    """Test cases for ModelPerformanceMetrics model."""
    
    def test_model_performance_metrics_creation(self):
        """Test ModelPerformanceMetrics creation."""
        metrics = ModelPerformanceMetrics(
            accuracy=0.95,
            precision=0.88,
            recall=0.92,
            f1_score=0.90,
            auc_roc=0.96,
            auc_pr=0.89,
            confusion_matrix=[[850, 50], [30, 70]],
            classification_report={
                "0": {"precision": 0.97, "recall": 0.94, "f1-score": 0.96},
                "1": {"precision": 0.58, "recall": 0.70, "f1-score": 0.64}
            },
            feature_importance={"amount": 0.35, "merchant_risk": 0.25, "velocity": 0.40},
            threshold_metrics={
                "0.3": {"precision": 0.75, "recall": 0.95},
                "0.5": {"precision": 0.85, "recall": 0.88},
                "0.7": {"precision": 0.92, "recall": 0.75}
            }
        )
        
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.88
        assert metrics.recall == 0.92
        assert metrics.f1_score == 0.90
        assert metrics.auc_roc == 0.96
        assert metrics.auc_pr == 0.89
        assert len(metrics.confusion_matrix) == 2
        assert "0" in metrics.classification_report
        assert "amount" in metrics.feature_importance
        assert "0.5" in metrics.threshold_metrics
    
    def test_model_performance_metrics_validation(self):
        """Test ModelPerformanceMetrics validation."""
        # Test accuracy out of range
        with pytest.raises(ValidationError, match="Accuracy must be between 0 and 1"):
            ModelPerformanceMetrics(
                accuracy=1.5,
                precision=0.8,
                recall=0.9,
                f1_score=0.85,
                auc_roc=0.92
            )
        
        # Test negative precision
        with pytest.raises(ValidationError, match="Precision must be between 0 and 1"):
            ModelPerformanceMetrics(
                accuracy=0.9,
                precision=-0.1,
                recall=0.9,
                f1_score=0.85,
                auc_roc=0.92
            )
    
    def test_model_performance_metrics_defaults(self):
        """Test ModelPerformanceMetrics with default values."""
        metrics = ModelPerformanceMetrics(
            accuracy=0.9,
            precision=0.85,
            recall=0.88,
            f1_score=0.86,
            auc_roc=0.93
        )
        
        assert metrics.auc_pr is None
        assert metrics.confusion_matrix == []
        assert metrics.classification_report == {}
        assert metrics.feature_importance == {}
        assert metrics.threshold_metrics == {}


class TestFeatureImportance:
    """Test cases for FeatureImportance model."""
    
    def test_feature_importance_creation(self):
        """Test FeatureImportance creation."""
        importance = FeatureImportance(
            feature_name="transaction_amount",
            importance_score=0.35,
            rank=1,
            category="transaction",
            description="The monetary amount of the transaction"
        )
        
        assert importance.feature_name == "transaction_amount"
        assert importance.importance_score == 0.35
        assert importance.rank == 1
        assert importance.category == "transaction"
        assert importance.description == "The monetary amount of the transaction"
    
    def test_feature_importance_validation(self):
        """Test FeatureImportance validation."""
        # Test negative importance score
        with pytest.raises(ValidationError, match="Importance score must be non-negative"):
            FeatureImportance(
                feature_name="test_feature",
                importance_score=-0.1,
                rank=1
            )
        
        # Test zero rank
        with pytest.raises(ValidationError, match="Rank must be positive"):
            FeatureImportance(
                feature_name="test_feature",
                importance_score=0.5,
                rank=0
            )
    
    def test_feature_importance_defaults(self):
        """Test FeatureImportance with default values."""
        importance = FeatureImportance(
            feature_name="test_feature",
            importance_score=0.25,
            rank=2
        )
        
        assert importance.category == "unknown"
        assert importance.description == ""


class TestPredictionResult:
    """Test cases for PredictionResult model."""
    
    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        result = PredictionResult(
            transaction_id="txn_123",
            fraud_probability=0.75,
            risk_level=RiskLevel.HIGH,
            decision="DECLINE",
            confidence_score=0.88,
            feature_contributions={
                "amount": 0.3,
                "merchant_risk": 0.25,
                "velocity": 0.2
            },
            model_version="v1.2.0",
            processing_time_ms=45.2,
            timestamp=datetime.utcnow(),
            explanation="High fraud probability due to unusual transaction pattern"
        )
        
        assert result.transaction_id == "txn_123"
        assert result.fraud_probability == 0.75
        assert result.risk_level == RiskLevel.HIGH
        assert result.decision == "DECLINE"
        assert result.confidence_score == 0.88
        assert len(result.feature_contributions) == 3
        assert result.model_version == "v1.2.0"
        assert result.processing_time_ms == 45.2
        assert isinstance(result.timestamp, datetime)
        assert "unusual transaction pattern" in result.explanation
    
    def test_prediction_result_validation(self):
        """Test PredictionResult validation."""
        # Test fraud probability out of range
        with pytest.raises(ValidationError, match="Fraud probability must be between 0 and 1"):
            PredictionResult(
                transaction_id="txn_123",
                fraud_probability=1.5,
                risk_level=RiskLevel.HIGH,
                decision="DECLINE"
            )
        
        # Test negative processing time
        with pytest.raises(ValidationError, match="Processing time must be non-negative"):
            PredictionResult(
                transaction_id="txn_123",
                fraud_probability=0.5,
                risk_level=RiskLevel.MEDIUM,
                decision="REVIEW",
                processing_time_ms=-10.0
            )
    
    def test_prediction_result_defaults(self):
        """Test PredictionResult with default values."""
        result = PredictionResult(
            transaction_id="txn_123",
            fraud_probability=0.6,
            risk_level=RiskLevel.MEDIUM,
            decision="REVIEW"
        )
        
        assert result.confidence_score == 0.0
        assert result.feature_contributions == {}
        assert result.model_version == "unknown"
        assert result.processing_time_ms == 0.0
        assert isinstance(result.timestamp, datetime)
        assert result.explanation == ""


class TestBatchPredictionResult:
    """Test cases for BatchPredictionResult model."""
    
    def test_batch_prediction_result_creation(self):
        """Test BatchPredictionResult creation."""
        predictions = [
            PredictionResult(
                transaction_id="txn_1",
                fraud_probability=0.3,
                risk_level=RiskLevel.LOW,
                decision="APPROVE"
            ),
            PredictionResult(
                transaction_id="txn_2",
                fraud_probability=0.8,
                risk_level=RiskLevel.HIGH,
                decision="DECLINE"
            )
        ]
        
        batch_result = BatchPredictionResult(
            batch_id="batch_123",
            predictions=predictions,
            total_count=2,
            success_count=2,
            error_count=0,
            processing_time_ms=125.5,
            timestamp=datetime.utcnow(),
            model_version="v1.0.0"
        )
        
        assert batch_result.batch_id == "batch_123"
        assert len(batch_result.predictions) == 2
        assert batch_result.total_count == 2
        assert batch_result.success_count == 2
        assert batch_result.error_count == 0
        assert batch_result.processing_time_ms == 125.5
        assert isinstance(batch_result.timestamp, datetime)
        assert batch_result.model_version == "v1.0.0"
    
    def test_batch_prediction_result_validation(self):
        """Test BatchPredictionResult validation."""
        predictions = [
            PredictionResult(
                transaction_id="txn_1",
                fraud_probability=0.3,
                risk_level=RiskLevel.LOW,
                decision="APPROVE"
            )
        ]
        
        # Test count mismatch
        with pytest.raises(ValidationError, match="Total count must equal success count plus error count"):
            BatchPredictionResult(
                batch_id="batch_123",
                predictions=predictions,
                total_count=5,
                success_count=2,
                error_count=2  # 2 + 2 != 5
            )
    
    def test_batch_prediction_result_defaults(self):
        """Test BatchPredictionResult with default values."""
        batch_result = BatchPredictionResult(
            batch_id="batch_123",
            predictions=[],
            total_count=0,
            success_count=0,
            error_count=0
        )
        
        assert batch_result.processing_time_ms == 0.0
        assert isinstance(batch_result.timestamp, datetime)
        assert batch_result.model_version == "unknown"


class TestModelTrainingRequest:
    """Test cases for ModelTrainingRequest model."""
    
    def test_model_training_request_creation(self):
        """Test ModelTrainingRequest creation."""
        request = ModelTrainingRequest(
            model_name="fraud_detector_v2",
            model_type="xgboost",
            training_data_path="/data/training_set.csv",
            validation_data_path="/data/validation_set.csv",
            test_data_path="/data/test_set.csv",
            feature_columns=["amount", "merchant_risk", "velocity"],
            target_column="is_fraud",
            hyperparameters={
                "max_depth": 6,
                "n_estimators": 100,
                "learning_rate": 0.1
            },
            cross_validation_folds=5,
            random_seed=42,
            save_model_path="/models/fraud_detector_v2.pkl"
        )
        
        assert request.model_name == "fraud_detector_v2"
        assert request.model_type == "xgboost"
        assert request.training_data_path == "/data/training_set.csv"
        assert len(request.feature_columns) == 3
        assert request.target_column == "is_fraud"
        assert request.hyperparameters["max_depth"] == 6
        assert request.cross_validation_folds == 5
        assert request.random_seed == 42
    
    def test_model_training_request_validation(self):
        """Test ModelTrainingRequest validation."""
        # Test empty feature columns
        with pytest.raises(ValidationError, match="Feature columns cannot be empty"):
            ModelTrainingRequest(
                model_name="test_model",
                model_type="sklearn",
                training_data_path="/data/train.csv",
                feature_columns=[],
                target_column="target"
            )
        
        # Test invalid cross-validation folds
        with pytest.raises(ValidationError, match="Cross validation folds must be at least 2"):
            ModelTrainingRequest(
                model_name="test_model",
                model_type="sklearn",
                training_data_path="/data/train.csv",
                feature_columns=["feature1"],
                target_column="target",
                cross_validation_folds=1
            )
    
    def test_model_training_request_defaults(self):
        """Test ModelTrainingRequest with default values."""
        request = ModelTrainingRequest(
            model_name="test_model",
            model_type="sklearn",
            training_data_path="/data/train.csv",
            feature_columns=["feature1"],
            target_column="target"
        )
        
        assert request.validation_data_path is None
        assert request.test_data_path is None
        assert request.hyperparameters == {}
        assert request.cross_validation_folds == 5
        assert request.random_seed == 42
        assert request.save_model_path is None


class TestHealthCheckResponse:
    """Test cases for HealthCheckResponse model."""
    
    def test_health_check_response_creation(self):
        """Test HealthCheckResponse creation."""
        response = HealthCheckResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime_seconds=3600,
            dependencies={
                "database": "connected",
                "model_loader": "ready",
                "kafka": "connected"
            },
            metrics={
                "requests_per_minute": 150,
                "average_response_time_ms": 25.5,
                "error_rate": 0.02
            }
        )
        
        assert response.status == "healthy"
        assert isinstance(response.timestamp, datetime)
        assert response.version == "1.0.0"
        assert response.uptime_seconds == 3600
        assert len(response.dependencies) == 3
        assert "requests_per_minute" in response.metrics
    
    def test_health_check_response_defaults(self):
        """Test HealthCheckResponse with default values."""
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=1800
        )
        
        assert isinstance(response.timestamp, datetime)
        assert response.dependencies == {}
        assert response.metrics == {}


class TestServiceMetrics:
    """Test cases for ServiceMetrics model."""
    
    def test_service_metrics_creation(self):
        """Test ServiceMetrics creation."""
        metrics = ServiceMetrics(
            total_requests=10000,
            successful_requests=9850,
            failed_requests=150,
            average_response_time_ms=32.5,
            requests_per_minute=125,
            error_rate=0.015,
            cache_hit_rate=0.85,
            active_models=3,
            memory_usage_mb=512.0,
            cpu_usage_percent=25.5,
            timestamp=datetime.utcnow()
        )
        
        assert metrics.total_requests == 10000
        assert metrics.successful_requests == 9850
        assert metrics.failed_requests == 150
        assert metrics.average_response_time_ms == 32.5
        assert metrics.requests_per_minute == 125
        assert metrics.error_rate == 0.015
        assert metrics.cache_hit_rate == 0.85
        assert metrics.active_models == 3
        assert metrics.memory_usage_mb == 512.0
        assert metrics.cpu_usage_percent == 25.5
        assert isinstance(metrics.timestamp, datetime)
    
    def test_service_metrics_validation(self):
        """Test ServiceMetrics validation."""
        # Test negative total requests
        with pytest.raises(ValidationError, match="Total requests must be non-negative"):
            ServiceMetrics(
                total_requests=-1,
                successful_requests=100,
                failed_requests=10
            )
        
        # Test error rate out of range
        with pytest.raises(ValidationError, match="Error rate must be between 0 and 1"):
            ServiceMetrics(
                total_requests=1000,
                successful_requests=900,
                failed_requests=100,
                error_rate=1.5
            )
    
    def test_service_metrics_defaults(self):
        """Test ServiceMetrics with default values."""
        metrics = ServiceMetrics(
            total_requests=1000,
            successful_requests=950,
            failed_requests=50
        )
        
        assert metrics.average_response_time_ms == 0.0
        assert metrics.requests_per_minute == 0
        assert metrics.error_rate == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.active_models == 0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert isinstance(metrics.timestamp, datetime)