"""Pytest configuration and shared fixtures for fraud detection tests."""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import pandas as pd
from pathlib import Path

# Test environment setup
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/fraud_detection_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9092"

# Import after environment setup
from service.models import (
    Transaction,
    EnrichedTransaction,
    ModelScore,
    FeatureImportance,
    ScoringResponse,
    RiskLevel,
)
# Import ML components from available modules
try:
    from service.ml_inference import MLInferenceService
    from service.model_loader import ModelLoader
except ImportError:
    # Fallback for testing without full ML stack
    MLInferenceService = None
    ModelLoader = None


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (deselect with '-m \"not unit\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "external: marks tests that require external services"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_model_dir():
    """Temporary directory for model files during testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Transaction fixtures
@pytest.fixture
def sample_transaction():
    """Sample legitimate transaction."""
    return Transaction(
        transaction_id="txn_sample_001",
        user_id="user_sample_001",
        merchant_id="merchant_sample_001",
        amount=Decimal("125.50"),
        currency="USD",
        transaction_time=datetime.now(timezone.utc),
        transaction_type="purchase",
        payment_method="credit_card"
    )


@pytest.fixture
def high_risk_transaction():
    """Sample high-risk transaction."""
    return Transaction(
        transaction_id="txn_high_risk_001",
        user_id="user_high_risk_001",
        merchant_id="merchant_high_risk_001",
        amount=Decimal("2500.00"),  # High amount
        currency="USD",
        transaction_time=datetime.now(timezone.utc),
        transaction_type="purchase",
        payment_method="credit_card"
    )


@pytest.fixture
def international_transaction():
    """Sample international transaction."""
    return Transaction(
        transaction_id="txn_intl_001",
        user_id="user_intl_001",
        merchant_id="merchant_intl_001",
        amount=Decimal("350.75"),
        currency="EUR",
        transaction_time=datetime.now(timezone.utc),
        transaction_type="purchase",
        payment_method="credit_card"
    )


@pytest.fixture
def batch_transactions():
    """Batch of sample transactions for testing."""
    transactions = []
    
    for i in range(10):
        transaction = Transaction(
            transaction_id=f"txn_batch_{i:03d}",
            user_id=f"user_batch_{i % 3}",  # 3 different users
            merchant_id=f"merchant_batch_{i % 5}",  # 5 different merchants
            amount=Decimal(str(50.0 + i * 25.5)),
            currency="USD",
            transaction_time=datetime.now(timezone.utc),
            transaction_type="purchase",
            payment_method="credit_card"
        )
        transactions.append(transaction)
    
    return transactions


# Profile fixtures (simplified for testing)
@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "user_id": "user_sample_001",
        "account_age_days": 365,
        "avg_transaction_amount": 125.00,
        "transaction_count_30d": 45,
        "unique_merchants_30d": 12,
        "risk_score": 0.15
    }


@pytest.fixture
def sample_merchant_data():
    """Sample merchant data for testing."""
    return {
        "merchant_id": "merchant_sample_001",
        "merchant_name": "Sample Grocery Store",
        "category": "grocery",
        "avg_transaction_amount": 85.50,
        "transaction_count_30d": 1250,
        "fraud_rate_30d": 0.008,
        "risk_score": 0.12
    }


# ML fixtures
@pytest.fixture
def sample_features():
    """Sample feature matrix for ML testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.rand(100, 10)


@pytest.fixture
def sample_labels():
    """Sample labels for ML testing."""
    np.random.seed(42)  # For reproducible tests
    # 10% fraud rate
    return np.random.choice([0, 1], size=100, p=[0.9, 0.1])


@pytest.fixture
def trained_xgboost_model(sample_features, sample_labels):
    """Pre-trained XGBoost model for testing."""
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(random_state=42)
        model.fit(sample_features, sample_labels)
        return model
    except ImportError:
        # Mock model for testing without XGBoost
        class MockModel:
            def predict_proba(self, X):
                return np.random.rand(len(X), 2)
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
        return MockModel()


@pytest.fixture
def mock_ensemble():
    """Mock ensemble model for testing."""
    from unittest.mock import Mock
    
    # Create a simple mock ensemble
    ensemble = Mock()
    ensemble.predict.return_value = np.array([0])
    ensemble.predict_proba.return_value = np.array([[0.8, 0.2]])
    ensemble.models = []
    
    for i in range(3):
        model = Mock()
        model.model_name = f"test_model_{i}"
        model.model_version = "1.0.0"
        model.predict.return_value = np.array([0])
        model.predict_proba.return_value = np.array([[0.9 - i * 0.1, 0.1 + i * 0.1]])
        ensemble.models.append(model)
    
    return ensemble


@pytest.fixture
def sample_transaction_features():
    """Sample transaction features as dictionary."""
    return {
        "transaction_id": "txn_features_001",
        "amount_zscore": 1.5,
        "frequency_1h": 2,
        "frequency_24h": 8,
        "avg_amount_ratio": 1.2,
        "merchant_risk_score": 0.15,
        "user_risk_score": 0.12,
        "location_risk_score": 0.08,
        "time_since_last_transaction": 3600,
        "is_weekend": False,
        "is_night_time": False,
        "velocity_score": 0.25,
        "category_frequency": 0.3,
        "payment_method_risk": 0.1,
        "cross_border": False,
        "high_amount_flag": False
    }


# Prediction fixtures
@pytest.fixture
def sample_fraud_prediction():
    """Sample fraud prediction result."""
    return {
        "transaction_id": "txn_pred_001",
        "is_fraud": False,
        "fraud_probability": 0.15,
        "risk_score": 0.2,
        "confidence": 0.85,
        "model_version": "ensemble_v1.0.0",
        "prediction_timestamp": datetime.now(timezone.utc),
        "features_used": ["amount_zscore", "frequency_1h", "merchant_risk_score"],
        "model_scores": [
            {
                "model_name": "xgboost_v1",
                "model_version": "1.0.0",
                "score": 0.12,
                "confidence": 0.88,
                "prediction_time_ms": 15.5
            },
            {
                "model_name": "random_forest_v1",
                "model_version": "1.0.0",
                "score": 0.18,
                "confidence": 0.82,
                "prediction_time_ms": 12.3
            }
        ],
        "processing_time_ms": 28.7
    }


@pytest.fixture
def high_risk_fraud_prediction():
    """Sample high-risk fraud prediction."""
    return {
        "transaction_id": "txn_fraud_001",
        "is_fraud": True,
        "fraud_probability": 0.92,
        "risk_score": 0.95,
        "confidence": 0.88,
        "model_version": "ensemble_v1.0.0",
        "prediction_timestamp": datetime.now(timezone.utc),
        "features_used": ["amount_zscore", "velocity_score", "location_risk_score"],
        "model_scores": [
            {
                "model_name": "xgboost_v1",
                "model_version": "1.0.0",
                "score": 0.94,
                "confidence": 0.90,
                "prediction_time_ms": 16.2
            },
            {
                "model_name": "random_forest_v1",
                "model_version": "1.0.0",
                "score": 0.89,
                "confidence": 0.85,
                "prediction_time_ms": 13.1
            }
        ],
        "processing_time_ms": 31.4
    }


# Mock fixtures
@pytest.fixture
def mock_database():
    """Mock database connection."""
    with patch('service.database.get_connection') as mock_conn:
        mock_conn.return_value.__enter__.return_value = Mock()
        yield mock_conn


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    with patch('service.cache.redis_client') as mock_redis:
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.exists.return_value = False
        yield mock_redis


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer."""
    with patch('streaming.kafka_producer.KafkaProducer') as mock_producer:
        mock_instance = Mock()
        mock_producer.return_value = mock_instance
        mock_instance.send.return_value = Mock()
        yield mock_instance


@pytest.fixture
def mock_feature_store():
    """Mock feature store."""
    with patch('service.feature_store.FeatureStore') as mock_store:
        mock_instance = Mock()
        mock_store.return_value = mock_instance
        
        # Mock feature retrieval
        mock_instance.get_user_features.return_value = {
            'avg_transaction_amount': Decimal('100.00'),
            'transaction_count_30d': 50,
            'risk_score': 0.2
        }
        
        mock_instance.get_merchant_features.return_value = {
            'avg_transaction_amount': Decimal('85.00'),
            'fraud_rate_30d': 0.01,
            'risk_score': 0.15
        }
        
        yield mock_instance


# Test data generators
@pytest.fixture
def transaction_generator():
    """Generator for creating test transactions."""
    def _generate_transaction(
        transaction_id=None,
        user_id=None,
        merchant_id=None,
        amount=None,
        is_fraud=False,
        **kwargs
    ):
        """Generate a transaction with specified parameters."""
        defaults = {
            "transaction_id": transaction_id or f"txn_gen_{np.random.randint(1000, 9999)}",
            "user_id": user_id or f"user_gen_{np.random.randint(100, 999)}",
            "merchant_id": merchant_id or f"merchant_gen_{np.random.randint(100, 999)}",
            "amount": amount or Decimal(str(np.random.uniform(10, 1000))),
            "currency": "USD",
            "timestamp": datetime.now(timezone.utc),
            "merchant_category": np.random.choice(["grocery", "gas", "restaurant", "retail"]),
            "payment_method": "credit_card",
            "card_type": np.random.choice(["visa", "mastercard", "amex"]),
            "location": {
                "latitude": 40.7128 + np.random.uniform(-1, 1),
                "longitude": -74.0060 + np.random.uniform(-1, 1),
                "city": "New York",
                "country": "US"
            }
        }
        
        # Override with any provided kwargs
        defaults.update(kwargs)
        
        # Adjust for fraud characteristics
        if is_fraud:
            defaults["amount"] = Decimal(str(np.random.uniform(1000, 5000)))  # Higher amounts
            defaults["location"]["country"] = np.random.choice(["US", "RU", "CN", "NG"])  # Risky countries
        
        return Transaction(**defaults)
    
    return _generate_transaction


# Performance test helpers
@pytest.fixture
def performance_timer():
    """Timer utility for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        def elapsed_ms(self):
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started/stopped")
            return (self.end_time - self.start_time) * 1000
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, *args):
            self.stop()
    
    return Timer


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    # Cleanup logic here if needed
    pass


# Skip markers for external dependencies
def pytest_runtest_setup(item):
    """Skip tests based on markers and environment."""
    if "external" in item.keywords:
        # Skip external tests if not in CI or if external services not available
        if not os.getenv("CI") and not os.getenv("RUN_EXTERNAL_TESTS"):
            pytest.skip("External service tests skipped (set RUN_EXTERNAL_TESTS=1 to run)")