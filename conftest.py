"""Global pytest configuration and fixtures."""

import pytest
import asyncio
import os
import sys
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test fixtures and utilities
from tests.fixtures.mock_objects import (
    MockMLInferenceService,
    MockModelLoader,
    MockKafkaProducer,
    MockKafkaConsumer,
    MockDatabase,
    MockFeaturePipeline
)
from tests.fixtures.test_data import (
    sample_transactions,
    sample_users,
    sample_merchants,
    sample_predictions,
    sample_model_metadata,
    generate_random_transaction,
    generate_batch_transactions
)
from tests.fixtures.database_fixtures import (
    sync_engine,
    async_engine,
    sync_session,
    async_session,
    populated_sync_session,
    populated_async_session,
    DatabaseTestHelper,
    AsyncDatabaseTestHelper
)


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    
    # Configure asyncio for testing
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def pytest_unconfigure(config):
    """Clean up after pytest run."""
    # Clean up environment variables
    os.environ.pop("TESTING", None)
    os.environ.pop("LOG_LEVEL", None)
    os.environ.pop("DATABASE_URL", None)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add markers based on test name patterns
        if "database" in item.name.lower():
            item.add_marker(pytest.mark.database)
        if "kafka" in item.name.lower():
            item.add_marker(pytest.mark.kafka)
        if "ml" in item.name.lower() or "model" in item.name.lower():
            item.add_marker(pytest.mark.ml)
        if "api" in item.name.lower() or "endpoint" in item.name.lower():
            item.add_marker(pytest.mark.api)
        if "async" in item.name.lower():
            item.add_marker(pytest.mark.asyncio)


# ============================================================================
# Event Loop Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_loop():
    """Provide event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_ml_service():
    """Provide mock ML inference service."""
    return MockMLInferenceService()


@pytest.fixture
def mock_model_loader():
    """Provide mock model loader."""
    return MockModelLoader()


@pytest.fixture
def mock_database():
    """Provide mock database."""
    return MockDatabase()


@pytest.fixture
def mock_feature_pipeline():
    """Provide mock feature pipeline."""
    return MockFeaturePipeline()


@pytest.fixture
def mock_kafka_producer():
    """Provide mock Kafka producer."""
    return MockKafkaProducer()


@pytest.fixture
def mock_kafka_consumer():
    """Provide mock Kafka consumer."""
    return MockKafkaConsumer(["transactions", "fraud_predictions"])


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_transaction():
    """Provide a single sample transaction."""
    return sample_transactions[0]


@pytest.fixture
def sample_user():
    """Provide a single sample user."""
    return sample_users[0]


@pytest.fixture
def sample_merchant():
    """Provide a single sample merchant."""
    return sample_merchants[0]


@pytest.fixture
def sample_prediction():
    """Provide a single sample fraud prediction."""
    return sample_predictions[0]


@pytest.fixture
def sample_model_meta():
    """Provide sample model metadata."""
    return sample_model_metadata[0]


@pytest.fixture
def batch_transactions():
    """Provide batch of sample transactions."""
    return sample_transactions[:10]


@pytest.fixture
def random_transaction():
    """Generate a random transaction for testing."""
    return generate_random_transaction()


@pytest.fixture
def random_batch_transactions():
    """Generate a batch of random transactions."""
    return generate_batch_transactions(20)


# ============================================================================
# Environment and Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "database_url": "sqlite:///:memory:",
        "kafka_bootstrap_servers": "localhost:9092",
        "model_path": "/tmp/test_models",
        "log_level": "DEBUG",
        "api_host": "localhost",
        "api_port": 8000,
        "redis_url": "redis://localhost:6379/0",
        "feature_store_url": "http://localhost:8080",
        "model_registry_url": "http://localhost:8081"
    }


@pytest.fixture
def test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "TESTING": "true",
        "DATABASE_URL": "sqlite:///:memory:",
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        "REDIS_URL": "redis://localhost:6379/0",
        "MODEL_PATH": "/tmp/test_models",
        "LOG_LEVEL": "DEBUG"
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Time and Date Fixtures
# ============================================================================

@pytest.fixture
def fixed_datetime():
    """Provide a fixed datetime for consistent testing."""
    return datetime(2024, 1, 15, 12, 0, 0)


@pytest.fixture
def mock_datetime(fixed_datetime):
    """Mock datetime.utcnow() to return fixed datetime."""
    with patch('datetime.datetime') as mock_dt:
        mock_dt.utcnow.return_value = fixed_datetime
        mock_dt.now.return_value = fixed_datetime
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield mock_dt


@pytest.fixture
def time_range():
    """Provide a time range for testing."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = datetime(2024, 1, 31, 23, 59, 59)
    return start_time, end_time


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def api_client():
    """Provide API test client."""
    # This would typically use FastAPI TestClient or similar
    class MockAPIClient:
        def __init__(self):
            self.base_url = "http://testserver"
            self.headers = {"Content-Type": "application/json"}
        
        def get(self, url, **kwargs):
            return Mock(status_code=200, json=lambda: {"status": "ok"})
        
        def post(self, url, **kwargs):
            return Mock(status_code=201, json=lambda: {"status": "created"})
        
        def put(self, url, **kwargs):
            return Mock(status_code=200, json=lambda: {"status": "updated"})
        
        def delete(self, url, **kwargs):
            return Mock(status_code=204, json=lambda: {})
    
    return MockAPIClient()


@pytest.fixture
def auth_headers():
    """Provide authentication headers for API testing."""
    return {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json"
    }


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def performance_config():
    """Provide performance testing configuration."""
    return {
        "max_latency_ms": 100,
        "min_throughput_rps": 50,
        "max_error_rate_percent": 1.0,
        "load_test_duration_seconds": 30,
        "concurrent_users": 10,
        "ramp_up_time_seconds": 5
    }


@pytest.fixture
def benchmark_data():
    """Provide benchmark data for performance comparisons."""
    return {
        "baseline_latency_ms": 50,
        "baseline_throughput_rps": 100,
        "baseline_error_rate_percent": 0.1,
        "performance_threshold_percent": 10  # Allow 10% degradation
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically clean up temporary files after each test."""
    temp_files = []
    temp_dirs = []
    
    yield
    
    # Clean up temporary files
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {file_path}: {e}")
    
    # Clean up temporary directories
    for dir_path in temp_dirs:
        try:
            if os.path.exists(dir_path):
                import shutil
                shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Warning: Could not remove temp dir {dir_path}: {e}")


@pytest.fixture
def temp_file():
    """Provide a temporary file path that gets cleaned up."""
    import tempfile
    
    fd, path = tempfile.mkstemp()
    os.close(fd)
    
    yield path
    
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Warning: Could not remove temp file {path}: {e}")


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that gets cleaned up."""
    import tempfile
    import shutil
    
    dir_path = tempfile.mkdtemp()
    
    yield dir_path
    
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Warning: Could not remove temp dir {dir_path}: {e}")


# ============================================================================
# Logging Fixtures
# ============================================================================

@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    
    yield log_capture
    
    # Clean up
    root_logger.removeHandler(handler)
    root_logger.setLevel(original_level)


# ============================================================================
# Parametrized Fixtures
# ============================================================================

@pytest.fixture(params=["low", "medium", "high"])
def risk_level(request):
    """Parametrized fixture for different risk levels."""
    return request.param


@pytest.fixture(params=["CREDIT_CARD", "DEBIT_CARD", "BANK_TRANSFER", "DIGITAL_WALLET"])
def payment_method(request):
    """Parametrized fixture for different payment methods."""
    return request.param


@pytest.fixture(params=[10, 50, 100, 500])
def batch_size(request):
    """Parametrized fixture for different batch sizes."""
    return request.param


# ============================================================================
# Skip Conditions
# ============================================================================

@pytest.fixture
def skip_if_no_kafka():
    """Skip test if Kafka is not available."""
    # In a real implementation, this would check if Kafka is running
    kafka_available = os.environ.get("KAFKA_AVAILABLE", "false").lower() == "true"
    
    if not kafka_available:
        pytest.skip("Kafka not available for testing")


@pytest.fixture
def skip_if_no_database():
    """Skip test if database is not available."""
    # In a real implementation, this would check database connectivity
    db_available = os.environ.get("DATABASE_AVAILABLE", "true").lower() == "true"
    
    if not db_available:
        pytest.skip("Database not available for testing")


@pytest.fixture
def skip_slow_tests():
    """Skip slow tests unless explicitly enabled."""
    run_slow = os.environ.get("RUN_SLOW_TESTS", "false").lower() == "true"
    
    if not run_slow:
        pytest.skip("Slow tests disabled (set RUN_SLOW_TESTS=true to enable)")