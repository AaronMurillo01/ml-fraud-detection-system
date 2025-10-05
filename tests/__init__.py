"""Test package for Credit Card Fraud Detection System.

This package contains comprehensive tests for all components of the fraud detection system:
- Unit tests for individual components
- Integration tests for service interactions
- Performance tests for latency and throughput requirements
- End-to-end tests for complete workflows
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", 
    "postgresql://test_user:test_pass@localhost:5432/test_fraud_detection"
)
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
TEST_KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "TEST_KAFKA_BOOTSTRAP_SERVERS", 
    "localhost:9092"
)

# Test data paths
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_MODELS_DIR = project_root / "tests" / "models"
TEST_FIXTURES_DIR = project_root / "tests" / "fixtures"

# Create test directories if they don't exist
for directory in [TEST_DATA_DIR, TEST_MODELS_DIR, TEST_FIXTURES_DIR]:
    directory.mkdir(exist_ok=True)

__all__ = [
    "TEST_DATABASE_URL",
    "TEST_REDIS_URL", 
    "TEST_KAFKA_BOOTSTRAP_SERVERS",
    "TEST_DATA_DIR",
    "TEST_MODELS_DIR",
    "TEST_FIXTURES_DIR",
]