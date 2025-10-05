"""Testing environment configuration."""

from typing import List
from pydantic import Field

from .base import BaseConfig, Environment, LogLevel


class TestingConfig(BaseConfig):
    """Testing environment configuration."""
    
    # Environment
    environment: Environment = Environment.TESTING
    
    # Application
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    enable_docs: bool = False  # Disable docs during testing
    
    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8001  # Different port to avoid conflicts
    api_workers: int = 1
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]  # Allow all origins for testing
    
    # Database Configuration (In-memory or test database)
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "fraud_detection_test"
    database_user: str = "test_user"
    database_password: str = "test_pass"
    database_pool_size: int = 1  # Minimal pool for testing
    database_max_overflow: int = 2
    database_echo: bool = False  # Disable SQL logging during tests
    
    # Redis Configuration (Test Redis or mock)
    redis_url: str = "redis://localhost:6379/15"  # Use different DB for testing
    redis_max_connections: int = 5
    
    # Kafka Configuration (Mock or test Kafka)
    kafka_bootstrap_servers: List[str] = ["localhost:9092"]
    kafka_consumer_group: str = "fraud-detection-test"
    
    # ML Models Configuration (Test models)
    model_path: str = "tests/fixtures/models/"
    model_cache_ttl: int = 60  # Short TTL for testing
    model_cache_size: int = 5
    model_max_workers: int = 1
    enable_model_preloading: bool = False  # Disable for faster test startup
    preload_models: List[str] = []
    
    # Security Configuration (Relaxed for testing)
    secret_key: str = "test-secret-key-not-for-production"
    require_authentication: bool = False  # Disable auth for easier testing
    enable_rate_limiting: bool = False  # Disable rate limiting for testing
    access_token_expire_minutes: int = 5  # Short expiry for testing
    
    # Rate Limiting Configuration (Disabled)
    rate_limit_per_minute: int = 10000  # Very high limits
    rate_limit_burst: int = 20000
    api_key_rate_limit: int = 100000
    ip_rate_limit: int = 10000
    
    # Monitoring Configuration (Disabled for testing)
    enable_metrics: bool = False
    enable_tracing: bool = False
    health_check_interval: int = 5  # Faster health checks
    
    # Fraud Detection Configuration (Test-friendly values)
    default_fraud_threshold: float = 0.5
    high_risk_threshold: float = 0.7
    low_risk_threshold: float = 0.3
    enable_model_explanation: bool = True
    max_batch_size: int = 10  # Small batches for testing
    
    # Performance Configuration
    max_request_size: int = 1024 * 1024  # 1MB for testing
    request_timeout: int = 10  # Shorter timeout for faster tests
    enable_compression: bool = False
    
    # Caching Configuration (Very short TTLs for testing)
    prediction_cache_ttl: int = 30  # 30 seconds
    user_profile_cache_ttl: int = 60  # 1 minute
    model_metadata_cache_ttl: int = 120  # 2 minutes
    feature_cache_ttl: int = 30  # 30 seconds
    
    # Feature Store Configuration (Disabled for testing)
    feature_store_url: str = None
    enable_feature_caching: bool = False  # Disable for predictable tests
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env.testing"
        env_file_encoding = "utf-8"
