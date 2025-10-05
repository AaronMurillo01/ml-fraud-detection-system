"""Development environment configuration."""

from typing import List
from pydantic import Field

from .base import BaseConfig, Environment, LogLevel


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    
    # Application
    debug: bool = True
    log_level: LogLevel = LogLevel.DEBUG
    enable_docs: bool = True
    
    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_workers: int = 1
    enable_cors: bool = True
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]
    
    # Database Configuration (Local PostgreSQL)
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "fraud_detection_dev"
    database_user: str = "fraud_dev_user"
    database_password: str = "fraud_dev_pass"
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_echo: bool = True  # Enable SQL logging in development
    
    # Redis Configuration (Local Redis)
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 10
    
    # Kafka Configuration (Local Kafka)
    kafka_servers: str = "localhost:9092"
    kafka_consumer_group: str = "fraud-detection-dev"
    
    # ML Models Configuration
    model_path: str = "models/dev/"
    model_cache_ttl: int = 300  # 5 minutes for faster development
    model_cache_size: int = 10  # Smaller cache for development
    model_max_workers: int = 2
    enable_model_preloading: bool = False  # Disable for faster startup
    preload_models: List[str] = []
    
    # Security Configuration (Relaxed for development)
    require_authentication: bool = False  # Disable auth for easier development
    enable_rate_limiting: bool = False  # Disable rate limiting for development
    access_token_expire_minutes: int = 60  # Longer token expiry
    
    # Rate Limiting Configuration (Relaxed)
    rate_limit_per_minute: int = 1000
    rate_limit_burst: int = 2000
    api_key_rate_limit: int = 10000
    ip_rate_limit: int = 600
    
    # Monitoring Configuration
    enable_metrics: bool = False  # Disable metrics collection in development
    enable_tracing: bool = False  # Disable tracing in development
    
    # Fraud Detection Configuration
    default_fraud_threshold: float = 0.5  # Lower threshold for testing
    enable_model_explanation: bool = True
    max_batch_size: int = 100  # Smaller batches for development
    
    # Performance Configuration
    request_timeout: int = 60  # Longer timeout for debugging
    enable_compression: bool = False  # Disable compression for easier debugging
    
    # Caching Configuration (Shorter TTLs for development)
    prediction_cache_ttl: int = 300  # 5 minutes
    user_profile_cache_ttl: int = 600  # 10 minutes
    model_metadata_cache_ttl: int = 1800  # 30 minutes
    feature_cache_ttl: int = 600  # 10 minutes
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env.development"
        env_file_encoding = "utf-8"
