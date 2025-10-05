"""Staging environment configuration."""

from typing import List
from pydantic import Field

from .base import BaseConfig, Environment, LogLevel


class StagingConfig(BaseConfig):
    """Staging environment configuration."""
    
    # Environment
    environment: Environment = Environment.STAGING
    
    # Application
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    enable_docs: bool = True  # Keep docs enabled for testing
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
    enable_cors: bool = True
    cors_origins: List[str] = [
        "https://staging-frontend.fraud-detection.com",
        "https://staging-admin.fraud-detection.com"
    ]
    trusted_hosts: List[str] = [
        "staging-api.fraud-detection.com",
        "staging.fraud-detection.com"
    ]
    
    # Database Configuration (Staging Database)
    database_host: str = Field(env="DATABASE_HOST")
    database_port: int = 5432
    database_name: str = "fraud_detection_staging"
    database_user: str = Field(env="DATABASE_USER")
    database_password: str = Field(env="DATABASE_PASSWORD")
    database_pool_size: int = 15
    database_max_overflow: int = 25
    database_echo: bool = False
    
    # Redis Configuration (Staging Redis)
    redis_url: str = Field(env="REDIS_URL")
    redis_max_connections: int = 15
    redis_socket_timeout: int = 10
    redis_socket_connect_timeout: int = 10
    
    # Kafka Configuration (Staging Kafka)
    kafka_bootstrap_servers: List[str] = Field(env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_consumer_group: str = "fraud-detection-staging"
    
    # ML Models Configuration
    model_path: str = "models/staging/"
    model_cache_ttl: int = 1800  # 30 minutes
    model_cache_size: int = 25
    model_max_workers: int = 3
    enable_model_preloading: bool = True
    preload_models: List[str] = ["fraud_detector_v1", "fraud_detector_v2"]
    
    # Security Configuration
    secret_key: str = Field(env="SECRET_KEY")
    require_authentication: bool = True
    enable_rate_limiting: bool = True
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Rate Limiting Configuration
    rate_limit_per_minute: int = 200
    rate_limit_burst: int = 400
    api_key_rate_limit: int = 2000
    ip_rate_limit: int = 120
    
    # Monitoring Configuration
    prometheus_port: int = 9090
    enable_metrics: bool = True
    enable_tracing: bool = True
    jaeger_endpoint: str = Field(default=None, env="JAEGER_ENDPOINT")
    health_check_interval: int = 30
    
    # Fraud Detection Configuration
    default_fraud_threshold: float = 0.75
    high_risk_threshold: float = 0.85
    low_risk_threshold: float = 0.35
    enable_model_explanation: bool = True
    max_batch_size: int = 500
    
    # Performance Configuration
    max_request_size: int = 5 * 1024 * 1024  # 5MB
    request_timeout: int = 30
    enable_compression: bool = True
    
    # Caching Configuration
    prediction_cache_ttl: int = 1200  # 20 minutes
    user_profile_cache_ttl: int = 2400  # 40 minutes
    model_metadata_cache_ttl: int = 43200  # 12 hours
    feature_cache_ttl: int = 3600  # 1 hour
    
    # Feature Store Configuration
    feature_store_url: str = Field(default=None, env="FEATURE_STORE_URL")
    enable_feature_caching: bool = True
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env.staging"
        env_file_encoding = "utf-8"
