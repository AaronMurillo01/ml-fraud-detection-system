"""Configuration settings for the fraud detection system."""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "Fraud Detection System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    trusted_hosts: Optional[List[str]] = Field(default=None, env="TRUSTED_HOSTS")
    
    # Database
    database_url: str = Field(
        default="postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db",
        env="DATABASE_URL"
    )
    database_host: str = Field(default="localhost", env="DATABASE_HOST")
    database_port: int = Field(default=5432, env="DATABASE_PORT")
    database_name: str = Field(default="fraud_db", env="DATABASE_NAME")
    database_user: str = Field(default="fraud_user", env="DATABASE_USER")
    database_password: str = Field(default="fraud_pass", env="DATABASE_PASSWORD")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    redis_socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    redis_health_check_interval: int = Field(default=30, env="REDIS_HEALTH_CHECK_INTERVAL")
    
    # Kafka
    kafka_bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], env="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_consumer_group: str = Field(default="fraud-detection", env="KAFKA_CONSUMER_GROUP")
    kafka_auto_offset_reset: str = Field(default="latest", env="KAFKA_AUTO_OFFSET_RESET")
    
    # ML Models
    model_path: str = Field(default="models/", env="MODEL_PATH")
    model_cache_ttl: int = Field(default=3600, env="MODEL_CACHE_TTL")  # 1 hour
    
    # Feature Store
    feature_store_url: Optional[str] = Field(default=None, env="FEATURE_STORE_URL")
    
    # Monitoring
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    
    # Security
    # IMPORTANT: SECRET_KEY must be set via environment variable in production
    # Generate a secure key using: python -c "import secrets; print(secrets.token_urlsafe(32))"
    secret_key: str = Field(
        default="",
        env="SECRET_KEY",
        description="Secret key for JWT signing - MUST be set via environment variable in production"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    require_authentication: bool = Field(default=False, env="REQUIRE_AUTHENTICATION")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    
    # Fraud Detection
    fraud_threshold: float = Field(default=0.5, env="FRAUD_THRESHOLD")
    high_risk_threshold: float = Field(default=0.8, env="HIGH_RISK_THRESHOLD")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
# NOTE: This is deprecated. Use `from config import get_settings` instead.
# settings = Settings()


def get_settings():
    """Get application settings. This is deprecated, use config.get_settings() instead."""
    from config import get_settings as get_main_settings
    return get_main_settings()