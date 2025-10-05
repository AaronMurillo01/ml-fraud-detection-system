"""Production environment configuration."""

from typing import List
from pydantic import Field

from .base import BaseConfig, Environment, LogLevel


class ProductionConfig(BaseConfig):
    """Production environment configuration."""
    
    # Environment
    environment: Environment = Environment.PRODUCTION
    
    # Application
    debug: bool = False
    log_level: LogLevel = LogLevel.WARNING
    enable_docs: bool = False  # Disable docs in production for security
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = Field(default=4, env="API_WORKERS")  # Scale based on CPU cores
    enable_cors: bool = True
    cors_origins: List[str] = Field(env="CORS_ORIGINS")  # Must be explicitly set
    trusted_hosts: List[str] = Field(env="TRUSTED_HOSTS")  # Must be explicitly set
    
    # Database Configuration (Production Database with required secrets)
    database_host: str = Field(env="DATABASE_HOST")
    database_port: int = Field(default=5432, env="DATABASE_PORT")
    database_name: str = Field(env="DATABASE_NAME")
    database_user: str = Field(env="DATABASE_USER")
    database_password: str = Field(env="DATABASE_PASSWORD")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    database_echo: bool = False  # Never log SQL in production
    
    # Redis Configuration (Production Redis with required secrets)
    redis_url: str = Field(env="REDIS_URL")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    redis_health_check_interval: int = 30
    redis_retry_on_timeout: bool = True
    redis_max_retries: int = 3
    
    # Kafka Configuration (Production Kafka)
    kafka_bootstrap_servers: List[str] = Field(env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_consumer_group: str = "fraud-detection-prod"
    kafka_auto_offset_reset: str = "earliest"  # More reliable for production
    kafka_enable_auto_commit: bool = False  # Manual commit for reliability
    kafka_session_timeout_ms: int = 30000
    
    # ML Models Configuration
    model_path: str = Field(default="models/production/", env="MODEL_PATH")
    model_cache_ttl: int = 3600  # 1 hour
    model_cache_size: int = 100  # Larger cache for production
    model_max_workers: int = Field(default=8, env="MODEL_MAX_WORKERS")
    enable_model_preloading: bool = True
    preload_models: List[str] = Field(
        default=["fraud_detector_v1", "fraud_detector_v2", "fraud_detector_v3"],
        env="PRELOAD_MODELS"
    )
    
    # Security Configuration (Strict production security)
    secret_key: str = Field(env="SECRET_KEY")  # Must be set from environment
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15  # Shorter expiry for security
    refresh_token_expire_days: int = 1  # Shorter refresh token expiry
    require_authentication: bool = True
    enable_rate_limiting: bool = True
    
    # Rate Limiting Configuration (Strict limits)
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=150, env="RATE_LIMIT_BURST")
    api_key_rate_limit: int = Field(default=1000, env="API_KEY_RATE_LIMIT")
    ip_rate_limit: int = Field(default=60, env="IP_RATE_LIMIT")
    
    # Monitoring Configuration (Full monitoring in production)
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    enable_metrics: bool = True
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    jaeger_endpoint: str = Field(env="JAEGER_ENDPOINT")
    health_check_interval: int = 30
    
    # Fraud Detection Configuration
    default_fraud_threshold: float = Field(default=0.8, env="DEFAULT_FRAUD_THRESHOLD")
    high_risk_threshold: float = Field(default=0.9, env="HIGH_RISK_THRESHOLD")
    low_risk_threshold: float = Field(default=0.3, env="LOW_RISK_THRESHOLD")
    enable_model_explanation: bool = Field(default=True, env="ENABLE_MODEL_EXPLANATION")
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    
    # Performance Configuration (Optimized for production)
    max_request_size: int = Field(default=10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    enable_compression: bool = True
    
    # Caching Configuration (Optimized TTLs for production)
    prediction_cache_ttl: int = Field(default=1800, env="PREDICTION_CACHE_TTL")  # 30 minutes
    user_profile_cache_ttl: int = Field(default=3600, env="USER_PROFILE_CACHE_TTL")  # 1 hour
    model_metadata_cache_ttl: int = Field(default=86400, env="MODEL_METADATA_CACHE_TTL")  # 24 hours
    feature_cache_ttl: int = Field(default=7200, env="FEATURE_CACHE_TTL")  # 2 hours
    
    # Feature Store Configuration
    feature_store_url: str = Field(env="FEATURE_STORE_URL")
    enable_feature_caching: bool = True
    
    # Additional Production Settings
    enable_ssl: bool = Field(default=True, env="ENABLE_SSL")
    ssl_cert_path: str = Field(default=None, env="SSL_CERT_PATH")
    ssl_key_path: str = Field(default=None, env="SSL_KEY_PATH")
    
    # Backup and Recovery
    enable_database_backup: bool = Field(default=True, env="ENABLE_DATABASE_BACKUP")
    backup_schedule: str = Field(default="0 2 * * *", env="BACKUP_SCHEDULE")  # Daily at 2 AM
    backup_retention_days: int = Field(default=30, env="BACKUP_RETENTION_DAYS")
    
    # Alerting Configuration
    alert_webhook_url: str = Field(default=None, env="ALERT_WEBHOOK_URL")
    alert_email_recipients: List[str] = Field(default=[], env="ALERT_EMAIL_RECIPIENTS")
    enable_error_alerts: bool = Field(default=True, env="ENABLE_ERROR_ALERTS")
    enable_performance_alerts: bool = Field(default=True, env="ENABLE_PERFORMANCE_ALERTS")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env.production"
        env_file_encoding = "utf-8"
