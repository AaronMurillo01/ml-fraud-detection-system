"""Base configuration settings for the fraud detection system."""

import os
import secrets
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BaseConfig(BaseSettings):
    """Base configuration settings shared across all environments."""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    
    # Application
    app_name: str = "Fraud Detection System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    cors_origins: List[str] = Field(default=["*"])
    trusted_hosts: Optional[List[str]] = Field(default=None)
    
    # Database Configuration
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_host: str = Field(default="localhost", env="DATABASE_HOST")
    database_port: int = Field(default=5432, env="DATABASE_PORT")
    database_name: str = Field(default="fraud_db", env="DATABASE_NAME")
    database_user: str = Field(default="fraud_user", env="DATABASE_USER")
    database_password: str = Field(default="fraud_pass", env="DATABASE_PASSWORD")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    redis_socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    redis_health_check_interval: int = Field(default=30, env="REDIS_HEALTH_CHECK_INTERVAL")
    redis_retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    redis_max_retries: int = Field(default=3, env="REDIS_MAX_RETRIES")
    
    # Kafka Configuration
    kafka_servers: Optional[str] = Field(
        default="localhost:9092"
    )
    kafka_consumer_group: str = Field(default="fraud-detection", env="KAFKA_CONSUMER_GROUP")
    kafka_auto_offset_reset: str = Field(default="latest", env="KAFKA_AUTO_OFFSET_RESET")
    kafka_enable_auto_commit: bool = Field(default=True, env="KAFKA_ENABLE_AUTO_COMMIT")
    kafka_session_timeout_ms: int = Field(default=30000, env="KAFKA_SESSION_TIMEOUT_MS")
    
    # ML Models Configuration
    model_path: str = Field(default="models/", env="MODEL_PATH")
    model_cache_ttl: int = Field(default=3600, env="MODEL_CACHE_TTL")  # 1 hour
    model_cache_size: int = Field(default=50, env="MODEL_CACHE_SIZE")
    model_max_workers: int = Field(default=4, env="MODEL_MAX_WORKERS")
    enable_model_preloading: bool = Field(default=True, env="ENABLE_MODEL_PRELOADING")
    preload_models: List[str] = Field(default=["fraud_detector_v1"], env="PRELOAD_MODELS_LIST")
    
    # Feature Store Configuration
    feature_store_url: Optional[str] = Field(default=None, env="FEATURE_STORE_URL")
    feature_cache_ttl: int = Field(default=7200, env="FEATURE_CACHE_TTL")  # 2 hours
    enable_feature_caching: bool = Field(default=True, env="ENABLE_FEATURE_CACHING")
    
    # Security Configuration
    # IMPORTANT: In production, SECRET_KEY MUST be set via environment variable
    # For development, a random key is generated. For production, use a persistent secure key.
    # Generate one using: python -c "import secrets; print(secrets.token_urlsafe(32))"
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        env="SECRET_KEY",
        description="Secret key for cryptographic operations - set via SECRET_KEY environment variable"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    require_authentication: bool = Field(default=True, env="REQUIRE_AUTHENTICATION")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    
    # Rate Limiting Configuration
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=200, env="RATE_LIMIT_BURST")
    api_key_rate_limit: int = Field(default=1000, env="API_KEY_RATE_LIMIT")
    ip_rate_limit: int = Field(default=60, env="IP_RATE_LIMIT")
    
    # Monitoring Configuration
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Fraud Detection Configuration
    default_fraud_threshold: float = Field(default=0.8, env="DEFAULT_FRAUD_THRESHOLD")
    high_risk_threshold: float = Field(default=0.9, env="HIGH_RISK_THRESHOLD")
    low_risk_threshold: float = Field(default=0.3, env="LOW_RISK_THRESHOLD")
    enable_model_explanation: bool = Field(default=True, env="ENABLE_MODEL_EXPLANATION")
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    
    # Performance Configuration
    max_request_size: int = Field(default=10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")  # 30 seconds
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    
    # Caching Configuration
    prediction_cache_ttl: int = Field(default=1800, env="PREDICTION_CACHE_TTL")  # 30 minutes
    user_profile_cache_ttl: int = Field(default=3600, env="USER_PROFILE_CACHE_TTL")  # 1 hour
    model_metadata_cache_ttl: int = Field(default=86400, env="MODEL_METADATA_CACHE_TTL")  # 24 hours
    
    @validator("database_url", pre=True)
    def build_database_url(cls, v, values):
        """Build database URL from components if not provided."""
        if v:
            return v
        
        user = values.get("database_user", "fraud_user")
        password = values.get("database_password", "fraud_pass")
        host = values.get("database_host", "localhost")
        port = values.get("database_port", 5432)
        name = values.get("database_name", "fraud_db")
        
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
    
    def __init__(self, **kwargs):
        # Handle CORS origins from environment variable
        cors_origins_env = os.getenv("CORS_ORIGINS")
        if cors_origins_env:
            kwargs["cors_origins"] = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

        # Handle Kafka bootstrap servers from environment variable
        kafka_servers_env = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if kafka_servers_env is not None and kafka_servers_env.strip():
            kwargs["kafka_servers"] = kafka_servers_env.strip()

        # Handle preload models from environment variable
        preload_models_env = os.getenv("PRELOAD_MODELS")
        if preload_models_env is not None:
            if preload_models_env.strip() in ["", "[]"]:
                kwargs["preload_models"] = []
            else:
                # Try to parse as JSON first, then fall back to comma-separated
                try:
                    import json
                    kwargs["preload_models"] = json.loads(preload_models_env)
                except (json.JSONDecodeError, ValueError):
                    kwargs["preload_models"] = [model.strip() for model in preload_models_env.split(",") if model.strip()]

        # Handle trusted hosts from environment variable
        trusted_hosts_env = os.getenv("TRUSTED_HOSTS")
        if trusted_hosts_env is not None:
            if trusted_hosts_env.strip() in ["", "[]"]:
                kwargs["trusted_hosts"] = None
            else:
                try:
                    import json
                    kwargs["trusted_hosts"] = json.loads(trusted_hosts_env)
                except (json.JSONDecodeError, ValueError):
                    kwargs["trusted_hosts"] = [host.strip() for host in trusted_hosts_env.split(",") if host.strip()]

        super().__init__(**kwargs)
    
    @validator("kafka_servers", pre=True)
    def parse_kafka_servers(cls, v):
        """Parse Kafka bootstrap servers from string or list."""
        if isinstance(v, str):
            # Return as string, will be split when used
            return v
        return v
    
    @validator("preload_models", pre=True)
    def parse_preload_models(cls, v):
        """Parse preload models from string or list."""
        if isinstance(v, str):
            return [model.strip() for model in v.split(",")]
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True
    
    def get_database_url(self) -> str:
        """Get the complete database URL."""
        return self.database_url or self.build_database_url(None, self.__dict__)
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration based on environment."""
        base_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": self.log_level.value,
                "handlers": ["default"],
            },
        }
        
        if self.is_production():
            # Production logging with structured format
            base_config["formatters"]["json"] = {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            }
            base_config["handlers"]["default"]["formatter"] = "json"
        
        return base_config
