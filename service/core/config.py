"""Application configuration using Pydantic settings."""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

from pydantic import BaseSettings, Field, validator
from pydantic.networks import AnyHttpUrl, PostgresDsn, RedisDsn


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application Settings
    app_name: str = Field(default="Fraud Detection API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=True, env="RELOAD")
    
    # Security Settings
    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[AnyHttpUrl] = Field(default=[], env="CORS_ORIGINS")
    
    # Database Settings
    postgres_server: str = Field(env="POSTGRES_SERVER")
    postgres_user: str = Field(env="POSTGRES_USER")
    postgres_password: str = Field(env="POSTGRES_PASSWORD")
    postgres_db: str = Field(env="POSTGRES_DB")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    database_url: Optional[PostgresDsn] = None
    
    # Redis Settings
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_url: Optional[RedisDsn] = None
    
    # Kafka Settings
    kafka_bootstrap_servers: List[str] = Field(
        default=["localhost:9092"], 
        env="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_topic_transactions: str = Field(
        default="transactions", 
        env="KAFKA_TOPIC_TRANSACTIONS"
    )
    kafka_topic_scores: str = Field(
        default="fraud_scores", 
        env="KAFKA_TOPIC_SCORES"
    )
    kafka_consumer_group: str = Field(
        default="fraud_detection", 
        env="KAFKA_CONSUMER_GROUP"
    )
    kafka_auto_offset_reset: str = Field(
        default="latest", 
        env="KAFKA_AUTO_OFFSET_RESET"
    )
    
    # ML Model Settings
    model_path: str = Field(
        default="/app/models/fraud_detector.pkl", 
        env="MODEL_PATH"
    )
    model_version: str = Field(default="v2.1", env="MODEL_VERSION")
    model_threshold: float = Field(default=0.5, env="MODEL_THRESHOLD")
    feature_store_enabled: bool = Field(default=True, env="FEATURE_STORE_ENABLED")
    
    # Performance Settings
    max_request_size: int = Field(default=16777216, env="MAX_REQUEST_SIZE")  # 16MB
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    batch_size_limit: int = Field(default=1000, env="BATCH_SIZE_LIMIT")
    inference_timeout_ms: int = Field(default=50, env="INFERENCE_TIMEOUT_MS")
    
    # Monitoring Settings
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    jaeger_enabled: bool = Field(default=False, env="JAEGER_ENABLED")
    jaeger_agent_host: str = Field(default="localhost", env="JAEGER_AGENT_HOST")
    jaeger_agent_port: int = Field(default=6831, env="JAEGER_AGENT_PORT")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # External Services
    external_api_timeout: int = Field(default=10, env="EXTERNAL_API_TIMEOUT")
    external_api_retries: int = Field(default=3, env="EXTERNAL_API_RETRIES")
    
    # Feature Engineering Settings
    feature_cache_ttl: int = Field(default=300, env="FEATURE_CACHE_TTL")  # 5 minutes
    feature_computation_timeout: int = Field(default=10, env="FEATURE_COMPUTATION_TIMEOUT")
    
    # Alert Thresholds
    alert_high_fraud_rate: float = Field(default=0.1, env="ALERT_HIGH_FRAUD_RATE")
    alert_low_model_confidence: float = Field(default=0.7, env="ALERT_LOW_MODEL_CONFIDENCE")
    alert_high_latency_ms: int = Field(default=100, env="ALERT_HIGH_LATENCY_MS")
    
    @validator("database_url", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        """Assemble database URL from components."""
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("postgres_user"),
            password=values.get("postgres_password"),
            host=values.get("postgres_server"),
            port=str(values.get("postgres_port")),
            path=f"/{values.get('postgres_db') or ''}",
        )
    
    @validator("redis_url", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        """Assemble Redis URL from components."""
        if isinstance(v, str):
            return v
        
        password = values.get("redis_password")
        auth_part = f":{password}@" if password else ""
        
        return f"redis://{auth_part}{values.get('redis_host')}:{values.get('redis_port')}/{values.get('redis_db')}"
    
    @validator("kafka_bootstrap_servers", pre=True)
    def parse_kafka_servers(cls, v):
        """Parse Kafka bootstrap servers from string or list."""
        if isinstance(v, str):
            return [server.strip() for server in v.split(",")]
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    @property
    def kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration dictionary."""
        return {
            "bootstrap_servers": self.kafka_bootstrap_servers,
            "auto_offset_reset": self.kafka_auto_offset_reset,
            "group_id": self.kafka_consumer_group,
            "enable_auto_commit": True,
            "auto_commit_interval_ms": 1000,
            "session_timeout_ms": 30000,
            "heartbeat_interval_ms": 10000,
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary."""
        config = {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db,
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True,
        }
        
        if self.redis_password:
            config["password"] = self.redis_password
        
        return config
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": str(self.database_url),
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"
    

class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    workers: int = 4
    

class TestingSettings(Settings):
    """Testing environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    postgres_db: str = "fraud_detection_test"
    redis_db: int = 1


def get_settings_for_environment(env: str) -> Settings:
    """Get settings for specific environment."""
    settings_map = {
        "development": DevelopmentSettings,
        "production": ProductionSettings,
        "testing": TestingSettings,
    }
    
    settings_class = settings_map.get(env.lower(), Settings)
    return settings_class()