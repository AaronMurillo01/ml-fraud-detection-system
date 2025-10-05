"""Configuration factory for environment-specific settings."""

import os
import logging
from typing import Type, Dict, Any
from functools import lru_cache

from .base import BaseConfig, Environment, LogLevel
from .development import DevelopmentConfig
from .staging import StagingConfig
from .production import ProductionConfig
from .testing import TestingConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass


class ConfigFactory:
    """Factory for creating environment-specific configurations."""
    
    _config_map: Dict[Environment, Type[BaseConfig]] = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.STAGING: StagingConfig,
        Environment.PRODUCTION: ProductionConfig,
        Environment.TESTING: TestingConfig,
    }
    
    @classmethod
    def get_config_class(cls, environment: Environment) -> Type[BaseConfig]:
        """Get configuration class for the specified environment.
        
        Args:
            environment: Target environment
            
        Returns:
            Configuration class for the environment
            
        Raises:
            ConfigurationError: If environment is not supported
        """
        if environment not in cls._config_map:
            raise ConfigurationError(f"Unsupported environment: {environment}")
        
        return cls._config_map[environment]
    
    @classmethod
    def create_config(cls, environment: Environment = None) -> BaseConfig:
        """Create configuration instance for the specified environment.

        Args:
            environment: Target environment. If None, will be determined from
                        ENVIRONMENT environment variable or default to development.

        Returns:
            Configuration instance

        Raises:
            ConfigurationError: If configuration creation fails
        """
        # Check for test mode bypass
        if os.getenv("TEST_MODE", "").lower() in ("true", "1", "yes"):
            return cls.create_test_config()

        if environment is None:
            env_str = os.getenv("ENVIRONMENT", "development").lower()
            try:
                environment = Environment(env_str)
            except ValueError:
                logger.warning(f"Invalid environment '{env_str}', defaulting to development")
                environment = Environment.DEVELOPMENT
        
        try:
            config_class = cls.get_config_class(environment)
            config = config_class()
            
            logger.info(f"Loaded {environment.value} configuration")
            
            # Validate critical production settings
            if environment == Environment.PRODUCTION:
                cls._validate_production_config(config)
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create {environment.value} configuration: {e}")

    @classmethod
    def create_test_config(cls):
        """Create a simple test configuration that bypasses Pydantic Settings.

        Returns:
            Simple test configuration object
        """
        class SimpleTestConfig:
            """Simple test configuration that doesn't use Pydantic Settings."""

            def __init__(self):
                # Environment
                self.environment = Environment.TESTING
                self.debug = True
                self.log_level = LogLevel.DEBUG

                # API Configuration
                self.api_host = "127.0.0.1"
                self.api_port = 8000
                self.api_workers = 1
                self.enable_cors = True
                self.cors_origins = ["*"]
                self.trusted_hosts = None

                # Database Configuration (In-memory SQLite for testing)
                self.database_url = "sqlite:///:memory:"
                self.database_host = "localhost"
                self.database_port = 5432
                self.database_name = "test_db"
                self.database_user = "test_user"
                self.database_password = "test_pass"
                self.database_pool_size = 1
                self.database_max_overflow = 0
                self.database_echo = False

                # Redis Configuration (Mock Redis for testing)
                self.redis_url = "redis://localhost:6379/15"  # Use DB 15 for testing
                self.redis_max_connections = 1

                # Kafka Configuration (Mock Kafka for testing)
                self.kafka_bootstrap_servers = "localhost:9092"
                self.kafka_group_id = "fraud-detection-test"

                # ML Models Configuration
                self.model_path = "tests/models/"
                self.model_cache_ttl = 60
                self.model_cache_size = 1
                self.model_max_workers = 1
                self.enable_model_preloading = False
                self.preload_models = []

                # Security Configuration (Minimal for testing)
                self.secret_key = "test-secret-key-not-for-production"
                self.require_authentication = False
                self.enable_rate_limiting = False
                self.access_token_expire_minutes = 30
                self.algorithm = "HS256"

                # Rate Limiting Configuration (Disabled for testing)
                self.rate_limit_per_minute = 10000
                self.rate_limit_burst = 20000
                self.api_key_rate_limit = 100000
                self.ip_rate_limit = 60000

                # Monitoring Configuration (Disabled for testing)
                self.enable_metrics = False
                self.enable_tracing = False
                self.jaeger_endpoint = None

                # Fraud Detection Configuration
                self.default_fraud_threshold = 0.5
                self.high_risk_threshold = 0.7
                self.low_risk_threshold = 0.3
                self.enable_model_explanation = True
                self.max_batch_size = 10

                # Performance Configuration
                self.request_timeout = 30
                self.enable_compression = False

                # Caching Configuration (Short TTLs for testing)
                self.prediction_cache_ttl = 60
                self.user_profile_cache_ttl = 60
                self.model_metadata_cache_ttl = 60
                self.feature_cache_ttl = 60

                # Application Configuration
                self.app_name = "Fraud Detection API"
                self.app_version = "1.0.0-test"
                self.enable_docs = True

                # Encryption Configuration
                self.encryption_key = None  # Will be generated if needed

                # Additional test-specific settings
                self.test_mode = True
                self.mock_external_services = True
                self.disable_background_tasks = True

            def is_production(self) -> bool:
                """Check if running in production environment."""
                return self.environment == Environment.PRODUCTION

            def is_development(self) -> bool:
                """Check if running in development environment."""
                return self.environment == Environment.DEVELOPMENT

            def is_testing(self) -> bool:
                """Check if running in testing environment."""
                return self.environment == Environment.TESTING

        return SimpleTestConfig()

    @classmethod
    def _validate_production_config(cls, config: BaseConfig) -> None:
        """Validate production configuration for security and completeness.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        required_fields = [
            "secret_key",
            "database_host",
            "database_user",
            "database_password",
            "redis_url",
        ]
        
        missing_fields = []
        for field in required_fields:
            value = getattr(config, field, None)
            if not value or (isinstance(value, str) and value.strip() == ""):
                missing_fields.append(field)
        
        if missing_fields:
            raise ConfigurationError(
                f"Missing required production configuration fields: {', '.join(missing_fields)}"
            )
        
        # Validate security settings
        if config.secret_key == "test-secret-key-not-for-production":
            raise ConfigurationError("Production secret key must be changed from default")
        
        if len(config.secret_key) < 32:
            raise ConfigurationError("Production secret key must be at least 32 characters")
        
        if config.debug:
            raise ConfigurationError("Debug mode must be disabled in production")
        
        if config.enable_docs:
            logger.warning("API documentation is enabled in production - consider disabling for security")
        
        if "*" in config.cors_origins:
            raise ConfigurationError("CORS origins must be explicitly set in production (no wildcards)")
        
        logger.info("Production configuration validation passed")


@lru_cache(maxsize=1)
def get_settings() -> BaseConfig:
    """Get cached application settings.
    
    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    
    Returns:
        Application configuration instance
    """
    return ConfigFactory.create_config()


def get_settings_for_environment(environment: Environment) -> BaseConfig:
    """Get settings for a specific environment (not cached).
    
    Args:
        environment: Target environment
        
    Returns:
        Configuration instance for the specified environment
    """
    return ConfigFactory.create_config(environment)


def reload_settings() -> BaseConfig:
    """Reload settings by clearing cache and creating new instance.
    
    Returns:
        New configuration instance
    """
    get_settings.cache_clear()
    return get_settings()


def validate_environment_file(environment: Environment) -> Dict[str, Any]:
    """Validate that required environment file exists and is readable.
    
    Args:
        environment: Environment to validate
        
    Returns:
        Dictionary with validation results
    """
    config_class = ConfigFactory.get_config_class(environment)
    env_file = getattr(config_class.Config, 'env_file', None)
    
    result = {
        "environment": environment.value,
        "env_file": env_file,
        "exists": False,
        "readable": False,
        "errors": []
    }
    
    if not env_file:
        result["errors"].append("No environment file specified")
        return result
    
    try:
        if os.path.exists(env_file):
            result["exists"] = True
            
            # Try to read the file
            with open(env_file, 'r', encoding='utf-8') as f:
                f.read()
            result["readable"] = True
            
        else:
            result["errors"].append(f"Environment file {env_file} does not exist")
            
    except PermissionError:
        result["errors"].append(f"Permission denied reading {env_file}")
    except Exception as e:
        result["errors"].append(f"Error reading {env_file}: {e}")
    
    return result


def get_config_summary(config: BaseConfig = None) -> Dict[str, Any]:
    """Get a summary of current configuration (safe for logging).
    
    Args:
        config: Configuration instance. If None, uses current settings.
        
    Returns:
        Dictionary with configuration summary (sensitive data masked)
    """
    if config is None:
        config = get_settings()
    
    # Fields to mask for security
    sensitive_fields = {
        "secret_key", "database_password", "redis_url", "kafka_bootstrap_servers"
    }
    
    summary = {
        "environment": config.environment.value,
        "app_name": config.app_name,
        "app_version": config.app_version,
        "debug": config.debug,
        "log_level": config.log_level.value,
        "api_host": config.api_host,
        "api_port": config.api_port,
        "api_workers": config.api_workers,
        "database_host": config.database_host,
        "database_name": config.database_name,
        "database_pool_size": config.database_pool_size,
        "enable_authentication": config.require_authentication,
        "enable_rate_limiting": config.enable_rate_limiting,
        "enable_metrics": config.enable_metrics,
        "enable_tracing": config.enable_tracing,
        "model_cache_size": config.model_cache_size,
        "max_batch_size": config.max_batch_size,
    }
    
    # Add masked sensitive fields
    for field in sensitive_fields:
        value = getattr(config, field, None)
        if value:
            if isinstance(value, str):
                summary[f"{field}_masked"] = f"{value[:4]}***{value[-4:]}" if len(value) > 8 else "***"
            else:
                summary[f"{field}_masked"] = "***"
    
    return summary
