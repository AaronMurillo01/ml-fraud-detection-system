"""Configuration validation utilities."""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path

from .base import BaseConfig, Environment

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigValidator:
    """Validator for configuration settings."""
    
    @staticmethod
    def validate_database_url(url: str) -> Tuple[bool, Optional[str]]:
        """Validate database URL format.
        
        Args:
            url: Database URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                return False, "Database URL must include scheme (postgresql, mysql, etc.)"
            
            if not parsed.hostname:
                return False, "Database URL must include hostname"
            
            if not parsed.username:
                return False, "Database URL must include username"
            
            if not parsed.password:
                return False, "Database URL must include password"
            
            if not parsed.path or parsed.path == '/':
                return False, "Database URL must include database name"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid database URL format: {e}"
    
    @staticmethod
    def validate_redis_url(url: str) -> Tuple[bool, Optional[str]]:
        """Validate Redis URL format.
        
        Args:
            url: Redis URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = urlparse(url)
            
            if parsed.scheme not in ['redis', 'rediss']:
                return False, "Redis URL must use 'redis://' or 'rediss://' scheme"
            
            if not parsed.hostname:
                return False, "Redis URL must include hostname"
            
            # Port is optional (defaults to 6379)
            if parsed.port and (parsed.port < 1 or parsed.port > 65535):
                return False, "Redis port must be between 1 and 65535"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid Redis URL format: {e}"
    
    @staticmethod
    def validate_secret_key(key: str, min_length: int = 32) -> Tuple[bool, Optional[str]]:
        """Validate secret key strength.
        
        Args:
            key: Secret key to validate
            min_length: Minimum required length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not key:
            return False, "Secret key cannot be empty"
        
        if len(key) < min_length:
            return False, f"Secret key must be at least {min_length} characters long"
        
        # Check for common weak keys
        weak_keys = [
            "secret",
            "password",
            "key",
            "test",
            "dev",
            "development",
            "production",
            "admin",
            "default"
        ]
        
        if key.lower() in weak_keys:
            return False, "Secret key is too weak (common word)"
        
        # Check for sufficient entropy (basic check)
        unique_chars = len(set(key))
        if unique_chars < min_length // 4:
            return False, "Secret key has insufficient entropy"
        
        return True, None
    
    @staticmethod
    def validate_cors_origins(origins: List[str], environment: Environment) -> Tuple[bool, Optional[str]]:
        """Validate CORS origins configuration.
        
        Args:
            origins: List of CORS origins
            environment: Current environment
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not origins:
            return False, "CORS origins cannot be empty"
        
        # In production, wildcards should not be allowed
        if environment == Environment.PRODUCTION:
            if "*" in origins:
                return False, "Wildcard CORS origins not allowed in production"
            
            # Validate each origin is a proper URL
            for origin in origins:
                if not origin.startswith(('http://', 'https://')):
                    return False, f"Invalid CORS origin format: {origin}"
                
                # Production should use HTTPS
                if origin.startswith('http://') and not origin.startswith('http://localhost'):
                    return False, f"Production CORS origins should use HTTPS: {origin}"
        
        return True, None
    
    @staticmethod
    def validate_file_path(path: str, must_exist: bool = False) -> Tuple[bool, Optional[str]]:
        """Validate file path.
        
        Args:
            path: File path to validate
            must_exist: Whether the file must exist
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not path:
            return False, "File path cannot be empty"
        
        try:
            path_obj = Path(path)
            
            if must_exist and not path_obj.exists():
                return False, f"File does not exist: {path}"
            
            # Check if parent directory is writable (if file doesn't exist)
            if not path_obj.exists():
                parent = path_obj.parent
                if not parent.exists():
                    return False, f"Parent directory does not exist: {parent}"
                
                if not os.access(parent, os.W_OK):
                    return False, f"Parent directory is not writable: {parent}"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid file path: {e}"
    
    @staticmethod
    def validate_port(port: int) -> Tuple[bool, Optional[str]]:
        """Validate port number.
        
        Args:
            port: Port number to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if port < 1 or port > 65535:
            return False, f"Port must be between 1 and 65535, got {port}"
        
        # Check for privileged ports in production
        if port < 1024:
            return True, f"Warning: Using privileged port {port} (requires root privileges)"
        
        return True, None
    
    @classmethod
    def validate_config(cls, config: BaseConfig) -> List[str]:
        """Validate entire configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate database URL
        if config.database_url:
            is_valid, error = cls.validate_database_url(config.database_url)
            if not is_valid:
                errors.append(f"Database URL: {error}")
        
        # Validate Redis URL
        is_valid, error = cls.validate_redis_url(config.redis_url)
        if not is_valid:
            errors.append(f"Redis URL: {error}")
        
        # Validate secret key
        is_valid, error = cls.validate_secret_key(config.secret_key)
        if not is_valid:
            errors.append(f"Secret key: {error}")
        
        # Validate CORS origins
        is_valid, error = cls.validate_cors_origins(config.cors_origins, config.environment)
        if not is_valid:
            errors.append(f"CORS origins: {error}")
        
        # Validate ports
        is_valid, error = cls.validate_port(config.api_port)
        if not is_valid:
            errors.append(f"API port: {error}")
        
        # Validate model path
        is_valid, error = cls.validate_file_path(config.model_path, must_exist=False)
        if not is_valid:
            errors.append(f"Model path: {error}")
        
        # Environment-specific validations
        if config.environment == Environment.PRODUCTION:
            errors.extend(cls._validate_production_config(config))
        
        return errors
    
    @classmethod
    def _validate_production_config(cls, config: BaseConfig) -> List[str]:
        """Validate production-specific configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Security checks
        if config.debug:
            errors.append("Debug mode must be disabled in production")
        
        if config.enable_docs:
            errors.append("API documentation should be disabled in production")
        
        # Required fields for production
        required_fields = [
            ("database_host", "Database host"),
            ("database_user", "Database user"),
            ("database_password", "Database password"),
            ("redis_url", "Redis URL"),
        ]
        
        for field, description in required_fields:
            value = getattr(config, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                errors.append(f"{description} is required in production")
        
        # SSL/TLS checks
        if hasattr(config, 'enable_ssl') and not config.enable_ssl:
            errors.append("SSL should be enabled in production")
        
        # Rate limiting should be enabled
        if not config.enable_rate_limiting:
            errors.append("Rate limiting should be enabled in production")
        
        # Authentication should be required
        if not config.require_authentication:
            errors.append("Authentication should be required in production")
        
        return errors


def validate_environment_variables() -> Dict[str, Any]:
    """Validate required environment variables.
    
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "missing_optional": []
    }
    
    # Get current environment
    env_str = os.getenv("ENVIRONMENT", "development")
    
    try:
        environment = Environment(env_str.lower())
    except ValueError:
        result["errors"].append(f"Invalid ENVIRONMENT value: {env_str}")
        result["valid"] = False
        return result
    
    # Environment-specific required variables
    required_vars = {
        Environment.PRODUCTION: [
            "SECRET_KEY",
            "DATABASE_HOST",
            "DATABASE_USER", 
            "DATABASE_PASSWORD",
            "REDIS_URL"
        ],
        Environment.STAGING: [
            "SECRET_KEY",
            "DATABASE_HOST",
            "DATABASE_USER",
            "DATABASE_PASSWORD",
            "REDIS_URL"
        ]
    }
    
    # Check required variables for current environment
    if environment in required_vars:
        for var in required_vars[environment]:
            if not os.getenv(var):
                result["errors"].append(f"Required environment variable missing: {var}")
                result["valid"] = False
    
    # Optional but recommended variables
    optional_vars = [
        "CORS_ORIGINS",
        "TRUSTED_HOSTS",
        "JAEGER_ENDPOINT",
        "FEATURE_STORE_URL"
    ]
    
    for var in optional_vars:
        if not os.getenv(var):
            result["missing_optional"].append(var)
    
    return result


def generate_config_report(config: BaseConfig) -> Dict[str, Any]:
    """Generate comprehensive configuration report.
    
    Args:
        config: Configuration to analyze
        
    Returns:
        Dictionary with configuration report
    """
    validator = ConfigValidator()
    validation_errors = validator.validate_config(config)
    env_validation = validate_environment_variables()
    
    report = {
        "environment": config.environment.value,
        "validation": {
            "config_valid": len(validation_errors) == 0,
            "config_errors": validation_errors,
            "env_valid": env_validation["valid"],
            "env_errors": env_validation["errors"],
            "env_warnings": env_validation["warnings"]
        },
        "security": {
            "debug_enabled": config.debug,
            "docs_enabled": config.enable_docs,
            "authentication_required": config.require_authentication,
            "rate_limiting_enabled": config.enable_rate_limiting,
            "ssl_enabled": getattr(config, 'enable_ssl', False)
        },
        "performance": {
            "api_workers": config.api_workers,
            "database_pool_size": config.database_pool_size,
            "redis_max_connections": config.redis_max_connections,
            "model_cache_size": config.model_cache_size,
            "enable_compression": config.enable_compression
        },
        "monitoring": {
            "metrics_enabled": config.enable_metrics,
            "tracing_enabled": config.enable_tracing,
            "health_checks_enabled": True
        }
    }
    
    return report
