"""Configuration package for the fraud detection system."""

from .base import BaseConfig, Environment, LogLevel
from .development import DevelopmentConfig
from .staging import StagingConfig
from .production import ProductionConfig
from .testing import TestingConfig
from .factory import (
    ConfigFactory,
    ConfigurationError,
    get_settings,
    get_settings_for_environment,
    reload_settings,
    validate_environment_file,
    get_config_summary
)

# Create global settings instance using factory (lazy initialization)
_settings = None

def get_global_settings():
    """Get global settings instance with lazy initialization."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings

# For backward compatibility
settings = get_global_settings()

__all__ = [
    # Base classes
    "BaseConfig",
    "Environment", 
    "LogLevel",
    
    # Environment-specific configs
    "DevelopmentConfig",
    "StagingConfig", 
    "ProductionConfig",
    "TestingConfig",
    
    # Factory and utilities
    "ConfigFactory",
    "ConfigurationError",
    "get_settings",
    "get_settings_for_environment", 
    "reload_settings",
    "validate_environment_file",
    "get_config_summary",
    
    # Global settings instance
    "settings"
]
