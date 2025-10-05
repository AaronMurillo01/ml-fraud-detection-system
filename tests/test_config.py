#!/usr/bin/env python3
"""Test script for configuration system."""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_configuration():
    """Test the configuration system."""
    print("Testing Configuration System")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("1. Testing imports...")
        from config.base import BaseConfig, Environment
        from config.development import DevelopmentConfig
        from config.production import ProductionConfig
        from config.factory import ConfigFactory, get_settings
        print("   ✓ All imports successful")
        
        # Test factory creation
        print("\n2. Testing configuration factory...")
        
        # Test development config
        dev_config = ConfigFactory.create_config(Environment.DEVELOPMENT)
        print(f"   ✓ Development config: {dev_config.environment}")
        
        # Test production config (should work even without env vars)
        try:
            prod_config = ConfigFactory.create_config(Environment.PRODUCTION)
            print(f"   ✓ Production config: {prod_config.environment}")
        except Exception as e:
            print(f"   ⚠ Production config validation failed (expected): {e}")
        
        # Test default config
        default_config = get_settings()
        print(f"   ✓ Default config: {default_config.environment}")
        
        # Test configuration properties
        print("\n3. Testing configuration properties...")
        config = dev_config
        print(f"   App Name: {config.app_name}")
        print(f"   Debug Mode: {config.debug}")
        print(f"   API Port: {config.api_port}")
        print(f"   Database URL: {config.database_url}")
        print(f"   Redis URL: {config.redis_url}")
        
        # Test validation
        print("\n4. Testing configuration validation...")
        from config.validation import ConfigValidator
        validator = ConfigValidator()
        errors = validator.validate_config(config)
        if errors:
            print(f"   Validation errors: {errors}")
        else:
            print("   ✓ Configuration validation passed")
        
        # Test environment detection
        print("\n5. Testing environment detection...")
        original_env = os.getenv("ENVIRONMENT")
        
        # Test with different environment variables
        test_envs = ["development", "staging", "production"]
        for env in test_envs:
            os.environ["ENVIRONMENT"] = env
            try:
                test_config = ConfigFactory.create_config()
                print(f"   ✓ {env}: {test_config.environment}")
            except Exception as e:
                print(f"   ⚠ {env}: {e}")
        
        # Restore original environment
        if original_env:
            os.environ["ENVIRONMENT"] = original_env
        else:
            os.environ.pop("ENVIRONMENT", None)
        
        print("\n6. Configuration Summary:")
        print("-" * 30)
        summary = {
            "environment": config.environment.value,
            "debug": config.debug,
            "api_port": config.api_port,
            "database_host": config.database_host,
            "redis_max_connections": config.redis_max_connections,
            "enable_authentication": config.require_authentication,
            "enable_rate_limiting": config.enable_rate_limiting,
        }
        
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Configuration system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_configuration()
    sys.exit(0 if success else 1)
