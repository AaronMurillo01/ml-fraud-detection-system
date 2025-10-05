#!/usr/bin/env python3
"""Debug configuration loading issues."""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """Test configuration loading step by step."""
    print("Testing configuration loading...")
    
    # Set environment for testing
    os.environ["ENVIRONMENT"] = "development"
    
    try:
        print("1. Testing base config import...")
        from config.base import BaseConfig
        print("   [OK] BaseConfig imported")
        
        print("2. Testing development config import...")
        from config.development import DevelopmentConfig
        print("   [OK] DevelopmentConfig imported")
        
        print("3. Testing factory import...")
        from config.factory import ConfigFactory
        print("   [OK] ConfigFactory imported")
        
        print("4. Testing config creation...")
        config = ConfigFactory.create_config()
        print(f"   [OK] Config created: {type(config).__name__}")
        print(f"   Environment: {config.environment}")
        print(f"   CORS Origins: {config.cors_origins}")
        
        print("5. Testing settings import...")
        from config import settings
        print(f"   [OK] Settings imported: {type(settings).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cors_parsing():
    """Test CORS origins parsing specifically."""
    print("\nTesting CORS origins parsing...")
    
    try:
        from config.base import BaseConfig
        
        # Test different CORS origin formats
        test_cases = [
            "http://localhost:3000,http://localhost:8080",
            ["http://localhost:3000", "http://localhost:8080"],
            "*",
            ["*"]
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"   Test case {i}: {test_case}")
            
            # Create a temporary config with this CORS value
            os.environ["CORS_ORIGINS"] = str(test_case) if isinstance(test_case, str) else ",".join(test_case)
            
            try:
                config = BaseConfig()
                print(f"   [OK] Parsed to: {config.cors_origins}")
            except Exception as e:
                print(f"   [ERROR] Failed to parse: {e}")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] CORS parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Configuration Debug Test")
    print("=" * 40)
    
    success1 = test_config_loading()
    success2 = test_cors_parsing()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("All configuration tests passed!")
        sys.exit(0)
    else:
        print("Some configuration tests failed!")
        sys.exit(1)
