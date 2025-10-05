#!/usr/bin/env python3
"""Test core functionality without configuration dependencies."""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_core_models():
    """Test core model functionality."""
    print("Testing core models...")
    
    try:
        # Test basic model imports
        from shared.models import Transaction, PredictionResult, RiskLevel
        print("   [OK] Core models imported")
        
        # Test creating a transaction
        from decimal import Decimal
        from datetime import datetime, timezone
        
        transaction = Transaction(
            transaction_id="test_123",
            user_id="user_456", 
            merchant_id="merchant_789",
            amount=Decimal("100.00"),
            currency="USD",
            transaction_type="purchase",
            payment_method="credit_card",
            transaction_time=datetime.now(timezone.utc),
            location_lat=40.7128,
            location_lon=-74.0060,
            country_code="US",
            ip_address="192.168.1.100",
            device_id="device_test"
        )
        print(f"   [OK] Transaction created: {transaction.transaction_id}")
        
        # Test creating a prediction result
        prediction = PredictionResult(
            transaction_id="test_123",
            fraud_probability=0.25,
            risk_level=RiskLevel.LOW,
            decision="APPROVE",
            confidence_score=0.85,
            feature_contributions={
                "amount": 0.15,
                "merchant_risk": 0.30
            },
            model_version="v1.0.0-test",
            processing_time_ms=25.0,
            explanation="Test prediction"
        )
        print(f"   [OK] Prediction created: {prediction.risk_level}")
        
        # Test model validation
        assert transaction.amount > 0
        assert prediction.fraud_probability >= 0 and prediction.fraud_probability <= 1
        assert prediction.confidence_score >= 0 and prediction.confidence_score <= 1
        print("   [OK] Model validation passed")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Core models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unit_tests_directly():
    """Test running unit tests directly with pytest."""
    print("\nTesting unit tests directly...")
    
    try:
        # Try to run a simple unit test
        import subprocess
        import tempfile
        
        # Create a simple test file
        test_content = '''
import pytest
from decimal import Decimal
from datetime import datetime, timezone
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_transaction_creation():
    """Test transaction model creation."""
    from shared.models import Transaction
    
    transaction = Transaction(
        transaction_id="test_123",
        user_id="user_456",
        merchant_id="merchant_789", 
        amount=Decimal("100.00"),
        currency="USD",
        transaction_type="purchase",
        payment_method="credit_card",
        transaction_time=datetime.now(timezone.utc),
        location_lat=40.7128,
        location_lon=-74.0060,
        country_code="US",
        ip_address="192.168.1.100",
        device_id="device_test"
    )
    
    assert transaction.transaction_id == "test_123"
    assert transaction.amount == Decimal("100.00")
    assert transaction.currency == "USD"

def test_risk_level_enum():
    """Test RiskLevel enum."""
    from shared.models import RiskLevel
    
    assert RiskLevel.LOW == "low"
    assert RiskLevel.MEDIUM == "medium"
    assert RiskLevel.HIGH == "high"
    assert RiskLevel.CRITICAL == "critical"

def test_prediction_result():
    """Test prediction result model."""
    from shared.models import PredictionResult, RiskLevel
    
    prediction = PredictionResult(
        transaction_id="test_123",
        fraud_probability=0.25,
        risk_level=RiskLevel.LOW,
        decision="APPROVE",
        confidence_score=0.85,
        feature_contributions={"amount": 0.15},
        model_version="v1.0.0",
        processing_time_ms=25.0,
        explanation="Test"
    )
    
    assert prediction.fraud_probability == 0.25
    assert prediction.risk_level == RiskLevel.LOW
    assert prediction.decision == "APPROVE"
'''
        
        # Write test to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as f:
            f.write(test_content)
            temp_test_file = f.name
        
        try:
            # Run pytest on the temporary test file
            result = subprocess.run([
                sys.executable, '-m', 'pytest', temp_test_file, '-v'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("   [OK] Unit tests passed")
                print(f"   Output: {result.stdout.split('=')[-1].strip()}")
                return True
            else:
                print(f"   [ERROR] Unit tests failed: {result.stderr}")
                return False
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_test_file)
            except:
                pass
        
    except Exception as e:
        print(f"   [ERROR] Unit test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_existing_unit_tests():
    """Test existing unit test files."""
    print("\nTesting existing unit test files...")
    
    try:
        # Check if we can import test modules
        test_files = [
            "tests.unit.test_models",
        ]
        
        for test_module in test_files:
            try:
                __import__(test_module)
                print(f"   [OK] {test_module} imported successfully")
            except Exception as e:
                print(f"   [ERROR] {test_module} import failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Existing unit tests check failed: {e}")
        return False

if __name__ == "__main__":
    print("Core Functionality Test")
    print("=" * 40)
    
    results = []
    
    # Test core models
    results.append(test_core_models())
    
    # Test unit tests directly
    results.append(test_unit_tests_directly())
    
    # Test existing unit tests
    results.append(test_existing_unit_tests())
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 All core functionality tests passed!")
        print("The basic models and testing infrastructure are working.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed, but core functionality may still work.")
        sys.exit(1)
