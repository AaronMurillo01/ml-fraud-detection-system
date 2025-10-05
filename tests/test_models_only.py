#!/usr/bin/env python3
"""Test only the core models without any configuration dependencies."""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_transaction_model():
    """Test Transaction model."""
    print("Testing Transaction model...")
    
    try:
        from shared.models import Transaction, PaymentMethod, TransactionStatus
        from decimal import Decimal
        from datetime import datetime, timezone
        
        transaction = Transaction(
            transaction_id="test_123",
            user_id="user_456", 
            merchant_id="merchant_789",
            amount=Decimal("100.00"),
            currency="USD",
            transaction_type="purchase",
            payment_method=PaymentMethod.CREDIT_CARD,
            timestamp=datetime.now(timezone.utc),
            location="New York, NY",
            device_id="device_test",
            ip_address="192.168.1.100",
            merchant_category="retail",
            status=TransactionStatus.PENDING
        )
        
        print(f"   [OK] Transaction created: {transaction.transaction_id}")
        print(f"   [OK] Amount: {transaction.amount}")
        print(f"   [OK] Payment method: {transaction.payment_method}")
        print(f"   [OK] Status: {transaction.status}")
        
        # Test validation
        assert transaction.amount > 0
        assert transaction.transaction_id == "test_123"
        assert transaction.payment_method == PaymentMethod.CREDIT_CARD
        
        print("   [OK] Transaction validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Transaction model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fraud_prediction_model():
    """Test FraudPrediction model."""
    print("\nTesting FraudPrediction model...")
    
    try:
        from shared.models import FraudPrediction
        from datetime import datetime, timezone
        
        prediction = FraudPrediction(
            prediction_id="pred_123",
            transaction_id="test_123",
            user_id="user_456",
            model_version="1.0.0",
            fraud_probability=0.25,
            risk_score=0.25,
            prediction_timestamp=datetime.now(timezone.utc),
            decision="APPROVE",
            confidence_score=0.85,
            explanation={"amount": 0.15, "merchant_risk": 0.30}
        )
        
        print(f"   [OK] Fraud prediction created: {prediction.prediction_id}")
        print(f"   [OK] Decision: {prediction.decision}")
        print(f"   [OK] Fraud probability: {prediction.fraud_probability}")
        print(f"   [OK] Risk score: {prediction.risk_score}")
        
        # Test validation
        assert 0 <= prediction.fraud_probability <= 1
        assert 0 <= prediction.risk_score <= 1
        assert prediction.decision == "APPROVE"
        
        print("   [OK] FraudPrediction validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] FraudPrediction model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_user_model():
    """Test User model."""
    print("\nTesting User model...")
    
    try:
        from shared.models import User
        from datetime import datetime, timezone
        
        user = User(
            user_id="user_456",
            email="test@example.com",
            phone_number="+1234567890",
            first_name="John",
            last_name="Doe",
            date_of_birth=datetime(1990, 1, 1, tzinfo=timezone.utc),
            kyc_status="verified"
        )
        
        print(f"   [OK] User created: {user.user_id}")
        print(f"   [OK] Email: {user.email}")
        print(f"   [OK] KYC status: {user.kyc_status}")
        
        # Test validation
        assert user.user_id == "user_456"
        assert "@" in user.email
        
        print("   [OK] User validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] User model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_merchant_model():
    """Test Merchant model."""
    print("\nTesting Merchant model...")
    
    try:
        from shared.models import Merchant
        
        merchant = Merchant(
            merchant_id="merchant_789",
            merchant_name="Test Store",
            category="retail",
            country="US",
            risk_score=0.1
        )
        
        print(f"   [OK] Merchant created: {merchant.merchant_id}")
        print(f"   [OK] Name: {merchant.merchant_name}")
        print(f"   [OK] Category: {merchant.category}")
        print(f"   [OK] Risk score: {merchant.risk_score}")
        
        # Test validation
        assert merchant.merchant_id == "merchant_789"
        assert merchant.category == "retail"
        
        print("   [OK] Merchant validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Merchant model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_metadata():
    """Test ModelMetadata model."""
    print("\nTesting ModelMetadata model...")
    
    try:
        from shared.models import ModelMetadata
        from datetime import datetime, timezone
        
        metadata = ModelMetadata(
            model_id="fraud_model_v1",
            model_name="Fraud Detection Model",
            model_version="1.0.0",
            model_type="xgboost",
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            is_active=True,
            deployment_date=datetime.now(timezone.utc)
        )
        
        print(f"   [OK] ModelMetadata created: {metadata.model_id}")
        print(f"   [OK] Name: {metadata.model_name}")
        print(f"   [OK] Version: {metadata.model_version}")
        print(f"   [OK] Accuracy: {metadata.accuracy}")
        
        # Test validation
        assert metadata.model_id == "fraud_model_v1"
        assert 0 <= metadata.accuracy <= 1
        
        print("   [OK] ModelMetadata validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] ModelMetadata model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enums():
    """Test enum classes."""
    print("\nTesting enum classes...")
    
    try:
        from shared.models import RiskLevel, PaymentMethod, TransactionStatus
        
        # Test RiskLevel
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"
        print("   [OK] RiskLevel enum")
        
        # Test PaymentMethod
        assert PaymentMethod.CREDIT_CARD == "credit_card"
        assert PaymentMethod.DEBIT_CARD == "debit_card"
        assert PaymentMethod.BANK_TRANSFER == "bank_transfer"
        print("   [OK] PaymentMethod enum")
        
        # Test TransactionStatus
        assert TransactionStatus.PENDING == "pending"
        assert TransactionStatus.APPROVED == "approved"
        assert TransactionStatus.DECLINED == "declined"
        print("   [OK] TransactionStatus enum")
        
        print("   [OK] All enums validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Enums test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Core Models Only")
    print("=" * 40)
    
    results = []
    
    # Test all models
    results.append(test_transaction_model())
    results.append(test_fraud_prediction_model())
    results.append(test_user_model())
    results.append(test_merchant_model())
    results.append(test_model_metadata())
    results.append(test_enums())
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✅ All core model tests passed!")
        print("The shared models are working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(results) - sum(results)} model tests failed.")
        print("Some models may need fixes.")
        sys.exit(1)
