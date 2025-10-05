#!/usr/bin/env python3
"""Test service models directly without configuration dependencies."""

import os
import sys
from pathlib import Path

# Set test mode before importing anything
os.environ["TEST_MODE"] = "true"
os.environ["ENVIRONMENT"] = "testing"

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_service_transaction_model():
    """Test service Transaction model."""
    print("Testing service Transaction model...")
    
    try:
        from service.models import Transaction, PaymentMethod, TransactionType, TransactionStatus
        from decimal import Decimal
        from datetime import datetime, timezone
        
        transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_456",
            merchant_id="merchant_789",
            amount=Decimal("100.50"),
            currency="USD",
            transaction_type=TransactionType.PURCHASE,
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
        assert transaction.transaction_id == "txn_123"
        
        print("   [OK] Service Transaction validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Service Transaction model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_scoring_models():
    """Test service scoring models."""
    print("\nTesting service scoring models...")
    
    try:
        from service.models import ModelScore, ScoringResponse, RiskLevel, FeatureImportance
        from datetime import datetime, timezone
        
        # Test FeatureImportance
        feature_importance = FeatureImportance(
            feature_name="amount",
            importance_score=0.25,
            feature_value=100.50,
            contribution=0.15
        )
        print(f"   [OK] FeatureImportance created: {feature_importance.feature_name}")
        
        # Test ModelScore
        model_score = ModelScore(
            model_id="fraud_model_v1",
            model_version="1.0.0",
            fraud_probability=0.25,
            risk_level=RiskLevel.LOW,
            confidence_score=0.85,
            feature_importances=[feature_importance],
            processing_time_ms=25.0
        )
        print(f"   [OK] ModelScore created: {model_score.model_id}")
        
        # Test ScoringResponse
        scoring_response = ScoringResponse(
            transaction_id="txn_123",
            user_id="user_456",
            model_scores=[model_score],
            final_decision="APPROVE",
            risk_level=RiskLevel.LOW,
            explanation="Low risk transaction",
            timestamp=datetime.now(timezone.utc)
        )
        print(f"   [OK] ScoringResponse created: {scoring_response.transaction_id}")
        print(f"   [OK] Final decision: {scoring_response.final_decision}")
        print(f"   [OK] Risk level: {scoring_response.risk_level}")
        
        # Test validation
        assert 0 <= model_score.fraud_probability <= 1
        assert 0 <= model_score.confidence_score <= 1
        assert scoring_response.final_decision == "APPROVE"
        
        print("   [OK] Service scoring models validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Service scoring models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_prediction_models():
    """Test service prediction models."""
    print("\nTesting service prediction models...")
    
    try:
        from service.models import PredictionResult, HealthCheckResponse, ServiceMetrics
        from datetime import datetime, timezone
        
        # Test PredictionResult
        prediction_result = PredictionResult(
            prediction_id="pred_123",
            transaction_id="txn_123",
            model_version="1.0.0",
            fraud_probability=0.25,
            risk_score=0.25,
            decision="APPROVE",
            confidence_score=0.85,
            explanation={"amount": 0.15, "merchant_risk": 0.30},
            processing_time_ms=25.0,
            timestamp=datetime.now(timezone.utc)
        )
        print(f"   [OK] PredictionResult created: {prediction_result.prediction_id}")
        print(f"   [OK] Decision: {prediction_result.decision}")
        
        # Test HealthCheckResponse
        health_check = HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            uptime_seconds=3600,
            checks={
                "database": "healthy",
                "redis": "healthy",
                "model": "healthy"
            }
        )
        print(f"   [OK] HealthCheckResponse created: {health_check.status}")
        
        # Test ServiceMetrics
        service_metrics = ServiceMetrics(
            requests_total=1000,
            requests_per_second=10.5,
            average_response_time_ms=25.0,
            error_rate=0.01,
            timestamp=datetime.now(timezone.utc)
        )
        print(f"   [OK] ServiceMetrics created: {service_metrics.requests_total}")
        
        # Test validation
        assert 0 <= prediction_result.fraud_probability <= 1
        assert health_check.status == "healthy"
        assert service_metrics.requests_total > 0
        
        print("   [OK] Service prediction models validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Service prediction models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_enums():
    """Test service enum classes."""
    print("\nTesting service enum classes...")
    
    try:
        from service.models import RiskLevel, PaymentMethod, TransactionStatus, TransactionType
        
        # Test RiskLevel
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"
        print("   [OK] RiskLevel enum")
        
        # Test PaymentMethod
        assert PaymentMethod.CREDIT_CARD == "credit_card"
        assert PaymentMethod.DEBIT_CARD == "debit_card"
        print("   [OK] PaymentMethod enum")
        
        # Test TransactionStatus
        assert TransactionStatus.PENDING == "pending"
        assert TransactionStatus.APPROVED == "approved"
        assert TransactionStatus.DECLINED == "declined"
        print("   [OK] TransactionStatus enum")
        
        # Test TransactionType
        assert TransactionType.PURCHASE == "purchase"
        assert TransactionType.REFUND == "refund"
        print("   [OK] TransactionType enum")
        
        print("   [OK] All service enums validation passed")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Service enums test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Service Models")
    print("=" * 40)
    
    results = []
    
    # Test all service models
    results.append(test_service_transaction_model())
    results.append(test_service_scoring_models())
    results.append(test_service_prediction_models())
    results.append(test_service_enums())
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\nAll service model tests passed!")
        print("The service models are working correctly.")
        sys.exit(0)
    else:
        print(f"\n{len(results) - sum(results)} service model tests failed.")
        print("Some service models may need fixes.")
        sys.exit(1)
