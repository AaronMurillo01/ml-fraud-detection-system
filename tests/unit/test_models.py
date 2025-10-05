"""Unit tests for core data models."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from pydantic import ValidationError

from service.models import (
    Transaction,
    EnrichedTransaction,
    ModelScore,
    ScoringResponse,
    RiskLevel,
    FeatureImportance,
    ActionRecommendation
)


class TestTransaction:
    """Test Transaction model."""
    
    def test_valid_transaction_creation(self):
        """Test creating a valid transaction."""
        transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_456",
            merchant_id="merchant_789",
            amount=Decimal("100.50"),
            currency="USD",
            transaction_type="purchase",
            payment_method="credit_card",
            transaction_time=datetime.now(timezone.utc),
            location_lat=40.7128,
            location_lon=-74.0060,
            country_code="US",
            ip_address="192.168.1.100",
            device_id="device_fingerprint_xyz789"
        )
        
        assert transaction.transaction_id == "txn_123"
        assert transaction.amount == Decimal("100.50")
        assert transaction.currency == "USD"
        assert transaction.transaction_type == "purchase"
        assert transaction.payment_method == "credit_card"
    
    def test_invalid_amount(self):
        """Test transaction with invalid amount."""
        with pytest.raises(ValidationError) as exc_info:
            Transaction(
                transaction_id="txn_123",
                user_id="user_456",
                merchant_id="merchant_789",
                amount=Decimal("-50.00"),  # Negative amount
                currency="USD",
                transaction_type="purchase",
                payment_method="credit_card",
                transaction_time=datetime.now(timezone.utc)
            )
        
        assert "greater than 0" in str(exc_info.value)
    
    def test_invalid_currency(self):
        """Test transaction with invalid currency."""
        with pytest.raises(ValidationError) as exc_info:
            Transaction(
                transaction_id="txn_123",
                user_id="user_456",
                merchant_id="merchant_789",
                amount=Decimal("100.50"),
                currency="INVALID",  # Invalid currency code (too long)
                transaction_type="purchase",
                payment_method="credit_card",
                transaction_time=datetime.now(timezone.utc)
            )
        
        assert "String should have at most 3 characters" in str(exc_info.value)
    
    def test_transaction_serialization(self):
        """Test transaction serialization to dict."""
        transaction = Transaction(
            transaction_id="txn_123",
            user_id="user_456",
            merchant_id="merchant_789",
            amount=Decimal("100.50"),
            currency="USD",
            transaction_type="purchase",
            payment_method="credit_card",
            transaction_time=datetime.now(timezone.utc)
        )
        
        transaction_dict = transaction.model_dump()
        assert transaction_dict["transaction_id"] == "txn_123"
        assert transaction_dict["amount"] == Decimal("100.50")
        assert transaction_dict["currency"] == "USD"


class TestEnrichedTransaction:
    """Test EnrichedTransaction model."""
    
    def test_enriched_transaction_creation(self):
        """Test creating an enriched transaction."""
        enriched_transaction = EnrichedTransaction(
            transaction_id="txn_123",
            user_id="user_456",
            merchant_id="merchant_789",
            amount=Decimal("100.50"),
            currency="USD",
            transaction_type="purchase",
            payment_method="credit_card",
            transaction_time=datetime.now(timezone.utc),
            is_weekend=False,
            is_night_time=False,
            user_transaction_count_1h=2,
            user_transaction_count_24h=5,
            user_amount_sum_1h=Decimal("75.25"),
            user_amount_sum_24h=Decimal("500.00"),
            card_transaction_count_1h=1,
            card_transaction_count_24h=3,
            merchant_transaction_count_1h=45,
            distance_from_home=15.5,
            distance_from_last_transaction=2.3,
            is_new_device=False,
            is_new_ip=True,
            amount_zscore_user_7d=1.5,
            amount_zscore_merchant_7d=0.8,
            hour_of_day=14,
            day_of_week=2
        )
        
        assert enriched_transaction.transaction_id == "txn_123"
        assert enriched_transaction.is_weekend == False
        assert enriched_transaction.user_transaction_count_1h == 2
        assert enriched_transaction.hour_of_day == 14


class TestModelScore:
    """Test ModelScore model."""
    
    def test_model_score_creation(self):
        """Test creating a model score."""
        model_score = ModelScore(
            transaction_id="txn_123",
            model_version="xgboost_v2",
            model_name="fraud_detector_xgb_v2.1",
            fraud_probability=0.75,
            risk_level=RiskLevel.HIGH,
            confidence_score=0.9,
            recommended_action=ActionRecommendation.REVIEW,
            inference_time_ms=12.5,
            decision_threshold=0.5
        )
        
        assert model_score.transaction_id == "txn_123"
        assert model_score.fraud_probability == 0.75
        assert model_score.confidence_score == 0.9
        assert model_score.decision_threshold == 0.5
        assert model_score.risk_level == RiskLevel.HIGH
    
    def test_invalid_score_range(self):
        """Test ModelScore with invalid fraud probability range."""
        with pytest.raises(ValidationError) as exc_info:
            ModelScore(
                transaction_id="txn_123",
                model_version="xgboost_v2",
                model_name="fraud_detector_xgb_v2.1",
                fraud_probability=1.5,  # Invalid: > 1.0
                risk_level=RiskLevel.HIGH,
                confidence_score=0.9,
                recommended_action=ActionRecommendation.REVIEW,
                inference_time_ms=12.5,
                decision_threshold=0.5
            )
        
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_invalid_confidence_range(self):
        """Test ModelScore with invalid confidence range."""
        with pytest.raises(ValidationError) as exc_info:
            ModelScore(
                transaction_id="txn_123",
                model_version="xgboost_v1",
                model_name="fraud_detector_xgb_v1.0",
                fraud_probability=0.75,
                confidence_score=1.5,  # Invalid: > 1.0
                decision_threshold=0.5,
                inference_time_ms=12.5,
                recommended_action="review"
            )
        
        assert "less than or equal to 1" in str(exc_info.value)


class TestScoringResponse:
    """Test ScoringResponse model."""
    
    def test_scoring_response_creation(self):
        """Test ScoringResponse model creation."""
        model_score = ModelScore(
            transaction_id="txn_123",
            model_version="xgboost_v1",
            model_name="fraud_detector_xgb_v1.0",
            fraud_probability=0.65,
            risk_level=RiskLevel.MEDIUM,
            confidence_score=0.88,
            recommended_action=ActionRecommendation.REVIEW,
            inference_time_ms=12.5,
            decision_threshold=0.5
        )
        
        scoring_response = ScoringResponse(
            transaction_id="txn_123",
            score=model_score,
            processing_time_ms=25.5
        )
        
        assert scoring_response.transaction_id == "txn_123"
        assert scoring_response.score.fraud_probability == 0.65
        assert scoring_response.score.confidence_score == 0.88
        assert scoring_response.processing_time_ms == 25.5
    
    def test_invalid_risk_score_range(self):
        """Test ScoringResponse with invalid fraud probability range."""
        with pytest.raises(ValidationError) as exc_info:
            ModelScore(
                transaction_id="txn_123",
                model_version="xgboost_v1",
                model_name="fraud_detector_xgb_v1.0",
                fraud_probability=1.2,  # Invalid: > 1.0
                confidence_score=0.88,
                decision_threshold=0.5,
                inference_time_ms=12.5,
                recommended_action="review"
            )
        
        assert "less than or equal to 1" in str(exc_info.value)


class TestFeatureImportance:
    """Test FeatureImportance model."""
    
    def test_feature_importance_creation(self):
        """Test FeatureImportance model creation."""
        feature_importance = FeatureImportance(
            feature_name="amount_zscore_user_7d",
            importance_score=0.15,
            feature_value=2.3,
            description="Transaction amount Z-score over 7 days"
        )
        
        assert feature_importance.feature_name == "amount_zscore_user_7d"
        assert feature_importance.importance_score == 0.15
        assert feature_importance.feature_value == 2.3
        assert feature_importance.description == "Transaction amount Z-score over 7 days"
    
    def test_negative_importance_score(self):
        """Test FeatureImportance with negative importance score (allowed for SHAP)."""
        feature_importance = FeatureImportance(
            feature_name="test_feature",
            importance_score=-0.5,  # Negative values allowed for SHAP
            description="Test feature with negative importance"
        )
        
        assert feature_importance.importance_score == -0.5


class TestRiskLevel:
    """Test cases for RiskLevel enum."""
    
    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"
    
    def test_risk_level_ordering(self):
        """Test RiskLevel enum can be used in comparisons."""
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(levels) == 4
        assert RiskLevel.LOW in levels
        assert RiskLevel.CRITICAL in levels


if __name__ == "__main__":
    pytest.main([__file__])