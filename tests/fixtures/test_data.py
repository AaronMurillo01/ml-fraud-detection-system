"""Test data fixtures and sample data generators."""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from decimal import Decimal

from shared.models import (
    Transaction, User, Merchant, FraudPrediction, ModelMetadata,
    PaymentMethod, TransactionStatus, RiskLevel
)
from service.models import (
    EnrichedTransaction, PredictionResult, ModelPerformanceMetrics,
    FeatureImportance, BatchPredictionResult
)


# Sample transaction data
sample_transactions = [
    {
        "transaction_id": "txn_001",
        "user_id": "user_001",
        "merchant_id": "merchant_001",
        "amount": 25.99,
        "timestamp": datetime(2024, 1, 15, 10, 30, 0),
        "payment_method": PaymentMethod.CREDIT_CARD,
        "transaction_type": "purchase",
        "status": TransactionStatus.APPROVED,
        "currency": "USD",
        "description": "Coffee shop purchase",
        "metadata": {
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "device_id": "device_001",
            "location": "New York, NY"
        }
    },
    {
        "transaction_id": "txn_002",
        "user_id": "user_002",
        "merchant_id": "merchant_002",
        "amount": 1250.00,
        "timestamp": datetime(2024, 1, 15, 14, 45, 0),
        "payment_method": PaymentMethod.DEBIT_CARD,
        "transaction_type": "purchase",
        "status": TransactionStatus.PENDING,
        "currency": "USD",
        "description": "Electronics store purchase",
        "metadata": {
            "ip_address": "10.0.0.50",
            "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)",
            "device_id": "device_002",
            "location": "Los Angeles, CA"
        }
    },
    {
        "transaction_id": "txn_003",
        "user_id": "user_001",
        "merchant_id": "merchant_003",
        "amount": 5000.00,
        "timestamp": datetime(2024, 1, 15, 23, 15, 0),
        "payment_method": PaymentMethod.CREDIT_CARD,
        "transaction_type": "purchase",
        "status": TransactionStatus.DECLINED,
        "currency": "USD",
        "description": "Suspicious high-value transaction",
        "metadata": {
            "ip_address": "203.0.113.1",
            "user_agent": "curl/7.68.0",
            "device_id": "unknown",
            "location": "Unknown"
        }
    },
    {
        "transaction_id": "txn_004",
        "user_id": "user_003",
        "merchant_id": "merchant_001",
        "amount": 3.50,
        "timestamp": datetime(2024, 1, 16, 8, 0, 0),
        "payment_method": PaymentMethod.DIGITAL_WALLET,
        "transaction_type": "purchase",
        "status": TransactionStatus.APPROVED,
        "currency": "USD",
        "description": "Small coffee purchase",
        "metadata": {
            "ip_address": "192.168.1.200",
            "user_agent": "PayPal/1.0",
            "device_id": "device_003",
            "location": "Chicago, IL"
        }
    },
    {
        "transaction_id": "txn_005",
        "user_id": "user_002",
        "merchant_id": "merchant_004",
        "amount": 89.99,
        "timestamp": datetime(2024, 1, 16, 16, 30, 0),
        "payment_method": PaymentMethod.BANK_TRANSFER,
        "transaction_type": "refund",
        "status": TransactionStatus.APPROVED,
        "currency": "USD",
        "description": "Product return refund",
        "metadata": {
            "ip_address": "10.0.0.50",
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "device_id": "device_002",
            "location": "Los Angeles, CA"
        }
    }
]

# Sample user data
sample_users = [
    {
        "user_id": "user_001",
        "email": "john.doe@example.com",
        "phone_number": "+1-555-0101",
        "registration_date": datetime(2023, 6, 15, 12, 0, 0),
        "last_login": datetime(2024, 1, 15, 9, 30, 0),
        "risk_score": 0.25,
        "status": "active",
        "user_data": {
            "age": 32,
            "location": "New York, NY",
            "account_type": "premium",
            "verification_status": "verified"
        }
    },
    {
        "user_id": "user_002",
        "email": "jane.smith@example.com",
        "phone_number": "+1-555-0102",
        "registration_date": datetime(2023, 8, 22, 14, 30, 0),
        "last_login": datetime(2024, 1, 16, 13, 45, 0),
        "risk_score": 0.15,
        "status": "active",
        "user_data": {
            "age": 28,
            "location": "Los Angeles, CA",
            "account_type": "standard",
            "verification_status": "verified"
        }
    },
    {
        "user_id": "user_003",
        "email": "bob.wilson@example.com",
        "phone_number": "+1-555-0103",
        "registration_date": datetime(2024, 1, 10, 10, 0, 0),
        "last_login": datetime(2024, 1, 16, 7, 15, 0),
        "risk_score": 0.60,
        "status": "active",
        "user_data": {
            "age": 45,
            "location": "Chicago, IL",
            "account_type": "basic",
            "verification_status": "pending"
        }
    },
    {
        "user_id": "user_004",
        "email": "suspicious@tempmail.com",
        "phone_number": "+1-555-9999",
        "registration_date": datetime(2024, 1, 15, 23, 0, 0),
        "last_login": datetime(2024, 1, 15, 23, 10, 0),
        "risk_score": 0.95,
        "status": "suspended",
        "user_data": {
            "age": None,
            "location": "Unknown",
            "account_type": "basic",
            "verification_status": "failed"
        }
    }
]

# Sample merchant data
sample_merchants = [
    {
        "merchant_id": "merchant_001",
        "merchant_name": "Central Coffee Co.",
        "category": "food_beverage",
        "location": "New York, NY",
        "risk_score": 0.10,
        "status": "active",
        "merchant_data": {
            "business_type": "restaurant",
            "years_in_business": 8,
            "average_transaction_amount": 15.50,
            "monthly_volume": 50000.00,
            "verification_status": "verified"
        }
    },
    {
        "merchant_id": "merchant_002",
        "merchant_name": "TechWorld Electronics",
        "category": "electronics",
        "location": "Los Angeles, CA",
        "risk_score": 0.30,
        "status": "active",
        "merchant_data": {
            "business_type": "retail",
            "years_in_business": 12,
            "average_transaction_amount": 450.00,
            "monthly_volume": 200000.00,
            "verification_status": "verified"
        }
    },
    {
        "merchant_id": "merchant_003",
        "merchant_name": "Luxury Goods Inc.",
        "category": "luxury",
        "location": "Miami, FL",
        "risk_score": 0.70,
        "status": "under_review",
        "merchant_data": {
            "business_type": "luxury_retail",
            "years_in_business": 2,
            "average_transaction_amount": 2500.00,
            "monthly_volume": 500000.00,
            "verification_status": "pending"
        }
    },
    {
        "merchant_id": "merchant_004",
        "merchant_name": "QuickMart Convenience",
        "category": "convenience",
        "location": "Chicago, IL",
        "risk_score": 0.20,
        "status": "active",
        "merchant_data": {
            "business_type": "convenience_store",
            "years_in_business": 5,
            "average_transaction_amount": 25.00,
            "monthly_volume": 75000.00,
            "verification_status": "verified"
        }
    }
]

# Sample fraud prediction data
sample_predictions = [
    {
        "prediction_id": "pred_001",
        "transaction_id": "txn_001",
        "user_id": "user_001",
        "fraud_probability": 0.15,
        "risk_level": RiskLevel.LOW,
        "decision": "APPROVE",
        "confidence_score": 0.92,
        "model_version": "v1.0.0",
        "model_features": {
            "amount": 25.99,
            "merchant_risk": 0.10,
            "user_risk": 0.25,
            "velocity_1h": 1,
            "velocity_24h": 3
        },
        "feature_importance": {
            "amount": 0.25,
            "merchant_risk": 0.20,
            "user_risk": 0.15,
            "velocity": 0.30,
            "contextual": 0.10
        },
        "created_at": datetime(2024, 1, 15, 10, 30, 5)
    },
    {
        "prediction_id": "pred_002",
        "transaction_id": "txn_002",
        "user_id": "user_002",
        "fraud_probability": 0.45,
        "risk_level": RiskLevel.MEDIUM,
        "decision": "REVIEW",
        "confidence_score": 0.78,
        "model_version": "v1.0.0",
        "model_features": {
            "amount": 1250.00,
            "merchant_risk": 0.30,
            "user_risk": 0.15,
            "velocity_1h": 1,
            "velocity_24h": 2
        },
        "feature_importance": {
            "amount": 0.35,
            "merchant_risk": 0.25,
            "user_risk": 0.10,
            "velocity": 0.20,
            "contextual": 0.10
        },
        "created_at": datetime(2024, 1, 15, 14, 45, 3)
    },
    {
        "prediction_id": "pred_003",
        "transaction_id": "txn_003",
        "user_id": "user_001",
        "fraud_probability": 0.95,
        "risk_level": RiskLevel.CRITICAL,
        "decision": "DECLINE",
        "confidence_score": 0.98,
        "model_version": "v1.0.0",
        "model_features": {
            "amount": 5000.00,
            "merchant_risk": 0.70,
            "user_risk": 0.25,
            "velocity_1h": 2,
            "velocity_24h": 4
        },
        "feature_importance": {
            "amount": 0.40,
            "merchant_risk": 0.30,
            "user_risk": 0.10,
            "velocity": 0.15,
            "contextual": 0.05
        },
        "created_at": datetime(2024, 1, 15, 23, 15, 2)
    }
]

# Sample model metadata
sample_model_metadata = [
    {
        "model_id": "model_001",
        "model_name": "fraud_detector_v1",
        "model_version": "1.0.0",
        "model_type": "xgboost",
        "model_path": "/models/fraud_detector_v1_1.0.0.pkl",
        "feature_names": [
            "amount", "merchant_risk_score", "user_risk_score",
            "velocity_1h", "velocity_24h", "hour_of_day", "day_of_week",
            "payment_method_encoded", "transaction_type_encoded"
        ],
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "performance_metrics": {
            "accuracy": 0.95,
            "precision": 0.88,
            "recall": 0.92,
            "f1_score": 0.90,
            "auc_roc": 0.96,
            "auc_pr": 0.89
        },
        "training_data_size": 100000,
        "validation_data_size": 20000,
        "test_data_size": 10000,
        "deployment_date": datetime(2024, 1, 1, 0, 0, 0),
        "created_at": datetime(2023, 12, 15, 10, 0, 0),
        "updated_at": datetime(2024, 1, 1, 0, 0, 0)
    },
    {
        "model_id": "model_002",
        "model_name": "fraud_detector_v2",
        "model_version": "2.0.0",
        "model_type": "lightgbm",
        "model_path": "/models/fraud_detector_v2_2.0.0.pkl",
        "feature_names": [
            "amount", "merchant_risk_score", "user_risk_score",
            "velocity_1h", "velocity_24h", "velocity_7d", "hour_of_day",
            "day_of_week", "payment_method_encoded", "transaction_type_encoded",
            "location_risk_score", "device_risk_score"
        ],
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9
        },
        "performance_metrics": {
            "accuracy": 0.97,
            "precision": 0.91,
            "recall": 0.94,
            "f1_score": 0.925,
            "auc_roc": 0.98,
            "auc_pr": 0.92
        },
        "training_data_size": 200000,
        "validation_data_size": 40000,
        "test_data_size": 20000,
        "deployment_date": datetime(2024, 2, 1, 0, 0, 0),
        "created_at": datetime(2024, 1, 15, 14, 0, 0),
        "updated_at": datetime(2024, 2, 1, 0, 0, 0)
    }
]


def generate_random_transaction(user_id: str = None, merchant_id: str = None) -> Dict[str, Any]:
    """Generate a random transaction for testing."""
    transaction_id = f"txn_{random.randint(10000, 99999)}"
    
    return {
        "transaction_id": transaction_id,
        "user_id": user_id or f"user_{random.randint(1000, 9999)}",
        "merchant_id": merchant_id or f"merchant_{random.randint(100, 999)}",
        "amount": round(random.uniform(1.0, 2000.0), 2),
        "timestamp": datetime.utcnow() - timedelta(
            hours=random.randint(0, 72),
            minutes=random.randint(0, 59)
        ),
        "payment_method": random.choice(list(PaymentMethod)),
        "transaction_type": random.choice(["purchase", "refund", "transfer"]),
        "status": random.choice(list(TransactionStatus)),
        "currency": "USD",
        "description": f"Random transaction {transaction_id}",
        "metadata": {
            "ip_address": f"192.168.1.{random.randint(1, 254)}",
            "user_agent": "TestAgent/1.0",
            "device_id": f"device_{random.randint(1000, 9999)}",
            "location": random.choice(["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX"])
        }
    }


def generate_random_user() -> Dict[str, Any]:
    """Generate a random user for testing."""
    user_id = f"user_{random.randint(10000, 99999)}"
    
    return {
        "user_id": user_id,
        "email": f"user{random.randint(1000, 9999)}@example.com",
        "phone_number": f"+1-555-{random.randint(1000, 9999)}",
        "registration_date": datetime.utcnow() - timedelta(
            days=random.randint(1, 365)
        ),
        "last_login": datetime.utcnow() - timedelta(
            hours=random.randint(1, 48)
        ),
        "risk_score": round(random.uniform(0.0, 1.0), 2),
        "status": random.choice(["active", "inactive", "suspended"]),
        "user_data": {
            "age": random.randint(18, 80),
            "location": random.choice(["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX"]),
            "account_type": random.choice(["basic", "standard", "premium"]),
            "verification_status": random.choice(["verified", "pending", "failed"])
        }
    }


def generate_random_merchant() -> Dict[str, Any]:
    """Generate a random merchant for testing."""
    merchant_id = f"merchant_{random.randint(1000, 9999)}"
    
    return {
        "merchant_id": merchant_id,
        "merchant_name": f"Test Merchant {random.randint(100, 999)}",
        "category": random.choice([
            "food_beverage", "electronics", "clothing", "automotive",
            "health_beauty", "home_garden", "sports_outdoors", "books_media"
        ]),
        "location": random.choice(["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX"]),
        "risk_score": round(random.uniform(0.0, 1.0), 2),
        "status": random.choice(["active", "inactive", "under_review"]),
        "merchant_data": {
            "business_type": random.choice(["retail", "restaurant", "service", "online"]),
            "years_in_business": random.randint(1, 20),
            "average_transaction_amount": round(random.uniform(10.0, 500.0), 2),
            "monthly_volume": round(random.uniform(10000.0, 1000000.0), 2),
            "verification_status": random.choice(["verified", "pending", "failed"])
        }
    }


def generate_batch_transactions(count: int = 10) -> List[Dict[str, Any]]:
    """Generate a batch of random transactions for testing."""
    return [generate_random_transaction() for _ in range(count)]


def generate_batch_users(count: int = 5) -> List[Dict[str, Any]]:
    """Generate a batch of random users for testing."""
    return [generate_random_user() for _ in range(count)]


def generate_batch_merchants(count: int = 5) -> List[Dict[str, Any]]:
    """Generate a batch of random merchants for testing."""
    return [generate_random_merchant() for _ in range(count)]


def create_enriched_transaction_from_sample(transaction_data: Dict[str, Any]) -> EnrichedTransaction:
    """Create an enriched transaction from sample data."""
    return EnrichedTransaction(
        **transaction_data,
        velocity_features={
            "txn_count_1h": random.randint(1, 10),
            "txn_count_24h": random.randint(5, 50),
            "amount_sum_1h": round(random.uniform(100.0, 5000.0), 2),
            "amount_sum_24h": round(random.uniform(500.0, 20000.0), 2)
        },
        risk_features={
            "merchant_risk_score": round(random.uniform(0.0, 1.0), 2),
            "user_risk_score": round(random.uniform(0.0, 1.0), 2),
            "location_risk_score": round(random.uniform(0.0, 1.0), 2)
        },
        behavioral_features={
            "avg_transaction_amount": round(random.uniform(50.0, 500.0), 2),
            "transaction_frequency": round(random.uniform(1.0, 10.0), 1)
        },
        contextual_features={
            "is_weekend": random.choice([True, False]),
            "hour_of_day": random.randint(0, 23),
            "day_of_week": random.randint(0, 6)
        }
    )


def create_prediction_result_from_sample(prediction_data: Dict[str, Any]) -> PredictionResult:
    """Create a prediction result from sample data."""
    return PredictionResult(
        transaction_id=prediction_data["transaction_id"],
        fraud_probability=prediction_data["fraud_probability"],
        risk_level=prediction_data["risk_level"],
        decision=prediction_data["decision"],
        confidence_score=prediction_data["confidence_score"],
        feature_contributions=prediction_data.get("feature_importance", {}),
        model_version=prediction_data["model_version"],
        processing_time_ms=round(random.uniform(10.0, 100.0), 1),
        explanation=f"Prediction for {prediction_data['transaction_id']}"
    )


# Export commonly used test data
__all__ = [
    'sample_transactions',
    'sample_users', 
    'sample_merchants',
    'sample_predictions',
    'sample_model_metadata',
    'generate_random_transaction',
    'generate_random_user',
    'generate_random_merchant',
    'generate_batch_transactions',
    'generate_batch_users',
    'generate_batch_merchants',
    'create_enriched_transaction_from_sample',
    'create_prediction_result_from_sample'
]