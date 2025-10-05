-- Migration: Seed initial data
-- Description: Insert initial data for development and testing
-- Created: Initial migration

-- Insert sample model metadata
INSERT INTO model_metadata (
    model_name,
    model_version,
    model_type,
    model_status,
    model_path,
    feature_columns,
    target_column,
    hyperparameters,
    training_metrics,
    validation_metrics,
    test_metrics,
    threshold_config,
    deployed_at,
    created_by,
    notes
) VALUES 
(
    'fraud_detector_v1',
    '1.0.0',
    'XGBOOST',
    'ACTIVE',
    '/models/fraud_detector_v1_1.0.0.pkl',
    ARRAY['amount', 'merchant_category', 'hour_of_day', 'day_of_week', 'user_avg_amount', 'merchant_risk_score', 'location_risk_score', 'velocity_1h', 'velocity_24h'],
    'is_fraud',
    '{
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }'::JSONB,
    '{
        "accuracy": 0.9456,
        "precision": 0.8923,
        "recall": 0.8734,
        "f1_score": 0.8827,
        "auc": 0.9612,
        "log_loss": 0.1234
    }'::JSONB,
    '{
        "accuracy": 0.9389,
        "precision": 0.8856,
        "recall": 0.8645,
        "f1_score": 0.8749,
        "auc": 0.9534,
        "log_loss": 0.1345
    }'::JSONB,
    '{
        "accuracy": 0.9401,
        "precision": 0.8878,
        "recall": 0.8667,
        "f1_score": 0.8771,
        "auc": 0.9556,
        "log_loss": 0.1298
    }'::JSONB,
    '{
        "low_risk": 0.3,
        "medium_risk": 0.6,
        "high_risk": 0.8,
        "critical_risk": 0.95
    }'::JSONB,
    CURRENT_TIMESTAMP - INTERVAL '7 days',
    'system',
    'Initial production model for fraud detection'
),
(
    'fraud_detector_v2',
    '2.0.0',
    'XGBOOST',
    'TESTING',
    '/models/fraud_detector_v2_2.0.0.pkl',
    ARRAY['amount', 'merchant_category', 'hour_of_day', 'day_of_week', 'user_avg_amount', 'merchant_risk_score', 'location_risk_score', 'velocity_1h', 'velocity_24h', 'device_fingerprint', 'ip_reputation'],
    'is_fraud',
    '{
        "n_estimators": 150,
        "max_depth": 8,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42
    }'::JSONB,
    '{
        "accuracy": 0.9523,
        "precision": 0.9012,
        "recall": 0.8856,
        "f1_score": 0.8933,
        "auc": 0.9678,
        "log_loss": 0.1156
    }'::JSONB,
    '{
        "accuracy": 0.9467,
        "precision": 0.8934,
        "recall": 0.8789,
        "f1_score": 0.8861,
        "auc": 0.9601,
        "log_loss": 0.1234
    }'::JSONB,
    '{
        "accuracy": 0.9478,
        "precision": 0.8945,
        "recall": 0.8801,
        "f1_score": 0.8872,
        "auc": 0.9612,
        "log_loss": 0.1198
    }'::JSONB,
    '{
        "low_risk": 0.25,
        "medium_risk": 0.55,
        "high_risk": 0.75,
        "critical_risk": 0.92
    }'::JSONB,
    NULL,
    'system',
    'Enhanced model with additional features - currently in testing phase'
);

-- Insert sample user profiles
INSERT INTO user_profiles (
    user_id,
    profile_data,
    avg_transaction_amount,
    transaction_frequency,
    preferred_merchants,
    preferred_categories,
    typical_locations,
    spending_patterns,
    risk_factors,
    account_age_days,
    total_transactions,
    total_amount,
    fraud_history_count
) VALUES 
(
    'user_001',
    '{
        "age_group": "25-34",
        "income_bracket": "middle",
        "credit_score_range": "good",
        "account_type": "premium"
    }'::JSONB,
    125.50,
    15,
    ARRAY['merchant_001', 'merchant_005', 'merchant_012'],
    ARRAY['grocery', 'gas_station', 'restaurant'],
    '{
        "primary_city": "New York",
        "primary_country": "US",
        "frequent_locations": [
            {"city": "New York", "country": "US", "frequency": 0.8},
            {"city": "Boston", "country": "US", "frequency": 0.15},
            {"city": "Philadelphia", "country": "US", "frequency": 0.05}
        ]
    }'::JSONB,
    '{
        "peak_hours": [12, 13, 18, 19],
        "peak_days": ["monday", "wednesday", "friday"],
        "seasonal_patterns": {
            "holiday_spending": 1.4,
            "summer_travel": 1.2
        }
    }'::JSONB,
    '{
        "risk_score": 0.15,
        "factors": ["stable_location", "consistent_spending"]
    }'::JSONB,
    365,
    1250,
    156875.00,
    0
),
(
    'user_002',
    '{
        "age_group": "35-44",
        "income_bracket": "high",
        "credit_score_range": "excellent",
        "account_type": "business"
    }'::JSONB,
    450.75,
    8,
    ARRAY['merchant_003', 'merchant_007', 'merchant_015'],
    ARRAY['business_services', 'travel', 'electronics'],
    '{
        "primary_city": "San Francisco",
        "primary_country": "US",
        "frequent_locations": [
            {"city": "San Francisco", "country": "US", "frequency": 0.7},
            {"city": "Los Angeles", "country": "US", "frequency": 0.2},
            {"city": "Seattle", "country": "US", "frequency": 0.1}
        ]
    }'::JSONB,
    '{
        "peak_hours": [9, 10, 14, 15],
        "peak_days": ["tuesday", "wednesday", "thursday"],
        "business_patterns": {
            "expense_reports": true,
            "travel_heavy": true
        }
    }'::JSONB,
    '{
        "risk_score": 0.08,
        "factors": ["high_income", "business_account", "excellent_credit"]
    }'::JSONB,
    1095,
    890,
    401167.50,
    0
),
(
    'user_003',
    '{
        "age_group": "18-24",
        "income_bracket": "low",
        "credit_score_range": "fair",
        "account_type": "basic"
    }'::JSONB,
    45.25,
    25,
    ARRAY['merchant_002', 'merchant_008', 'merchant_011'],
    ARRAY['fast_food', 'entertainment', 'clothing'],
    '{
        "primary_city": "Austin",
        "primary_country": "US",
        "frequent_locations": [
            {"city": "Austin", "country": "US", "frequency": 0.95},
            {"city": "Houston", "country": "US", "frequency": 0.05}
        ]
    }'::JSONB,
    '{
        "peak_hours": [19, 20, 21, 22],
        "peak_days": ["friday", "saturday", "sunday"],
        "student_patterns": {
            "weekend_heavy": true,
            "small_amounts": true
        }
    }'::JSONB,
    '{
        "risk_score": 0.25,
        "factors": ["young_account", "low_income", "high_frequency"]
    }'::JSONB,
    180,
    2150,
    97287.50,
    1
);

-- Insert sample transactions
INSERT INTO transactions (
    transaction_id,
    user_id,
    merchant_id,
    amount,
    currency,
    transaction_type,
    payment_method,
    card_type,
    card_last_four,
    merchant_category,
    merchant_name,
    location_country,
    location_city,
    location_latitude,
    location_longitude,
    device_id,
    ip_address,
    session_id,
    timestamp
) VALUES 
(
    'txn_001',
    'user_001',
    'merchant_001',
    89.99,
    'USD',
    'purchase',
    'credit_card',
    'visa',
    '1234',
    'grocery',
    'Fresh Market',
    'US',
    'New York',
    40.7128,
    -74.0060,
    'device_001',
    '192.168.1.100',
    'session_001',
    CURRENT_TIMESTAMP - INTERVAL '2 hours'
),
(
    'txn_002',
    'user_002',
    'merchant_003',
    1250.00,
    'USD',
    'purchase',
    'credit_card',
    'mastercard',
    '5678',
    'business_services',
    'Tech Solutions Inc',
    'US',
    'San Francisco',
    37.7749,
    -122.4194,
    'device_002',
    '10.0.0.50',
    'session_002',
    CURRENT_TIMESTAMP - INTERVAL '4 hours'
),
(
    'txn_003',
    'user_003',
    'merchant_002',
    15.75,
    'USD',
    'purchase',
    'debit_card',
    'visa',
    '9876',
    'fast_food',
    'Quick Burger',
    'US',
    'Austin',
    30.2672,
    -97.7431,
    'device_003',
    '172.16.0.25',
    'session_003',
    CURRENT_TIMESTAMP - INTERVAL '1 hour'
);

-- Insert sample fraud scores
INSERT INTO fraud_scores (
    transaction_id,
    model_name,
    model_version,
    fraud_score,
    risk_level,
    confidence_score,
    feature_importance,
    model_features,
    processing_time_ms,
    threshold_used,
    decision,
    decision_reason
) VALUES 
(
    'txn_001',
    'fraud_detector_v1',
    '1.0.0',
    0.15,
    'LOW',
    0.92,
    '{
        "amount": 0.12,
        "merchant_category": 0.08,
        "user_avg_amount": 0.25,
        "location_risk_score": 0.05,
        "velocity_24h": 0.18,
        "hour_of_day": 0.15,
        "merchant_risk_score": 0.10,
        "day_of_week": 0.07
    }'::JSONB,
    '{
        "amount": 89.99,
        "merchant_category": "grocery",
        "user_avg_amount": 125.50,
        "location_risk_score": 0.05,
        "velocity_24h": 3,
        "hour_of_day": 14,
        "merchant_risk_score": 0.08,
        "day_of_week": 3
    }'::JSONB,
    45,
    0.30,
    'APPROVE',
    'Low fraud score, consistent with user profile'
),
(
    'txn_002',
    'fraud_detector_v1',
    '1.0.0',
    0.25,
    'LOW',
    0.88,
    '{
        "amount": 0.35,
        "merchant_category": 0.15,
        "user_avg_amount": 0.20,
        "location_risk_score": 0.03,
        "velocity_24h": 0.12,
        "hour_of_day": 0.08,
        "merchant_risk_score": 0.05,
        "day_of_week": 0.02
    }'::JSONB,
    '{
        "amount": 1250.00,
        "merchant_category": "business_services",
        "user_avg_amount": 450.75,
        "location_risk_score": 0.03,
        "velocity_24h": 2,
        "hour_of_day": 10,
        "merchant_risk_score": 0.05,
        "day_of_week": 2
    }'::JSONB,
    52,
    0.30,
    'APPROVE',
    'Higher amount but consistent with business user profile'
),
(
    'txn_003',
    'fraud_detector_v1',
    '1.0.0',
    0.08,
    'LOW',
    0.95,
    '{
        "amount": 0.08,
        "merchant_category": 0.12,
        "user_avg_amount": 0.18,
        "location_risk_score": 0.02,
        "velocity_24h": 0.22,
        "hour_of_day": 0.20,
        "merchant_risk_score": 0.10,
        "day_of_week": 0.08
    }'::JSONB,
    '{
        "amount": 15.75,
        "merchant_category": "fast_food",
        "user_avg_amount": 45.25,
        "location_risk_score": 0.02,
        "velocity_24h": 8,
        "hour_of_day": 19,
        "merchant_risk_score": 0.10,
        "day_of_week": 5
    }'::JSONB,
    38,
    0.30,
    'APPROVE',
    'Very low fraud score, typical student spending pattern'
);

-- Insert sample audit log entries
SELECT create_audit_log(
    'TRANSACTION_CREATED',
    'TRANSACTION',
    'transaction',
    'txn_001',
    'CREATE',
    'user_001',
    NULL,
    '{
        "transaction_id": "txn_001",
        "amount": 89.99,
        "merchant_id": "merchant_001",
        "status": "completed"
    }'::JSONB,
    '{
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "session_id": "session_001"
    }'::JSONB,
    'INFO',
    'SUCCESS',
    'fraud_detection',
    'transaction_service'
);

SELECT create_audit_log(
    'FRAUD_SCORE_CALCULATED',
    'FRAUD_DETECTION',
    'fraud_score',
    'txn_001',
    'CALCULATE',
    NULL,
    NULL,
    '{
        "fraud_score": 0.15,
        "risk_level": "LOW",
        "decision": "APPROVE",
        "model_name": "fraud_detector_v1",
        "processing_time_ms": 45
    }'::JSONB,
    '{
        "model_version": "1.0.0",
        "confidence_score": 0.92
    }'::JSONB,
    'INFO',
    'SUCCESS',
    'fraud_detection',
    'ml_service'
);

-- Create indexes on timestamp columns for better query performance
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp_desc ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_created_at_desc ON fraud_scores(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_profiles_updated_at_desc ON user_profiles(updated_at DESC);

-- Update table statistics
ANALYZE transactions;
ANALYZE fraud_scores;
ANALYZE user_profiles;
ANALYZE model_metadata;
ANALYZE audit_log;

-- Comments
COMMENT ON COLUMN transactions.timestamp IS 'Transaction timestamp - indexed for performance';
COMMENT ON COLUMN fraud_scores.created_at IS 'Score calculation timestamp - indexed for performance';
COMMENT ON COLUMN user_profiles.updated_at IS 'Profile update timestamp - indexed for performance';