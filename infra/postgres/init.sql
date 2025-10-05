-- Fraud Detection Database Initialization Script
-- This script sets up the initial database schema and configuration

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS fraud_detection;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Set search path
SET search_path TO fraud_detection, public;

-- Create enum types
CREATE TYPE transaction_status AS ENUM ('pending', 'approved', 'declined', 'flagged');
CREATE TYPE fraud_prediction AS ENUM ('legitimate', 'fraud', 'suspicious');
CREATE TYPE model_status AS ENUM ('active', 'inactive', 'training', 'deprecated');

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    merchant_id VARCHAR(255),
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    transaction_time TIMESTAMP WITH TIME ZONE NOT NULL,
    merchant_category VARCHAR(100),
    location_country VARCHAR(2),
    location_city VARCHAR(100),
    location_lat DECIMAL(10, 8),
    location_lon DECIMAL(11, 8),
    payment_method VARCHAR(50),
    card_type VARCHAR(50),
    is_weekend BOOLEAN,
    is_night BOOLEAN,
    status transaction_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fraud predictions table
CREATE TABLE IF NOT EXISTS fraud_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    prediction fraud_prediction NOT NULL,
    confidence_score DECIMAL(5, 4) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    risk_score DECIMAL(5, 4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    feature_importance JSONB,
    processing_time_ms INTEGER,
    prediction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

-- User behavior profiles
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    avg_transaction_amount DECIMAL(15, 2),
    transaction_frequency DECIMAL(8, 2),
    preferred_merchants TEXT[],
    common_locations JSONB,
    spending_patterns JSONB,
    risk_level VARCHAR(20) DEFAULT 'low',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Merchant profiles
CREATE TABLE IF NOT EXISTS merchant_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    merchant_id VARCHAR(255) UNIQUE NOT NULL,
    merchant_name VARCHAR(255),
    category VARCHAR(100),
    risk_level VARCHAR(20) DEFAULT 'low',
    fraud_rate DECIMAL(5, 4) DEFAULT 0,
    avg_transaction_amount DECIMAL(15, 2),
    location_country VARCHAR(2),
    location_city VARCHAR(100),
    is_high_risk BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model versions and metadata
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    status model_status DEFAULT 'inactive',
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    auc_score DECIMAL(5, 4),
    training_data_size INTEGER,
    feature_count INTEGER,
    hyperparameters JSONB,
    deployment_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

-- Feature store
CREATE TABLE IF NOT EXISTS feature_store (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15, 6),
    feature_type VARCHAR(50),
    extraction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

-- Audit schema tables
CREATE TABLE IF NOT EXISTS audit.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(255) UNIQUE NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id VARCHAR(255),
    api_key_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    request_body JSONB,
    response_status INTEGER,
    response_time_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit.fraud_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) NOT NULL,
    decision fraud_prediction NOT NULL,
    confidence_score DECIMAL(5, 4) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    decision_factors JSONB,
    reviewer_id VARCHAR(255),
    review_notes TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Monitoring schema tables
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    metric_type VARCHAR(50),
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(transaction_time);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_location ON transactions(location_country, location_city);

CREATE INDEX IF NOT EXISTS idx_fraud_predictions_transaction_id ON fraud_predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_fraud_predictions_model_version ON fraud_predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_fraud_predictions_prediction ON fraud_predictions(prediction);
CREATE INDEX IF NOT EXISTS idx_fraud_predictions_time ON fraud_predictions(prediction_time);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_risk_level ON user_profiles(risk_level);

CREATE INDEX IF NOT EXISTS idx_merchant_profiles_merchant_id ON merchant_profiles(merchant_id);
CREATE INDEX IF NOT EXISTS idx_merchant_profiles_category ON merchant_profiles(category);
CREATE INDEX IF NOT EXISTS idx_merchant_profiles_risk_level ON merchant_profiles(risk_level);

CREATE INDEX IF NOT EXISTS idx_feature_store_transaction_id ON feature_store(transaction_id);
CREATE INDEX IF NOT EXISTS idx_feature_store_feature_name ON feature_store(feature_name);

CREATE INDEX IF NOT EXISTS idx_audit_api_requests_timestamp ON audit.api_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_api_requests_endpoint ON audit.api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_audit_api_requests_user_id ON audit.api_requests(user_id);

CREATE INDEX IF NOT EXISTS idx_audit_fraud_decisions_transaction_id ON audit.fraud_decisions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_fraud_decisions_timestamp ON audit.fraud_decisions(timestamp);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_merchant_profiles_updated_at BEFORE UPDATE ON merchant_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW fraud_detection.transaction_summary AS
SELECT 
    t.transaction_id,
    t.user_id,
    t.merchant_id,
    t.amount,
    t.currency,
    t.transaction_time,
    t.status,
    fp.prediction,
    fp.confidence_score,
    fp.risk_score,
    fp.model_version
FROM transactions t
LEFT JOIN fraud_predictions fp ON t.transaction_id = fp.transaction_id;

CREATE OR REPLACE VIEW fraud_detection.daily_fraud_stats AS
SELECT 
    DATE(prediction_time) as date,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN prediction = 'fraud' THEN 1 END) as fraud_count,
    COUNT(CASE WHEN prediction = 'legitimate' THEN 1 END) as legitimate_count,
    COUNT(CASE WHEN prediction = 'suspicious' THEN 1 END) as suspicious_count,
    AVG(confidence_score) as avg_confidence,
    AVG(risk_score) as avg_risk_score
FROM fraud_predictions
GROUP BY DATE(prediction_time)
ORDER BY date DESC;

-- Insert sample data for testing (optional)
-- This can be removed in production
INSERT INTO model_versions (version, model_type, status, accuracy, precision_score, recall, f1_score, auc_score, created_by)
VALUES 
    ('v1.0.0', 'xgboost', 'active', 0.9234, 0.8876, 0.9012, 0.8943, 0.9567, 'system'),
    ('v0.9.0', 'xgboost', 'deprecated', 0.9123, 0.8765, 0.8934, 0.8849, 0.9456, 'system')
ON CONFLICT (version) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA fraud_detection TO fraud_user;
GRANT USAGE ON SCHEMA audit TO fraud_user;
GRANT USAGE ON SCHEMA monitoring TO fraud_user;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA fraud_detection TO fraud_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA audit TO fraud_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA monitoring TO fraud_user;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA fraud_detection TO fraud_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO fraud_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA monitoring TO fraud_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA fraud_detection GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fraud_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fraud_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fraud_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA fraud_detection GRANT USAGE, SELECT ON SEQUENCES TO fraud_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT USAGE, SELECT ON SEQUENCES TO fraud_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA monitoring GRANT USAGE, SELECT ON SEQUENCES TO fraud_user;

-- Create database statistics and maintenance functions
CREATE OR REPLACE FUNCTION fraud_detection.cleanup_old_audit_data(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM audit.api_requests 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to update user profiles based on transaction history
CREATE OR REPLACE FUNCTION fraud_detection.update_user_profile(p_user_id VARCHAR(255))
RETURNS VOID AS $$
BEGIN
    INSERT INTO user_profiles (user_id, avg_transaction_amount, transaction_frequency, last_updated)
    SELECT 
        p_user_id,
        AVG(amount),
        COUNT(*)::DECIMAL / EXTRACT(DAYS FROM (MAX(transaction_time) - MIN(transaction_time) + INTERVAL '1 day')),
        CURRENT_TIMESTAMP
    FROM transactions 
    WHERE user_id = p_user_id
    ON CONFLICT (user_id) DO UPDATE SET
        avg_transaction_amount = EXCLUDED.avg_transaction_amount,
        transaction_frequency = EXCLUDED.transaction_frequency,
        last_updated = EXCLUDED.last_updated;
END;
$$ LANGUAGE plpgsql;

-- Log successful initialization
INSERT INTO monitoring.system_metrics (metric_name, metric_value, metric_type)
VALUES ('database_initialized', 1, 'counter');

COMMIT;