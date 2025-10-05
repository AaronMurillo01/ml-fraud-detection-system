-- Migration: Create fraud_scores table
-- Description: Table for storing ML model fraud detection scores
-- Created: Initial migration

CREATE TABLE IF NOT EXISTS fraud_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    fraud_score DECIMAL(5, 4) NOT NULL CHECK (fraud_score >= 0 AND fraud_score <= 1),
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    confidence_score DECIMAL(5, 4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    feature_importance JSONB,
    model_features JSONB,
    processing_time_ms INTEGER,
    threshold_used DECIMAL(5, 4),
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('APPROVE', 'DECLINE', 'REVIEW', 'PENDING')),
    decision_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Foreign key constraint
ALTER TABLE fraud_scores 
ADD CONSTRAINT fk_fraud_scores_transaction 
FOREIGN KEY (transaction_id) 
REFERENCES transactions(transaction_id) 
ON DELETE CASCADE;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_fraud_scores_transaction_id ON fraud_scores(transaction_id);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_model_name ON fraud_scores(model_name);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_fraud_score ON fraud_scores(fraud_score DESC);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_risk_level ON fraud_scores(risk_level);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_decision ON fraud_scores(decision);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_created_at ON fraud_scores(created_at);

-- Composite indexes
CREATE INDEX IF NOT EXISTS idx_fraud_scores_model_version ON fraud_scores(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_score_decision ON fraud_scores(fraud_score DESC, decision);

-- Trigger to update updated_at timestamp
CREATE TRIGGER update_fraud_scores_updated_at
    BEFORE UPDATE ON fraud_scores
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE fraud_scores IS 'ML model fraud detection scores and decisions';
COMMENT ON COLUMN fraud_scores.fraud_score IS 'Probability of fraud (0.0 to 1.0)';
COMMENT ON COLUMN fraud_scores.risk_level IS 'Categorical risk assessment';
COMMENT ON COLUMN fraud_scores.confidence_score IS 'Model confidence in the prediction';
COMMENT ON COLUMN fraud_scores.feature_importance IS 'JSON object with feature importance scores';
COMMENT ON COLUMN fraud_scores.model_features IS 'JSON object with features used by the model';
COMMENT ON COLUMN fraud_scores.processing_time_ms IS 'Time taken to generate the score in milliseconds';