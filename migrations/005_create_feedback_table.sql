-- Migration: Create feedback table
-- Description: Table for storing user feedback on fraud detection decisions
-- Created: Initial migration

CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_id VARCHAR(255) UNIQUE NOT NULL,
    transaction_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    fraud_score_id UUID,
    alert_id VARCHAR(255),
    feedback_type VARCHAR(50) NOT NULL CHECK (feedback_type IN ('TRUE_POSITIVE', 'FALSE_POSITIVE', 'TRUE_NEGATIVE', 'FALSE_NEGATIVE', 'DISPUTED', 'CONFIRMED')),
    original_decision VARCHAR(20) NOT NULL CHECK (original_decision IN ('APPROVE', 'DECLINE', 'REVIEW', 'PENDING')),
    actual_outcome VARCHAR(20) NOT NULL CHECK (actual_outcome IN ('LEGITIMATE', 'FRAUDULENT', 'DISPUTED', 'UNKNOWN')),
    confidence_level INTEGER CHECK (confidence_level >= 1 AND confidence_level <= 5),
    feedback_source VARCHAR(50) NOT NULL CHECK (feedback_source IN ('USER', 'ANALYST', 'AUTOMATED', 'EXTERNAL', 'CHARGEBACK')),
    feedback_details JSONB,
    comments TEXT,
    evidence_provided BOOLEAN DEFAULT FALSE,
    evidence_details JSONB,
    impact_assessment JSONB,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    original_fraud_score DECIMAL(5, 4),
    processing_time_ms INTEGER,
    reviewer_id VARCHAR(255),
    review_notes TEXT,
    follow_up_required BOOLEAN DEFAULT FALSE,
    follow_up_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Foreign key constraints
ALTER TABLE feedback 
ADD CONSTRAINT fk_feedback_transaction 
FOREIGN KEY (transaction_id) 
REFERENCES transactions(transaction_id) 
ON DELETE CASCADE;

ALTER TABLE feedback 
ADD CONSTRAINT fk_feedback_fraud_score 
FOREIGN KEY (fraud_score_id) 
REFERENCES fraud_scores(id) 
ON DELETE SET NULL;

ALTER TABLE feedback 
ADD CONSTRAINT fk_feedback_alert 
FOREIGN KEY (alert_id) 
REFERENCES fraud_alerts(alert_id) 
ON DELETE SET NULL;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_feedback_transaction_id ON feedback(transaction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_fraud_score_id ON feedback(fraud_score_id);
CREATE INDEX IF NOT EXISTS idx_feedback_alert_id ON feedback(alert_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_original_decision ON feedback(original_decision);
CREATE INDEX IF NOT EXISTS idx_feedback_actual_outcome ON feedback(actual_outcome);
CREATE INDEX IF NOT EXISTS idx_feedback_source ON feedback(feedback_source);
CREATE INDEX IF NOT EXISTS idx_feedback_model_name ON feedback(model_name);
CREATE INDEX IF NOT EXISTS idx_feedback_reviewer_id ON feedback(reviewer_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_follow_up ON feedback(follow_up_required);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_feedback_type_outcome ON feedback(feedback_type, actual_outcome);
CREATE INDEX IF NOT EXISTS idx_feedback_model_version ON feedback(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_feedback_decision_outcome ON feedback(original_decision, actual_outcome);
CREATE INDEX IF NOT EXISTS idx_feedback_source_type ON feedback(feedback_source, feedback_type);

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_feedback_details ON feedback USING GIN (feedback_details);
CREATE INDEX IF NOT EXISTS idx_feedback_evidence_details ON feedback USING GIN (evidence_details);
CREATE INDEX IF NOT EXISTS idx_feedback_impact_assessment ON feedback USING GIN (impact_assessment);

-- Trigger to update updated_at timestamp
CREATE TRIGGER update_feedback_updated_at
    BEFORE UPDATE ON feedback
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate feedback metrics
CREATE OR REPLACE FUNCTION calculate_model_accuracy_metrics(p_model_name VARCHAR, p_model_version VARCHAR DEFAULT NULL)
RETURNS TABLE (
    model_name VARCHAR,
    model_version VARCHAR,
    total_feedback INTEGER,
    true_positives INTEGER,
    false_positives INTEGER,
    true_negatives INTEGER,
    false_negatives INTEGER,
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4)
) AS $$
BEGIN
    RETURN QUERY
    WITH feedback_stats AS (
        SELECT 
            f.model_name,
            f.model_version,
            COUNT(*) as total_feedback,
            COUNT(*) FILTER (WHERE f.feedback_type = 'TRUE_POSITIVE') as true_positives,
            COUNT(*) FILTER (WHERE f.feedback_type = 'FALSE_POSITIVE') as false_positives,
            COUNT(*) FILTER (WHERE f.feedback_type = 'TRUE_NEGATIVE') as true_negatives,
            COUNT(*) FILTER (WHERE f.feedback_type = 'FALSE_NEGATIVE') as false_negatives
        FROM feedback f
        WHERE f.model_name = p_model_name
        AND (p_model_version IS NULL OR f.model_version = p_model_version)
        GROUP BY f.model_name, f.model_version
    )
    SELECT 
        fs.model_name,
        fs.model_version,
        fs.total_feedback,
        fs.true_positives,
        fs.false_positives,
        fs.true_negatives,
        fs.false_negatives,
        CASE WHEN fs.total_feedback > 0 THEN
            ROUND((fs.true_positives + fs.true_negatives)::DECIMAL / fs.total_feedback, 4)
        ELSE 0 END as accuracy,
        CASE WHEN (fs.true_positives + fs.false_positives) > 0 THEN
            ROUND(fs.true_positives::DECIMAL / (fs.true_positives + fs.false_positives), 4)
        ELSE 0 END as precision_score,
        CASE WHEN (fs.true_positives + fs.false_negatives) > 0 THEN
            ROUND(fs.true_positives::DECIMAL / (fs.true_positives + fs.false_negatives), 4)
        ELSE 0 END as recall_score,
        CASE WHEN (fs.true_positives + fs.false_positives + fs.false_negatives) > 0 THEN
            ROUND(2.0 * fs.true_positives::DECIMAL / (2 * fs.true_positives + fs.false_positives + fs.false_negatives), 4)
        ELSE 0 END as f1_score
    FROM feedback_stats fs;
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON TABLE feedback IS 'User and analyst feedback on fraud detection decisions';
COMMENT ON COLUMN feedback.feedback_type IS 'Classification of the feedback (TP, FP, TN, FN)';
COMMENT ON COLUMN feedback.original_decision IS 'The original decision made by the fraud detection system';
COMMENT ON COLUMN feedback.actual_outcome IS 'The actual outcome as determined by investigation';
COMMENT ON COLUMN feedback.confidence_level IS 'Confidence level of the feedback provider (1-5 scale)';
COMMENT ON COLUMN feedback.feedback_source IS 'Source of the feedback';
COMMENT ON COLUMN feedback.evidence_provided IS 'Whether supporting evidence was provided';
COMMENT ON COLUMN feedback.impact_assessment IS 'JSON object with impact analysis of the decision';