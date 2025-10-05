-- Migration: Create fraud_alerts table
-- Description: Table for storing fraud alert notifications and their status
-- Created: Initial migration

CREATE TABLE IF NOT EXISTS fraud_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    transaction_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN ('HIGH_RISK', 'SUSPICIOUS_PATTERN', 'VELOCITY_CHECK', 'LOCATION_ANOMALY', 'AMOUNT_ANOMALY', 'MERCHANT_RISK', 'DEVICE_RISK')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    fraud_score DECIMAL(5, 4) NOT NULL,
    alert_message TEXT NOT NULL,
    alert_details JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'INVESTIGATING', 'RESOLVED', 'FALSE_POSITIVE', 'CONFIRMED_FRAUD')),
    assigned_to VARCHAR(255),
    resolution_notes TEXT,
    auto_resolved BOOLEAN DEFAULT FALSE,
    escalated BOOLEAN DEFAULT FALSE,
    escalation_reason TEXT,
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channels TEXT[],
    response_required BOOLEAN DEFAULT TRUE,
    response_deadline TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Foreign key constraints
ALTER TABLE fraud_alerts 
ADD CONSTRAINT fk_fraud_alerts_transaction 
FOREIGN KEY (transaction_id) 
REFERENCES transactions(transaction_id) 
ON DELETE CASCADE;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_transaction_id ON fraud_alerts(transaction_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_user_id ON fraud_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_alert_type ON fraud_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_severity ON fraud_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_fraud_score ON fraud_alerts(fraud_score DESC);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_created_at ON fraud_alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_assigned_to ON fraud_alerts(assigned_to);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_escalated ON fraud_alerts(escalated);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_response_deadline ON fraud_alerts(response_deadline);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status_severity ON fraud_alerts(status, severity);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_user_status ON fraud_alerts(user_id, status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_type_severity ON fraud_alerts(alert_type, severity);

-- GIN index for JSONB column
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_details ON fraud_alerts USING GIN (alert_details);

-- GIN index for array column
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_notification_channels ON fraud_alerts USING GIN (notification_channels);

-- Trigger to update updated_at timestamp
CREATE TRIGGER update_fraud_alerts_updated_at
    BEFORE UPDATE ON fraud_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to set resolved_at timestamp when status changes to resolved
CREATE OR REPLACE FUNCTION set_alert_resolved_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    -- Set resolved_at when status changes to a resolved state
    IF NEW.status IN ('RESOLVED', 'FALSE_POSITIVE', 'CONFIRMED_FRAUD') AND 
       OLD.status NOT IN ('RESOLVED', 'FALSE_POSITIVE', 'CONFIRMED_FRAUD') THEN
        NEW.resolved_at = CURRENT_TIMESTAMP;
    END IF;
    
    -- Clear resolved_at if status changes back to unresolved
    IF NEW.status IN ('PENDING', 'INVESTIGATING') AND 
       OLD.status IN ('RESOLVED', 'FALSE_POSITIVE', 'CONFIRMED_FRAUD') THEN
        NEW.resolved_at = NULL;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically set resolved timestamp
CREATE TRIGGER set_fraud_alert_resolved_timestamp
    BEFORE UPDATE ON fraud_alerts
    FOR EACH ROW
    EXECUTE FUNCTION set_alert_resolved_timestamp();

-- Comments
COMMENT ON TABLE fraud_alerts IS 'Fraud detection alerts and their investigation status';
COMMENT ON COLUMN fraud_alerts.alert_id IS 'Unique identifier for the alert';
COMMENT ON COLUMN fraud_alerts.alert_type IS 'Type of fraud pattern detected';
COMMENT ON COLUMN fraud_alerts.severity IS 'Alert severity level';
COMMENT ON COLUMN fraud_alerts.alert_details IS 'JSON object with detailed alert information';
COMMENT ON COLUMN fraud_alerts.status IS 'Current investigation status of the alert';
COMMENT ON COLUMN fraud_alerts.auto_resolved IS 'Whether the alert was automatically resolved';
COMMENT ON COLUMN fraud_alerts.escalated IS 'Whether the alert has been escalated';
COMMENT ON COLUMN fraud_alerts.response_deadline IS 'Deadline for responding to the alert';