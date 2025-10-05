-- Migration: Create audit_log table
-- Description: Table for storing audit trail of all system activities
-- Created: Initial migration

CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50) NOT NULL CHECK (event_category IN ('TRANSACTION', 'FRAUD_DETECTION', 'USER_ACTION', 'SYSTEM', 'MODEL', 'ALERT', 'FEEDBACK', 'AUTHENTICATION', 'AUTHORIZATION')),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    changes JSONB,
    metadata JSONB,
    request_id VARCHAR(255),
    correlation_id VARCHAR(255),
    trace_id VARCHAR(255),
    severity VARCHAR(20) NOT NULL DEFAULT 'INFO' CHECK (severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('SUCCESS', 'FAILURE', 'PENDING', 'PARTIAL')),
    error_message TEXT,
    error_code VARCHAR(50),
    processing_time_ms INTEGER,
    source_system VARCHAR(100) NOT NULL,
    source_component VARCHAR(100),
    environment VARCHAR(50) NOT NULL DEFAULT 'production',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance (audit logs are write-heavy, read-light)
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_category ON audit_log(event_category);
CREATE INDEX IF NOT EXISTS idx_audit_log_entity_type ON audit_log(entity_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_entity_id ON audit_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_severity ON audit_log(severity);
CREATE INDEX IF NOT EXISTS idx_audit_log_status ON audit_log(status);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_source_system ON audit_log(source_system);
CREATE INDEX IF NOT EXISTS idx_audit_log_environment ON audit_log(environment);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_audit_log_entity_created ON audit_log(entity_type, entity_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_created ON audit_log(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_category_created ON audit_log(event_category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_severity_created ON audit_log(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_status_created ON audit_log(status, created_at DESC);

-- Correlation and tracing indexes
CREATE INDEX IF NOT EXISTS idx_audit_log_request_id ON audit_log(request_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_correlation_id ON audit_log(correlation_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_trace_id ON audit_log(trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_session_id ON audit_log(session_id);

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_audit_log_old_values ON audit_log USING GIN (old_values);
CREATE INDEX IF NOT EXISTS idx_audit_log_new_values ON audit_log USING GIN (new_values);
CREATE INDEX IF NOT EXISTS idx_audit_log_changes ON audit_log USING GIN (changes);
CREATE INDEX IF NOT EXISTS idx_audit_log_metadata ON audit_log USING GIN (metadata);

-- Partial indexes for error tracking
CREATE INDEX IF NOT EXISTS idx_audit_log_errors ON audit_log(created_at DESC) WHERE status = 'FAILURE';
CREATE INDEX IF NOT EXISTS idx_audit_log_critical ON audit_log(created_at DESC) WHERE severity = 'CRITICAL';

-- Function to create audit log entry
CREATE OR REPLACE FUNCTION create_audit_log(
    p_event_type VARCHAR,
    p_event_category VARCHAR,
    p_entity_type VARCHAR,
    p_entity_id VARCHAR,
    p_action VARCHAR,
    p_user_id VARCHAR DEFAULT NULL,
    p_old_values JSONB DEFAULT NULL,
    p_new_values JSONB DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL,
    p_severity VARCHAR DEFAULT 'INFO',
    p_status VARCHAR DEFAULT 'SUCCESS',
    p_source_system VARCHAR DEFAULT 'fraud_detection',
    p_source_component VARCHAR DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    audit_id UUID;
    event_uuid VARCHAR;
BEGIN
    -- Generate unique event ID
    event_uuid := 'evt_' || EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::BIGINT || '_' || SUBSTRING(gen_random_uuid()::TEXT, 1, 8);
    
    -- Insert audit log entry
    INSERT INTO audit_log (
        event_id,
        event_type,
        event_category,
        entity_type,
        entity_id,
        user_id,
        action,
        old_values,
        new_values,
        changes,
        metadata,
        severity,
        status,
        source_system,
        source_component
    ) VALUES (
        event_uuid,
        p_event_type,
        p_event_category,
        p_entity_type,
        p_entity_id,
        p_user_id,
        p_action,
        p_old_values,
        p_new_values,
        CASE 
            WHEN p_old_values IS NOT NULL AND p_new_values IS NOT NULL THEN
                jsonb_diff(p_old_values, p_new_values)
            ELSE NULL
        END,
        p_metadata,
        p_severity,
        p_status,
        p_source_system,
        p_source_component
    ) RETURNING id INTO audit_id;
    
    RETURN audit_id;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate JSONB differences
CREATE OR REPLACE FUNCTION jsonb_diff(old_data JSONB, new_data JSONB)
RETURNS JSONB AS $$
DECLARE
    result JSONB := '{}'::JSONB;
    key TEXT;
    old_val JSONB;
    new_val JSONB;
BEGIN
    -- Find changed and new keys
    FOR key IN SELECT jsonb_object_keys(new_data) LOOP
        new_val := new_data -> key;
        old_val := old_data -> key;
        
        IF old_val IS NULL OR old_val != new_val THEN
            result := result || jsonb_build_object(
                key, 
                jsonb_build_object(
                    'old', old_val,
                    'new', new_val
                )
            );
        END IF;
    END LOOP;
    
    -- Find deleted keys
    FOR key IN SELECT jsonb_object_keys(old_data) LOOP
        IF new_data -> key IS NULL THEN
            result := result || jsonb_build_object(
                key,
                jsonb_build_object(
                    'old', old_data -> key,
                    'new', NULL
                )
            );
        END IF;
    END LOOP;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to get audit trail for an entity
CREATE OR REPLACE FUNCTION get_audit_trail(
    p_entity_type VARCHAR,
    p_entity_id VARCHAR,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    event_id VARCHAR,
    event_type VARCHAR,
    action VARCHAR,
    user_id VARCHAR,
    changes JSONB,
    metadata JSONB,
    severity VARCHAR,
    status VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        al.event_id,
        al.event_type,
        al.action,
        al.user_id,
        al.changes,
        al.metadata,
        al.severity,
        al.status,
        al.created_at
    FROM audit_log al
    WHERE al.entity_type = p_entity_type
    AND al.entity_id = p_entity_id
    ORDER BY al.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Partitioning setup for large audit logs (optional, for high-volume systems)
-- This creates monthly partitions for the audit_log table
-- Uncomment if you expect high audit log volume

/*
-- Enable partitioning extension
CREATE EXTENSION IF NOT EXISTS pg_partman;

-- Convert to partitioned table
ALTER TABLE audit_log RENAME TO audit_log_template;

CREATE TABLE audit_log (
    LIKE audit_log_template INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create initial partitions
SELECT partman.create_parent(
    p_parent_table => 'public.audit_log',
    p_control => 'created_at',
    p_type => 'range',
    p_interval => 'monthly',
    p_premake => 2
);
*/

-- Comments
COMMENT ON TABLE audit_log IS 'Comprehensive audit trail for all system activities';
COMMENT ON COLUMN audit_log.event_id IS 'Unique identifier for the audit event';
COMMENT ON COLUMN audit_log.event_type IS 'Type of event that occurred';
COMMENT ON COLUMN audit_log.event_category IS 'Category classification of the event';
COMMENT ON COLUMN audit_log.entity_type IS 'Type of entity that was affected';
COMMENT ON COLUMN audit_log.entity_id IS 'Identifier of the affected entity';
COMMENT ON COLUMN audit_log.action IS 'Action that was performed';
COMMENT ON COLUMN audit_log.old_values IS 'JSON object with values before the change';
COMMENT ON COLUMN audit_log.new_values IS 'JSON object with values after the change';
COMMENT ON COLUMN audit_log.changes IS 'JSON object with calculated differences';
COMMENT ON COLUMN audit_log.correlation_id IS 'Identifier to correlate related events';
COMMENT ON COLUMN audit_log.trace_id IS 'Distributed tracing identifier';