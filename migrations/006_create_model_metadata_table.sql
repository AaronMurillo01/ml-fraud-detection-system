-- Migration: Create model_metadata table
-- Description: Table for storing ML model metadata, versions, and performance metrics
-- Created: Initial migration

CREATE TABLE IF NOT EXISTS model_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('XGBOOST', 'RANDOM_FOREST', 'NEURAL_NETWORK', 'LOGISTIC_REGRESSION', 'ENSEMBLE')),
    model_status VARCHAR(20) NOT NULL DEFAULT 'TRAINING' CHECK (model_status IN ('TRAINING', 'TESTING', 'ACTIVE', 'DEPRECATED', 'ARCHIVED')),
    model_path VARCHAR(500),
    model_size_bytes BIGINT,
    training_dataset_info JSONB,
    feature_columns TEXT[],
    target_column VARCHAR(100),
    hyperparameters JSONB,
    training_metrics JSONB,
    validation_metrics JSONB,
    test_metrics JSONB,
    performance_benchmarks JSONB,
    feature_importance JSONB,
    model_artifacts JSONB,
    deployment_config JSONB,
    threshold_config JSONB,
    training_start_time TIMESTAMP WITH TIME ZONE,
    training_end_time TIMESTAMP WITH TIME ZONE,
    training_duration_minutes INTEGER,
    deployed_at TIMESTAMP WITH TIME ZONE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(255),
    notes TEXT,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique combination of model_name and model_version
    CONSTRAINT unique_model_name_version UNIQUE (model_name, model_version)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_model_metadata_name ON model_metadata(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metadata_version ON model_metadata(model_version);
CREATE INDEX IF NOT EXISTS idx_model_metadata_type ON model_metadata(model_type);
CREATE INDEX IF NOT EXISTS idx_model_metadata_status ON model_metadata(model_status);
CREATE INDEX IF NOT EXISTS idx_model_metadata_deployed_at ON model_metadata(deployed_at);
CREATE INDEX IF NOT EXISTS idx_model_metadata_created_by ON model_metadata(created_by);
CREATE INDEX IF NOT EXISTS idx_model_metadata_created_at ON model_metadata(created_at);

-- Composite indexes
CREATE INDEX IF NOT EXISTS idx_model_metadata_name_status ON model_metadata(model_name, model_status);
CREATE INDEX IF NOT EXISTS idx_model_metadata_type_status ON model_metadata(model_type, model_status);

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_model_metadata_training_dataset ON model_metadata USING GIN (training_dataset_info);
CREATE INDEX IF NOT EXISTS idx_model_metadata_hyperparameters ON model_metadata USING GIN (hyperparameters);
CREATE INDEX IF NOT EXISTS idx_model_metadata_training_metrics ON model_metadata USING GIN (training_metrics);
CREATE INDEX IF NOT EXISTS idx_model_metadata_validation_metrics ON model_metadata USING GIN (validation_metrics);
CREATE INDEX IF NOT EXISTS idx_model_metadata_test_metrics ON model_metadata USING GIN (test_metrics);
CREATE INDEX IF NOT EXISTS idx_model_metadata_feature_importance ON model_metadata USING GIN (feature_importance);
CREATE INDEX IF NOT EXISTS idx_model_metadata_deployment_config ON model_metadata USING GIN (deployment_config);
CREATE INDEX IF NOT EXISTS idx_model_metadata_threshold_config ON model_metadata USING GIN (threshold_config);

-- GIN indexes for array columns
CREATE INDEX IF NOT EXISTS idx_model_metadata_feature_columns ON model_metadata USING GIN (feature_columns);
CREATE INDEX IF NOT EXISTS idx_model_metadata_tags ON model_metadata USING GIN (tags);

-- Trigger to update updated_at timestamp
CREATE TRIGGER update_model_metadata_updated_at
    BEFORE UPDATE ON model_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate training duration
CREATE OR REPLACE FUNCTION calculate_training_duration()
RETURNS TRIGGER AS $$
BEGIN
    -- Calculate training duration when both start and end times are set
    IF NEW.training_start_time IS NOT NULL AND NEW.training_end_time IS NOT NULL THEN
        NEW.training_duration_minutes = EXTRACT(EPOCH FROM (NEW.training_end_time - NEW.training_start_time)) / 60;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically calculate training duration
CREATE TRIGGER calculate_model_training_duration
    BEFORE INSERT OR UPDATE ON model_metadata
    FOR EACH ROW
    EXECUTE FUNCTION calculate_training_duration();

-- Function to get active model for a given model name
CREATE OR REPLACE FUNCTION get_active_model(p_model_name VARCHAR)
RETURNS TABLE (
    model_name VARCHAR,
    model_version VARCHAR,
    model_path VARCHAR,
    deployment_config JSONB,
    threshold_config JSONB,
    deployed_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        mm.model_name,
        mm.model_version,
        mm.model_path,
        mm.deployment_config,
        mm.threshold_config,
        mm.deployed_at
    FROM model_metadata mm
    WHERE mm.model_name = p_model_name
    AND mm.model_status = 'ACTIVE'
    ORDER BY mm.deployed_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to get model performance comparison
CREATE OR REPLACE FUNCTION compare_model_performance(p_model_name VARCHAR)
RETURNS TABLE (
    model_version VARCHAR,
    model_status VARCHAR,
    training_accuracy DECIMAL,
    validation_accuracy DECIMAL,
    test_accuracy DECIMAL,
    training_auc DECIMAL,
    validation_auc DECIMAL,
    test_auc DECIMAL,
    deployed_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        mm.model_version,
        mm.model_status,
        (mm.training_metrics->>'accuracy')::DECIMAL as training_accuracy,
        (mm.validation_metrics->>'accuracy')::DECIMAL as validation_accuracy,
        (mm.test_metrics->>'accuracy')::DECIMAL as test_accuracy,
        (mm.training_metrics->>'auc')::DECIMAL as training_auc,
        (mm.validation_metrics->>'auc')::DECIMAL as validation_auc,
        (mm.test_metrics->>'auc')::DECIMAL as test_auc,
        mm.deployed_at
    FROM model_metadata mm
    WHERE mm.model_name = p_model_name
    ORDER BY mm.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Comments
COMMENT ON TABLE model_metadata IS 'Metadata and performance metrics for ML models';
COMMENT ON COLUMN model_metadata.model_name IS 'Name of the ML model';
COMMENT ON COLUMN model_metadata.model_version IS 'Version identifier for the model';
COMMENT ON COLUMN model_metadata.model_type IS 'Type of machine learning algorithm used';
COMMENT ON COLUMN model_metadata.model_status IS 'Current status of the model in the lifecycle';
COMMENT ON COLUMN model_metadata.model_path IS 'File system path to the serialized model';
COMMENT ON COLUMN model_metadata.training_dataset_info IS 'JSON object with training dataset information';
COMMENT ON COLUMN model_metadata.feature_columns IS 'Array of feature column names used by the model';
COMMENT ON COLUMN model_metadata.hyperparameters IS 'JSON object with model hyperparameters';
COMMENT ON COLUMN model_metadata.training_metrics IS 'JSON object with training performance metrics';
COMMENT ON COLUMN model_metadata.validation_metrics IS 'JSON object with validation performance metrics';
COMMENT ON COLUMN model_metadata.test_metrics IS 'JSON object with test performance metrics';
COMMENT ON COLUMN model_metadata.feature_importance IS 'JSON object with feature importance scores';
COMMENT ON COLUMN model_metadata.threshold_config IS 'JSON object with decision thresholds configuration';