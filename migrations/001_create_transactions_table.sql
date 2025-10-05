-- Migration: Create transactions table
-- Description: Core table for storing transaction data
-- Created: Initial migration

CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    merchant_id VARCHAR(255) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    transaction_type VARCHAR(50) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    card_type VARCHAR(50),
    card_last_four VARCHAR(4),
    merchant_category VARCHAR(100),
    merchant_name VARCHAR(255),
    location_country VARCHAR(3),
    location_city VARCHAR(100),
    location_latitude DECIMAL(10, 8),
    location_longitude DECIMAL(11, 8),
    device_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_payment_method ON transactions(payment_method);
CREATE INDEX IF NOT EXISTS idx_transactions_location_country ON transactions(location_country);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transactions_user_timestamp ON transactions(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_timestamp ON transactions(merchant_id, timestamp DESC);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE transactions IS 'Core transaction data for fraud detection';
COMMENT ON COLUMN transactions.transaction_id IS 'Unique identifier for the transaction';
COMMENT ON COLUMN transactions.amount IS 'Transaction amount in the specified currency';
COMMENT ON COLUMN transactions.timestamp IS 'When the transaction occurred';
COMMENT ON COLUMN transactions.ip_address IS 'IP address from which the transaction was initiated';