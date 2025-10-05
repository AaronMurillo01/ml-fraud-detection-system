-- Migration: Create user_profiles table
-- Description: Table for storing user behavior profiles and patterns
-- Created: Initial migration

CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    profile_data JSONB NOT NULL,
    avg_transaction_amount DECIMAL(15, 2),
    transaction_frequency INTEGER DEFAULT 0,
    preferred_merchants TEXT[],
    preferred_categories TEXT[],
    typical_locations JSONB,
    spending_patterns JSONB,
    risk_factors JSONB,
    account_age_days INTEGER,
    last_transaction_date TIMESTAMP WITH TIME ZONE,
    total_transactions INTEGER DEFAULT 0,
    total_amount DECIMAL(15, 2) DEFAULT 0,
    fraud_history_count INTEGER DEFAULT 0,
    last_fraud_date TIMESTAMP WITH TIME ZONE,
    profile_version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_avg_amount ON user_profiles(avg_transaction_amount);
CREATE INDEX IF NOT EXISTS idx_user_profiles_frequency ON user_profiles(transaction_frequency);
CREATE INDEX IF NOT EXISTS idx_user_profiles_account_age ON user_profiles(account_age_days);
CREATE INDEX IF NOT EXISTS idx_user_profiles_fraud_count ON user_profiles(fraud_history_count);
CREATE INDEX IF NOT EXISTS idx_user_profiles_last_transaction ON user_profiles(last_transaction_date);
CREATE INDEX IF NOT EXISTS idx_user_profiles_updated_at ON user_profiles(updated_at);

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_user_profiles_profile_data ON user_profiles USING GIN (profile_data);
CREATE INDEX IF NOT EXISTS idx_user_profiles_typical_locations ON user_profiles USING GIN (typical_locations);
CREATE INDEX IF NOT EXISTS idx_user_profiles_spending_patterns ON user_profiles USING GIN (spending_patterns);
CREATE INDEX IF NOT EXISTS idx_user_profiles_risk_factors ON user_profiles USING GIN (risk_factors);

-- GIN index for array columns
CREATE INDEX IF NOT EXISTS idx_user_profiles_preferred_merchants ON user_profiles USING GIN (preferred_merchants);
CREATE INDEX IF NOT EXISTS idx_user_profiles_preferred_categories ON user_profiles USING GIN (preferred_categories);

-- Trigger to update updated_at timestamp
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to update profile statistics
CREATE OR REPLACE FUNCTION update_user_profile_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update user profile when new transaction is added
    INSERT INTO user_profiles (user_id, total_transactions, total_amount, last_transaction_date)
    VALUES (NEW.user_id, 1, NEW.amount, NEW.timestamp)
    ON CONFLICT (user_id) DO UPDATE SET
        total_transactions = user_profiles.total_transactions + 1,
        total_amount = user_profiles.total_amount + NEW.amount,
        last_transaction_date = GREATEST(user_profiles.last_transaction_date, NEW.timestamp),
        avg_transaction_amount = (user_profiles.total_amount + NEW.amount) / (user_profiles.total_transactions + 1),
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to automatically update user profiles on new transactions
CREATE TRIGGER update_user_profile_on_transaction
    AFTER INSERT ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_user_profile_stats();

-- Comments
COMMENT ON TABLE user_profiles IS 'User behavior profiles and spending patterns';
COMMENT ON COLUMN user_profiles.profile_data IS 'JSON object containing detailed user profile information';
COMMENT ON COLUMN user_profiles.avg_transaction_amount IS 'Average transaction amount for the user';
COMMENT ON COLUMN user_profiles.transaction_frequency IS 'Number of transactions per day (average)';
COMMENT ON COLUMN user_profiles.preferred_merchants IS 'Array of frequently used merchant IDs';
COMMENT ON COLUMN user_profiles.preferred_categories IS 'Array of preferred merchant categories';
COMMENT ON COLUMN user_profiles.typical_locations IS 'JSON object with typical transaction locations';
COMMENT ON COLUMN user_profiles.spending_patterns IS 'JSON object with spending behavior patterns';
COMMENT ON COLUMN user_profiles.risk_factors IS 'JSON object with identified risk factors';