#!/usr/bin/env python3
"""Data loading utilities for fraud detection model training.

This module provides comprehensive data loading capabilities including:
- Loading data from various sources (CSV, Parquet, databases)
- Data validation and quality checks
- Data preprocessing and cleaning
- Synthetic data generation for testing
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import psycopg2
from faker import Faker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Data loading and preprocessing utilities."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize DataLoader.
        
        Args:
            db_url: Database connection URL for loading from database
        """
        self.db_url = db_url
        self.engine = None
        
        if db_url:
            try:
                self.engine = create_engine(db_url)
                logger.info("Database connection established")
            except Exception as e:
                logger.warning(f"Failed to connect to database: {e}")
    
    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """Load training data from file or database.
        
        Args:
            data_path: Path to data file or SQL query
            
        Returns:
            DataFrame with transaction data
        """
        logger.info(f"Loading training data from: {data_path}")
        
        if data_path.endswith('.csv'):
            return self._load_csv(data_path)
        elif data_path.endswith('.parquet'):
            return self._load_parquet(data_path)
        elif data_path.startswith('SELECT') or data_path.startswith('select'):
            return self._load_from_database(data_path)
        else:
            # Try to load as CSV by default
            return self._load_csv(data_path)
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from CSV")
            return self._validate_and_clean_data(df)
        except FileNotFoundError:
            logger.warning(f"CSV file not found: {file_path}")
            logger.info("Generating synthetic data for training...")
            return self.generate_synthetic_data()
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def _load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load data from Parquet file."""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} records from Parquet")
            return self._validate_and_clean_data(df)
        except FileNotFoundError:
            logger.warning(f"Parquet file not found: {file_path}")
            logger.info("Generating synthetic data for training...")
            return self.generate_synthetic_data()
        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise
    
    def _load_from_database(self, query: str) -> pd.DataFrame:
        """Load data from database using SQL query."""
        if not self.engine:
            raise ValueError("Database connection not available")
        
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} records from database")
            return self._validate_and_clean_data(df)
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            raise
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean loaded data."""
        logger.info("Validating and cleaning data...")
        
        # Check required columns
        required_columns = [
            'transaction_id', 'user_id', 'merchant_id', 'amount',
            'timestamp', 'is_fraud'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            # Try to map common column variations
            df = self._map_column_names(df)
        
        # Data type conversions
        df = self._convert_data_types(df)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['transaction_id'])
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate transactions")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Data quality checks
        self._perform_quality_checks(df)
        
        logger.info(f"Data validation completed. Final dataset: {len(df)} records")
        return df
    
    def _map_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column name variations to standard names."""
        column_mapping = {
            'txn_id': 'transaction_id',
            'trans_id': 'transaction_id',
            'id': 'transaction_id',
            'customer_id': 'user_id',
            'client_id': 'user_id',
            'shop_id': 'merchant_id',
            'store_id': 'merchant_id',
            'vendor_id': 'merchant_id',
            'transaction_amount': 'amount',
            'amt': 'amount',
            'value': 'amount',
            'transaction_time': 'timestamp',
            'trans_time': 'timestamp',
            'datetime': 'timestamp',
            'date': 'timestamp',
            'fraud': 'is_fraud',
            'fraud_flag': 'is_fraud',
            'is_fraudulent': 'is_fraud',
            'fraudulent': 'is_fraud'
        }
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        # Convert fraud indicators to boolean
        if 'is_fraud' in df.columns:
            df['is_fraud'] = df['is_fraud'].astype(bool)
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert categorical columns
        categorical_columns = ['user_id', 'merchant_id', 'transaction_id']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Remove rows with missing critical information
        critical_columns = ['transaction_id', 'amount', 'is_fraud']
        for col in critical_columns:
            if col in df.columns:
                initial_count = len(df)
                df = df.dropna(subset=[col])
                if len(df) < initial_count:
                    logger.info(f"Removed {initial_count - len(df)} rows with missing {col}")
        
        # Fill missing values for other columns
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].fillna('unknown_user')
        
        if 'merchant_id' in df.columns:
            df['merchant_id'] = df['merchant_id'].fillna('unknown_merchant')
        
        if 'timestamp' in df.columns:
            # Fill missing timestamps with a default value
            df['timestamp'] = df['timestamp'].fillna(datetime.now())
        
        return df
    
    def _perform_quality_checks(self, df: pd.DataFrame) -> None:
        """Perform data quality checks and log warnings."""
        # Check for negative amounts
        if 'amount' in df.columns:
            negative_amounts = (df['amount'] < 0).sum()
            if negative_amounts > 0:
                logger.warning(f"Found {negative_amounts} transactions with negative amounts")
        
        # Check fraud rate
        if 'is_fraud' in df.columns:
            fraud_rate = df['is_fraud'].mean()
            logger.info(f"Fraud rate: {fraud_rate:.4f}")
            
            if fraud_rate < 0.001:
                logger.warning("Very low fraud rate detected (<0.1%)")
            elif fraud_rate > 0.1:
                logger.warning("Very high fraud rate detected (>10%)")
        
        # Check for duplicate transaction IDs
        if 'transaction_id' in df.columns:
            duplicate_ids = df['transaction_id'].duplicated().sum()
            if duplicate_ids > 0:
                logger.warning(f"Found {duplicate_ids} duplicate transaction IDs")
        
        # Check timestamp range
        if 'timestamp' in df.columns:
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            logger.info(f"Date range: {min_date} to {max_date}")
            
            # Check for future dates
            future_dates = (df['timestamp'] > datetime.now()).sum()
            if future_dates > 0:
                logger.warning(f"Found {future_dates} transactions with future dates")
    
    def generate_synthetic_data(self, n_samples: int = 10000, 
                              fraud_rate: float = 0.02) -> pd.DataFrame:
        """Generate synthetic transaction data for training.
        
        Args:
            n_samples: Number of samples to generate
            fraud_rate: Proportion of fraudulent transactions
            
        Returns:
            DataFrame with synthetic transaction data
        """
        logger.info(f"Generating {n_samples} synthetic transactions...")
        
        fake = Faker()
        np.random.seed(42)
        
        # Generate base transaction data
        data = []
        
        for i in range(n_samples):
            # Determine if transaction is fraudulent
            is_fraud = np.random.random() < fraud_rate
            
            # Generate transaction details
            transaction = {
                'transaction_id': f"txn_{i:08d}",
                'user_id': f"user_{np.random.randint(1, 5000):06d}",
                'merchant_id': f"merchant_{np.random.randint(1, 1000):04d}",
                'timestamp': fake.date_time_between(
                    start_date='-2y', end_date='now'
                ),
                'is_fraud': is_fraud
            }
            
            # Generate amount based on fraud status
            if is_fraud:
                # Fraudulent transactions tend to be higher amounts
                if np.random.random() < 0.3:
                    # High-value fraud
                    amount = np.random.lognormal(mean=8, sigma=1)
                else:
                    # Regular fraud
                    amount = np.random.lognormal(mean=5, sigma=1.5)
            else:
                # Normal transactions
                amount = np.random.lognormal(mean=4, sigma=1.2)
            
            transaction['amount'] = round(max(amount, 1.0), 2)
            
            # Add additional features
            transaction.update({
                'currency': np.random.choice(['USD', 'EUR', 'GBP'], p=[0.7, 0.2, 0.1]),
                'merchant_category': np.random.choice([
                    'grocery', 'gas_station', 'restaurant', 'retail', 
                    'online', 'atm', 'pharmacy', 'entertainment'
                ]),
                'payment_method': np.random.choice([
                    'credit_card', 'debit_card', 'mobile_payment', 'bank_transfer'
                ], p=[0.5, 0.3, 0.15, 0.05]),
                'country': fake.country_code(),
                'city': fake.city(),
                'device_id': f"device_{np.random.randint(1, 10000):05d}",
                'ip_address': fake.ipv4(),
                'user_agent': fake.user_agent()
            })
            
            # Add fraud-specific patterns
            if is_fraud:
                # Fraudulent transactions more likely at night
                if np.random.random() < 0.4:
                    night_hour = np.random.randint(0, 6)
                    transaction['timestamp'] = transaction['timestamp'].replace(
                        hour=night_hour
                    )
                
                # More likely to be online or ATM
                if np.random.random() < 0.6:
                    transaction['merchant_category'] = np.random.choice(
                        ['online', 'atm']
                    )
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated synthetic dataset with {len(df)} transactions")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.4f}")
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_path: str, 
                  format: str = 'csv') -> None:
        """Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            format: Output format ('csv' or 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to {output_path}")
    
    def load_batch_data(self, batch_size: int = 1000, 
                       data_path: Optional[str] = None) -> pd.DataFrame:
        """Load data in batches for memory-efficient processing.
        
        Args:
            batch_size: Number of records per batch
            data_path: Path to data file
            
        Yields:
            DataFrame batches
        """
        if data_path and data_path.endswith('.csv'):
            # Use chunked reading for large CSV files
            for chunk in pd.read_csv(data_path, chunksize=batch_size):
                yield self._validate_and_clean_data(chunk)
        else:
            # Load full dataset and yield batches
            df = self.load_training_data(data_path or "data/transactions.csv")
            
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i + batch_size]
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive statistics about the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_transactions': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
            },
            'fraud_statistics': {
                'total_fraud': df['is_fraud'].sum() if 'is_fraud' in df.columns else 0,
                'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else 0
            },
            'amount_statistics': {
                'mean': df['amount'].mean() if 'amount' in df.columns else 0,
                'median': df['amount'].median() if 'amount' in df.columns else 0,
                'std': df['amount'].std() if 'amount' in df.columns else 0,
                'min': df['amount'].min() if 'amount' in df.columns else 0,
                'max': df['amount'].max() if 'amount' in df.columns else 0
            },
            'unique_counts': {
                'users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
                'merchants': df['merchant_id'].nunique() if 'merchant_id' in df.columns else 0
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        return stats


def main():
    """Example usage of DataLoader."""
    # Initialize data loader
    loader = DataLoader()
    
    # Generate and save synthetic data
    df = loader.generate_synthetic_data(n_samples=5000)
    
    # Save to file
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    loader.save_data(df, "data/synthetic_transactions.csv")
    
    # Load and validate
    loaded_df = loader.load_training_data("data/synthetic_transactions.csv")
    
    # Print statistics
    stats = loader.get_data_statistics(loaded_df)
    print("Dataset Statistics:")
    print(f"Total transactions: {stats['total_transactions']}")
    print(f"Fraud rate: {stats['fraud_statistics']['fraud_rate']:.4f}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")


if __name__ == "__main__":
    main()