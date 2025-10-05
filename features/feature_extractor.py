"""Feature extraction module for fraud detection ML models.

This module provides comprehensive feature extraction capabilities including:
- Time-based aggregations
- Statistical features
- Behavioral patterns
- Categorical encodings
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler, LabelEncoder

from service.models import EnrichedTransaction

logger = logging.getLogger(__name__)


class AggregationType(str, Enum):
    """Types of aggregations for feature extraction."""
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    QUANTILE_25 = "q25"
    QUANTILE_75 = "q75"
    UNIQUE_COUNT = "nunique"
    MODE = "mode"


class FeatureType(str, Enum):
    """Types of features to extract."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"


@dataclass
class TimeWindowConfig:
    """Configuration for time-based feature windows."""
    window_size: int  # in minutes
    aggregations: List[AggregationType]
    features: List[str]  # feature names to aggregate
    name: str  # window name for feature naming


class FeatureConfig(BaseModel):
    """Configuration for feature extraction."""
    
    # Time windows for aggregation
    time_windows: List[TimeWindowConfig] = Field(default_factory=lambda: [
        TimeWindowConfig(5, [AggregationType.COUNT, AggregationType.SUM], ["amount"], "5m"),
        TimeWindowConfig(15, [AggregationType.COUNT, AggregationType.SUM, AggregationType.MEAN], ["amount"], "15m"),
        TimeWindowConfig(60, [AggregationType.COUNT, AggregationType.SUM, AggregationType.MEAN, AggregationType.STD], ["amount"], "1h"),
        TimeWindowConfig(1440, [AggregationType.COUNT, AggregationType.SUM, AggregationType.MEAN, AggregationType.STD], ["amount"], "1d"),
    ])
    
    # Categorical features to encode
    categorical_features: List[str] = Field(default_factory=lambda: [
        "merchant_category", "transaction_country", "risk_level", "amount_bin"
    ])
    
    # Numerical features to normalize
    numerical_features: List[str] = Field(default_factory=lambda: [
        "amount", "amount_log", "hour_of_day", "day_of_week", "merchant_risk_score"
    ])
    
    # Boolean features
    boolean_features: List[str] = Field(default_factory=lambda: [
        "is_weekend", "is_business_hours", "is_high_risk_country", "is_round_amount"
    ])
    
    # Statistical features to compute
    statistical_features: Dict[str, List[str]] = Field(default_factory=lambda: {
        "amount": ["zscore", "percentile_rank", "deviation_from_mean"],
        "hour_of_day": ["zscore", "deviation_from_mean"],
        "merchant_risk_score": ["zscore", "percentile_rank"]
    })
    
    # Feature scaling
    enable_scaling: bool = True
    scaling_method: str = "standard"  # "standard", "minmax", "robust"
    
    # Feature selection
    enable_feature_selection: bool = False
    max_features: Optional[int] = None
    
    # Missing value handling
    fill_missing_numerical: float = 0.0
    fill_missing_categorical: str = "unknown"
    
    class Config:
        use_enum_values = True


class FeatureExtractor:
    """Extracts features from enriched transactions for ML models."""
    
    def __init__(self, config: FeatureConfig):
        """Initialize feature extractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
        
        logger.info("Feature extractor initialized")
    
    def extract_features(self, 
                        transaction: EnrichedTransaction, 
                        historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Extract features from an enriched transaction.
        
        Args:
            transaction: Enriched transaction
            historical_data: Historical transaction data for time-based features
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Extract base features from transaction
            base_features = self._extract_base_features(transaction)
            features.update(base_features)
            
            # Extract time-based aggregation features
            if historical_data is not None:
                time_features = self._extract_time_features(transaction, historical_data)
                features.update(time_features)
            
            # Extract statistical features
            statistical_features = self._extract_statistical_features(transaction, historical_data)
            features.update(statistical_features)
            
            # Extract interaction features
            interaction_features = self._extract_interaction_features(features)
            features.update(interaction_features)
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Apply feature transformations if fitted
            if self.is_fitted:
                features = self._apply_transformations(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features()
    
    def _extract_base_features(self, transaction: EnrichedTransaction) -> Dict[str, Any]:
        """Extract base features from transaction.
        
        Args:
            transaction: Enriched transaction
            
        Returns:
            Dictionary of base features
        """
        features = {}
        
        # Convert transaction to dict for easier access
        tx_dict = transaction.dict()
        
        # Numerical features
        for feature in self.config.numerical_features:
            if feature in tx_dict:
                features[feature] = float(tx_dict[feature])
            else:
                features[feature] = self.config.fill_missing_numerical
        
        # Boolean features
        for feature in self.config.boolean_features:
            if feature in tx_dict:
                features[feature] = bool(tx_dict[feature])
            else:
                features[feature] = False
        
        # Categorical features (will be encoded later)
        for feature in self.config.categorical_features:
            if feature in tx_dict:
                features[feature] = str(tx_dict[feature])
            else:
                features[feature] = self.config.fill_missing_categorical
        
        # Derived numerical features
        if 'amount' in features:
            amount = features['amount']
            features['amount_squared'] = amount ** 2
            features['amount_log1p'] = np.log1p(amount)
            features['amount_sqrt'] = np.sqrt(max(amount, 0))
        
        # Time-based features
        if hasattr(transaction, 'timestamp'):
            timestamp = transaction.timestamp
            features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
            features['day_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
            features['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
        
        return features
    
    def _extract_time_features(self, 
                              transaction: EnrichedTransaction, 
                              historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract time-based aggregation features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical transaction data
            
        Returns:
            Dictionary of time-based features
        """
        features = {}
        
        if historical_data.empty:
            return self._get_default_time_features()
        
        current_time = transaction.timestamp
        user_id = transaction.user_id
        
        # Filter historical data for the user
        user_data = historical_data[historical_data['user_id'] == user_id]
        
        for window_config in self.config.time_windows:
            window_start = current_time - timedelta(minutes=window_config.window_size)
            window_data = user_data[
                (user_data['timestamp'] >= window_start) & 
                (user_data['timestamp'] < current_time)
            ]
            
            window_name = window_config.name
            
            for feature_name in window_config.features:
                if feature_name not in window_data.columns:
                    continue
                
                feature_values = window_data[feature_name].dropna()
                
                for agg_type in window_config.aggregations:
                    feature_key = f"{feature_name}_{agg_type.value}_{window_name}"
                    
                    if len(feature_values) == 0:
                        features[feature_key] = 0.0
                        continue
                    
                    if agg_type == AggregationType.COUNT:
                        features[feature_key] = len(feature_values)
                    elif agg_type == AggregationType.SUM:
                        features[feature_key] = float(feature_values.sum())
                    elif agg_type == AggregationType.MEAN:
                        features[feature_key] = float(feature_values.mean())
                    elif agg_type == AggregationType.MEDIAN:
                        features[feature_key] = float(feature_values.median())
                    elif agg_type == AggregationType.STD:
                        features[feature_key] = float(feature_values.std())
                    elif agg_type == AggregationType.MIN:
                        features[feature_key] = float(feature_values.min())
                    elif agg_type == AggregationType.MAX:
                        features[feature_key] = float(feature_values.max())
                    elif agg_type == AggregationType.QUANTILE_25:
                        features[feature_key] = float(feature_values.quantile(0.25))
                    elif agg_type == AggregationType.QUANTILE_75:
                        features[feature_key] = float(feature_values.quantile(0.75))
                    elif agg_type == AggregationType.UNIQUE_COUNT:
                        features[feature_key] = feature_values.nunique()
                    elif agg_type == AggregationType.MODE:
                        mode_values = feature_values.mode()
                        features[feature_key] = float(mode_values.iloc[0]) if len(mode_values) > 0 else 0.0
            
            # Additional window-specific features
            features[f"tx_frequency_{window_name}"] = len(window_data) / max(window_config.window_size, 1)
            
            if len(window_data) > 1:
                # Time between transactions
                time_diffs = window_data['timestamp'].diff().dt.total_seconds().dropna()
                if len(time_diffs) > 0:
                    features[f"avg_time_between_tx_{window_name}"] = float(time_diffs.mean())
                    features[f"std_time_between_tx_{window_name}"] = float(time_diffs.std())
                else:
                    features[f"avg_time_between_tx_{window_name}"] = 0.0
                    features[f"std_time_between_tx_{window_name}"] = 0.0
            else:
                features[f"avg_time_between_tx_{window_name}"] = 0.0
                features[f"std_time_between_tx_{window_name}"] = 0.0
        
        # Cross-window ratios
        self._add_cross_window_ratios(features)
        
        return features
    
    def _extract_statistical_features(self, 
                                    transaction: EnrichedTransaction, 
                                    historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Extract statistical features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical data for statistical context
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        if historical_data is None or historical_data.empty:
            return features
        
        user_id = transaction.user_id
        user_data = historical_data[historical_data['user_id'] == user_id]
        
        for feature_name, stat_types in self.config.statistical_features.items():
            if feature_name not in user_data.columns:
                continue
            
            current_value = getattr(transaction, feature_name, None)
            if current_value is None:
                continue
            
            historical_values = user_data[feature_name].dropna()
            
            if len(historical_values) == 0:
                continue
            
            for stat_type in stat_types:
                feature_key = f"{feature_name}_{stat_type}"
                
                if stat_type == "zscore":
                    mean_val = historical_values.mean()
                    std_val = historical_values.std()
                    if std_val > 0:
                        features[feature_key] = (current_value - mean_val) / std_val
                    else:
                        features[feature_key] = 0.0
                
                elif stat_type == "percentile_rank":
                    rank = (historical_values < current_value).sum()
                    features[feature_key] = rank / len(historical_values)
                
                elif stat_type == "deviation_from_mean":
                    mean_val = historical_values.mean()
                    features[feature_key] = abs(current_value - mean_val)
                
                elif stat_type == "deviation_from_median":
                    median_val = historical_values.median()
                    features[feature_key] = abs(current_value - median_val)
        
        return features
    
    def _extract_interaction_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interaction features between existing features.
        
        Args:
            features: Existing features
            
        Returns:
            Dictionary of interaction features
        """
        interaction_features = {}
        
        # Amount and time interactions
        if 'amount' in features and 'hour_of_day' in features:
            interaction_features['amount_hour_interaction'] = features['amount'] * features['hour_of_day']
        
        if 'amount' in features and 'is_weekend' in features:
            interaction_features['amount_weekend_interaction'] = features['amount'] * (1 if features['is_weekend'] else 0)
        
        # Risk score interactions
        if 'merchant_risk_score' in features and 'amount' in features:
            interaction_features['risk_amount_interaction'] = features['merchant_risk_score'] * features['amount']
        
        # Velocity interactions
        amount_5m_key = 'amount_sum_5m'
        amount_1h_key = 'amount_sum_1h'
        
        if amount_5m_key in features and amount_1h_key in features:
            if features[amount_1h_key] > 0:
                interaction_features['amount_velocity_ratio'] = features[amount_5m_key] / features[amount_1h_key]
            else:
                interaction_features['amount_velocity_ratio'] = 0.0
        
        # Boolean feature combinations
        if 'is_weekend' in features and 'is_late_night' in features:
            interaction_features['weekend_late_night'] = (
                features['is_weekend'] and features.get('is_late_night', False)
            )
        
        return interaction_features
    
    def _add_cross_window_ratios(self, features: Dict[str, Any]):
        """Add cross-window ratio features.
        
        Args:
            features: Features dictionary to update
        """
        # Transaction count ratios
        count_5m = features.get('amount_count_5m', 0)
        count_1h = features.get('amount_count_1h', 0)
        count_1d = features.get('amount_count_1d', 0)
        
        if count_1h > 0:
            features['tx_ratio_5m_1h'] = count_5m / count_1h
        else:
            features['tx_ratio_5m_1h'] = 0.0
        
        if count_1d > 0:
            features['tx_ratio_1h_1d'] = count_1h / count_1d
        else:
            features['tx_ratio_1h_1d'] = 0.0
        
        # Amount sum ratios
        sum_5m = features.get('amount_sum_5m', 0)
        sum_1h = features.get('amount_sum_1h', 0)
        sum_1d = features.get('amount_sum_1d', 0)
        
        if sum_1h > 0:
            features['amount_ratio_5m_1h'] = sum_5m / sum_1h
        else:
            features['amount_ratio_5m_1h'] = 0.0
        
        if sum_1d > 0:
            features['amount_ratio_1h_1d'] = sum_1h / sum_1d
        else:
            features['amount_ratio_1h_1d'] = 0.0
    
    def _handle_missing_values(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing values in features.
        
        Args:
            features: Features dictionary
            
        Returns:
            Features dictionary with missing values handled
        """
        for key, value in features.items():
            if pd.isna(value) or value is None:
                if key in self.config.numerical_features or '_' in key:  # Assume derived features are numerical
                    features[key] = self.config.fill_missing_numerical
                elif key in self.config.categorical_features:
                    features[key] = self.config.fill_missing_categorical
                elif key in self.config.boolean_features:
                    features[key] = False
                else:
                    # Default handling based on type
                    if isinstance(value, (int, float)):
                        features[key] = self.config.fill_missing_numerical
                    elif isinstance(value, str):
                        features[key] = self.config.fill_missing_categorical
                    else:
                        features[key] = False
        
        return features
    
    def _apply_transformations(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fitted transformations to features.
        
        Args:
            features: Raw features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            return features
        
        transformed_features = features.copy()
        
        # Apply categorical encodings
        for feature in self.config.categorical_features:
            if feature in features and feature in self.encoders:
                try:
                    encoded_value = self.encoders[feature].transform([features[feature]])[0]
                    transformed_features[feature] = encoded_value
                except ValueError:
                    # Handle unseen categories
                    transformed_features[feature] = 0  # Default encoding
        
        # Apply numerical scaling
        if self.config.enable_scaling:
            numerical_features = [f for f in features.keys() 
                                if f in self.config.numerical_features or 
                                any(f.startswith(prefix) for prefix in ['amount_', 'tx_', 'avg_', 'std_'])]
            
            for feature in numerical_features:
                if feature in self.scalers:
                    try:
                        scaled_value = self.scalers[feature].transform([[features[feature]]])[0][0]
                        transformed_features[feature] = scaled_value
                    except Exception:
                        # Keep original value if scaling fails
                        pass
        
        return transformed_features
    
    def fit(self, transactions: List[EnrichedTransaction], historical_data: Optional[pd.DataFrame] = None):
        """Fit the feature extractor on training data.
        
        Args:
            transactions: List of enriched transactions
            historical_data: Historical transaction data
        """
        logger.info(f"Fitting feature extractor on {len(transactions)} transactions")
        
        # Extract features from all transactions
        all_features = []
        for transaction in transactions:
            features = self.extract_features(transaction, historical_data)
            all_features.append(features)
        
        if not all_features:
            logger.warning("No features extracted for fitting")
            return
        
        # Convert to DataFrame for easier processing
        features_df = pd.DataFrame(all_features)
        
        # Fit categorical encoders
        for feature in self.config.categorical_features:
            if feature in features_df.columns:
                encoder = LabelEncoder()
                encoder.fit(features_df[feature].astype(str))
                self.encoders[feature] = encoder
        
        # Fit numerical scalers
        if self.config.enable_scaling:
            numerical_features = [f for f in features_df.columns 
                                if f in self.config.numerical_features or 
                                any(f.startswith(prefix) for prefix in ['amount_', 'tx_', 'avg_', 'std_'])]
            
            for feature in numerical_features:
                if feature in features_df.columns:
                    scaler = StandardScaler()
                    scaler.fit(features_df[[feature]])
                    self.scalers[feature] = scaler
        
        # Compute feature statistics
        self._compute_feature_stats(features_df)
        
        self.is_fitted = True
        logger.info("Feature extractor fitted successfully")
    
    def _compute_feature_stats(self, features_df: pd.DataFrame):
        """Compute feature statistics for monitoring.
        
        Args:
            features_df: Features DataFrame
        """
        for column in features_df.columns:
            if pd.api.types.is_numeric_dtype(features_df[column]):
                self.feature_stats[column] = {
                    'mean': float(features_df[column].mean()),
                    'std': float(features_df[column].std()),
                    'min': float(features_df[column].min()),
                    'max': float(features_df[column].max()),
                    'median': float(features_df[column].median())
                }
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default features when extraction fails.
        
        Returns:
            Dictionary of default features
        """
        default_features = {}
        
        # Default numerical features
        for feature in self.config.numerical_features:
            default_features[feature] = self.config.fill_missing_numerical
        
        # Default boolean features
        for feature in self.config.boolean_features:
            default_features[feature] = False
        
        # Default categorical features
        for feature in self.config.categorical_features:
            default_features[feature] = self.config.fill_missing_categorical
        
        return default_features
    
    def _get_default_time_features(self) -> Dict[str, Any]:
        """Get default time-based features when historical data is unavailable.
        
        Returns:
            Dictionary of default time features
        """
        features = {}
        
        for window_config in self.config.time_windows:
            window_name = window_config.name
            
            for feature_name in window_config.features:
                for agg_type in window_config.aggregations:
                    feature_key = f"{feature_name}_{agg_type.value}_{window_name}"
                    features[feature_key] = 0.0
            
            features[f"tx_frequency_{window_name}"] = 0.0
            features[f"avg_time_between_tx_{window_name}"] = 0.0
            features[f"std_time_between_tx_{window_name}"] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Base features
        feature_names.extend(self.config.numerical_features)
        feature_names.extend(self.config.boolean_features)
        feature_names.extend(self.config.categorical_features)
        
        # Derived features
        feature_names.extend([
            'amount_squared', 'amount_log1p', 'amount_sqrt',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ])
        
        # Time-based features
        for window_config in self.config.time_windows:
            window_name = window_config.name
            for feature_name in window_config.features:
                for agg_type in window_config.aggregations:
                    feature_names.append(f"{feature_name}_{agg_type.value}_{window_name}")
            
            feature_names.extend([
                f"tx_frequency_{window_name}",
                f"avg_time_between_tx_{window_name}",
                f"std_time_between_tx_{window_name}"
            ])
        
        # Statistical features
        for feature_name, stat_types in self.config.statistical_features.items():
            for stat_type in stat_types:
                feature_names.append(f"{feature_name}_{stat_type}")
        
        # Interaction features
        feature_names.extend([
            'amount_hour_interaction', 'amount_weekend_interaction',
            'risk_amount_interaction', 'amount_velocity_ratio', 'weekend_late_night'
        ])
        
        # Cross-window ratios
        feature_names.extend([
            'tx_ratio_5m_1h', 'tx_ratio_1h_1d',
            'amount_ratio_5m_1h', 'amount_ratio_1h_1d'
        ])
        
        return feature_names
    
    def get_feature_importance_names(self) -> List[str]:
        """Get feature names suitable for model training.
        
        Returns:
            List of feature names for model training
        """
        # This would return the final feature names after all transformations
        return self.get_feature_names()


def create_feature_extractor(config: Optional[FeatureConfig] = None) -> FeatureExtractor:
    """Create feature extractor with default or custom configuration.
    
    Args:
        config: Optional feature extraction configuration
        
    Returns:
        Feature extractor instance
    """
    if config is None:
        config = FeatureConfig()
    
    return FeatureExtractor(config)


# Global feature extractor instance
_extractor_instance: Optional[FeatureExtractor] = None


def get_feature_extractor() -> FeatureExtractor:
    """Get global feature extractor instance.
    
    Returns:
        Global feature extractor instance
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = create_feature_extractor()
    return _extractor_instance


def set_feature_extractor(extractor: FeatureExtractor):
    """Set global feature extractor instance.
    
    Args:
        extractor: Feature extractor instance to set as global
    """
    global _extractor_instance
    _extractor_instance = extractor