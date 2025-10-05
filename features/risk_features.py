"""Risk feature calculation module for fraud detection.

This module provides comprehensive risk feature calculation including:
- Transaction velocity features
- Amount-based risk indicators
- Location risk features
- Merchant risk features
- Time-based risk features
- Network and device risk features
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter
import math

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from geopy.distance import geodesic

from service.models import EnrichedTransaction

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    """Risk feature categories."""
    VELOCITY = "velocity"
    AMOUNT = "amount"
    LOCATION = "location"
    MERCHANT = "merchant"
    TEMPORAL = "temporal"
    NETWORK = "network"
    DEVICE = "device"
    BEHAVIORAL = "behavioral"


class RiskLevel(str, Enum):
    """Risk levels for features."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskFeature:
    """Represents a calculated risk feature."""
    name: str
    value: float
    category: RiskCategory
    risk_level: RiskLevel
    confidence: float
    description: str
    metadata: Dict[str, Any]


class VelocityFeatures(BaseModel):
    """Velocity-based risk features."""
    
    # Transaction count features
    txn_count_1h: int = 0
    txn_count_6h: int = 0
    txn_count_24h: int = 0
    txn_count_7d: int = 0
    
    # Amount velocity features
    amount_sum_1h: float = 0.0
    amount_sum_6h: float = 0.0
    amount_sum_24h: float = 0.0
    amount_sum_7d: float = 0.0
    
    # Average amount features
    avg_amount_1h: float = 0.0
    avg_amount_6h: float = 0.0
    avg_amount_24h: float = 0.0
    avg_amount_7d: float = 0.0
    
    # Unique merchant/location counts
    unique_merchants_1h: int = 0
    unique_merchants_24h: int = 0
    unique_countries_1h: int = 0
    unique_countries_24h: int = 0
    
    # Velocity ratios
    txn_velocity_ratio_1h_24h: float = 0.0  # 1h count / (24h count / 24)
    amount_velocity_ratio_1h_24h: float = 0.0  # 1h sum / (24h sum / 24)
    
    class Config:
        use_enum_values = True


class AmountFeatures(BaseModel):
    """Amount-based risk features."""
    
    # Current transaction amount features
    amount: float = 0.0
    amount_log: float = 0.0
    amount_rounded: bool = False  # Ends in 00
    amount_category: str = "medium"  # small, medium, large, very_large
    
    # Historical comparison features
    amount_vs_user_avg_ratio: float = 1.0
    amount_vs_user_max_ratio: float = 1.0
    amount_vs_merchant_avg_ratio: float = 1.0
    amount_vs_category_avg_ratio: float = 1.0
    
    # Statistical features
    amount_zscore_user: float = 0.0
    amount_zscore_merchant: float = 0.0
    amount_zscore_category: float = 0.0
    
    # Percentile features
    amount_percentile_user: float = 50.0
    amount_percentile_merchant: float = 50.0
    amount_percentile_global: float = 50.0
    
    # Amount pattern features
    is_amount_outlier: bool = False
    amount_deviation_score: float = 0.0
    
    class Config:
        use_enum_values = True


class LocationFeatures(BaseModel):
    """Location-based risk features."""
    
    # Current location features
    transaction_country: Optional[str] = None
    is_high_risk_country: bool = False
    country_risk_score: float = 0.0
    
    # Distance features
    distance_from_home_km: float = 0.0
    distance_from_last_txn_km: float = 0.0
    impossible_travel_flag: bool = False
    travel_velocity_kmh: float = 0.0
    
    # Location pattern features
    is_new_country: bool = False
    is_frequent_country: bool = False
    country_transaction_count: int = 0
    country_first_seen_days_ago: int = 0
    
    # Geographic risk features
    timezone_offset_hours: float = 0.0
    is_unusual_timezone: bool = False
    location_entropy: float = 0.0  # Diversity of locations
    
    # Travel pattern features
    countries_visited_24h: int = 0
    max_distance_24h_km: float = 0.0
    total_travel_distance_24h_km: float = 0.0
    
    class Config:
        use_enum_values = True


class MerchantFeatures(BaseModel):
    """Merchant-based risk features."""
    
    # Merchant identification
    merchant_id: Optional[str] = None
    merchant_category: Optional[str] = None
    
    # Merchant risk features
    merchant_risk_score: float = 0.0
    is_high_risk_merchant: bool = False
    merchant_fraud_rate: float = 0.0
    
    # Merchant familiarity features
    is_new_merchant: bool = False
    is_frequent_merchant: bool = False
    merchant_transaction_count: int = 0
    merchant_first_seen_days_ago: int = 0
    
    # Category features
    is_high_risk_category: bool = False
    category_risk_score: float = 0.0
    category_fraud_rate: float = 0.0
    
    # Merchant pattern features
    merchant_avg_amount: float = 0.0
    merchant_std_amount: float = 0.0
    amount_vs_merchant_pattern: float = 0.0
    
    # Merchant diversity features
    unique_merchants_user_24h: int = 0
    merchant_diversity_score: float = 0.0
    
    class Config:
        use_enum_values = True


class TemporalFeatures(BaseModel):
    """Time-based risk features."""
    
    # Time components
    hour_of_day: int = 0
    day_of_week: int = 0
    day_of_month: int = 1
    month_of_year: int = 1
    
    # Time pattern features
    is_weekend: bool = False
    is_business_hours: bool = True
    is_unusual_hour: bool = False
    hour_risk_score: float = 0.0
    
    # Temporal velocity features
    time_since_last_txn_minutes: float = 0.0
    time_since_last_txn_category: str = "normal"  # very_fast, fast, normal, slow
    
    # Seasonal features
    is_holiday_period: bool = False
    is_month_end: bool = False
    is_payday_period: bool = False
    
    # User temporal patterns
    is_user_active_hour: bool = True
    is_user_active_day: bool = True
    hour_deviation_from_user_pattern: float = 0.0
    
    # Time-based anomalies
    is_night_transaction: bool = False
    is_early_morning_transaction: bool = False
    temporal_anomaly_score: float = 0.0
    
    class Config:
        use_enum_values = True


class NetworkFeatures(BaseModel):
    """Network and device-based risk features."""
    
    # IP and network features
    ip_address: Optional[str] = None
    is_vpn: bool = False
    is_tor: bool = False
    is_proxy: bool = False
    ip_risk_score: float = 0.0
    
    # Device features
    device_id: Optional[str] = None
    is_new_device: bool = False
    device_risk_score: float = 0.0
    device_transaction_count: int = 0
    
    # Session features
    session_id: Optional[str] = None
    session_transaction_count: int = 0
    session_duration_minutes: float = 0.0
    
    # Network pattern features
    ip_country_mismatch: bool = False
    device_location_mismatch: bool = False
    network_anomaly_score: float = 0.0
    
    class Config:
        use_enum_values = True


class RiskFeatureConfig(BaseModel):
    """Configuration for risk feature calculation."""
    
    # Velocity windows (in hours)
    velocity_windows: List[int] = Field(default=[1, 6, 24, 168])  # 1h, 6h, 24h, 7d
    
    # Amount thresholds
    small_amount_threshold: float = 50.0
    large_amount_threshold: float = 1000.0
    very_large_amount_threshold: float = 5000.0
    
    # Distance thresholds (in km)
    local_distance_threshold: float = 100.0
    domestic_distance_threshold: float = 1000.0
    impossible_travel_speed_kmh: float = 1000.0  # Speed indicating impossible travel
    
    # Time thresholds
    business_hours_start: int = 9
    business_hours_end: int = 17
    night_hours_start: int = 22
    night_hours_end: int = 6
    
    # Risk scoring weights
    velocity_weight: float = 0.25
    amount_weight: float = 0.20
    location_weight: float = 0.20
    merchant_weight: float = 0.15
    temporal_weight: float = 0.10
    network_weight: float = 0.10
    
    # Statistical thresholds
    outlier_zscore_threshold: float = 3.0
    high_percentile_threshold: float = 95.0
    low_percentile_threshold: float = 5.0
    
    # Pattern detection parameters
    min_transactions_for_pattern: int = 10
    pattern_confidence_threshold: float = 0.7
    
    class Config:
        use_enum_values = True


class RiskFeatureCalculator:
    """Calculates comprehensive risk features for fraud detection."""
    
    def __init__(self, config: RiskFeatureConfig):
        """Initialize risk feature calculator.
        
        Args:
            config: Risk feature calculation configuration
        """
        self.config = config
        
        # High-risk countries (example list)
        self.high_risk_countries = {
            'AF', 'IQ', 'LY', 'SO', 'SY', 'YE', 'MM', 'KP', 'IR'
        }
        
        # High-risk merchant categories
        self.high_risk_categories = {
            'gambling', 'adult_entertainment', 'cryptocurrency', 
            'money_transfer', 'cash_advance', 'pawn_shop'
        }
        
        logger.info("Risk feature calculator initialized")
    
    def calculate_risk_features(self, 
                              transaction: EnrichedTransaction, 
                              historical_data: Optional[pd.DataFrame] = None,
                              user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate comprehensive risk features for a transaction.
        
        Args:
            transaction: Transaction to analyze
            historical_data: Historical transaction data
            user_profile: User behavior profile
            
        Returns:
            Dictionary containing all risk features
        """
        try:
            risk_features = {}
            
            # Calculate velocity features
            velocity_features = self._calculate_velocity_features(transaction, historical_data)
            risk_features.update(velocity_features.dict())
            
            # Calculate amount features
            amount_features = self._calculate_amount_features(transaction, historical_data, user_profile)
            risk_features.update(amount_features.dict())
            
            # Calculate location features
            location_features = self._calculate_location_features(transaction, historical_data, user_profile)
            risk_features.update(location_features.dict())
            
            # Calculate merchant features
            merchant_features = self._calculate_merchant_features(transaction, historical_data)
            risk_features.update(merchant_features.dict())
            
            # Calculate temporal features
            temporal_features = self._calculate_temporal_features(transaction, historical_data, user_profile)
            risk_features.update(temporal_features.dict())
            
            # Calculate network features
            network_features = self._calculate_network_features(transaction, historical_data)
            risk_features.update(network_features.dict())
            
            # Calculate composite risk scores
            composite_scores = self._calculate_composite_scores(risk_features)
            risk_features.update(composite_scores)
            
            return risk_features
            
        except Exception as e:
            logger.error(f"Risk feature calculation failed: {e}")
            return self._get_default_risk_features()
    
    def _calculate_velocity_features(self, 
                                   transaction: EnrichedTransaction, 
                                   historical_data: Optional[pd.DataFrame]) -> VelocityFeatures:
        """Calculate velocity-based risk features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical transaction data
            
        Returns:
            Velocity features
        """
        features = VelocityFeatures()
        
        if historical_data is None or historical_data.empty:
            return features
        
        user_data = historical_data[historical_data['user_id'] == transaction.user_id]
        current_time = transaction.timestamp
        
        # Calculate features for each time window
        for hours in self.config.velocity_windows:
            window_start = current_time - timedelta(hours=hours)
            window_data = user_data[user_data['timestamp'] >= window_start]
            
            if hours == 1:
                features.txn_count_1h = len(window_data)
                features.amount_sum_1h = window_data['amount'].sum() if not window_data.empty else 0.0
                features.avg_amount_1h = window_data['amount'].mean() if not window_data.empty else 0.0
                features.unique_merchants_1h = window_data['merchant_id'].nunique() if 'merchant_id' in window_data.columns else 0
                features.unique_countries_1h = window_data['transaction_country'].nunique() if 'transaction_country' in window_data.columns else 0
            elif hours == 6:
                features.txn_count_6h = len(window_data)
                features.amount_sum_6h = window_data['amount'].sum() if not window_data.empty else 0.0
                features.avg_amount_6h = window_data['amount'].mean() if not window_data.empty else 0.0
            elif hours == 24:
                features.txn_count_24h = len(window_data)
                features.amount_sum_24h = window_data['amount'].sum() if not window_data.empty else 0.0
                features.avg_amount_24h = window_data['amount'].mean() if not window_data.empty else 0.0
                features.unique_merchants_24h = window_data['merchant_id'].nunique() if 'merchant_id' in window_data.columns else 0
                features.unique_countries_24h = window_data['transaction_country'].nunique() if 'transaction_country' in window_data.columns else 0
            elif hours == 168:  # 7 days
                features.txn_count_7d = len(window_data)
                features.amount_sum_7d = window_data['amount'].sum() if not window_data.empty else 0.0
                features.avg_amount_7d = window_data['amount'].mean() if not window_data.empty else 0.0
        
        # Calculate velocity ratios
        if features.txn_count_24h > 0:
            expected_1h_count = features.txn_count_24h / 24.0
            features.txn_velocity_ratio_1h_24h = features.txn_count_1h / max(expected_1h_count, 0.1)
        
        if features.amount_sum_24h > 0:
            expected_1h_amount = features.amount_sum_24h / 24.0
            features.amount_velocity_ratio_1h_24h = features.amount_sum_1h / max(expected_1h_amount, 1.0)
        
        return features
    
    def _calculate_amount_features(self, 
                                 transaction: EnrichedTransaction, 
                                 historical_data: Optional[pd.DataFrame],
                                 user_profile: Optional[Dict[str, Any]]) -> AmountFeatures:
        """Calculate amount-based risk features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical transaction data
            user_profile: User behavior profile
            
        Returns:
            Amount features
        """
        features = AmountFeatures()
        amount = transaction.amount
        
        # Basic amount features
        features.amount = amount
        features.amount_log = math.log10(max(amount, 1.0))
        features.amount_rounded = (amount % 100) == 0
        
        # Amount category
        if amount < self.config.small_amount_threshold:
            features.amount_category = "small"
        elif amount < self.config.large_amount_threshold:
            features.amount_category = "medium"
        elif amount < self.config.very_large_amount_threshold:
            features.amount_category = "large"
        else:
            features.amount_category = "very_large"
        
        if historical_data is not None and not historical_data.empty:
            # User-specific features
            user_data = historical_data[historical_data['user_id'] == transaction.user_id]
            if not user_data.empty:
                user_amounts = user_data['amount']
                user_avg = user_amounts.mean()
                user_std = user_amounts.std()
                user_max = user_amounts.max()
                
                features.amount_vs_user_avg_ratio = amount / max(user_avg, 1.0)
                features.amount_vs_user_max_ratio = amount / max(user_max, 1.0)
                
                if user_std > 0:
                    features.amount_zscore_user = (amount - user_avg) / user_std
                
                features.amount_percentile_user = (user_amounts < amount).mean() * 100
                features.is_amount_outlier = abs(features.amount_zscore_user) > self.config.outlier_zscore_threshold
            
            # Merchant-specific features
            merchant_id = getattr(transaction, 'merchant_id', None)
            if merchant_id:
                merchant_data = historical_data[historical_data.get('merchant_id') == merchant_id]
                if not merchant_data.empty:
                    merchant_amounts = merchant_data['amount']
                    merchant_avg = merchant_amounts.mean()
                    merchant_std = merchant_amounts.std()
                    
                    features.amount_vs_merchant_avg_ratio = amount / max(merchant_avg, 1.0)
                    features.merchant_avg_amount = merchant_avg
                    features.merchant_std_amount = merchant_std
                    
                    if merchant_std > 0:
                        features.amount_zscore_merchant = (amount - merchant_avg) / merchant_std
                    
                    features.amount_percentile_merchant = (merchant_amounts < amount).mean() * 100
            
            # Category-specific features
            category = getattr(transaction, 'merchant_category', None)
            if category:
                category_data = historical_data[historical_data.get('merchant_category') == category]
                if not category_data.empty:
                    category_amounts = category_data['amount']
                    category_avg = category_amounts.mean()
                    category_std = category_amounts.std()
                    
                    features.amount_vs_category_avg_ratio = amount / max(category_avg, 1.0)
                    
                    if category_std > 0:
                        features.amount_zscore_category = (amount - category_avg) / category_std
            
            # Global percentile
            all_amounts = historical_data['amount']
            features.amount_percentile_global = (all_amounts < amount).mean() * 100
        
        # Amount deviation score (composite)
        deviation_factors = [
            abs(features.amount_zscore_user),
            abs(features.amount_zscore_merchant),
            abs(features.amount_zscore_category)
        ]
        features.amount_deviation_score = max([f for f in deviation_factors if f > 0], default=0.0)
        
        return features
    
    def _calculate_location_features(self, 
                                   transaction: EnrichedTransaction, 
                                   historical_data: Optional[pd.DataFrame],
                                   user_profile: Optional[Dict[str, Any]]) -> LocationFeatures:
        """Calculate location-based risk features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical transaction data
            user_profile: User behavior profile
            
        Returns:
            Location features
        """
        features = LocationFeatures()
        
        country = getattr(transaction, 'transaction_country', None)
        if country:
            features.transaction_country = country
            features.is_high_risk_country = country in self.high_risk_countries
            features.country_risk_score = 0.8 if features.is_high_risk_country else 0.2
        
        if historical_data is not None and not historical_data.empty:
            user_data = historical_data[historical_data['user_id'] == transaction.user_id]
            
            if not user_data.empty and 'transaction_country' in user_data.columns:
                user_countries = user_data['transaction_country'].dropna()
                
                if len(user_countries) > 0:
                    # Country familiarity features
                    country_counts = user_countries.value_counts()
                    features.is_new_country = country not in country_counts.index
                    features.is_frequent_country = country in country_counts.head(3).index
                    features.country_transaction_count = country_counts.get(country, 0)
                    
                    # Home country detection
                    home_country = country_counts.index[0] if len(country_counts) > 0 else None
                    
                    # Location diversity
                    unique_countries = len(country_counts)
                    total_transactions = len(user_countries)
                    features.location_entropy = -sum([(count/total_transactions) * math.log2(count/total_transactions) 
                                                     for count in country_counts.values()])
                    
                    # Recent travel patterns
                    recent_24h = user_data[user_data['timestamp'] >= transaction.timestamp - timedelta(hours=24)]
                    if not recent_24h.empty and 'transaction_country' in recent_24h.columns:
                        recent_countries = recent_24h['transaction_country'].dropna()
                        features.countries_visited_24h = recent_countries.nunique()
        
        # Calculate distance and travel features (simplified - would need actual coordinates)
        # This is a placeholder implementation
        if hasattr(transaction, 'latitude') and hasattr(transaction, 'longitude'):
            # Distance calculations would go here
            # For now, using simplified logic
            features.distance_from_home_km = 0.0
            features.distance_from_last_txn_km = 0.0
            features.impossible_travel_flag = False
            features.travel_velocity_kmh = 0.0
        
        return features
    
    def _calculate_merchant_features(self, 
                                   transaction: EnrichedTransaction, 
                                   historical_data: Optional[pd.DataFrame]) -> MerchantFeatures:
        """Calculate merchant-based risk features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical transaction data
            
        Returns:
            Merchant features
        """
        features = MerchantFeatures()
        
        merchant_id = getattr(transaction, 'merchant_id', None)
        category = getattr(transaction, 'merchant_category', None)
        
        features.merchant_id = merchant_id
        features.merchant_category = category
        
        # Category risk assessment
        if category:
            features.is_high_risk_category = category.lower() in self.high_risk_categories
            features.category_risk_score = 0.8 if features.is_high_risk_category else 0.2
        
        if historical_data is not None and not historical_data.empty:
            user_data = historical_data[historical_data['user_id'] == transaction.user_id]
            
            # Merchant familiarity features
            if merchant_id and 'merchant_id' in historical_data.columns:
                user_merchants = user_data['merchant_id'].dropna() if not user_data.empty else pd.Series([])
                merchant_counts = user_merchants.value_counts()
                
                features.is_new_merchant = merchant_id not in merchant_counts.index
                features.is_frequent_merchant = merchant_id in merchant_counts.head(10).index
                features.merchant_transaction_count = merchant_counts.get(merchant_id, 0)
                
                # Merchant diversity for user
                recent_24h = user_data[user_data['timestamp'] >= transaction.timestamp - timedelta(hours=24)]
                if not recent_24h.empty and 'merchant_id' in recent_24h.columns:
                    features.unique_merchants_user_24h = recent_24h['merchant_id'].nunique()
                
                if len(user_merchants) > 0:
                    features.merchant_diversity_score = len(merchant_counts) / len(user_merchants)
            
            # Global merchant statistics
            if merchant_id:
                merchant_data = historical_data[historical_data.get('merchant_id') == merchant_id]
                if not merchant_data.empty:
                    # Merchant risk scoring based on fraud rate (simplified)
                    # In practice, this would come from a merchant risk database
                    merchant_transaction_count = len(merchant_data)
                    features.merchant_risk_score = min(0.1 + (1000 / max(merchant_transaction_count, 1)) * 0.1, 0.9)
                    features.is_high_risk_merchant = features.merchant_risk_score > 0.6
                    
                    # Merchant amount patterns
                    merchant_amounts = merchant_data['amount']
                    features.merchant_avg_amount = merchant_amounts.mean()
                    features.merchant_std_amount = merchant_amounts.std()
                    
                    if features.merchant_avg_amount > 0:
                        features.amount_vs_merchant_pattern = transaction.amount / features.merchant_avg_amount
            
            # Category statistics
            if category:
                category_data = historical_data[historical_data.get('merchant_category') == category]
                if not category_data.empty:
                    # Simplified fraud rate calculation
                    category_transaction_count = len(category_data)
                    base_fraud_rate = 0.05 if features.is_high_risk_category else 0.01
                    features.category_fraud_rate = base_fraud_rate
        
        return features
    
    def _calculate_temporal_features(self, 
                                   transaction: EnrichedTransaction, 
                                   historical_data: Optional[pd.DataFrame],
                                   user_profile: Optional[Dict[str, Any]]) -> TemporalFeatures:
        """Calculate time-based risk features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical transaction data
            user_profile: User behavior profile
            
        Returns:
            Temporal features
        """
        features = TemporalFeatures()
        
        timestamp = transaction.timestamp
        
        # Basic time components
        features.hour_of_day = timestamp.hour
        features.day_of_week = timestamp.weekday()
        features.day_of_month = timestamp.day
        features.month_of_year = timestamp.month
        
        # Time pattern features
        features.is_weekend = features.day_of_week >= 5
        features.is_business_hours = self.config.business_hours_start <= features.hour_of_day < self.config.business_hours_end
        features.is_night_transaction = (features.hour_of_day >= self.config.night_hours_start or 
                                       features.hour_of_day < self.config.night_hours_end)
        features.is_early_morning_transaction = 3 <= features.hour_of_day < 7
        
        # Seasonal features
        features.is_month_end = features.day_of_month >= 28
        features.is_payday_period = features.day_of_month in [1, 2, 15, 16]  # Common payday periods
        
        # Hour risk scoring (simplified)
        night_hours = list(range(22, 24)) + list(range(0, 6))
        if features.hour_of_day in night_hours:
            features.hour_risk_score = 0.7
            features.is_unusual_hour = True
        elif features.hour_of_day in [6, 7, 8, 21]:
            features.hour_risk_score = 0.4
        else:
            features.hour_risk_score = 0.2
        
        if historical_data is not None and not historical_data.empty:
            user_data = historical_data[historical_data['user_id'] == transaction.user_id]
            
            if not user_data.empty:
                # Time since last transaction
                user_timestamps = pd.to_datetime(user_data['timestamp']).sort_values()
                if len(user_timestamps) > 0:
                    last_timestamp = user_timestamps.iloc[-1]
                    time_diff = (timestamp - last_timestamp).total_seconds() / 60  # minutes
                    features.time_since_last_txn_minutes = time_diff
                    
                    # Categorize time difference
                    if time_diff < 5:
                        features.time_since_last_txn_category = "very_fast"
                    elif time_diff < 60:
                        features.time_since_last_txn_category = "fast"
                    elif time_diff < 1440:  # 24 hours
                        features.time_since_last_txn_category = "normal"
                    else:
                        features.time_since_last_txn_category = "slow"
                
                # User temporal patterns
                user_hours = pd.to_datetime(user_data['timestamp']).dt.hour
                user_days = pd.to_datetime(user_data['timestamp']).dt.dayofweek
                
                if len(user_hours) >= self.config.min_transactions_for_pattern:
                    # Check if current hour is typical for user
                    hour_counts = user_hours.value_counts()
                    top_hours = hour_counts.head(max(1, len(hour_counts) // 3)).index
                    features.is_user_active_hour = features.hour_of_day in top_hours
                    
                    # Calculate hour deviation
                    if len(top_hours) > 0:
                        hour_distances = [abs(features.hour_of_day - h) for h in top_hours]
                        features.hour_deviation_from_user_pattern = min(hour_distances)
                    
                    # Check if current day is typical for user
                    day_counts = user_days.value_counts()
                    top_days = day_counts.head(max(1, len(day_counts) // 2)).index
                    features.is_user_active_day = features.day_of_week in top_days
        
        # Temporal anomaly score (composite)
        anomaly_factors = []
        if features.is_night_transaction:
            anomaly_factors.append(0.6)
        if features.is_early_morning_transaction:
            anomaly_factors.append(0.4)
        if not features.is_user_active_hour:
            anomaly_factors.append(0.3)
        if features.time_since_last_txn_category == "very_fast":
            anomaly_factors.append(0.5)
        
        features.temporal_anomaly_score = max(anomaly_factors, default=0.0)
        
        return features
    
    def _calculate_network_features(self, 
                                  transaction: EnrichedTransaction, 
                                  historical_data: Optional[pd.DataFrame]) -> NetworkFeatures:
        """Calculate network and device-based risk features.
        
        Args:
            transaction: Current transaction
            historical_data: Historical transaction data
            
        Returns:
            Network features
        """
        features = NetworkFeatures()
        
        # Extract network information from transaction
        features.ip_address = getattr(transaction, 'ip_address', None)
        features.device_id = getattr(transaction, 'device_id', None)
        features.session_id = getattr(transaction, 'session_id', None)
        
        # Network risk indicators (simplified - would need actual IP intelligence)
        features.is_vpn = getattr(transaction, 'is_vpn', False)
        features.is_tor = getattr(transaction, 'is_tor', False)
        features.is_proxy = getattr(transaction, 'is_proxy', False)
        
        # IP risk scoring
        if features.is_tor:
            features.ip_risk_score = 0.9
        elif features.is_vpn:
            features.ip_risk_score = 0.6
        elif features.is_proxy:
            features.ip_risk_score = 0.4
        else:
            features.ip_risk_score = 0.1
        
        if historical_data is not None and not historical_data.empty:
            user_data = historical_data[historical_data['user_id'] == transaction.user_id]
            
            # Device familiarity
            if features.device_id and 'device_id' in user_data.columns:
                user_devices = user_data['device_id'].dropna()
                device_counts = user_devices.value_counts()
                
                features.is_new_device = features.device_id not in device_counts.index
                features.device_transaction_count = device_counts.get(features.device_id, 0)
                
                # Device risk scoring
                if features.is_new_device:
                    features.device_risk_score = 0.5
                else:
                    # More transactions = lower risk
                    features.device_risk_score = max(0.1, 0.5 - (features.device_transaction_count * 0.02))
            
            # Session analysis
            if features.session_id and 'session_id' in user_data.columns:
                session_data = user_data[user_data['session_id'] == features.session_id]
                features.session_transaction_count = len(session_data)
                
                if len(session_data) > 1:
                    session_timestamps = pd.to_datetime(session_data['timestamp'])
                    session_duration = (session_timestamps.max() - session_timestamps.min()).total_seconds() / 60
                    features.session_duration_minutes = session_duration
        
        # Geographic consistency checks
        transaction_country = getattr(transaction, 'transaction_country', None)
        ip_country = getattr(transaction, 'ip_country', None)
        
        if transaction_country and ip_country:
            features.ip_country_mismatch = transaction_country != ip_country
        
        # Network anomaly score
        anomaly_factors = []
        if features.is_tor:
            anomaly_factors.append(0.9)
        elif features.is_vpn:
            anomaly_factors.append(0.6)
        if features.is_new_device:
            anomaly_factors.append(0.4)
        if features.ip_country_mismatch:
            anomaly_factors.append(0.3)
        
        features.network_anomaly_score = max(anomaly_factors, default=0.0)
        
        return features
    
    def _calculate_composite_scores(self, risk_features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite risk scores.

        This method combines individual risk features into composite scores using weighted
        averages and normalization. Each composite score represents a different risk dimension
        and is normalized to a 0-1 scale for consistent interpretation.

        Args:
            risk_features: Dictionary of calculated risk features

        Returns:
            Dictionary of composite scores (0-1 scale)
        """
        composite_scores = {}

        # Velocity risk score - measures transaction frequency and speed anomalies
        # Higher ratios indicate unusual bursts of activity compared to baseline
        velocity_factors = [
            # Transaction frequency ratio (1h vs 24h baseline) - cap at 5x normal
            min(risk_features.get('txn_velocity_ratio_1h_24h', 0) / 5.0, 1.0),
            # Amount velocity ratio - spending rate compared to normal patterns
            min(risk_features.get('amount_velocity_ratio_1h_24h', 0) / 5.0, 1.0),
            # Geographic diversity - multiple countries in short time is suspicious
            min(risk_features.get('unique_countries_1h', 0) / 3.0, 1.0),
            # Merchant diversity - too many different merchants quickly is unusual
            min(risk_features.get('unique_merchants_1h', 0) / 10.0, 1.0)
        ]
        composite_scores['velocity_risk_score'] = sum(velocity_factors) / len(velocity_factors)

        # Amount risk score - measures transaction amount anomalies
        # Combines statistical outliers with user-specific spending patterns
        amount_factors = [
            # Z-score normalization - how many standard deviations from user's mean
            min(abs(risk_features.get('amount_zscore_user', 0)) / 3.0, 1.0),
            # Binary outlier flag - statistical outlier detection result
            1.0 if risk_features.get('is_amount_outlier', False) else 0.0,
            # Deviation score - custom metric for amount unusualness
            min(risk_features.get('amount_deviation_score', 0) / 3.0, 1.0)
        ]
        composite_scores['amount_risk_score'] = sum(amount_factors) / len(amount_factors)

        # Location risk score - measures geographic risk factors
        # Combines country risk, travel patterns, and impossible travel detection
        location_factors = [
            # Country-specific risk score (0-1) based on fraud rates
            risk_features.get('country_risk_score', 0),
            # New country flag - first time user transacts in this country
            1.0 if risk_features.get('is_new_country', False) else 0.0,
            # Impossible travel - physically impossible to travel between locations
            1.0 if risk_features.get('impossible_travel_flag', False) else 0.0,
            # Country hopping - multiple countries in 24h (normalized by 5 max)
            min(risk_features.get('countries_visited_24h', 0) / 5.0, 1.0)
        ]
        composite_scores['location_risk_score'] = sum(location_factors) / len(location_factors)

        # Merchant risk score - measures merchant-related risk factors
        # Combines merchant reputation, category risk, and novelty
        merchant_factors = [
            # Merchant-specific risk score based on historical fraud rates
            risk_features.get('merchant_risk_score', 0),
            # Category risk score - some categories (e.g., online) are riskier
            risk_features.get('category_risk_score', 0),
            # New merchant penalty - moderate risk for first-time merchant interaction
            0.3 if risk_features.get('is_new_merchant', False) else 0.0
        ]
        composite_scores['merchant_risk_score'] = sum(merchant_factors) / len(merchant_factors)

        # Temporal risk score - measures time-based risk factors
        # Considers hour-of-day patterns and user activity schedules
        temporal_factors = [
            # Hour risk score - some hours (e.g., 3 AM) are inherently riskier
            risk_features.get('hour_risk_score', 0),
            # Temporal anomaly - transaction at unusual time for this user
            risk_features.get('temporal_anomaly_score', 0),
            # Inactive hour penalty - user typically not active at this time
            0.3 if not risk_features.get('is_user_active_hour', True) else 0.0
        ]
        composite_scores['temporal_risk_score'] = sum(temporal_factors) / len(temporal_factors)

        # Network risk score - measures device and network-related risks
        # Combines IP reputation, device fingerprinting, and network anomalies
        network_factors = [
            # IP address risk score based on reputation databases
            risk_features.get('ip_risk_score', 0),
            # Device risk score based on device fingerprinting and history
            risk_features.get('device_risk_score', 0),
            # Network anomaly score - unusual network behavior patterns
            risk_features.get('network_anomaly_score', 0)
        ]
        composite_scores['network_risk_score'] = sum(network_factors) / len(network_factors)
        
        # Overall risk score (weighted combination)
        overall_risk = (
            composite_scores['velocity_risk_score'] * self.config.velocity_weight +
            composite_scores['amount_risk_score'] * self.config.amount_weight +
            composite_scores['location_risk_score'] * self.config.location_weight +
            composite_scores['merchant_risk_score'] * self.config.merchant_weight +
            composite_scores['temporal_risk_score'] * self.config.temporal_weight +
            composite_scores['network_risk_score'] * self.config.network_weight
        )
        composite_scores['overall_risk_score'] = min(overall_risk, 1.0)
        
        # Risk level classification
        if composite_scores['overall_risk_score'] >= 0.8:
            composite_scores['risk_level'] = 'critical'
        elif composite_scores['overall_risk_score'] >= 0.6:
            composite_scores['risk_level'] = 'high'
        elif composite_scores['overall_risk_score'] >= 0.3:
            composite_scores['risk_level'] = 'medium'
        else:
            composite_scores['risk_level'] = 'low'
        
        return composite_scores
    
    def _get_default_risk_features(self) -> Dict[str, Any]:
        """Get default risk features when calculation fails.
        
        Returns:
            Dictionary of default risk features
        """
        return {
            # Velocity features
            'txn_count_1h': 0, 'txn_count_6h': 0, 'txn_count_24h': 0, 'txn_count_7d': 0,
            'amount_sum_1h': 0.0, 'amount_sum_6h': 0.0, 'amount_sum_24h': 0.0, 'amount_sum_7d': 0.0,
            'avg_amount_1h': 0.0, 'avg_amount_6h': 0.0, 'avg_amount_24h': 0.0, 'avg_amount_7d': 0.0,
            'unique_merchants_1h': 0, 'unique_merchants_24h': 0,
            'unique_countries_1h': 0, 'unique_countries_24h': 0,
            'txn_velocity_ratio_1h_24h': 0.0, 'amount_velocity_ratio_1h_24h': 0.0,
            
            # Amount features
            'amount': 0.0, 'amount_log': 0.0, 'amount_rounded': False, 'amount_category': 'medium',
            'amount_vs_user_avg_ratio': 1.0, 'amount_vs_user_max_ratio': 1.0,
            'amount_vs_merchant_avg_ratio': 1.0, 'amount_vs_category_avg_ratio': 1.0,
            'amount_zscore_user': 0.0, 'amount_zscore_merchant': 0.0, 'amount_zscore_category': 0.0,
            'amount_percentile_user': 50.0, 'amount_percentile_merchant': 50.0, 'amount_percentile_global': 50.0,
            'is_amount_outlier': False, 'amount_deviation_score': 0.0,
            
            # Location features
            'transaction_country': None, 'is_high_risk_country': False, 'country_risk_score': 0.0,
            'distance_from_home_km': 0.0, 'distance_from_last_txn_km': 0.0,
            'impossible_travel_flag': False, 'travel_velocity_kmh': 0.0,
            'is_new_country': False, 'is_frequent_country': False,
            'country_transaction_count': 0, 'country_first_seen_days_ago': 0,
            'timezone_offset_hours': 0.0, 'is_unusual_timezone': False, 'location_entropy': 0.0,
            'countries_visited_24h': 0, 'max_distance_24h_km': 0.0, 'total_travel_distance_24h_km': 0.0,
            
            # Merchant features
            'merchant_id': None, 'merchant_category': None,
            'merchant_risk_score': 0.0, 'is_high_risk_merchant': False, 'merchant_fraud_rate': 0.0,
            'is_new_merchant': False, 'is_frequent_merchant': False,
            'merchant_transaction_count': 0, 'merchant_first_seen_days_ago': 0,
            'is_high_risk_category': False, 'category_risk_score': 0.0, 'category_fraud_rate': 0.0,
            'merchant_avg_amount': 0.0, 'merchant_std_amount': 0.0, 'amount_vs_merchant_pattern': 0.0,
            'unique_merchants_user_24h': 0, 'merchant_diversity_score': 0.0,
            
            # Temporal features
            'hour_of_day': 12, 'day_of_week': 0, 'day_of_month': 1, 'month_of_year': 1,
            'is_weekend': False, 'is_business_hours': True, 'is_unusual_hour': False, 'hour_risk_score': 0.0,
            'time_since_last_txn_minutes': 0.0, 'time_since_last_txn_category': 'normal',
            'is_holiday_period': False, 'is_month_end': False, 'is_payday_period': False,
            'is_user_active_hour': True, 'is_user_active_day': True, 'hour_deviation_from_user_pattern': 0.0,
            'is_night_transaction': False, 'is_early_morning_transaction': False, 'temporal_anomaly_score': 0.0,
            
            # Network features
            'ip_address': None, 'is_vpn': False, 'is_tor': False, 'is_proxy': False, 'ip_risk_score': 0.0,
            'device_id': None, 'is_new_device': False, 'device_risk_score': 0.0, 'device_transaction_count': 0,
            'session_id': None, 'session_transaction_count': 0, 'session_duration_minutes': 0.0,
            'ip_country_mismatch': False, 'device_location_mismatch': False, 'network_anomaly_score': 0.0,
            
            # Composite scores
            'velocity_risk_score': 0.0, 'amount_risk_score': 0.0, 'location_risk_score': 0.0,
            'merchant_risk_score': 0.0, 'temporal_risk_score': 0.0, 'network_risk_score': 0.0,
            'overall_risk_score': 0.0, 'risk_level': 'low'
        }


def create_risk_calculator(config: Optional[RiskFeatureConfig] = None) -> RiskFeatureCalculator:
    """Create risk feature calculator with default or custom configuration.
    
    Args:
        config: Optional risk feature calculation configuration
        
    Returns:
        Risk feature calculator instance
    """
    if config is None:
        config = RiskFeatureConfig()
    
    return RiskFeatureCalculator(config)


# Global risk calculator instance
_calculator_instance: Optional[RiskFeatureCalculator] = None


def get_risk_calculator() -> RiskFeatureCalculator:
    """Get global risk feature calculator instance.
    
    Returns:
        Global risk calculator instance
    """
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = create_risk_calculator()
    return _calculator_instance


def set_risk_calculator(calculator: RiskFeatureCalculator):
    """Set global risk feature calculator instance.
    
    Args:
        calculator: Risk calculator instance to set as global
    """
    global _calculator_instance
    _calculator_instance = calculator