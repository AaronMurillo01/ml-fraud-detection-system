"""User behavior analysis module for fraud detection.

This module provides comprehensive user behavior analysis including:
- Spending pattern analysis
- Location behavior tracking
- Temporal pattern analysis
- Merchant preference analysis
- Anomaly detection in user behavior
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from service.models import EnrichedTransaction

logger = logging.getLogger(__name__)


class BehaviorAnomalyType(str, Enum):
    """Types of behavioral anomalies."""
    AMOUNT_ANOMALY = "amount_anomaly"
    LOCATION_ANOMALY = "location_anomaly"
    TIME_ANOMALY = "time_anomaly"
    MERCHANT_ANOMALY = "merchant_anomaly"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    PATTERN_ANOMALY = "pattern_anomaly"


class RiskLevel(str, Enum):
    """Risk levels for behavioral analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BehaviorPattern:
    """Represents a user behavior pattern."""
    pattern_type: str
    frequency: float
    confidence: float
    last_seen: datetime
    attributes: Dict[str, Any]


@dataclass
class BehaviorAnomaly:
    """Represents a detected behavioral anomaly."""
    anomaly_type: BehaviorAnomalyType
    severity: float  # 0-1 scale
    risk_level: RiskLevel
    description: str
    evidence: Dict[str, Any]
    confidence: float


class UserBehaviorProfile(BaseModel):
    """User behavior profile containing patterns and statistics."""
    
    user_id: str
    profile_created: datetime
    last_updated: datetime
    
    # Spending patterns
    avg_transaction_amount: float = 0.0
    median_transaction_amount: float = 0.0
    std_transaction_amount: float = 0.0
    max_transaction_amount: float = 0.0
    min_transaction_amount: float = 0.0
    
    # Temporal patterns
    preferred_hours: List[int] = Field(default_factory=list)
    preferred_days: List[int] = Field(default_factory=list)  # 0=Monday
    avg_transactions_per_day: float = 0.0
    max_transactions_per_day: int = 0
    
    # Location patterns
    frequent_countries: List[str] = Field(default_factory=list)
    home_country: Optional[str] = None
    travel_frequency: float = 0.0  # Transactions outside home country
    
    # Merchant patterns
    frequent_merchants: List[str] = Field(default_factory=list)
    frequent_categories: List[str] = Field(default_factory=list)
    merchant_diversity: float = 0.0  # Unique merchants / total transactions
    
    # Behavioral statistics
    transaction_count: int = 0
    days_active: int = 0
    avg_time_between_transactions: float = 0.0  # in minutes
    
    # Risk indicators
    anomaly_count: int = 0
    risk_score: float = 0.0
    last_anomaly_date: Optional[datetime] = None
    
    # Pattern confidence scores
    amount_pattern_confidence: float = 0.0
    time_pattern_confidence: float = 0.0
    location_pattern_confidence: float = 0.0
    merchant_pattern_confidence: float = 0.0
    
    class Config:
        use_enum_values = True


class BehaviorAnalysisConfig(BaseModel):
    """Configuration for user behavior analysis."""
    
    # Analysis windows
    short_term_days: int = 7
    medium_term_days: int = 30
    long_term_days: int = 90
    
    # Anomaly detection thresholds
    amount_anomaly_threshold: float = 3.0  # Z-score threshold
    frequency_anomaly_threshold: float = 2.5
    location_anomaly_threshold: float = 0.1  # Probability threshold
    time_anomaly_threshold: float = 0.05
    
    # Pattern detection parameters
    min_transactions_for_pattern: int = 10
    pattern_confidence_threshold: float = 0.7
    
    # Clustering parameters for location analysis
    location_eps: float = 0.1  # DBSCAN epsilon
    location_min_samples: int = 3
    
    # Risk scoring weights
    amount_weight: float = 0.3
    frequency_weight: float = 0.2
    location_weight: float = 0.2
    time_weight: float = 0.15
    merchant_weight: float = 0.15
    
    # Profile update settings
    min_transactions_for_profile: int = 5
    profile_decay_factor: float = 0.95  # For aging patterns
    
    class Config:
        use_enum_values = True


class UserBehaviorAnalyzer:
    """Analyzes user behavior patterns and detects anomalies."""
    
    def __init__(self, config: BehaviorAnalysisConfig):
        """Initialize behavior analyzer.
        
        Args:
            config: Behavior analysis configuration
        """
        self.config = config
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.behavior_patterns: Dict[str, List[BehaviorPattern]] = defaultdict(list)
        
        logger.info("User behavior analyzer initialized")
    
    def analyze_transaction(self, 
                          transaction: EnrichedTransaction, 
                          historical_data: Optional[pd.DataFrame] = None) -> Tuple[List[BehaviorAnomaly], Dict[str, Any]]:
        """Analyze a transaction for behavioral anomalies.
        
        Args:
            transaction: Transaction to analyze
            historical_data: Historical transaction data for the user
            
        Returns:
            Tuple of (detected anomalies, behavior features)
        """
        user_id = transaction.user_id
        anomalies = []
        behavior_features = {}
        
        try:
            # Get or create user profile
            profile = self._get_or_create_profile(user_id, historical_data)
            
            # Update profile with new transaction
            self._update_profile(profile, transaction)
            
            # Detect anomalies
            amount_anomalies = self._detect_amount_anomalies(transaction, profile)
            time_anomalies = self._detect_time_anomalies(transaction, profile)
            location_anomalies = self._detect_location_anomalies(transaction, profile)
            merchant_anomalies = self._detect_merchant_anomalies(transaction, profile)
            frequency_anomalies = self._detect_frequency_anomalies(transaction, profile, historical_data)
            
            anomalies.extend(amount_anomalies)
            anomalies.extend(time_anomalies)
            anomalies.extend(location_anomalies)
            anomalies.extend(merchant_anomalies)
            anomalies.extend(frequency_anomalies)
            
            # Extract behavior features
            behavior_features = self._extract_behavior_features(transaction, profile, anomalies)
            
            # Update profile with anomalies
            if anomalies:
                profile.anomaly_count += len(anomalies)
                profile.last_anomaly_date = transaction.timestamp
                profile.risk_score = self._calculate_risk_score(profile, anomalies)
            
            # Store updated profile
            self.user_profiles[user_id] = profile
            
            return anomalies, behavior_features
            
        except Exception as e:
            logger.error(f"Behavior analysis failed for user {user_id}: {e}")
            return [], self._get_default_behavior_features()
    
    def _get_or_create_profile(self, user_id: str, historical_data: Optional[pd.DataFrame]) -> UserBehaviorProfile:
        """Get existing profile or create new one.
        
        Args:
            user_id: User identifier
            historical_data: Historical transaction data
            
        Returns:
            User behavior profile
        """
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Create new profile
        profile = UserBehaviorProfile(
            user_id=user_id,
            profile_created=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Initialize profile from historical data if available
        if historical_data is not None and not historical_data.empty:
            user_data = historical_data[historical_data['user_id'] == user_id]
            if len(user_data) >= self.config.min_transactions_for_profile:
                self._initialize_profile_from_history(profile, user_data)
        
        return profile
    
    def _initialize_profile_from_history(self, profile: UserBehaviorProfile, user_data: pd.DataFrame):
        """Initialize profile from historical data.
        
        Args:
            profile: User profile to initialize
            user_data: Historical user transaction data
        """
        if user_data.empty:
            return
        
        # Amount patterns
        amounts = user_data['amount'].dropna()
        if len(amounts) > 0:
            profile.avg_transaction_amount = float(amounts.mean())
            profile.median_transaction_amount = float(amounts.median())
            profile.std_transaction_amount = float(amounts.std())
            profile.max_transaction_amount = float(amounts.max())
            profile.min_transaction_amount = float(amounts.min())
            profile.amount_pattern_confidence = min(len(amounts) / 50.0, 1.0)
        
        # Temporal patterns
        if 'timestamp' in user_data.columns:
            timestamps = pd.to_datetime(user_data['timestamp'])
            hours = timestamps.dt.hour
            days = timestamps.dt.dayofweek
            
            # Find preferred hours (top 25% most frequent)
            hour_counts = hours.value_counts()
            top_hours = hour_counts.head(max(1, len(hour_counts) // 4))
            profile.preferred_hours = top_hours.index.tolist()
            
            # Find preferred days
            day_counts = days.value_counts()
            top_days = day_counts.head(max(1, len(day_counts) // 2))
            profile.preferred_days = top_days.index.tolist()
            
            # Calculate daily transaction statistics
            daily_counts = timestamps.dt.date.value_counts()
            profile.avg_transactions_per_day = float(daily_counts.mean())
            profile.max_transactions_per_day = int(daily_counts.max())
            profile.days_active = len(daily_counts)
            
            # Time between transactions
            if len(timestamps) > 1:
                time_diffs = timestamps.sort_values().diff().dt.total_seconds() / 60  # minutes
                profile.avg_time_between_transactions = float(time_diffs.mean())
            
            profile.time_pattern_confidence = min(len(timestamps) / 100.0, 1.0)
        
        # Location patterns
        if 'transaction_country' in user_data.columns:
            countries = user_data['transaction_country'].dropna()
            if len(countries) > 0:
                country_counts = countries.value_counts()
                profile.frequent_countries = country_counts.head(5).index.tolist()
                profile.home_country = country_counts.index[0] if len(country_counts) > 0 else None
                
                # Travel frequency (transactions outside home country)
                if profile.home_country:
                    travel_txns = (countries != profile.home_country).sum()
                    profile.travel_frequency = travel_txns / len(countries)
                
                profile.location_pattern_confidence = min(len(countries) / 30.0, 1.0)
        
        # Merchant patterns
        if 'merchant_id' in user_data.columns:
            merchants = user_data['merchant_id'].dropna()
            if len(merchants) > 0:
                merchant_counts = merchants.value_counts()
                profile.frequent_merchants = merchant_counts.head(10).index.tolist()
                profile.merchant_diversity = len(merchant_counts) / len(merchants)
                profile.merchant_pattern_confidence = min(len(merchants) / 50.0, 1.0)
        
        if 'merchant_category' in user_data.columns:
            categories = user_data['merchant_category'].dropna()
            if len(categories) > 0:
                category_counts = categories.value_counts()
                profile.frequent_categories = category_counts.head(5).index.tolist()
        
        # General statistics
        profile.transaction_count = len(user_data)
        profile.last_updated = datetime.utcnow()
    
    def _update_profile(self, profile: UserBehaviorProfile, transaction: EnrichedTransaction):
        """Update profile with new transaction.
        
        Args:
            profile: User profile to update
            transaction: New transaction
        """
        # Update transaction count
        profile.transaction_count += 1
        profile.last_updated = transaction.timestamp
        
        # Update amount statistics (incremental)
        amount = transaction.amount
        n = profile.transaction_count
        
        if n == 1:
            profile.avg_transaction_amount = amount
            profile.min_transaction_amount = amount
            profile.max_transaction_amount = amount
        else:
            # Incremental mean update
            old_mean = profile.avg_transaction_amount
            profile.avg_transaction_amount = old_mean + (amount - old_mean) / n
            
            # Update min/max
            profile.min_transaction_amount = min(profile.min_transaction_amount, amount)
            profile.max_transaction_amount = max(profile.max_transaction_amount, amount)
        
        # Update temporal patterns
        hour = transaction.timestamp.hour
        day = transaction.timestamp.weekday()
        
        if hour not in profile.preferred_hours and len(profile.preferred_hours) < 12:
            profile.preferred_hours.append(hour)
        
        if day not in profile.preferred_days and len(profile.preferred_days) < 4:
            profile.preferred_days.append(day)
        
        # Update location patterns
        country = getattr(transaction, 'transaction_country', None)
        if country:
            if country not in profile.frequent_countries:
                profile.frequent_countries.append(country)
                if len(profile.frequent_countries) > 10:
                    profile.frequent_countries = profile.frequent_countries[:10]
            
            if not profile.home_country:
                profile.home_country = country
        
        # Update merchant patterns
        merchant_id = getattr(transaction, 'merchant_id', None)
        if merchant_id and merchant_id not in profile.frequent_merchants:
            profile.frequent_merchants.append(merchant_id)
            if len(profile.frequent_merchants) > 20:
                profile.frequent_merchants = profile.frequent_merchants[:20]
        
        category = getattr(transaction, 'merchant_category', None)
        if category and category not in profile.frequent_categories:
            profile.frequent_categories.append(category)
            if len(profile.frequent_categories) > 10:
                profile.frequent_categories = profile.frequent_categories[:10]
    
    def _detect_amount_anomalies(self, transaction: EnrichedTransaction, profile: UserBehaviorProfile) -> List[BehaviorAnomaly]:
        """Detect amount-based anomalies.
        
        Args:
            transaction: Current transaction
            profile: User behavior profile
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if profile.transaction_count < self.config.min_transactions_for_pattern:
            return anomalies
        
        amount = transaction.amount
        
        # Z-score based anomaly detection
        if profile.std_transaction_amount > 0:
            z_score = abs(amount - profile.avg_transaction_amount) / profile.std_transaction_amount
            
            if z_score > self.config.amount_anomaly_threshold:
                severity = min(z_score / 5.0, 1.0)  # Normalize to 0-1
                risk_level = self._get_risk_level(severity)
                
                anomaly = BehaviorAnomaly(
                    anomaly_type=BehaviorAnomalyType.AMOUNT_ANOMALY,
                    severity=severity,
                    risk_level=risk_level,
                    description=f"Transaction amount ${amount:.2f} is {z_score:.1f} standard deviations from user's average ${profile.avg_transaction_amount:.2f}",
                    evidence={
                        "amount": amount,
                        "user_avg": profile.avg_transaction_amount,
                        "user_std": profile.std_transaction_amount,
                        "z_score": z_score
                    },
                    confidence=min(profile.amount_pattern_confidence, 1.0)
                )
                anomalies.append(anomaly)
        
        # Extreme amount checks
        if amount > profile.max_transaction_amount * 2:
            severity = min(amount / (profile.max_transaction_amount * 2), 1.0)
            risk_level = self._get_risk_level(severity)
            
            anomaly = BehaviorAnomaly(
                anomaly_type=BehaviorAnomalyType.AMOUNT_ANOMALY,
                severity=severity,
                risk_level=risk_level,
                description=f"Transaction amount ${amount:.2f} is more than double the user's previous maximum ${profile.max_transaction_amount:.2f}",
                evidence={
                    "amount": amount,
                    "previous_max": profile.max_transaction_amount,
                    "ratio": amount / max(profile.max_transaction_amount, 1)
                },
                confidence=0.9
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_time_anomalies(self, transaction: EnrichedTransaction, profile: UserBehaviorProfile) -> List[BehaviorAnomaly]:
        """Detect time-based anomalies.
        
        Args:
            transaction: Current transaction
            profile: User behavior profile
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if profile.transaction_count < self.config.min_transactions_for_pattern:
            return anomalies
        
        hour = transaction.timestamp.hour
        day = transaction.timestamp.weekday()
        
        # Check if transaction is outside preferred hours
        if profile.preferred_hours and hour not in profile.preferred_hours:
            # Calculate how unusual this hour is
            hour_distance = min([abs(hour - ph) for ph in profile.preferred_hours])
            if hour_distance > 6:  # More than 6 hours from any preferred hour
                severity = min(hour_distance / 12.0, 1.0)
                risk_level = self._get_risk_level(severity)
                
                anomaly = BehaviorAnomaly(
                    anomaly_type=BehaviorAnomalyType.TIME_ANOMALY,
                    severity=severity,
                    risk_level=risk_level,
                    description=f"Transaction at hour {hour} is unusual for user (preferred hours: {profile.preferred_hours})",
                    evidence={
                        "transaction_hour": hour,
                        "preferred_hours": profile.preferred_hours,
                        "hour_distance": hour_distance
                    },
                    confidence=profile.time_pattern_confidence
                )
                anomalies.append(anomaly)
        
        # Check if transaction is on unusual day
        if profile.preferred_days and day not in profile.preferred_days:
            severity = 0.3  # Lower severity for day anomalies
            risk_level = self._get_risk_level(severity)
            
            anomaly = BehaviorAnomaly(
                anomaly_type=BehaviorAnomalyType.TIME_ANOMALY,
                severity=severity,
                risk_level=risk_level,
                description=f"Transaction on day {day} is unusual for user (preferred days: {profile.preferred_days})",
                evidence={
                    "transaction_day": day,
                    "preferred_days": profile.preferred_days
                },
                confidence=profile.time_pattern_confidence * 0.7
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_location_anomalies(self, transaction: EnrichedTransaction, profile: UserBehaviorProfile) -> List[BehaviorAnomaly]:
        """Detect location-based anomalies.
        
        Args:
            transaction: Current transaction
            profile: User behavior profile
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        country = getattr(transaction, 'transaction_country', None)
        if not country or not profile.frequent_countries:
            return anomalies
        
        # Check if transaction is in an unusual country
        if country not in profile.frequent_countries:
            # Calculate severity based on travel frequency
            if profile.travel_frequency < 0.1:  # User rarely travels
                severity = 0.8
            elif profile.travel_frequency < 0.3:
                severity = 0.5
            else:
                severity = 0.2
            
            risk_level = self._get_risk_level(severity)
            
            anomaly = BehaviorAnomaly(
                anomaly_type=BehaviorAnomalyType.LOCATION_ANOMALY,
                severity=severity,
                risk_level=risk_level,
                description=f"Transaction in {country} is unusual for user (frequent countries: {profile.frequent_countries[:3]})",
                evidence={
                    "transaction_country": country,
                    "frequent_countries": profile.frequent_countries,
                    "travel_frequency": profile.travel_frequency
                },
                confidence=profile.location_pattern_confidence
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_merchant_anomalies(self, transaction: EnrichedTransaction, profile: UserBehaviorProfile) -> List[BehaviorAnomaly]:
        """Detect merchant-based anomalies.
        
        Args:
            transaction: Current transaction
            profile: User behavior profile
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        merchant_id = getattr(transaction, 'merchant_id', None)
        category = getattr(transaction, 'merchant_category', None)
        
        # Check merchant familiarity
        if merchant_id and profile.frequent_merchants:
            if merchant_id not in profile.frequent_merchants:
                # New merchant - check if category is familiar
                category_familiar = category in profile.frequent_categories if category else False
                
                if not category_familiar:
                    severity = 0.4
                    risk_level = self._get_risk_level(severity)
                    
                    anomaly = BehaviorAnomaly(
                        anomaly_type=BehaviorAnomalyType.MERCHANT_ANOMALY,
                        severity=severity,
                        risk_level=risk_level,
                        description=f"Transaction with unfamiliar merchant {merchant_id} in unfamiliar category {category}",
                        evidence={
                            "merchant_id": merchant_id,
                            "merchant_category": category,
                            "frequent_merchants": profile.frequent_merchants[:5],
                            "frequent_categories": profile.frequent_categories
                        },
                        confidence=profile.merchant_pattern_confidence
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_frequency_anomalies(self, 
                                   transaction: EnrichedTransaction, 
                                   profile: UserBehaviorProfile, 
                                   historical_data: Optional[pd.DataFrame]) -> List[BehaviorAnomaly]:
        """Detect frequency-based anomalies.
        
        Args:
            transaction: Current transaction
            profile: User behavior profile
            historical_data: Historical transaction data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if historical_data is None or historical_data.empty:
            return anomalies
        
        user_data = historical_data[historical_data['user_id'] == transaction.user_id]
        if len(user_data) < self.config.min_transactions_for_pattern:
            return anomalies
        
        current_time = transaction.timestamp
        
        # Check recent transaction frequency
        recent_window = current_time - timedelta(hours=1)
        recent_txns = user_data[user_data['timestamp'] >= recent_window]
        
        if len(recent_txns) > profile.max_transactions_per_day / 24 * 2:  # More than 2x hourly average
            severity = min(len(recent_txns) / 10.0, 1.0)
            risk_level = self._get_risk_level(severity)
            
            anomaly = BehaviorAnomaly(
                anomaly_type=BehaviorAnomalyType.FREQUENCY_ANOMALY,
                severity=severity,
                risk_level=risk_level,
                description=f"High transaction frequency: {len(recent_txns)} transactions in the last hour",
                evidence={
                    "recent_transaction_count": len(recent_txns),
                    "time_window_hours": 1,
                    "user_avg_per_hour": profile.max_transactions_per_day / 24
                },
                confidence=0.8
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _extract_behavior_features(self, 
                                  transaction: EnrichedTransaction, 
                                  profile: UserBehaviorProfile, 
                                  anomalies: List[BehaviorAnomaly]) -> Dict[str, Any]:
        """Extract behavioral features for ML model.
        
        Args:
            transaction: Current transaction
            profile: User behavior profile
            anomalies: Detected anomalies
            
        Returns:
            Dictionary of behavioral features
        """
        features = {}
        
        # Profile-based features
        features['user_avg_amount'] = profile.avg_transaction_amount
        features['user_std_amount'] = profile.std_transaction_amount
        features['user_max_amount'] = profile.max_transaction_amount
        features['user_transaction_count'] = profile.transaction_count
        features['user_days_active'] = profile.days_active
        features['user_avg_txns_per_day'] = profile.avg_transactions_per_day
        features['user_travel_frequency'] = profile.travel_frequency
        features['user_merchant_diversity'] = profile.merchant_diversity
        features['user_risk_score'] = profile.risk_score
        features['user_anomaly_count'] = profile.anomaly_count
        
        # Pattern confidence features
        features['amount_pattern_confidence'] = profile.amount_pattern_confidence
        features['time_pattern_confidence'] = profile.time_pattern_confidence
        features['location_pattern_confidence'] = profile.location_pattern_confidence
        features['merchant_pattern_confidence'] = profile.merchant_pattern_confidence
        
        # Transaction vs profile comparison
        if profile.avg_transaction_amount > 0:
            features['amount_vs_user_avg_ratio'] = transaction.amount / profile.avg_transaction_amount
        else:
            features['amount_vs_user_avg_ratio'] = 1.0
        
        if profile.std_transaction_amount > 0:
            features['amount_zscore'] = (transaction.amount - profile.avg_transaction_amount) / profile.std_transaction_amount
        else:
            features['amount_zscore'] = 0.0
        
        # Time-based features
        hour = transaction.timestamp.hour
        day = transaction.timestamp.weekday()
        
        features['is_preferred_hour'] = hour in profile.preferred_hours
        features['is_preferred_day'] = day in profile.preferred_days
        
        if profile.preferred_hours:
            features['hour_distance_from_preferred'] = min([abs(hour - ph) for ph in profile.preferred_hours])
        else:
            features['hour_distance_from_preferred'] = 0
        
        # Location features
        country = getattr(transaction, 'transaction_country', None)
        features['is_frequent_country'] = country in profile.frequent_countries if country else False
        features['is_home_country'] = country == profile.home_country if country and profile.home_country else False
        
        # Merchant features
        merchant_id = getattr(transaction, 'merchant_id', None)
        category = getattr(transaction, 'merchant_category', None)
        
        features['is_frequent_merchant'] = merchant_id in profile.frequent_merchants if merchant_id else False
        features['is_frequent_category'] = category in profile.frequent_categories if category else False
        
        # Anomaly features
        features['anomaly_count'] = len(anomalies)
        features['max_anomaly_severity'] = max([a.severity for a in anomalies], default=0.0)
        features['has_amount_anomaly'] = any(a.anomaly_type == BehaviorAnomalyType.AMOUNT_ANOMALY for a in anomalies)
        features['has_time_anomaly'] = any(a.anomaly_type == BehaviorAnomalyType.TIME_ANOMALY for a in anomalies)
        features['has_location_anomaly'] = any(a.anomaly_type == BehaviorAnomalyType.LOCATION_ANOMALY for a in anomalies)
        features['has_merchant_anomaly'] = any(a.anomaly_type == BehaviorAnomalyType.MERCHANT_ANOMALY for a in anomalies)
        features['has_frequency_anomaly'] = any(a.anomaly_type == BehaviorAnomalyType.FREQUENCY_ANOMALY for a in anomalies)
        
        # Risk level features
        critical_anomalies = [a for a in anomalies if a.risk_level == RiskLevel.CRITICAL]
        high_anomalies = [a for a in anomalies if a.risk_level == RiskLevel.HIGH]
        
        features['has_critical_anomaly'] = len(critical_anomalies) > 0
        features['has_high_risk_anomaly'] = len(high_anomalies) > 0
        features['critical_anomaly_count'] = len(critical_anomalies)
        features['high_risk_anomaly_count'] = len(high_anomalies)
        
        return features
    
    def _calculate_risk_score(self, profile: UserBehaviorProfile, anomalies: List[BehaviorAnomaly]) -> float:
        """Calculate overall risk score for user.
        
        Args:
            profile: User behavior profile
            anomalies: Recent anomalies
            
        Returns:
            Risk score between 0 and 1
        """
        if not anomalies:
            return profile.risk_score * 0.95  # Decay existing risk score
        
        # Calculate anomaly-based risk
        anomaly_risk = 0.0
        for anomaly in anomalies:
            weight = 1.0
            if anomaly.anomaly_type == BehaviorAnomalyType.AMOUNT_ANOMALY:
                weight = self.config.amount_weight
            elif anomaly.anomaly_type == BehaviorAnomalyType.FREQUENCY_ANOMALY:
                weight = self.config.frequency_weight
            elif anomaly.anomaly_type == BehaviorAnomalyType.LOCATION_ANOMALY:
                weight = self.config.location_weight
            elif anomaly.anomaly_type == BehaviorAnomalyType.TIME_ANOMALY:
                weight = self.config.time_weight
            elif anomaly.anomaly_type == BehaviorAnomalyType.MERCHANT_ANOMALY:
                weight = self.config.merchant_weight
            
            anomaly_risk += anomaly.severity * weight * anomaly.confidence
        
        # Combine with existing risk score
        new_risk = min(anomaly_risk, 1.0)
        combined_risk = max(profile.risk_score * 0.7 + new_risk * 0.3, new_risk)
        
        return min(combined_risk, 1.0)
    
    def _get_risk_level(self, severity: float) -> RiskLevel:
        """Convert severity to risk level.
        
        Args:
            severity: Severity score (0-1)
            
        Returns:
            Risk level
        """
        if severity >= 0.8:
            return RiskLevel.CRITICAL
        elif severity >= 0.6:
            return RiskLevel.HIGH
        elif severity >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_default_behavior_features(self) -> Dict[str, Any]:
        """Get default behavioral features when analysis fails.
        
        Returns:
            Dictionary of default features
        """
        return {
            'user_avg_amount': 0.0,
            'user_std_amount': 0.0,
            'user_max_amount': 0.0,
            'user_transaction_count': 0,
            'user_days_active': 0,
            'user_avg_txns_per_day': 0.0,
            'user_travel_frequency': 0.0,
            'user_merchant_diversity': 0.0,
            'user_risk_score': 0.0,
            'user_anomaly_count': 0,
            'amount_pattern_confidence': 0.0,
            'time_pattern_confidence': 0.0,
            'location_pattern_confidence': 0.0,
            'merchant_pattern_confidence': 0.0,
            'amount_vs_user_avg_ratio': 1.0,
            'amount_zscore': 0.0,
            'is_preferred_hour': False,
            'is_preferred_day': False,
            'hour_distance_from_preferred': 0,
            'is_frequent_country': False,
            'is_home_country': False,
            'is_frequent_merchant': False,
            'is_frequent_category': False,
            'anomaly_count': 0,
            'max_anomaly_severity': 0.0,
            'has_amount_anomaly': False,
            'has_time_anomaly': False,
            'has_location_anomaly': False,
            'has_merchant_anomaly': False,
            'has_frequency_anomaly': False,
            'has_critical_anomaly': False,
            'has_high_risk_anomaly': False,
            'critical_anomaly_count': 0,
            'high_risk_anomaly_count': 0
        }
    
    def get_user_profile(self, user_id: str) -> Optional[UserBehaviorProfile]:
        """Get user behavior profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User behavior profile if exists
        """
        return self.user_profiles.get(user_id)
    
    def update_user_profile(self, user_id: str, profile: UserBehaviorProfile):
        """Update user behavior profile.
        
        Args:
            user_id: User identifier
            profile: Updated profile
        """
        self.user_profiles[user_id] = profile
    
    def get_behavior_summary(self, user_id: str) -> Dict[str, Any]:
        """Get behavior analysis summary for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Behavior summary dictionary
        """
        profile = self.get_user_profile(user_id)
        if not profile:
            return {"error": "User profile not found"}
        
        return {
            "user_id": user_id,
            "profile_age_days": (datetime.utcnow() - profile.profile_created).days,
            "transaction_count": profile.transaction_count,
            "risk_score": profile.risk_score,
            "anomaly_count": profile.anomaly_count,
            "spending_pattern": {
                "avg_amount": profile.avg_transaction_amount,
                "max_amount": profile.max_transaction_amount,
                "pattern_confidence": profile.amount_pattern_confidence
            },
            "temporal_pattern": {
                "preferred_hours": profile.preferred_hours,
                "preferred_days": profile.preferred_days,
                "pattern_confidence": profile.time_pattern_confidence
            },
            "location_pattern": {
                "home_country": profile.home_country,
                "travel_frequency": profile.travel_frequency,
                "pattern_confidence": profile.location_pattern_confidence
            },
            "merchant_pattern": {
                "frequent_categories": profile.frequent_categories,
                "merchant_diversity": profile.merchant_diversity,
                "pattern_confidence": profile.merchant_pattern_confidence
            }
        }


def create_behavior_analyzer(config: Optional[BehaviorAnalysisConfig] = None) -> UserBehaviorAnalyzer:
    """Create behavior analyzer with default or custom configuration.
    
    Args:
        config: Optional behavior analysis configuration
        
    Returns:
        Behavior analyzer instance
    """
    if config is None:
        config = BehaviorAnalysisConfig()
    
    return UserBehaviorAnalyzer(config)


# Global behavior analyzer instance
_analyzer_instance: Optional[UserBehaviorAnalyzer] = None


def get_behavior_analyzer() -> UserBehaviorAnalyzer:
    """Get global behavior analyzer instance.
    
    Returns:
        Global behavior analyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = create_behavior_analyzer()
    return _analyzer_instance


def set_behavior_analyzer(analyzer: UserBehaviorAnalyzer):
    """Set global behavior analyzer instance.
    
    Args:
        analyzer: Behavior analyzer instance to set as global
    """
    global _analyzer_instance
    _analyzer_instance = analyzer