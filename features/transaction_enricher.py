"""Transaction enrichment module for fraud detection.

This module enriches raw transactions with additional contextual information
including merchant data, location analysis, time-based features, and historical patterns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

from service.models import Transaction, EnrichedTransaction

logger = logging.getLogger(__name__)


class MerchantCategory(str, Enum):
    """Merchant category classifications."""
    GROCERY = "grocery"
    GAS_STATION = "gas_station"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    ONLINE = "online"
    ATM = "atm"
    PHARMACY = "pharmacy"
    ENTERTAINMENT = "entertainment"
    TRAVEL = "travel"
    UTILITY = "utility"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    FINANCIAL = "financial"
    OTHER = "other"


class TransactionChannel(str, Enum):
    """Transaction channel types."""
    CARD_PRESENT = "card_present"
    CARD_NOT_PRESENT = "card_not_present"
    ONLINE = "online"
    MOBILE = "mobile"
    ATM = "atm"
    PHONE = "phone"
    MAIL_ORDER = "mail_order"
    RECURRING = "recurring"


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MerchantInfo:
    """Merchant information for enrichment."""
    merchant_id: str
    name: str
    category: MerchantCategory
    risk_score: float
    location: Optional[str] = None
    country: Optional[str] = None
    is_high_risk: bool = False
    fraud_rate: float = 0.0


@dataclass
class LocationInfo:
    """Location information for enrichment."""
    country: str
    city: Optional[str] = None
    region: Optional[str] = None
    is_high_risk_location: bool = False
    distance_from_home: Optional[float] = None
    timezone_offset: Optional[int] = None


class EnrichmentConfig(BaseModel):
    """Configuration for transaction enrichment."""
    
    # Time-based features
    enable_time_features: bool = True
    timezone: str = "UTC"
    business_hours_start: int = 9
    business_hours_end: int = 17
    
    # Location features
    enable_location_features: bool = True
    home_country: str = "US"
    high_risk_countries: List[str] = Field(default_factory=lambda: [
        "NG", "GH", "RO", "BG", "PK", "BD", "ID", "MY"
    ])
    
    # Merchant features
    enable_merchant_features: bool = True
    high_risk_categories: List[str] = Field(default_factory=lambda: [
        "online", "travel", "entertainment", "other"
    ])
    
    # Amount-based features
    enable_amount_features: bool = True
    round_amount_threshold: float = 0.01
    large_amount_threshold: float = 1000.0
    
    # Velocity features
    enable_velocity_features: bool = True
    velocity_windows: List[int] = Field(default_factory=lambda: [1, 5, 15, 60, 1440])  # minutes
    
    # Pattern features
    enable_pattern_features: bool = True
    
    class Config:
        use_enum_values = True


class TransactionEnricher:
    """Enriches transactions with additional contextual features."""
    
    def __init__(self, config: EnrichmentConfig):
        """Initialize transaction enricher.
        
        Args:
            config: Enrichment configuration
        """
        self.config = config
        self.merchant_cache: Dict[str, MerchantInfo] = {}
        self.location_cache: Dict[str, LocationInfo] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default merchant categories
        self._initialize_merchant_categories()
        
        logger.info("Transaction enricher initialized")
    
    def _initialize_merchant_categories(self):
        """Initialize default merchant category mappings."""
        # This would typically be loaded from a database or external service
        default_merchants = {
            "AMAZON": MerchantInfo("AMAZON", "Amazon", MerchantCategory.ONLINE, 0.3, is_high_risk=True),
            "WALMART": MerchantInfo("WALMART", "Walmart", MerchantCategory.RETAIL, 0.1),
            "SHELL": MerchantInfo("SHELL", "Shell", MerchantCategory.GAS_STATION, 0.05),
            "MCDONALDS": MerchantInfo("MCDONALDS", "McDonald's", MerchantCategory.RESTAURANT, 0.02),
            "ATM": MerchantInfo("ATM", "ATM", MerchantCategory.ATM, 0.15, is_high_risk=True),
        }
        
        for merchant_id, info in default_merchants.items():
            self.merchant_cache[merchant_id] = info
    
    def enrich_transaction(self, transaction: Transaction) -> EnrichedTransaction:
        """Enrich a single transaction with additional features.
        
        Args:
            transaction: Raw transaction to enrich
            
        Returns:
            Enriched transaction with additional features
        """
        try:
            # Start with base transaction data
            enriched_data = transaction.dict()
            
            # Add time-based features
            if self.config.enable_time_features:
                time_features = self._extract_time_features(transaction.timestamp)
                enriched_data.update(time_features)
            
            # Add merchant features
            if self.config.enable_merchant_features:
                merchant_features = self._extract_merchant_features(transaction.merchant_id)
                enriched_data.update(merchant_features)
            
            # Add location features
            if self.config.enable_location_features:
                location_features = self._extract_location_features(
                    transaction.merchant_country, transaction.user_id
                )
                enriched_data.update(location_features)
            
            # Add amount-based features
            if self.config.enable_amount_features:
                amount_features = self._extract_amount_features(transaction.amount)
                enriched_data.update(amount_features)
            
            # Add pattern features
            if self.config.enable_pattern_features:
                pattern_features = self._extract_pattern_features(transaction)
                enriched_data.update(pattern_features)
            
            # Add velocity features (requires historical data)
            if self.config.enable_velocity_features:
                velocity_features = self._extract_velocity_features(transaction)
                enriched_data.update(velocity_features)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(enriched_data)
            enriched_data['risk_score'] = risk_score
            enriched_data['risk_level'] = self._classify_risk_level(risk_score)
            
            # Add enrichment metadata
            enriched_data['enrichment_timestamp'] = datetime.utcnow()
            enriched_data['enrichment_version'] = "1.0"
            
            return EnrichedTransaction(**enriched_data)
            
        except Exception as e:
            logger.error(f"Transaction enrichment failed for {transaction.transaction_id}: {e}")
            # Return minimal enriched transaction on error
            return EnrichedTransaction(
                **transaction.dict(),
                risk_score=0.5,
                risk_level=RiskLevel.MEDIUM,
                enrichment_timestamp=datetime.utcnow(),
                enrichment_version="1.0"
            )
    
    def _extract_time_features(self, timestamp: datetime) -> Dict[str, Any]:
        """Extract time-based features from transaction timestamp.
        
        Args:
            timestamp: Transaction timestamp
            
        Returns:
            Dictionary of time-based features
        """
        features = {}
        
        # Basic time components
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()  # 0=Monday, 6=Sunday
        features['day_of_month'] = timestamp.day
        features['month'] = timestamp.month
        features['quarter'] = (timestamp.month - 1) // 3 + 1
        
        # Business hours
        is_business_hours = (
            self.config.business_hours_start <= timestamp.hour < self.config.business_hours_end
        )
        features['is_business_hours'] = is_business_hours
        
        # Weekend
        features['is_weekend'] = timestamp.weekday() >= 5
        
        # Late night (11 PM - 6 AM)
        features['is_late_night'] = timestamp.hour >= 23 or timestamp.hour < 6
        
        # Early morning (6 AM - 9 AM)
        features['is_early_morning'] = 6 <= timestamp.hour < 9
        
        # Time since epoch (for trend analysis)
        features['timestamp_epoch'] = timestamp.timestamp()
        
        return features
    
    def _extract_merchant_features(self, merchant_id: str) -> Dict[str, Any]:
        """Extract merchant-based features.
        
        Args:
            merchant_id: Merchant identifier
            
        Returns:
            Dictionary of merchant-based features
        """
        features = {}
        
        # Get merchant info (with fallback)
        merchant_info = self.merchant_cache.get(
            merchant_id,
            MerchantInfo(merchant_id, "Unknown", MerchantCategory.OTHER, 0.5)
        )
        
        features['merchant_category'] = merchant_info.category.value
        features['merchant_risk_score'] = merchant_info.risk_score
        features['is_high_risk_merchant'] = merchant_info.is_high_risk
        features['merchant_fraud_rate'] = merchant_info.fraud_rate
        
        # Category-based features
        features['is_online_merchant'] = merchant_info.category == MerchantCategory.ONLINE
        features['is_atm_transaction'] = merchant_info.category == MerchantCategory.ATM
        features['is_gas_station'] = merchant_info.category == MerchantCategory.GAS_STATION
        features['is_restaurant'] = merchant_info.category == MerchantCategory.RESTAURANT
        features['is_retail'] = merchant_info.category == MerchantCategory.RETAIL
        
        # High-risk category check
        features['is_high_risk_category'] = (
            merchant_info.category.value in self.config.high_risk_categories
        )
        
        return features
    
    def _extract_location_features(self, country: str, user_id: str) -> Dict[str, Any]:
        """Extract location-based features.
        
        Args:
            country: Transaction country
            user_id: User identifier
            
        Returns:
            Dictionary of location-based features
        """
        features = {}
        
        # Country-based features
        features['transaction_country'] = country
        features['is_domestic'] = country == self.config.home_country
        features['is_high_risk_country'] = country in self.config.high_risk_countries
        
        # User's typical location (would be loaded from user profile)
        user_profile = self.user_profiles.get(user_id, {})
        home_country = user_profile.get('home_country', self.config.home_country)
        
        features['is_foreign_transaction'] = country != home_country
        features['home_country'] = home_country
        
        # Distance-based features (simplified)
        if country != home_country:
            # This would typically use actual geographic distance calculation
            features['estimated_distance_km'] = self._estimate_country_distance(home_country, country)
            features['is_long_distance'] = features['estimated_distance_km'] > 1000
        else:
            features['estimated_distance_km'] = 0.0
            features['is_long_distance'] = False
        
        return features
    
    def _extract_amount_features(self, amount: float) -> Dict[str, Any]:
        """Extract amount-based features.
        
        Args:
            amount: Transaction amount
            
        Returns:
            Dictionary of amount-based features
        """
        features = {}
        
        # Basic amount features
        features['amount_log'] = pd.np.log1p(amount)  # Log transformation
        features['amount_sqrt'] = pd.np.sqrt(amount)  # Square root transformation
        
        # Round amount detection
        decimal_part = amount % 1
        features['is_round_amount'] = decimal_part < self.config.round_amount_threshold
        features['decimal_places'] = len(str(decimal_part).split('.')[-1]) if decimal_part > 0 else 0
        
        # Amount categories
        features['is_small_amount'] = amount < 10.0
        features['is_medium_amount'] = 10.0 <= amount < 100.0
        features['is_large_amount'] = amount >= self.config.large_amount_threshold
        features['is_very_large_amount'] = amount >= 5000.0
        
        # Amount patterns
        features['ends_with_00'] = str(int(amount * 100)).endswith('00')
        features['ends_with_99'] = str(int(amount * 100)).endswith('99')
        
        # Amount bins (for categorical analysis)
        if amount < 25:
            amount_bin = "0-25"
        elif amount < 100:
            amount_bin = "25-100"
        elif amount < 500:
            amount_bin = "100-500"
        elif amount < 1000:
            amount_bin = "500-1000"
        else:
            amount_bin = "1000+"
        
        features['amount_bin'] = amount_bin
        
        return features
    
    def _extract_pattern_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract pattern-based features.
        
        Args:
            transaction: Transaction object
            
        Returns:
            Dictionary of pattern-based features
        """
        features = {}
        
        # Transaction ID patterns
        tx_id = transaction.transaction_id
        features['tx_id_length'] = len(tx_id)
        features['tx_id_has_numbers'] = any(c.isdigit() for c in tx_id)
        features['tx_id_has_letters'] = any(c.isalpha() for c in tx_id)
        
        # Card number patterns (if available)
        if hasattr(transaction, 'card_number') and transaction.card_number:
            card_num = str(transaction.card_number)
            features['card_bin'] = card_num[:6] if len(card_num) >= 6 else card_num
            features['card_last_4'] = card_num[-4:] if len(card_num) >= 4 else card_num
        
        # Merchant name patterns
        merchant_name = transaction.merchant_id.upper()
        features['merchant_name_length'] = len(merchant_name)
        features['merchant_has_numbers'] = any(c.isdigit() for c in merchant_name)
        
        # Common fraud patterns
        features['has_test_pattern'] = any(word in merchant_name.lower() for word in ['test', 'temp', 'dummy'])
        
        return features
    
    def _extract_velocity_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract velocity-based features.
        
        Args:
            transaction: Transaction object
            
        Returns:
            Dictionary of velocity-based features
        """
        features = {}
        
        # This would typically query historical transactions from a database
        # For now, we'll use placeholder values
        
        user_id = transaction.user_id
        current_time = transaction.timestamp
        
        # Simulate velocity calculations for different time windows
        for window_minutes in self.config.velocity_windows:
            window_start = current_time - timedelta(minutes=window_minutes)
            
            # These would be actual database queries in production
            features[f'tx_count_{window_minutes}m'] = self._get_transaction_count(
                user_id, window_start, current_time
            )
            features[f'amount_sum_{window_minutes}m'] = self._get_amount_sum(
                user_id, window_start, current_time
            )
            features[f'unique_merchants_{window_minutes}m'] = self._get_unique_merchants(
                user_id, window_start, current_time
            )
            features[f'unique_countries_{window_minutes}m'] = self._get_unique_countries(
                user_id, window_start, current_time
            )
        
        # Cross-window ratios
        if features['tx_count_60m'] > 0:
            features['tx_ratio_5m_60m'] = features['tx_count_5m'] / features['tx_count_60m']
        else:
            features['tx_ratio_5m_60m'] = 0.0
        
        return features
    
    def _get_transaction_count(self, user_id: str, start_time: datetime, end_time: datetime) -> int:
        """Get transaction count for user in time window (placeholder)."""
        # This would query the database in production
        return 1  # Placeholder
    
    def _get_amount_sum(self, user_id: str, start_time: datetime, end_time: datetime) -> float:
        """Get sum of transaction amounts for user in time window (placeholder)."""
        # This would query the database in production
        return 100.0  # Placeholder
    
    def _get_unique_merchants(self, user_id: str, start_time: datetime, end_time: datetime) -> int:
        """Get unique merchant count for user in time window (placeholder)."""
        # This would query the database in production
        return 1  # Placeholder
    
    def _get_unique_countries(self, user_id: str, start_time: datetime, end_time: datetime) -> int:
        """Get unique country count for user in time window (placeholder)."""
        # This would query the database in production
        return 1  # Placeholder
    
    def _estimate_country_distance(self, country1: str, country2: str) -> float:
        """Estimate distance between countries (simplified).
        
        Args:
            country1: First country code
            country2: Second country code
            
        Returns:
            Estimated distance in kilometers
        """
        # Simplified distance estimation
        # In production, this would use actual geographic coordinates
        distance_map = {
            ('US', 'CA'): 500,
            ('US', 'MX'): 1000,
            ('US', 'GB'): 5500,
            ('US', 'DE'): 6000,
            ('US', 'JP'): 10000,
            ('US', 'AU'): 15000,
        }
        
        key = tuple(sorted([country1, country2]))
        return distance_map.get(key, 8000)  # Default to 8000km
    
    def _calculate_risk_score(self, enriched_data: Dict[str, Any]) -> float:
        """Calculate overall risk score based on enriched features.
        
        Args:
            enriched_data: Dictionary of enriched features
            
        Returns:
            Risk score between 0 and 1
        """
        risk_score = 0.0
        
        # Time-based risk factors
        if enriched_data.get('is_late_night', False):
            risk_score += 0.1
        if enriched_data.get('is_weekend', False):
            risk_score += 0.05
        if not enriched_data.get('is_business_hours', True):
            risk_score += 0.05
        
        # Location-based risk factors
        if enriched_data.get('is_high_risk_country', False):
            risk_score += 0.3
        if enriched_data.get('is_foreign_transaction', False):
            risk_score += 0.1
        if enriched_data.get('is_long_distance', False):
            risk_score += 0.15
        
        # Merchant-based risk factors
        if enriched_data.get('is_high_risk_merchant', False):
            risk_score += 0.2
        if enriched_data.get('is_high_risk_category', False):
            risk_score += 0.1
        
        # Amount-based risk factors
        if enriched_data.get('is_very_large_amount', False):
            risk_score += 0.2
        elif enriched_data.get('is_large_amount', False):
            risk_score += 0.1
        if enriched_data.get('is_round_amount', False):
            risk_score += 0.05
        
        # Velocity-based risk factors
        tx_count_5m = enriched_data.get('tx_count_5m', 0)
        if tx_count_5m > 3:
            risk_score += 0.3
        elif tx_count_5m > 1:
            risk_score += 0.1
        
        # Pattern-based risk factors
        if enriched_data.get('has_test_pattern', False):
            risk_score += 0.4
        
        # Normalize to 0-1 range
        return min(risk_score, 1.0)
    
    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk level based on risk score.
        
        Args:
            risk_score: Risk score between 0 and 1
            
        Returns:
            Risk level classification
        """
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Update user profile for enrichment.
        
        Args:
            user_id: User identifier
            profile_data: Profile data to update
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        self.user_profiles[user_id].update(profile_data)
        logger.debug(f"Updated profile for user {user_id}")
    
    def add_merchant_info(self, merchant_id: str, merchant_info: MerchantInfo):
        """Add merchant information to cache.
        
        Args:
            merchant_id: Merchant identifier
            merchant_info: Merchant information
        """
        self.merchant_cache[merchant_id] = merchant_info
        logger.debug(f"Added merchant info for {merchant_id}")
    
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics.
        
        Returns:
            Dictionary with enrichment statistics
        """
        return {
            "merchants_cached": len(self.merchant_cache),
            "locations_cached": len(self.location_cache),
            "user_profiles": len(self.user_profiles),
            "config": self.config.dict()
        }


def create_transaction_enricher(config: Optional[EnrichmentConfig] = None) -> TransactionEnricher:
    """Create transaction enricher with default or custom configuration.
    
    Args:
        config: Optional enrichment configuration
        
    Returns:
        Transaction enricher instance
    """
    if config is None:
        config = EnrichmentConfig()
    
    return TransactionEnricher(config)


# Global enricher instance
_enricher_instance: Optional[TransactionEnricher] = None


def get_enricher() -> TransactionEnricher:
    """Get global transaction enricher instance.
    
    Returns:
        Global transaction enricher instance
    """
    global _enricher_instance
    if _enricher_instance is None:
        _enricher_instance = create_transaction_enricher()
    return _enricher_instance


def set_enricher(enricher: TransactionEnricher):
    """Set global transaction enricher instance.
    
    Args:
        enricher: Transaction enricher instance to set as global
    """
    global _enricher_instance
    _enricher_instance = enricher