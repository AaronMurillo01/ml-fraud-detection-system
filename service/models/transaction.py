"""Transaction data models for fraud detection."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID

from pydantic import Field, validator, root_validator

from .base import BaseModel, EntityModel


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    REFUND = "refund"
    PAYMENT = "payment"
    DEPOSIT = "deposit"


class TransactionStatus(str, Enum):
    """Transaction status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PaymentMethod(str, Enum):
    """Payment method enumeration."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CASH = "cash"
    CHECK = "check"


class Transaction(EntityModel):
    """Core transaction model matching the database schema."""
    
    # Core transaction fields
    transaction_id: str = Field(
        ...,
        description="Unique transaction identifier from payment processor",
        min_length=1,
        max_length=100,
        json_schema_extra={"example": "txn_1234567890abcdef"}
    )
    
    user_id: str = Field(
        ...,
        description="User identifier who initiated the transaction",
        min_length=1,
        max_length=50,
        json_schema_extra={"example": "user_12345"}
    )
    
    card_id: Optional[str] = Field(
        None,
        description="Card identifier (hashed/tokenized)",
        max_length=100,
        json_schema_extra={"example": "card_hash_abcdef123456"}
    )
    
    merchant_id: str = Field(
        ...,
        description="Merchant identifier",
        min_length=1,
        max_length=50,
        json_schema_extra={"example": "merchant_67890"}
    )
    
    # Transaction details
    amount: Decimal = Field(
        ...,
        description="Transaction amount in the specified currency",
        gt=0,
        max_digits=12,
        decimal_places=2,
        json_schema_extra={"example": 125.50}
    )
    
    currency: str = Field(
        ...,
        description="ISO 4217 currency code",
        min_length=3,
        max_length=3,
        json_schema_extra={"example": "USD"}
    )
    
    transaction_type: TransactionType = Field(
        ...,
        description="Type of transaction",
        json_schema_extra={"example": "purchase"}
    )
    
    payment_method: PaymentMethod = Field(
        ...,
        description="Payment method used",
        json_schema_extra={"example": "credit_card"}
    )
    
    status: TransactionStatus = Field(
        default=TransactionStatus.PENDING,
        description="Current transaction status",
        json_schema_extra={"example": "approved"}
    )
    
    # Location and device information
    ip_address: Optional[str] = Field(
        None,
        description="IP address of the transaction origin",
        max_length=45,  # IPv6 max length
        json_schema_extra={"example": "192.168.1.100"}
    )
    
    device_id: Optional[str] = Field(
        None,
        description="Device identifier (fingerprint)",
        max_length=100,
        json_schema_extra={"example": "device_fingerprint_xyz789"}
    )
    
    location_lat: Optional[float] = Field(
        None,
        description="Latitude of transaction location",
        ge=-90,
        le=90,
        json_schema_extra={"example": 40.7128}
    )
    
    location_lon: Optional[float] = Field(
        None,
        description="Longitude of transaction location",
        ge=-180,
        le=180,
        json_schema_extra={"example": -74.0060}
    )
    
    country_code: Optional[str] = Field(
        None,
        description="ISO 3166-1 alpha-2 country code",
        min_length=2,
        max_length=2,
        json_schema_extra={"example": "US"}
    )
    
    # Timing information
    transaction_time: datetime = Field(
        ...,
        description="When the transaction occurred",
        json_schema_extra={"example": "2024-01-15T14:30:00Z"}
    )
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional transaction metadata",
        json_schema_extra={
            "example": {
                "channel": "online",
                "category": "retail",
                "description": "Online purchase at Example Store"
            }
        }
    )
    
    @validator('currency')
    def validate_currency(cls, v):
        """Validate currency code format."""
        if v:
            return v.upper()
        return v
    
    @validator('country_code')
    def validate_country_code(cls, v):
        """Validate country code format."""
        if v:
            return v.upper()
        return v
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        """Basic IP address format validation."""
        if v:
            # Basic validation - could be enhanced with ipaddress module
            parts = v.split('.')
            if len(parts) == 4:
                try:
                    for part in parts:
                        if not (0 <= int(part) <= 255):
                            raise ValueError("Invalid IP address")
                except ValueError:
                    raise ValueError("Invalid IP address format")
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_location_coordinates(cls, values):
        """Validate that both lat and lon are provided together."""
        lat = values.get('location_lat')
        lon = values.get('location_lon')
        
        if (lat is None) != (lon is None):
            raise ValueError("Both latitude and longitude must be provided together")
        
        return values


class EnrichedTransaction(Transaction):
    """Transaction model with computed features for ML inference."""
    
    # Risk indicators
    is_weekend: bool = Field(
        ...,
        description="Whether transaction occurred on weekend",
        json_schema_extra={"example": False}
    )
    
    is_night_time: bool = Field(
        ...,
        description="Whether transaction occurred during night hours (10PM-6AM)",
        json_schema_extra={"example": False}
    )
    
    # Velocity features (computed from historical data)
    user_transaction_count_1h: int = Field(
        default=0,
        description="Number of transactions by user in last 1 hour",
        ge=0,
        json_schema_extra={"example": 2}
    )
    
    user_transaction_count_24h: int = Field(
        default=0,
        description="Number of transactions by user in last 24 hours",
        ge=0,
        json_schema_extra={"example": 15}
    )
    
    user_amount_sum_1h: Decimal = Field(
        default=Decimal('0.00'),
        description="Total amount spent by user in last 1 hour",
        ge=0,
        max_digits=12,
        decimal_places=2,
        json_schema_extra={"example": 250.00}
    )
    
    user_amount_sum_24h: Decimal = Field(
        default=Decimal('0.00'),
        description="Total amount spent by user in last 24 hours",
        ge=0,
        max_digits=12,
        decimal_places=2,
        json_schema_extra={"example": 1500.00}
    )
    
    # Card-based features
    card_transaction_count_1h: int = Field(
        default=0,
        description="Number of transactions with this card in last 1 hour",
        ge=0,
        json_schema_extra={"example": 1}
    )
    
    card_transaction_count_24h: int = Field(
        default=0,
        description="Number of transactions with this card in last 24 hours",
        ge=0,
        json_schema_extra={"example": 8}
    )
    
    # Merchant-based features
    merchant_transaction_count_1h: int = Field(
        default=0,
        description="Number of transactions at this merchant in last 1 hour",
        ge=0,
        json_schema_extra={"example": 45}
    )
    
    # Location-based features
    distance_from_home: Optional[float] = Field(
        None,
        description="Distance from user's home location in kilometers",
        ge=0,
        json_schema_extra={"example": 15.5}
    )
    
    distance_from_last_transaction: Optional[float] = Field(
        None,
        description="Distance from last transaction location in kilometers",
        ge=0,
        json_schema_extra={"example": 2.3}
    )
    
    # Device and IP features
    is_new_device: bool = Field(
        default=False,
        description="Whether this is a new device for the user",
        json_schema_extra={"example": False}
    )
    
    is_new_ip: bool = Field(
        default=False,
        description="Whether this is a new IP address for the user",
        json_schema_extra={"example": True}
    )
    
    # Statistical features
    amount_zscore_user_7d: Optional[float] = Field(
        None,
        description="Z-score of amount compared to user's 7-day average",
        json_schema_extra={"example": 1.5}
    )
    
    amount_zscore_merchant_7d: Optional[float] = Field(
        None,
        description="Z-score of amount compared to merchant's 7-day average",
        json_schema_extra={"example": 0.8}
    )
    
    # Time-based features
    hour_of_day: int = Field(
        ...,
        description="Hour of day when transaction occurred (0-23)",
        ge=0,
        le=23,
        json_schema_extra={"example": 14}
    )
    
    day_of_week: int = Field(
        ...,
        description="Day of week when transaction occurred (0=Monday, 6=Sunday)",
        ge=0,
        le=6,
        json_schema_extra={"example": 2}
    )
    
    # Feature computation timestamp
    features_computed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the features were computed",
        json_schema_extra={"example": "2024-01-15T14:30:05Z"}
    )


class TransactionRequest(BaseModel):
    """Request model for transaction scoring API."""
    
    transaction: Transaction = Field(
        ...,
        description="Transaction data to be scored"
    )
    
    compute_explanation: bool = Field(
        default=False,
        description="Whether to compute SHAP explanations"
    )
    
    include_features: bool = Field(
        default=False,
        description="Whether to include computed features in response"
    )


class BatchTransactionRequest(BaseModel):
    """Request model for batch transaction scoring."""
    
    transactions: list[Transaction] = Field(
        ...,
        description="List of transactions to be scored",
        min_items=1,
        max_items=1000
    )
    
    compute_explanations: bool = Field(
        default=False,
        description="Whether to compute SHAP explanations for all transactions"
    )
    
    include_features: bool = Field(
        default=False,
        description="Whether to include computed features in response"
    )