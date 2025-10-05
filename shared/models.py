"""Shared data models for the fraud detection system."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class TransactionStatus(str, Enum):
    """Transaction status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    REVIEW = "review"


class PaymentMethod(str, Enum):
    """Payment method enumeration."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CASH = "cash"


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Transaction(BaseModel):
    """Transaction data model."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: Decimal = Field(..., description="Transaction amount", gt=0)
    currency: str = Field(default="USD", description="Currency code")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    location: Optional[str] = Field(None, description="Transaction location")
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    payment_method: PaymentMethod = Field(..., description="Payment method used")
    merchant_category: Optional[str] = Field(None, description="Merchant category")
    transaction_type: str = Field(..., description="Type of transaction")
    status: TransactionStatus = Field(default=TransactionStatus.PENDING, description="Transaction status")
    risk_score: Optional[float] = Field(None, description="Calculated risk score", ge=0, le=1)
    is_fraud: Optional[bool] = Field(None, description="Fraud flag")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v)
        }
        use_enum_values = True


class EnrichedTransaction(Transaction):
    """Enriched transaction with additional features."""
    
    user_profile: Optional[Dict[str, Any]] = Field(None, description="User profile data")
    merchant_profile: Optional[Dict[str, Any]] = Field(None, description="Merchant profile data")
    transaction_history: Optional[List[Dict[str, Any]]] = Field(None, description="User transaction history")
    velocity_features: Optional[Dict[str, float]] = Field(None, description="Velocity-based features")
    risk_features: Optional[Dict[str, float]] = Field(None, description="Risk-based features")
    location_features: Optional[Dict[str, Any]] = Field(None, description="Location-based features")
    device_features: Optional[Dict[str, Any]] = Field(None, description="Device-based features")


class FraudPrediction(BaseModel):
    """Fraud prediction result model."""
    
    prediction_id: str = Field(..., description="Unique prediction identifier")
    transaction_id: str = Field(..., description="Associated transaction ID")
    user_id: str = Field(..., description="User identifier")
    model_version: str = Field(..., description="Model version used")
    fraud_probability: float = Field(..., description="Fraud probability", ge=0, le=1)
    risk_score: float = Field(..., description="Risk score", ge=0, le=1)
    prediction_timestamp: datetime = Field(..., description="Prediction timestamp")
    model_features: Optional[Dict[str, Any]] = Field(None, description="Features used in prediction")
    decision: str = Field(..., description="Final decision (approve/decline/review)")
    confidence_score: Optional[float] = Field(None, description="Confidence in prediction", ge=0, le=1)
    explanation: Optional[Dict[str, Any]] = Field(None, description="Prediction explanation")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    class Config:
        """Pydantic configuration."""
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class User(BaseModel):
    """User data model."""
    
    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    phone_number: Optional[str] = Field(None, description="Phone number")
    registration_date: datetime = Field(..., description="Registration date")
    account_status: str = Field(default="active", description="Account status")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="User risk level")
    kyc_status: str = Field(default="pending", description="KYC verification status")
    profile_data: Optional[Dict[str, Any]] = Field(None, description="Additional profile data")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True


class Merchant(BaseModel):
    """Merchant data model."""
    
    merchant_id: str = Field(..., description="Unique merchant identifier")
    merchant_name: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Merchant category")
    location: Optional[str] = Field(None, description="Merchant location")
    risk_score: Optional[float] = Field(0.5, description="Merchant risk score", ge=0, le=1)
    registration_date: datetime = Field(..., description="Registration date")
    status: str = Field(default="active", description="Merchant status")
    merchant_data: Optional[Dict[str, Any]] = Field(None, description="Additional merchant data")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelMetadata(BaseModel):
    """ML model metadata model."""
    
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (e.g., xgboost, neural_network)")
    training_date: datetime = Field(..., description="Training date")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Model performance metrics")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    feature_importance: Optional[Dict[str, Any]] = Field(None, description="Feature importance scores")
    model_path: str = Field(..., description="Path to model file")
    is_active: bool = Field(default=False, description="Whether model is active")
    deployment_date: Optional[datetime] = Field(None, description="Deployment date")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        protected_namespaces = ()
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }