"""Database ORM models."""

from sqlalchemy import String, DateTime, Boolean, Float, JSON, Integer, DECIMAL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, List, Optional
from decimal import Decimal as DecimalType

class Base(DeclarativeBase):
    pass


class TransactionModel(Base):
    """Transaction model for storing transaction data."""
    
    __tablename__ = 'transactions'
    
    transaction_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    merchant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    amount: Mapped[float] = mapped_column(DECIMAL(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default='USD')
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    device_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payment_method: Mapped[str] = mapped_column(String, nullable=False)
    merchant_category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    transaction_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default='pending')
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_fraud: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now(), onupdate=func.now())


class UserModel(Base):
    """User model for storing user data."""
    __tablename__ = 'users'
    
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    phone_number: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    registration_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    account_status: Mapped[str] = mapped_column(String, nullable=False, default='active')
    risk_level: Mapped[str] = mapped_column(String, nullable=False, default='low')
    kyc_status: Mapped[str] = mapped_column(String, nullable=False, default='pending')
    profile_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now(), onupdate=func.now())


class MerchantModel(Base):
    """Merchant model for storing merchant data."""
    __tablename__ = 'merchants'
    
    merchant_id: Mapped[str] = mapped_column(String, primary_key=True)
    merchant_name: Mapped[str] = mapped_column(String, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.5)
    registration_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default='active')
    merchant_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now(), onupdate=func.now())


class FraudPredictionModel(Base):
    """Fraud prediction model for storing prediction results."""
    __tablename__ = 'fraud_predictions'
    
    prediction_id: Mapped[str] = mapped_column(String, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    model_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    decision: Mapped[str] = mapped_column(String, nullable=False)  # 'approve', 'decline', 'review'
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())


class ModelMetadataModel(Base):
    """Model metadata model for storing ML model information."""
    __tablename__ = 'model_metadata'
    
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    model_type: Mapped[str] = mapped_column(String, nullable=False)  # 'xgboost', 'neural_network', etc.
    training_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    performance_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    hyperparameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    feature_importance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    model_path: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    deployment_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=func.now(), onupdate=func.now())