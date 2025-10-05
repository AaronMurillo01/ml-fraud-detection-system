"""Input validation functions for the fraud detection API."""

import re
import logging
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Dict, Any

from config.settings import get_settings
from api.exceptions import ValidationException, ValidationErrorDetail
from shared.models import Transaction

logger = logging.getLogger(__name__)
settings = get_settings()

# Validation patterns
CURRENCY_PATTERN = re.compile(r'^[A-Z]{3}$')
USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')
MERCHANT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')
MODEL_VERSION_PATTERN = re.compile(r'^v?\d+\.\d+\.\d+$')

# Supported currencies
SUPPORTED_CURRENCIES = {
    'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'SEK', 'NZD'
}


def validate_amount(amount: Decimal, min_amount: Decimal = Decimal('0.01'), 
                   max_amount: Decimal = Decimal('1000000.00')) -> None:
    """Validate transaction amount.
    
    Args:
        amount: Transaction amount
        min_amount: Minimum allowed amount
        max_amount: Maximum allowed amount
        
    Raises:
        ValidationException: If amount is invalid
    """
    if amount <= 0:
        raise ValidationException(
            message="Invalid transaction amount",
            validation_errors=[
                ValidationErrorDetail(
                    field="amount",
                    message="Amount must be greater than 0",
                    invalid_value=str(amount)
                )
            ]
        )
    
    if amount < min_amount:
        raise ValidationException(
            message="Transaction amount too small",
            validation_errors=[
                ValidationErrorDetail(
                    field="amount",
                    message=f"Amount must be at least {min_amount}",
                    invalid_value=str(amount)
                )
            ]
        )
    
    if amount > max_amount:
        raise ValidationException(
            message="Transaction amount too large",
            validation_errors=[
                ValidationErrorDetail(
                    field="amount",
                    message=f"Amount cannot exceed {max_amount}",
                    invalid_value=str(amount)
                )
            ]
        )


def validate_currency(currency: str) -> None:
    """Validate currency code.
    
    Args:
        currency: Currency code (e.g., 'USD')
        
    Raises:
        ValidationException: If currency is invalid
    """
    if not currency:
        raise ValidationException(
            message="Currency is required",
            validation_errors=[
                ValidationErrorDetail(
                    field="currency",
                    message="Currency code is required",
                    invalid_value=currency
                )
            ]
        )
    
    if not CURRENCY_PATTERN.match(currency):
        raise ValidationException(
            message="Invalid currency format",
            validation_errors=[
                ValidationErrorDetail(
                    field="currency",
                    message="Currency must be a 3-letter ISO code (e.g., USD)",
                    invalid_value=currency
                )
            ]
        )
    
    if currency not in SUPPORTED_CURRENCIES:
        raise ValidationException(
            message="Unsupported currency",
            validation_errors=[
                ValidationErrorDetail(
                    field="currency",
                    message=f"Currency {currency} is not supported. Supported: {', '.join(SUPPORTED_CURRENCIES)}",
                    invalid_value=currency
                )
            ]
        )


def validate_timestamp(timestamp: datetime, max_age_hours: int = 24) -> None:
    """Validate transaction timestamp.
    
    Args:
        timestamp: Transaction timestamp
        max_age_hours: Maximum age in hours
        
    Raises:
        ValidationException: If timestamp is invalid
    """
    now = datetime.utcnow()
    
    # Check if timestamp is in the future
    if timestamp > now + timedelta(minutes=5):  # Allow 5 minutes clock skew
        raise ValidationException(
            message="Invalid timestamp",
            validation_errors=[
                ValidationErrorDetail(
                    field="timestamp",
                    message="Transaction timestamp cannot be in the future",
                    invalid_value=timestamp.isoformat()
                )
            ]
        )
    
    # Check if timestamp is too old
    max_age = timedelta(hours=max_age_hours)
    if timestamp < now - max_age:
        raise ValidationException(
            message="Transaction too old",
            validation_errors=[
                ValidationErrorDetail(
                    field="timestamp",
                    message=f"Transaction timestamp cannot be older than {max_age_hours} hours",
                    invalid_value=timestamp.isoformat()
                )
            ]
        )


def validate_user_id(user_id: str) -> None:
    """Validate user ID.
    
    Args:
        user_id: User identifier
        
    Raises:
        ValidationException: If user ID is invalid
    """
    if not user_id:
        raise ValidationException(
            message="User ID is required",
            validation_errors=[
                ValidationErrorDetail(
                    field="user_id",
                    message="User ID cannot be empty",
                    invalid_value=user_id
                )
            ]
        )
    
    if not USER_ID_PATTERN.match(user_id):
        raise ValidationException(
            message="Invalid user ID format",
            validation_errors=[
                ValidationErrorDetail(
                    field="user_id",
                    message="User ID can only contain letters, numbers, hyphens, and underscores (1-50 characters)",
                    invalid_value=user_id
                )
            ]
        )


def validate_merchant_id(merchant_id: str) -> None:
    """Validate merchant ID.
    
    Args:
        merchant_id: Merchant identifier
        
    Raises:
        ValidationException: If merchant ID is invalid
    """
    if not merchant_id:
        raise ValidationException(
            message="Merchant ID is required",
            validation_errors=[
                ValidationErrorDetail(
                    field="merchant_id",
                    message="Merchant ID cannot be empty",
                    invalid_value=merchant_id
                )
            ]
        )
    
    if not MERCHANT_ID_PATTERN.match(merchant_id):
        raise ValidationException(
            message="Invalid merchant ID format",
            validation_errors=[
                ValidationErrorDetail(
                    field="merchant_id",
                    message="Merchant ID can only contain letters, numbers, hyphens, and underscores (1-100 characters)",
                    invalid_value=merchant_id
                )
            ]
        )


def validate_model_version(model_version: Optional[str]) -> None:
    """Validate model version format.
    
    Args:
        model_version: Model version string
        
    Raises:
        ValidationException: If model version is invalid
    """
    if model_version and not MODEL_VERSION_PATTERN.match(model_version):
        raise ValidationException(
            message="Invalid model version format",
            validation_errors=[
                ValidationErrorDetail(
                    field="model_version",
                    message="Model version must follow semantic versioning (e.g., v1.2.3 or 1.2.3)",
                    invalid_value=model_version
                )
            ]
        )


def sanitize_string_input(input_str: str, max_length: int = 1000) -> str:
    """Sanitize string input to prevent injection attacks.
    
    Args:
        input_str: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationException: If input is invalid
    """
    if not input_str:
        return ""
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', input_str)
    
    # Trim whitespace
    sanitized = sanitized.strip()
    
    # Check length
    if len(sanitized) > max_length:
        raise ValidationException(
            message="Input too long",
            validation_errors=[
                ValidationErrorDetail(
                    field="input",
                    message=f"Input cannot exceed {max_length} characters",
                    invalid_value=f"{sanitized[:50]}..." if len(sanitized) > 50 else sanitized
                )
            ]
        )
    
    return sanitized


def validate_batch_size(batch_size: int, max_size: int = None) -> None:
    """Validate batch size.
    
    Args:
        batch_size: Number of items in batch
        max_size: Maximum allowed batch size
        
    Raises:
        ValidationException: If batch size is invalid
    """
    if max_size is None:
        max_size = getattr(settings, 'max_batch_size', 100)
    
    if batch_size <= 0:
        raise ValidationException(
            message="Invalid batch size",
            validation_errors=[
                ValidationErrorDetail(
                    field="batch_size",
                    message="Batch size must be greater than 0",
                    invalid_value=batch_size
                )
            ]
        )
    
    if batch_size > max_size:
        raise ValidationException(
            message="Batch size too large",
            validation_errors=[
                ValidationErrorDetail(
                    field="batch_size",
                    message=f"Batch size cannot exceed {max_size}",
                    invalid_value=batch_size
                )
            ]
        )


def validate_transaction(transaction: Transaction) -> None:
    """Validate a complete transaction object.
    
    Args:
        transaction: Transaction to validate
        
    Raises:
        ValidationException: If transaction is invalid
    """
    errors = []
    
    try:
        validate_amount(transaction.amount)
    except ValidationException as e:
        errors.extend(e.validation_errors)
    
    try:
        validate_currency(transaction.currency)
    except ValidationException as e:
        errors.extend(e.validation_errors)
    
    try:
        validate_timestamp(transaction.timestamp)
    except ValidationException as e:
        errors.extend(e.validation_errors)
    
    try:
        validate_user_id(transaction.user_id)
    except ValidationException as e:
        errors.extend(e.validation_errors)
    
    try:
        validate_merchant_id(transaction.merchant_id)
    except ValidationException as e:
        errors.extend(e.validation_errors)
    
    if errors:
        raise ValidationException(
            message="Transaction validation failed",
            validation_errors=errors,
            details=f"Found {len(errors)} validation error(s)"
        )
