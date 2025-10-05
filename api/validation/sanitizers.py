"""Input sanitization utilities for the fraud detection API."""

import re
import logging
from typing import Dict, Any, List
from copy import deepcopy

from shared.models import Transaction

logger = logging.getLogger(__name__)

# Sensitive data patterns
CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
SSN_PATTERN = re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')

# Fields that should be removed from logs/responses
SENSITIVE_FIELDS = {
    'password', 'secret', 'token', 'key', 'ssn', 'social_security',
    'credit_card', 'card_number', 'cvv', 'pin', 'account_number'
}


def sanitize_string_for_logging(text: str) -> str:
    """Sanitize string for safe logging by removing sensitive data.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text safe for logging
    """
    if not text:
        return text
    
    # Replace credit card numbers
    text = CREDIT_CARD_PATTERN.sub('****-****-****-****', text)
    
    # Replace SSNs
    text = SSN_PATTERN.sub('***-**-****', text)
    
    # Replace email addresses (partially)
    def mask_email(match):
        email = match.group(0)
        parts = email.split('@')
        if len(parts) == 2:
            username = parts[0]
            domain = parts[1]
            masked_username = username[0] + '*' * (len(username) - 2) + username[-1] if len(username) > 2 else '*' * len(username)
            return f"{masked_username}@{domain}"
        return email
    
    text = EMAIL_PATTERN.sub(mask_email, text)
    
    # Replace phone numbers
    text = PHONE_PATTERN.sub('***-***-****', text)
    
    return text


def sanitize_dict_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize dictionary for safe logging by removing/masking sensitive fields.
    
    Args:
        data: Input dictionary
        
    Returns:
        Sanitized dictionary safe for logging
    """
    if not isinstance(data, dict):
        return data
    
    sanitized = deepcopy(data)
    
    def sanitize_recursive(obj, path=""):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                current_path = f"{path}.{key}" if path else key
                
                # Check if field name contains sensitive keywords
                if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
                    obj[key] = "***REDACTED***"
                elif isinstance(value, str):
                    obj[key] = sanitize_string_for_logging(value)
                elif isinstance(value, (dict, list)):
                    sanitize_recursive(value, current_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    sanitize_recursive(item, f"{path}[{i}]")
                elif isinstance(item, str):
                    obj[i] = sanitize_string_for_logging(item)
    
    sanitize_recursive(sanitized)
    return sanitized


def remove_sensitive_data(data: Dict[str, Any], 
                         additional_fields: List[str] = None) -> Dict[str, Any]:
    """Remove sensitive fields from data dictionary.
    
    Args:
        data: Input data dictionary
        additional_fields: Additional field names to remove
        
    Returns:
        Data with sensitive fields removed
    """
    if not isinstance(data, dict):
        return data
    
    cleaned = deepcopy(data)
    fields_to_remove = SENSITIVE_FIELDS.copy()
    
    if additional_fields:
        fields_to_remove.update(additional_fields)
    
    def remove_recursive(obj):
        if isinstance(obj, dict):
            keys_to_remove = []
            for key, value in obj.items():
                if any(sensitive in key.lower() for sensitive in fields_to_remove):
                    keys_to_remove.append(key)
                elif isinstance(value, (dict, list)):
                    remove_recursive(value)
            
            for key in keys_to_remove:
                del obj[key]
        
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    remove_recursive(item)
    
    remove_recursive(cleaned)
    return cleaned


def sanitize_transaction_data(transaction: Transaction) -> Transaction:
    """Sanitize transaction data for processing.
    
    Args:
        transaction: Input transaction
        
    Returns:
        Sanitized transaction
    """
    # Create a copy to avoid modifying the original
    sanitized_data = transaction.dict()
    
    # Sanitize string fields
    if 'location' in sanitized_data and sanitized_data['location']:
        sanitized_data['location'] = sanitize_string_for_logging(sanitized_data['location'])
    
    # Remove any sensitive fields that might have been added
    sanitized_data = remove_sensitive_data(sanitized_data)
    
    # Recreate transaction object
    return Transaction(**sanitized_data)


def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks.
    
    Args:
        user_input: Raw user input
        max_length: Maximum allowed length
        
    Returns:
        Sanitized input
    """
    if not user_input:
        return ""
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', user_input)
    
    # Remove potential SQL injection patterns
    sql_patterns = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(--|#|/\*|\*/)',
        r'(\bOR\b.*=.*\bOR\b)',
        r'(\bAND\b.*=.*\bAND\b)',
        r'(\'.*\')',
        r'(;.*)',
    ]
    
    for pattern in sql_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    # Remove potential XSS patterns
    xss_patterns = [
        r'<script.*?>.*?</script>',
        r'<.*?on\w+.*?=.*?>',
        r'javascript:',
        r'vbscript:',
        r'data:text/html',
    ]
    
    for pattern in xss_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Trim whitespace and limit length
    sanitized = sanitized.strip()[:max_length]
    
    return sanitized


def mask_sensitive_response_data(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive data in API responses.
    
    Args:
        response_data: Response data dictionary
        
    Returns:
        Response data with sensitive fields masked
    """
    if not isinstance(response_data, dict):
        return response_data
    
    masked = deepcopy(response_data)
    
    # Fields that should be partially masked in responses
    partial_mask_fields = {
        'user_id': lambda x: f"{x[:3]}***{x[-3:]}" if len(x) > 6 else "***",
        'transaction_id': lambda x: f"{x[:6]}***{x[-4:]}" if len(x) > 10 else "***",
        'email': lambda x: f"{x.split('@')[0][:2]}***@{x.split('@')[1]}" if '@' in x else "***",
    }
    
    def mask_recursive(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in partial_mask_fields and isinstance(value, str):
                    obj[key] = partial_mask_fields[key.lower()](value)
                elif isinstance(value, (dict, list)):
                    mask_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    mask_recursive(item)
    
    mask_recursive(masked)
    return masked
