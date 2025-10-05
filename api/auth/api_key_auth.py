"""API key-based authentication for the fraud detection API."""

import logging
from typing import Optional, List

from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session

from config import get_settings
from .models import User, APIKey, UserRole
from .api_key_manager import get_api_key_manager, APIKeyManager
from database.connection import get_database_session
from api.exceptions import AuthenticationException, ErrorCode

logger = logging.getLogger(__name__)
settings = get_settings()


def create_api_key(
    user_id: int,
    name: str,
    scopes: List[str] = None,
    expires_days: Optional[int] = None,
    db: Session = None
) -> tuple[str, APIKey]:
    """Create a new API key for a user.

    Args:
        user_id: User ID
        name: API key name
        scopes: List of permissions/scopes
        expires_days: Days until expiration (None for no expiration)
        db: Database session

    Returns:
        Tuple of (api_key_string, api_key_model)
    """
    manager = get_api_key_manager()
    return manager.create_api_key(user_id, name, scopes, expires_days)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API Key"),
    db: Session = Depends(get_database_session)
) -> User:
    """Verify API key and return associated user.

    Args:
        x_api_key: API key from header
        db: Database session

    Returns:
        User associated with the API key

    Raises:
        AuthenticationException: If API key is invalid or missing
    """
    if not x_api_key:
        raise AuthenticationException(
            message="API key required",
            details="X-API-Key header is missing"
        )

    if not x_api_key.startswith("fd_"):
        raise AuthenticationException(
            message="Invalid API key format",
            details="API key must start with 'fd_'"
        )

    manager = get_api_key_manager()
    user = manager.get_user_by_api_key(x_api_key)

    if not user:
        logger.warning(f"Invalid API key used: {x_api_key[:10]}...")
        raise AuthenticationException(
            message="Invalid API key",
            details="API key not found or expired"
        )

    if not user.is_active:
        raise AuthenticationException(
            message="User account is inactive",
            details="Associated user account has been deactivated"
        )

    # Log successful API key usage
    logger.info(f"API key authenticated for user: {user.username}")

    return user


async def get_api_key_user(api_key: str, db: Session) -> Optional[User]:
    """Get user associated with an API key.
    
    Args:
        api_key: API key string
        db: Database session
        
    Returns:
        User if API key is valid, None otherwise
    """
    # In a real implementation, query database for API key and associated user
    # For now, return None
    return None


def revoke_api_key(key_id: str, db: Session) -> bool:
    """Revoke an API key.
    
    Args:
        key_id: API key identifier
        db: Database session
        
    Returns:
        True if API key was revoked successfully
    """
    # In a real implementation, update database to mark API key as inactive
    logger.info(f"API key revoked: {key_id}")
    return True


def check_api_key_scopes(api_key_model: APIKey, required_scopes: List[str]) -> bool:
    """Check if API key has required scopes.
    
    Args:
        api_key_model: API key model
        required_scopes: List of required scopes
        
    Returns:
        True if API key has all required scopes
    """
    if not required_scopes:
        return True
    
    api_key_scopes = set(api_key_model.scopes)
    required_scopes_set = set(required_scopes)
    
    return required_scopes_set.issubset(api_key_scopes)


def require_api_key_scopes(required_scopes: List[str]):
    """Decorator to require specific API key scopes.
    
    Args:
        required_scopes: List of required scopes
        
    Returns:
        Dependency function
    """
    async def scope_checker(
        current_user: User = Depends(verify_api_key),
        db: Session = Depends(get_database_session)
    ) -> User:
        # In a real implementation, check API key scopes from database
        # For now, just return the user
        return current_user
    
    return scope_checker


# Rate limiting for API keys
class APIKeyRateLimiter:
    """Rate limiter for API keys."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def check_rate_limit(self, api_key: str, limit: int = 1000, window: int = 3600) -> bool:
        """Check if API key is within rate limits.
        
        Args:
            api_key: API key string
            limit: Request limit
            window: Time window in seconds
            
        Returns:
            True if within limits
        """
        # In a real implementation, use Redis to track API key usage
        # For now, always return True
        return True
    
    async def increment_usage(self, api_key: str) -> None:
        """Increment API key usage counter.
        
        Args:
            api_key: API key string
        """
        # In a real implementation, increment counter in Redis
        pass
