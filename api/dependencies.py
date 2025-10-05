"""API dependencies for fraud detection system.

This module provides dependency injection for:
- ML inference service
- Feature pipeline
- Database connections
- Authentication and authorization
- Rate limiting
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncpg
from redis import Redis

from config import get_settings
from service.ml_inference import MLInferenceService, get_inference_service
from features.feature_pipeline import FeaturePipeline, get_feature_pipeline
from database.connection import DatabaseManager
from api.auth import (
    get_current_user,
    get_current_active_user,
    verify_api_key,
    require_role,
    require_any_role,
    User,
    UserRole
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Security
security = HTTPBearer(auto_error=False)


# ML Service Dependencies
def get_ml_service() -> MLInferenceService:
    """Get ML inference service instance.
    
    Returns:
        ML inference service
    """
    return get_inference_service()


def get_feature_pipeline_service() -> FeaturePipeline:
    """Get feature pipeline instance.
    
    Returns:
        Feature pipeline
    """
    return get_feature_pipeline()


# Database Dependencies
async def get_database():
    """FastAPI dependency to get database session.

    Yields:
        AsyncSession: Database session

    Raises:
        HTTPException: If database is unavailable
    """
    try:
        from database.connection import get_database_session
        async for session in get_database_session():
            yield session
    except Exception as e:
        logger.error(f"Failed to get database session: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable"
        )


def get_database_manager():
    """Get database manager instance.

    Returns:
        DatabaseManager instance

    Raises:
        HTTPException: If database manager is unavailable
    """
    try:
        from database.connection import get_database_manager as _get_db_manager
        return _get_db_manager()
    except Exception as e:
        logger.error(f"Failed to get database manager: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database manager unavailable"
        )


# Rate Limiting
class RateLimiter:
    """Rate limiter implementation using Redis."""

    def __init__(self, redis_client=None):
        self.redis_client = redis_client

    async def check_rate_limit(
        self,
        key: str,
        limit: int = 100,
        window: int = 60
    ) -> bool:
        """Check if request is within rate limits.

        Args:
            key: Rate limiting key (user ID, IP, etc.)
            limit: Request limit
            window: Time window in seconds

        Returns:
            True if within limits
        """
        # In a real implementation, use Redis for distributed rate limiting
        # For now, always return True
        return True


def get_rate_limiter():
    """Get rate limiter instance.

    Returns:
        Rate limiter instance
    """
    return RateLimiter()


# Role-based access control dependencies
def require_admin_role():
    """Require admin role.

    Returns:
        Dependency function that requires admin role
    """
    return require_role(UserRole.ADMIN)


def require_analyst_role():
    """Require analyst role or higher.

    Returns:
        Dependency function that requires analyst role
    """
    return require_any_role(UserRole.ANALYST, UserRole.ADMIN)


def require_api_access():
    """Require API access (any authenticated user).

    Returns:
        Dependency function for API access
    """
    return require_any_role(UserRole.API_USER, UserRole.VIEWER, UserRole.ANALYST, UserRole.ADMIN)