"""Rate limiting management endpoints."""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field

from api.auth import get_current_active_user, User
from api.dependencies import require_admin_role
from api.rate_limiting import (
    get_rate_limiter,
    RateLimitType,
    RateLimitWindow,
    check_rate_limit_dependency
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rate-limit", tags=["Rate Limiting"])


class RateLimitResetRequest(BaseModel):
    """Request to reset rate limits."""
    
    limit_type: RateLimitType
    identifier: str
    window: Optional[RateLimitWindow] = None
    confirm: bool = False


class RateLimitStatusRequest(BaseModel):
    """Request to get rate limit status."""
    
    limit_type: RateLimitType
    identifier: str


@router.get("/status/current")
async def get_current_rate_limit_status(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's rate limit status.
    
    Args:
        request: HTTP request
        current_user: Current authenticated user
        
    Returns:
        Current rate limit status
    """
    try:
        rate_limiter = await get_rate_limiter()
        
        # Get status for current user
        user_status = await rate_limiter.get_rate_limit_status(
            RateLimitType.USER,
            str(current_user.id)
        )
        
        # Get IP status
        client_ip = request.client.host if request.client else "unknown"
        ip_status = await rate_limiter.get_rate_limit_status(
            RateLimitType.IP_ADDRESS,
            client_ip
        )
        
        return {
            "status": "success",
            "user_id": current_user.id,
            "username": current_user.username,
            "rate_limits": {
                "user": user_status,
                "ip_address": ip_status
            },
            "message": "Rate limit status retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to retrieve rate limit status",
                "error": str(e)
            }
        )


@router.post("/check")
async def check_rate_limit_manual(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Manually check rate limit for current request.
    
    Args:
        request: HTTP request
        current_user: Current authenticated user
        
    Returns:
        Rate limit check result
    """
    try:
        # Perform rate limit check
        result = await check_rate_limit_dependency(request)
        
        return {
            "status": "success",
            "rate_limit": {
                "allowed": result.allowed,
                "limit": result.limit,
                "remaining": result.remaining,
                "reset_time": result.reset_time.isoformat(),
                "retry_after": result.retry_after
            },
            "message": "Rate limit check completed"
        }
        
    except HTTPException as e:
        # Rate limit exceeded
        return {
            "status": "rate_limited",
            "rate_limit": e.detail,
            "message": "Rate limit exceeded"
        }
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Rate limit check failed",
                "error": str(e)
            }
        )


@router.post("/status")
async def get_rate_limit_status(
    request: RateLimitStatusRequest,
    current_user: User = Depends(require_admin_role)
):
    """Get rate limit status for a specific identifier.
    
    Requires admin role.
    
    Args:
        request: Rate limit status request
        current_user: Current authenticated admin user
        
    Returns:
        Rate limit status
    """
    try:
        rate_limiter = await get_rate_limiter()
        
        status_info = await rate_limiter.get_rate_limit_status(
            request.limit_type,
            request.identifier
        )
        
        return {
            "status": "success",
            "limit_type": request.limit_type,
            "identifier": request.identifier,
            "rate_limits": status_info,
            "message": "Rate limit status retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get rate limit status for {request.limit_type}:{request.identifier}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to retrieve rate limit status",
                "error": str(e)
            }
        )


@router.post("/reset")
async def reset_rate_limit(
    request: RateLimitResetRequest,
    current_user: User = Depends(require_admin_role)
):
    """Reset rate limits for a specific identifier.
    
    Requires admin role and confirmation.
    
    Args:
        request: Rate limit reset request
        current_user: Current authenticated admin user
        
    Returns:
        Reset operation result
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "Rate limit reset requires confirmation",
                "hint": "Set 'confirm' to true to proceed"
            }
        )
    
    try:
        rate_limiter = await get_rate_limiter()
        
        success = await rate_limiter.reset_rate_limit(
            request.limit_type,
            request.identifier,
            request.window
        )
        
        if success:
            logger.info(f"Admin {current_user.username} reset rate limit for {request.limit_type}:{request.identifier}")
            
            return {
                "status": "success",
                "limit_type": request.limit_type,
                "identifier": request.identifier,
                "window": request.window,
                "message": "Rate limit reset successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "status": "error",
                    "message": "Failed to reset rate limit"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset rate limit for {request.limit_type}:{request.identifier}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to reset rate limit",
                "error": str(e)
            }
        )


@router.get("/config")
async def get_rate_limit_config(
    current_user: User = Depends(require_admin_role)
):
    """Get current rate limiting configuration.
    
    Requires admin role.
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        Rate limiting configuration
    """
    try:
        rate_limiter = await get_rate_limiter()
        
        # Get default configurations
        default_configs = {}
        for limit_type, config in rate_limiter.default_configs.items():
            default_configs[limit_type.value] = {
                "limit": config.limit,
                "window": config.window.value,
                "burst_limit": config.burst_limit,
                "burst_window": config.burst_window.value if config.burst_window else None
            }
        
        # Get endpoint-specific configurations
        endpoint_configs = {}
        for endpoint, config in rate_limiter.endpoint_configs.items():
            endpoint_configs[endpoint] = {
                "limit": config.limit,
                "window": config.window.value,
                "burst_limit": config.burst_limit,
                "burst_window": config.burst_window.value if config.burst_window else None
            }
        
        return {
            "status": "success",
            "configuration": {
                "default_configs": default_configs,
                "endpoint_configs": endpoint_configs,
                "enabled": True  # Assuming enabled if we can access this endpoint
            },
            "message": "Rate limiting configuration retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get rate limit configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to retrieve rate limiting configuration",
                "error": str(e)
            }
        )


@router.get("/health")
async def rate_limit_health_check():
    """Check rate limiting system health.
    
    Returns:
        Rate limiting health status
    """
    try:
        rate_limiter = await get_rate_limiter()
        
        # Test basic functionality
        test_result = await rate_limiter.check_rate_limit(
            RateLimitType.GLOBAL,
            "health_check",
            None
        )
        
        return {
            "status": "healthy",
            "rate_limiting": {
                "enabled": True,
                "redis_connected": test_result is not None,
                "test_check_passed": test_result.allowed if test_result else False
            },
            "message": "Rate limiting system is healthy"
        }
        
    except Exception as e:
        logger.error(f"Rate limiting health check failed: {e}")
        return {
            "status": "unhealthy",
            "rate_limiting": {
                "enabled": False,
                "redis_connected": False,
                "error": str(e)
            },
            "message": "Rate limiting system is unhealthy"
        }
