"""Cache management endpoints for the fraud detection API."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import get_current_active_user, User
from api.dependencies import require_admin_role
from cache import get_redis_manager, get_ml_cache_service
from shared.models import Transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["Cache"])


class CacheInvalidationRequest(BaseModel):
    """Request to invalidate cache entries."""
    
    pattern: str
    confirm: bool = False


class TransactionCacheRequest(BaseModel):
    """Request to invalidate cache for a specific transaction."""
    
    transaction: Transaction


@router.get("/health")
async def cache_health_check():
    """Check cache health and connectivity.
    
    Returns:
        Cache health status
    """
    try:
        redis_manager = await get_redis_manager()
        is_healthy = await redis_manager.health_check()
        
        if is_healthy:
            connection_info = await redis_manager.get_connection_info()
            return {
                "status": "healthy",
                "redis_info": connection_info
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Redis connection failed"
            }
            
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": "Cache health check failed",
                "error": str(e)
            }
        )


@router.get("/stats")
async def get_cache_stats(
    current_user: User = Depends(require_admin_role)
):
    """Get cache statistics and performance metrics.
    
    Requires admin role.
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        Cache statistics
    """
    try:
        ml_cache = await get_ml_cache_service()
        cache_stats = await ml_cache.get_cache_stats()
        
        return {
            "status": "success",
            "cache_stats": cache_stats,
            "message": "Cache statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to retrieve cache statistics",
                "error": str(e)
            }
        )


@router.post("/invalidate/pattern")
async def invalidate_cache_pattern(
    request: CacheInvalidationRequest,
    current_user: User = Depends(require_admin_role)
):
    """Invalidate cache entries matching a pattern.
    
    Requires admin role and confirmation.
    
    Args:
        request: Cache invalidation request
        current_user: Current authenticated admin user
        
    Returns:
        Invalidation result
    """
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "message": "Cache invalidation requires confirmation",
                "hint": "Set 'confirm' to true to proceed"
            }
        )
    
    try:
        redis_manager = await get_redis_manager()
        deleted_count = await redis_manager.clear_pattern(request.pattern)
        
        logger.info(f"Admin {current_user.username} invalidated {deleted_count} cache entries with pattern: {request.pattern}")
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "pattern": request.pattern,
            "message": f"Invalidated {deleted_count} cache entries"
        }
        
    except Exception as e:
        logger.error(f"Failed to invalidate cache pattern {request.pattern}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to invalidate cache entries",
                "error": str(e)
            }
        )


@router.post("/invalidate/transaction")
async def invalidate_transaction_cache(
    request: TransactionCacheRequest,
    current_user: User = Depends(require_admin_role)
):
    """Invalidate all cache entries for a specific transaction.
    
    Requires admin role.
    
    Args:
        request: Transaction cache invalidation request
        current_user: Current authenticated admin user
        
    Returns:
        Invalidation result
    """
    try:
        ml_cache = await get_ml_cache_service()
        deleted_count = await ml_cache.invalidate_transaction_cache(request.transaction)
        
        logger.info(f"Admin {current_user.username} invalidated cache for transaction: {request.transaction.transaction_id}")
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "transaction_id": request.transaction.transaction_id,
            "message": f"Invalidated {deleted_count} cache entries for transaction"
        }
        
    except Exception as e:
        logger.error(f"Failed to invalidate transaction cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to invalidate transaction cache",
                "error": str(e)
            }
        )


@router.post("/clear/predictions")
async def clear_prediction_cache(
    current_user: User = Depends(require_admin_role)
):
    """Clear all prediction cache entries.
    
    Requires admin role.
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        Clear operation result
    """
    try:
        redis_manager = await get_redis_manager()
        deleted_count = await redis_manager.clear_pattern("fraud:pred:*")
        
        logger.info(f"Admin {current_user.username} cleared all prediction cache entries: {deleted_count}")
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "cache_type": "predictions",
            "message": f"Cleared {deleted_count} prediction cache entries"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear prediction cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to clear prediction cache",
                "error": str(e)
            }
        )


@router.post("/clear/features")
async def clear_feature_cache(
    current_user: User = Depends(require_admin_role)
):
    """Clear all feature cache entries.
    
    Requires admin role.
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        Clear operation result
    """
    try:
        redis_manager = await get_redis_manager()
        deleted_count = await redis_manager.clear_pattern("fraud:feat:*")
        
        logger.info(f"Admin {current_user.username} cleared all feature cache entries: {deleted_count}")
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "cache_type": "features",
            "message": f"Cleared {deleted_count} feature cache entries"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear feature cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to clear feature cache",
                "error": str(e)
            }
        )


@router.get("/info")
async def get_cache_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get general cache information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Cache information
    """
    try:
        redis_manager = await get_redis_manager()
        connection_info = await redis_manager.get_connection_info()
        
        return {
            "status": "success",
            "cache_info": {
                "type": "Redis",
                "connection_info": connection_info,
                "features": {
                    "prediction_caching": True,
                    "feature_caching": True,
                    "model_metadata_caching": True,
                    "user_profile_caching": True
                }
            },
            "message": "Cache information retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to retrieve cache information",
                "error": str(e)
            }
        )
