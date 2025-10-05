"""Database management and health check endpoints."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.auth import get_current_active_user, User
from api.dependencies import get_database_manager, require_admin_role
from database.connection import check_database_connectivity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database", tags=["Database"])


@router.get("/health")
async def database_health_check():
    """Check database health and connectivity.
    
    Returns:
        Database health status and metrics
    """
    try:
        health_status = await check_database_connectivity()
        
        # Determine HTTP status code based on health
        status_code = status.HTTP_200_OK if health_status['status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return health_status
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": "Database health check failed",
                "error": str(e)
            }
        )


@router.get("/pool-status")
async def get_pool_status(
    current_user: User = Depends(require_admin_role)
):
    """Get database connection pool status.
    
    Requires admin role.
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        Connection pool status and statistics
    """
    try:
        db_manager = get_database_manager()
        pool_status = await db_manager.get_pool_status()
        
        return {
            "status": "success",
            "pool_status": pool_status,
            "message": "Pool status retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get pool status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to retrieve pool status",
                "error": str(e)
            }
        )


@router.post("/health-check")
async def force_health_check(
    current_user: User = Depends(require_admin_role)
):
    """Force a database health check.
    
    Requires admin role.
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        Health check result
    """
    try:
        db_manager = get_database_manager()
        is_healthy = await db_manager.health_check()
        
        return {
            "status": "success",
            "is_healthy": is_healthy,
            "message": "Health check completed"
        }
        
    except Exception as e:
        logger.error(f"Forced health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Health check failed",
                "error": str(e)
            }
        )


@router.get("/metrics")
async def get_database_metrics(
    current_user: User = Depends(require_admin_role)
):
    """Get comprehensive database metrics.
    
    Requires admin role.
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        Database performance metrics
    """
    try:
        # Get connectivity status
        connectivity = await check_database_connectivity()
        
        # Get pool status
        db_manager = get_database_manager()
        pool_status = await db_manager.get_pool_status()
        
        return {
            "status": "success",
            "metrics": {
                "connectivity": connectivity,
                "pool": pool_status,
                "engine_info": {
                    "driver": "asyncpg",
                    "pool_class": "QueuePool"
                }
            },
            "message": "Database metrics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get database metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to retrieve database metrics",
                "error": str(e)
            }
        )
