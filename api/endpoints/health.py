"""Health check and monitoring endpoints.

This module provides endpoints for:
- Application health checks
- System status monitoring
- Performance metrics
- Readiness and liveness probes
"""

import logging
import time
import psutil
from typing import Dict, Any, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from service.ml_inference import get_inference_service
from features.feature_pipeline import get_feature_pipeline
from api.dependencies import get_database
from api.middleware import get_performance_metrics
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/health", tags=["health"])


# Response Models
class HealthStatus(BaseModel):
    """Health status response model."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }


class DetailedHealthStatus(BaseModel):
    """Detailed health status response model."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    
    # Component health
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    
    # System metrics
    system: Dict[str, Any] = Field(..., description="System resource metrics")
    
    # Performance metrics
    performance: Dict[str, Any] = Field(..., description="Application performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "components": {
                    "database": {"status": "healthy", "response_time_ms": 5.2},
                    "ml_model": {"status": "healthy", "loaded": True},
                    "feature_pipeline": {"status": "healthy", "ready": True}
                },
                "system": {
                    "cpu_percent": 25.5,
                    "memory_percent": 45.2,
                    "disk_percent": 60.1
                },
                "performance": {
                    "request_count": 1250,
                    "average_response_time_ms": 42.3,
                    "error_rate_percent": 0.1
                }
            }
        }


class ReadinessStatus(BaseModel):
    """Readiness status response model."""
    
    ready: bool = Field(..., description="Whether application is ready to serve requests")
    timestamp: datetime = Field(..., description="Readiness check timestamp")
    components: Dict[str, bool] = Field(..., description="Component readiness status")
    message: str = Field(..., description="Readiness status message")
    
    class Config:
        schema_extra = {
            "example": {
                "ready": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "components": {
                    "database": True,
                    "ml_model": True,
                    "feature_pipeline": True
                },
                "message": "All components are ready"
            }
        }


class LivenessStatus(BaseModel):
    """Liveness status response model."""
    
    alive: bool = Field(..., description="Whether application is alive")
    timestamp: datetime = Field(..., description="Liveness check timestamp")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    message: str = Field(..., description="Liveness status message")
    
    class Config:
        schema_extra = {
            "example": {
                "alive": True,
                "timestamp": "2024-01-15T10:30:00Z",
                "uptime_seconds": 3600.5,
                "message": "Application is alive and running"
            }
        }


# Global variables for tracking
start_time = time.time()
last_health_check = None
health_check_cache = None
cache_duration = 30  # Cache health check results for 30 seconds


@router.get(
    "/",
    response_model=HealthStatus,
    summary="Basic health check",
    description="Get basic application health status."
)
async def health_check():
    """Basic health check endpoint.
    
    Returns:
        Basic health status
    """
    try:
        uptime = time.time() - start_time
        
        return HealthStatus(
            status="healthy",
            timestamp=datetime.now(),
            version=settings.app_version,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/detailed",
    response_model=DetailedHealthStatus,
    summary="Detailed health check",
    description="Get detailed application health status with component and system metrics."
)
async def detailed_health_check(
    ml_service = Depends(get_inference_service),
    feature_pipeline = Depends(get_feature_pipeline),
    db = Depends(get_database)
):
    """Detailed health check endpoint.
    
    Args:
        ml_service: ML inference service
        feature_pipeline: Feature processing pipeline
        db: Database connection
        
    Returns:
        Detailed health status
    """
    global last_health_check, health_check_cache
    
    try:
        # Check if we have cached results
        current_time = time.time()
        if (last_health_check and 
            health_check_cache and 
            current_time - last_health_check < cache_duration):
            return health_check_cache
        
        uptime = current_time - start_time
        timestamp = datetime.now()
        
        # Check component health
        components = {}
        overall_status = "healthy"
        
        # Database health
        try:
            db_start = time.time()
            await db.execute("SELECT 1")
            db_response_time = (time.time() - db_start) * 1000
            
            components["database"] = {
                "status": "healthy",
                "response_time_ms": round(db_response_time, 2),
                "connected": True
            }
        except Exception as e:
            components["database"] = {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }
            overall_status = "degraded"
        
        # ML Model health
        try:
            model_status = await ml_service.get_model_status()
            components["ml_model"] = {
                "status": "healthy" if model_status["loaded"] else "unhealthy",
                "loaded": model_status["loaded"],
                "version": model_status["version"],
                "type": model_status["type"]
            }
            
            if not model_status["loaded"]:
                overall_status = "degraded"
                
        except Exception as e:
            components["ml_model"] = {
                "status": "unhealthy",
                "error": str(e),
                "loaded": False
            }
            overall_status = "degraded"
        
        # Feature Pipeline health
        try:
            pipeline_status = await feature_pipeline.get_status()
            components["feature_pipeline"] = {
                "status": "healthy" if pipeline_status["ready"] else "unhealthy",
                "ready": pipeline_status["ready"],
                "processors_loaded": pipeline_status.get("processors_loaded", 0)
            }
            
            if not pipeline_status["ready"]:
                overall_status = "degraded"
                
        except Exception as e:
            components["feature_pipeline"] = {
                "status": "unhealthy",
                "error": str(e),
                "ready": False
            }
            overall_status = "degraded"
        
        # System metrics
        system_metrics = get_system_metrics()
        
        # Performance metrics
        performance_metrics = get_application_performance_metrics()
        
        # Build response
        response = DetailedHealthStatus(
            status=overall_status,
            timestamp=timestamp,
            version=settings.app_version,
            uptime_seconds=uptime,
            components=components,
            system=system_metrics,
            performance=performance_metrics
        )
        
        # Cache the result
        last_health_check = current_time
        health_check_cache = response
        
        return response
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/ready",
    response_model=ReadinessStatus,
    summary="Readiness probe",
    description="Check if application is ready to serve requests (Kubernetes readiness probe)."
)
async def readiness_probe(
    ml_service = Depends(get_inference_service),
    feature_pipeline = Depends(get_feature_pipeline),
    db = Depends(get_database)
):
    """Readiness probe for Kubernetes.
    
    Args:
        ml_service: ML inference service
        feature_pipeline: Feature processing pipeline
        db: Database connection
        
    Returns:
        Readiness status
    """
    try:
        timestamp = datetime.now()
        components = {}
        ready = True
        messages = []
        
        # Check database readiness
        try:
            await db.execute("SELECT 1")
            components["database"] = True
        except Exception as e:
            components["database"] = False
            ready = False
            messages.append(f"Database not ready: {str(e)}")
        
        # Check ML model readiness
        try:
            model_status = await ml_service.get_model_status()
            components["ml_model"] = model_status["loaded"]
            if not model_status["loaded"]:
                ready = False
                messages.append("ML model not loaded")
        except Exception as e:
            components["ml_model"] = False
            ready = False
            messages.append(f"ML model not ready: {str(e)}")
        
        # Check feature pipeline readiness
        try:
            pipeline_status = await feature_pipeline.get_status()
            components["feature_pipeline"] = pipeline_status["ready"]
            if not pipeline_status["ready"]:
                ready = False
                messages.append("Feature pipeline not ready")
        except Exception as e:
            components["feature_pipeline"] = False
            ready = False
            messages.append(f"Feature pipeline not ready: {str(e)}")
        
        message = "All components are ready" if ready else "; ".join(messages)
        
        response = ReadinessStatus(
            ready=ready,
            timestamp=timestamp,
            components=components,
            message=message
        )
        
        if not ready:
            raise HTTPException(
                status_code=503,
                detail=response.dict()
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/live",
    response_model=LivenessStatus,
    summary="Liveness probe",
    description="Check if application is alive (Kubernetes liveness probe)."
)
async def liveness_probe():
    """Liveness probe for Kubernetes.
    
    Returns:
        Liveness status
    """
    try:
        uptime = time.time() - start_time
        
        # Basic liveness check - if we can respond, we're alive
        return LivenessStatus(
            alive=True,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            message="Application is alive and running"
        )
        
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "alive": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/metrics",
    summary="Application metrics",
    description="Get application performance and system metrics."
)
async def get_metrics():
    """Get application metrics.
    
    Returns:
        Application and system metrics
    """
    try:
        uptime = time.time() - start_time
        
        # System metrics
        system_metrics = get_system_metrics()
        
        # Application performance metrics
        performance_metrics = get_application_performance_metrics()
        
        # Additional metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "version": settings.app_version,
            "system": system_metrics,
            "performance": performance_metrics,
            "health": {
                "last_check": last_health_check,
                "cache_duration": cache_duration
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "metrics_failed",
                "message": "Failed to retrieve metrics"
            }
        )


def get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics.
    
    Returns:
        System metrics dictionary
    """
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            network_metrics = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except Exception:
            network_metrics = {}
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "network": network_metrics
        }
        
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
        return {"error": str(e)}


def get_application_performance_metrics() -> Dict[str, Any]:
    """Get application performance metrics.
    
    Returns:
        Performance metrics dictionary
    """
    try:
        # This would typically come from your FastAPI app's middleware
        # For now, return placeholder metrics
        return {
            "request_count": 0,
            "average_response_time_ms": 0.0,
            "error_rate_percent": 0.0,
            "active_connections": 0,
            "slow_requests_count": 0
        }
        
    except Exception as e:
        logger.warning(f"Failed to get performance metrics: {e}")
        return {"error": str(e)}