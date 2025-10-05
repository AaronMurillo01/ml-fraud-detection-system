"""Health check endpoints for service monitoring."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import psycopg2
import redis
from kafka import KafkaProducer

from service.core.config import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]


class DependencyCheck(BaseModel):
    """Individual dependency check result."""
    status: str  # "healthy", "unhealthy", "unknown"
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Service start time for uptime calculation
START_TIME = time.time()


async def check_database(settings: Settings) -> DependencyCheck:
    """Check PostgreSQL database connectivity."""
    start_time = time.time()
    
    try:
        # Create connection string
        conn_str = (
            f"host={settings.db_host} "
            f"port={settings.db_port} "
            f"dbname={settings.db_name} "
            f"user={settings.db_user} "
            f"password={settings.db_password}"
        )
        
        # Test connection
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        response_time = (time.time() - start_time) * 1000
        
        return DependencyCheck(
            status="healthy",
            response_time_ms=response_time,
            details={"result": result[0] if result else None}
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Database health check failed: {e}")
        
        return DependencyCheck(
            status="unhealthy",
            response_time_ms=response_time,
            error=str(e)
        )


async def check_redis(settings: Settings) -> DependencyCheck:
    """Check Redis connectivity."""
    start_time = time.time()
    
    try:
        # Create Redis client
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True,
            socket_timeout=5
        )
        
        # Test connection with ping
        result = client.ping()
        client.close()
        
        response_time = (time.time() - start_time) * 1000
        
        return DependencyCheck(
            status="healthy",
            response_time_ms=response_time,
            details={"ping_result": result}
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Redis health check failed: {e}")
        
        return DependencyCheck(
            status="unhealthy",
            response_time_ms=response_time,
            error=str(e)
        )


async def check_kafka(settings: Settings) -> DependencyCheck:
    """Check Kafka connectivity."""
    start_time = time.time()
    
    try:
        # Create Kafka producer with short timeout
        producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            request_timeout_ms=5000,
            api_version=(0, 10, 1)
        )
        
        # Get cluster metadata (this tests connectivity)
        metadata = producer.list_topics(timeout=5)
        producer.close()
        
        response_time = (time.time() - start_time) * 1000
        
        return DependencyCheck(
            status="healthy",
            response_time_ms=response_time,
            details={
                "topics_count": len(metadata.topics),
                "brokers_count": len(metadata.brokers)
            }
        )
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Kafka health check failed: {e}")
        
        return DependencyCheck(
            status="unhealthy",
            response_time_ms=response_time,
            error=str(e)
        )


@router.get("/", response_model=HealthStatus)
async def health_check(settings: Settings = Depends(get_settings)):
    """Basic health check endpoint."""
    uptime = time.time() - START_TIME
    
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=uptime,
        checks={}
    )


@router.get("/live", response_model=HealthStatus)
async def liveness_check(settings: Settings = Depends(get_settings)):
    """Kubernetes liveness probe endpoint."""
    uptime = time.time() - START_TIME
    
    # Basic service health - just check if we can respond
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=uptime,
        checks={"service": {"status": "healthy"}}
    )


@router.get("/ready", response_model=HealthStatus)
async def readiness_check(settings: Settings = Depends(get_settings)):
    """Kubernetes readiness probe endpoint - checks all dependencies."""
    uptime = time.time() - START_TIME
    
    # Run all dependency checks concurrently
    db_check, redis_check, kafka_check = await asyncio.gather(
        check_database(settings),
        check_redis(settings),
        check_kafka(settings),
        return_exceptions=True
    )
    
    # Handle any exceptions from the checks
    checks = {}
    
    if isinstance(db_check, Exception):
        checks["database"] = DependencyCheck(
            status="unhealthy",
            error=str(db_check)
        ).dict()
    else:
        checks["database"] = db_check.dict()
    
    if isinstance(redis_check, Exception):
        checks["redis"] = DependencyCheck(
            status="unhealthy",
            error=str(redis_check)
        ).dict()
    else:
        checks["redis"] = redis_check.dict()
    
    if isinstance(kafka_check, Exception):
        checks["kafka"] = DependencyCheck(
            status="unhealthy",
            error=str(kafka_check)
        ).dict()
    else:
        checks["kafka"] = kafka_check.dict()
    
    # Determine overall status
    all_healthy = all(
        check["status"] == "healthy" 
        for check in checks.values()
    )
    
    overall_status = "healthy" if all_healthy else "unhealthy"
    
    response = HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=uptime,
        checks=checks
    )
    
    # Return 503 if not ready
    if not all_healthy:
        raise HTTPException(
            status_code=503,
            detail=response.dict()
        )
    
    return response


@router.get("/detailed")
async def detailed_health_check(settings: Settings = Depends(get_settings)):
    """Detailed health check with system information."""
    uptime = time.time() - START_TIME
    
    # Run dependency checks
    db_check, redis_check, kafka_check = await asyncio.gather(
        check_database(settings),
        check_redis(settings),
        check_kafka(settings),
        return_exceptions=True
    )
    
    # System information
    import psutil
    import os
    
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "process_id": os.getpid(),
        "thread_count": psutil.Process().num_threads(),
    }
    
    # Prepare checks
    checks = {
        "database": db_check.dict() if not isinstance(db_check, Exception) else {"status": "error", "error": str(db_check)},
        "redis": redis_check.dict() if not isinstance(redis_check, Exception) else {"status": "error", "error": str(redis_check)},
        "kafka": kafka_check.dict() if not isinstance(kafka_check, Exception) else {"status": "error", "error": str(kafka_check)},
        "system": system_info
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "uptime_seconds": uptime,
        "environment": settings.environment,
        "checks": checks
    }