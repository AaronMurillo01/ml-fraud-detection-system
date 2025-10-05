"""Comprehensive health check system for fraud detection API."""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import psutil
from sqlalchemy import text
from redis.exceptions import ConnectionError as RedisConnectionError

from config import settings
from database.connection import get_database_manager
from cache.redis_manager import get_redis_manager
from service.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0
    version: str = ""
    environment: str = ""


class HealthChecker:
    """Comprehensive health check system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self.register_default_checks()
    
    def register_check(self, name: str, check_func: Callable[[], Awaitable[HealthCheckResult]]):
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Async function that returns HealthCheckResult
        """
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def register_default_checks(self):
        """Register default health checks."""
        self.register_check("database", self.check_database)
        self.register_check("redis", self.check_redis)
        self.register_check("model_loader", self.check_model_loader)
        self.register_check("system_resources", self.check_system_resources)
        self.register_check("disk_space", self.check_disk_space)
        self.register_check("memory_usage", self.check_memory_usage)
    
    async def check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            db_manager = get_database_manager()
            if not db_manager:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Database manager not initialized",
                    duration_ms=(time.time() - start_time) * 1000,
                    error="Database manager not available"
                )
            
            # Test basic connectivity
            async with db_manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()
            
            # Check connection pool status
            pool_info = {
                "pool_size": db_manager.engine.pool.size(),
                "checked_in": db_manager.engine.pool.checkedin(),
                "checked_out": db_manager.engine.pool.checkedout(),
                "overflow": db_manager.engine.pool.overflow(),
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status based on performance
            if duration_ms > 1000:  # > 1 second
                status = HealthStatus.DEGRADED
                message = f"Database responding slowly ({duration_ms:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy ({duration_ms:.1f}ms)"
            
            return HealthCheckResult(
                name="database",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=pool_info
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Database health check failed: {e}")
            
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database connection failed",
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        
        try:
            redis_manager = get_redis_manager()
            if not redis_manager:
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.UNHEALTHY,
                    message="Redis manager not initialized",
                    duration_ms=(time.time() - start_time) * 1000,
                    error="Redis manager not available"
                )
            
            # Test basic connectivity with ping
            await redis_manager.ping()
            
            # Test set/get operations
            test_key = "health_check_test"
            test_value = str(time.time())
            
            await redis_manager.set(test_key, test_value, ttl=60)
            retrieved_value = await redis_manager.get(test_key)
            
            if retrieved_value != test_value:
                raise Exception("Redis set/get test failed")
            
            # Clean up test key
            await redis_manager.delete(test_key)
            
            # Get Redis info
            info = await redis_manager.info()
            redis_info = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status based on performance
            if duration_ms > 500:  # > 500ms
                status = HealthStatus.DEGRADED
                message = f"Redis responding slowly ({duration_ms:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis healthy ({duration_ms:.1f}ms)"
            
            return HealthCheckResult(
                name="redis",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=redis_info
            )
            
        except (RedisConnectionError, Exception) as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Redis health check failed: {e}")
            
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Redis connection failed",
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_model_loader(self) -> HealthCheckResult:
        """Check ML model loader status."""
        start_time = time.time()
        
        try:
            model_loader = get_model_loader()
            if not model_loader:
                return HealthCheckResult(
                    name="model_loader",
                    status=HealthStatus.UNHEALTHY,
                    message="Model loader not initialized",
                    duration_ms=(time.time() - start_time) * 1000,
                    error="Model loader not available"
                )
            
            # Get model loader statistics
            stats = model_loader.get_statistics()
            
            # Check if any models are loaded
            loaded_models = stats.get("loaded_models", 0)
            cache_hit_rate = stats.get("cache_hit_rate", 0)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if loaded_models == 0:
                status = HealthStatus.DEGRADED
                message = "No models currently loaded"
            elif cache_hit_rate < 0.5:  # Less than 50% cache hit rate
                status = HealthStatus.DEGRADED
                message = f"Low cache hit rate ({cache_hit_rate:.1%})"
            else:
                status = HealthStatus.HEALTHY
                message = f"Model loader healthy ({loaded_models} models loaded)"
            
            return HealthCheckResult(
                name="model_loader",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=stats
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Model loader health check failed: {e}")
            
            return HealthCheckResult(
                name="model_loader",
                status=HealthStatus.UNHEALTHY,
                message="Model loader check failed",
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        start_time = time.time()
        
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                load_1min = load_avg[0]
            except (AttributeError, OSError):
                load_1min = 0  # Windows doesn't have load average
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
            elif cpu_percent > 70 or memory_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"Moderate resource usage (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"System resources healthy (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "load_1min": load_1min
            }
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"System resources health check failed: {e}")
            
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message="System resources check failed",
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_disk_space(self) -> HealthCheckResult:
        """Check disk space usage."""
        start_time = time.time()
        
        try:
            # Check disk usage for current directory
            disk_usage = psutil.disk_usage('.')
            
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            used_percent = (used_gb / total_gb) * 100
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status based on disk usage
            if used_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space ({used_percent:.1f}% used, {free_gb:.1f}GB free)"
            elif used_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"Low disk space ({used_percent:.1f}% used, {free_gb:.1f}GB free)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space healthy ({used_percent:.1f}% used, {free_gb:.1f}GB free)"
            
            details = {
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "used_percent": used_percent
            }
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Disk space health check failed: {e}")
            
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message="Disk space check failed",
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def check_memory_usage(self) -> HealthCheckResult:
        """Check detailed memory usage."""
        start_time = time.time()
        
        try:
            import gc
            import sys
            
            # Get Python memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get garbage collection stats
            gc_stats = gc.get_stats()
            
            # Get object counts
            object_counts = {}
            for obj_type in [dict, list, tuple, set]:
                object_counts[obj_type.__name__] = len(gc.get_objects())
            
            duration_ms = (time.time() - start_time) * 1000
            
            rss_mb = memory_info.rss / (1024**2)  # Resident Set Size in MB
            vms_mb = memory_info.vms / (1024**2)  # Virtual Memory Size in MB
            
            # Determine status based on memory usage
            if rss_mb > 2048:  # > 2GB
                status = HealthStatus.DEGRADED
                message = f"High memory usage ({rss_mb:.1f}MB RSS)"
            elif rss_mb > 4096:  # > 4GB
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage ({rss_mb:.1f}MB RSS)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage healthy ({rss_mb:.1f}MB RSS)"
            
            details = {
                "rss_mb": rss_mb,
                "vms_mb": vms_mb,
                "gc_collections": sum(stat['collections'] for stat in gc_stats),
                "python_objects": len(gc.get_objects()),
                "object_counts": object_counts
            }
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Memory usage health check failed: {e}")
            
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message="Memory usage check failed",
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def run_all_checks(self, timeout_seconds: float = 30.0) -> SystemHealth:
        """Run all registered health checks.
        
        Args:
            timeout_seconds: Maximum time to wait for all checks
            
        Returns:
            SystemHealth with results of all checks
        """
        start_time = time.time()
        results = []
        
        try:
            # Run all checks concurrently with timeout
            tasks = [
                asyncio.wait_for(check_func(), timeout=timeout_seconds)
                for check_func in self.checks.values()
            ]
            
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(check_results):
                if isinstance(result, Exception):
                    check_name = list(self.checks.keys())[i]
                    results.append(HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message="Health check timed out or failed",
                        duration_ms=(time.time() - start_time) * 1000,
                        error=str(result)
                    ))
                else:
                    results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to run health checks: {e}")
            results.append(HealthCheckResult(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message="Health check system failed",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            ))
        
        # Determine overall system status
        overall_status = self._determine_overall_status(results)
        
        return SystemHealth(
            status=overall_status,
            checks=results,
            uptime_seconds=time.time() - self.start_time,
            version=settings.app_version,
            environment=settings.environment.value
        )
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system status from individual check results.
        
        Args:
            results: List of health check results
            
        Returns:
            Overall system health status
        """
        if not results:
            return HealthStatus.UNKNOWN
        
        # Count status types
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.HEALTHY] > 0:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance.
    
    Returns:
        Health checker instance
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def run_health_checks() -> SystemHealth:
    """Run all health checks and return system health.
    
    Returns:
        System health status
    """
    health_checker = get_health_checker()
    return await health_checker.run_all_checks()


def register_health_check(name: str, check_func: Callable[[], Awaitable[HealthCheckResult]]):
    """Register a custom health check.
    
    Args:
        name: Name of the health check
        check_func: Async function that returns HealthCheckResult
    """
    health_checker = get_health_checker()
    health_checker.register_check(name, check_func)
