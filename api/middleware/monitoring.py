"""Advanced monitoring middleware with Prometheus metrics integration.

This module provides comprehensive monitoring capabilities including:
- Prometheus metrics collection
- Request/response tracking
- Performance monitoring
- Error rate tracking
- Custom business metrics
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response as StarletteResponse

from config.settings import get_settings
from config.monitoring import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    REQUEST_SIZE,
    RESPONSE_SIZE,
    ACTIVE_CONNECTIONS,
    ERROR_COUNT,
    FRAUD_PREDICTIONS,
    MODEL_INFERENCE_TIME,
    FEATURE_PROCESSING_TIME,
    DATABASE_QUERY_TIME
)

logger = logging.getLogger(__name__)
settings = get_settings()


class PrometheusMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection and monitoring."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.active_requests = 0
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and collect metrics.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object with metrics collected
        """
        # Start timing
        start_time = time.time()
        
        # Track active connections
        self.active_requests += 1
        ACTIVE_CONNECTIONS.set(self.active_requests)
        
        # Get request info
        method = request.method
        path = request.url.path
        
        # Get request size
        request_size = 0
        if "content-length" in request.headers:
            try:
                request_size = int(request.headers["content-length"])
                REQUEST_SIZE.labels(method=method, endpoint=path).observe(request_size)
            except ValueError:
                pass
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            duration = time.time() - start_time
            
            # Get response info
            status_code = response.status_code
            status_class = f"{status_code // 100}xx"
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            # Track response size
            response_size = 0
            if hasattr(response, 'body'):
                try:
                    response_size = len(response.body)
                    RESPONSE_SIZE.labels(method=method, endpoint=path).observe(response_size)
                except (AttributeError, TypeError):
                    pass
            
            # Track errors
            if status_code >= 400:
                ERROR_COUNT.labels(
                    method=method,
                    endpoint=path,
                    status_code=status_code,
                    error_type=status_class
                ).inc()
            
            # Add monitoring headers
            response.headers["X-Response-Time"] = f"{duration:.4f}"
            response.headers["X-Request-Size"] = str(request_size)
            response.headers["X-Response-Size"] = str(response_size)
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            duration = time.time() - start_time
            
            # Record error metrics
            ERROR_COUNT.labels(
                method=method,
                endpoint=path,
                status_code=500,
                error_type="5xx"
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            logger.error(
                f"Request failed - Method: {method} - Path: {path} - "
                f"Duration: {duration:.4f}s - Error: {str(e)}",
                exc_info=True
            )
            
            raise
            
        finally:
            # Update active connections
            self.active_requests -= 1
            ACTIVE_CONNECTIONS.set(self.active_requests)


class BusinessMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking business-specific metrics."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Track business metrics for fraud detection.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        path = request.url.path
        method = request.method
        
        # Track fraud detection requests
        if "/fraud/predict" in path and method == "POST":
            start_time = time.time()
            
            try:
                response = await call_next(request)
                
                # Track prediction time
                prediction_time = time.time() - start_time
                MODEL_INFERENCE_TIME.observe(prediction_time)
                
                # Track fraud predictions if successful
                if response.status_code == 200:
                    FRAUD_PREDICTIONS.labels(model_version="current").inc()
                
                return response
                
            except Exception as e:
                logger.error(f"Fraud prediction failed: {e}")
                raise
        
        # Track feature processing requests
        elif "/features" in path:
            start_time = time.time()
            
            try:
                response = await call_next(request)
                
                # Track feature processing time
                processing_time = time.time() - start_time
                FEATURE_PROCESSING_TIME.observe(processing_time)
                
                return response
                
            except Exception as e:
                logger.error(f"Feature processing failed: {e}")
                raise
        
        else:
            return await call_next(request)


class DatabaseMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking database operation metrics."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Track database metrics.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        # Store start time for database operations
        request.state.db_start_time = time.time()
        
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise


def track_database_query(operation_type: str, table_name: str = "unknown"):
    """Decorator to track database query performance.
    
    Args:
        operation_type: Type of database operation (SELECT, INSERT, UPDATE, DELETE)
        table_name: Name of the table being queried
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track successful query time
                query_time = time.time() - start_time
                DATABASE_QUERY_TIME.labels(
                    operation=operation_type,
                    table=table_name,
                    status="success"
                ).observe(query_time)
                
                return result
                
            except Exception as e:
                # Track failed query time
                query_time = time.time() - start_time
                DATABASE_QUERY_TIME.labels(
                    operation=operation_type,
                    table=table_name,
                    status="error"
                ).observe(query_time)
                
                logger.error(
                    f"Database query failed - Operation: {operation_type} - "
                    f"Table: {table_name} - Time: {query_time:.4f}s - Error: {e}"
                )
                raise
        
        return wrapper
    return decorator


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Prometheus metrics in text format
    """
    try:
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content=f"Error generating metrics: {str(e)}",
            status_code=500,
            media_type="text/plain"
        )


def setup_monitoring_middleware(app: FastAPI):
    """Setup monitoring middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Add monitoring middleware in order
    if settings.enable_prometheus_metrics:
        app.add_middleware(PrometheusMonitoringMiddleware)
        logger.info("Prometheus monitoring middleware enabled")
    
    if settings.enable_business_metrics:
        app.add_middleware(BusinessMetricsMiddleware)
        logger.info("Business metrics middleware enabled")
    
    if settings.enable_database_metrics:
        app.add_middleware(DatabaseMetricsMiddleware)
        logger.info("Database metrics middleware enabled")
    
    # Add metrics endpoint
    app.add_route("/metrics", metrics_endpoint, methods=["GET"])
    logger.info("Metrics endpoint added at /metrics")


def get_monitoring_status() -> Dict[str, Any]:
    """Get current monitoring status and configuration.
    
    Returns:
        Monitoring status information
    """
    return {
        "prometheus_enabled": settings.enable_prometheus_metrics,
        "business_metrics_enabled": settings.enable_business_metrics,
        "database_metrics_enabled": settings.enable_database_metrics,
        "metrics_endpoint": "/metrics",
        "timestamp": datetime.now().isoformat()
    }