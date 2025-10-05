"""Monitoring and logging configuration.

This module provides:
- Prometheus metrics setup and custom metrics
- Structured logging configuration
- Performance monitoring utilities
- Health check metrics
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
import structlog
from structlog.stdlib import LoggerFactory

from config import get_settings

settings = get_settings()

# Prometheus metrics registry
registry = CollectorRegistry()

# Custom metrics
request_count = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

fraud_detection_count = Counter(
    'fraud_detection_requests_total',
    'Total number of fraud detection requests',
    ['result', 'confidence_level'],
    registry=registry
)

fraud_detection_duration = Histogram(
    'fraud_detection_duration_seconds',
    'Fraud detection processing time in seconds',
    ['model_version'],
    registry=registry
)

model_prediction_count = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['model_version', 'prediction_type'],
    registry=registry
)

model_load_duration = Histogram(
    'model_load_duration_seconds',
    'Model loading time in seconds',
    ['model_version'],
    registry=registry
)

feature_extraction_duration = Histogram(
    'feature_extraction_duration_seconds',
    'Feature extraction processing time in seconds',
    ['feature_type'],
    registry=registry
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    registry=registry
)

memory_usage = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['type'],
    registry=registry
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_version'],
    registry=registry
)

api_errors = Counter(
    'api_errors_total',
    'Total number of API errors',
    ['error_type', 'endpoint'],
    registry=registry
)

queue_size = Gauge(
    'queue_size',
    'Current queue size',
    ['queue_type'],
    registry=registry
)

processing_rate = Summary(
    'processing_rate_per_second',
    'Processing rate per second',
    ['operation_type'],
    registry=registry
)

# System info
system_info = Info(
    'system_info',
    'System information',
    registry=registry
)


class StructuredLogger:
    """Structured logging configuration."""
    
    def __init__(self):
        self.configure_logging()
    
    def configure_logging(self):
        """Configure structured logging with structlog."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        log_level_str = settings.log_level if isinstance(settings.log_level, str) else settings.log_level.value
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, log_level_str.upper()),
            handlers=[
                logging.StreamHandler()
            ]
        )
    
    def get_logger(self, name: str) -> structlog.stdlib.BoundLogger:
        """Get a structured logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Structured logger instance
        """
        return structlog.get_logger(name)


class MetricsCollector:
    """Metrics collection utilities."""
    
    def __init__(self):
        self.logger = StructuredLogger().get_logger("metrics")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        self.logger.info(
            "HTTP request recorded",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            duration=duration
        )
    
    def record_fraud_detection(self, result: str, confidence: float, duration: float, model_version: str):
        """Record fraud detection metrics.
        
        Args:
            result: Detection result (fraud/legitimate)
            confidence: Confidence score
            duration: Processing duration in seconds
            model_version: Model version used
        """
        # Determine confidence level
        if confidence >= 0.9:
            confidence_level = "high"
        elif confidence >= 0.7:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        fraud_detection_count.labels(
            result=result,
            confidence_level=confidence_level
        ).inc()
        
        fraud_detection_duration.labels(
            model_version=model_version
        ).observe(duration)
        
        self.logger.info(
            "Fraud detection recorded",
            result=result,
            confidence=confidence,
            confidence_level=confidence_level,
            duration=duration,
            model_version=model_version
        )
    
    def record_model_prediction(self, model_version: str, prediction_type: str):
        """Record model prediction metrics.
        
        Args:
            model_version: Model version
            prediction_type: Type of prediction (single/batch)
        """
        model_prediction_count.labels(
            model_version=model_version,
            prediction_type=prediction_type
        ).inc()
    
    def record_model_load(self, model_version: str, duration: float):
        """Record model loading metrics.
        
        Args:
            model_version: Model version
            duration: Loading duration in seconds
        """
        model_load_duration.labels(
            model_version=model_version
        ).observe(duration)
        
        self.logger.info(
            "Model load recorded",
            model_version=model_version,
            duration=duration
        )
    
    def record_feature_extraction(self, feature_type: str, duration: float):
        """Record feature extraction metrics.
        
        Args:
            feature_type: Type of feature extraction
            duration: Processing duration in seconds
        """
        feature_extraction_duration.labels(
            feature_type=feature_type
        ).observe(duration)
    
    def update_active_connections(self, count: int):
        """Update active connections gauge.
        
        Args:
            count: Number of active connections
        """
        active_connections.set(count)
    
    def update_memory_usage(self, memory_type: str, bytes_used: int):
        """Update memory usage gauge.
        
        Args:
            memory_type: Type of memory (heap/non_heap)
            bytes_used: Memory usage in bytes
        """
        memory_usage.labels(type=memory_type).set(bytes_used)
    
    def update_model_accuracy(self, model_version: str, accuracy: float):
        """Update model accuracy gauge.
        
        Args:
            model_version: Model version
            accuracy: Model accuracy score
        """
        model_accuracy.labels(model_version=model_version).set(accuracy)
    
    def record_api_error(self, error_type: str, endpoint: str):
        """Record API error metrics.
        
        Args:
            error_type: Type of error
            endpoint: API endpoint where error occurred
        """
        api_errors.labels(
            error_type=error_type,
            endpoint=endpoint
        ).inc()
        
        self.logger.error(
            "API error recorded",
            error_type=error_type,
            endpoint=endpoint
        )
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size gauge.
        
        Args:
            queue_type: Type of queue
            size: Current queue size
        """
        queue_size.labels(queue_type=queue_type).set(size)
    
    def record_processing_rate(self, operation_type: str, rate: float):
        """Record processing rate.
        
        Args:
            operation_type: Type of operation
            rate: Processing rate per second
        """
        processing_rate.labels(operation_type=operation_type).observe(rate)


# Global metrics collector instance
metrics_collector = MetricsCollector()


def setup_system_info():
    """Setup system information metrics."""
    import platform
    import sys
    
    system_info.info({
        'version': settings.app_version,
        'python_version': sys.version,
        'platform': platform.platform(),
        'environment': settings.environment.value
    })


# Decorators for automatic metrics collection
def monitor_execution_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to monitor function execution time.
    
    Args:
        metric_name: Name of the metric to record
        labels: Additional labels for the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics based on metric name
                if metric_name == "fraud_detection" and len(args) >= 2:
                    # Extract model version from args if available
                    model_version = getattr(args[1], 'version', 'unknown')
                    fraud_detection_duration.labels(
                        model_version=model_version
                    ).observe(duration)
                elif metric_name == "feature_extraction":
                    feature_type = labels.get('feature_type', 'unknown') if labels else 'unknown'
                    feature_extraction_duration.labels(
                        feature_type=feature_type
                    ).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    duration=duration,
                    error=str(e)
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics based on metric name
                if metric_name == "model_load" and len(args) >= 1:
                    model_version = args[0] if isinstance(args[0], str) else 'unknown'
                    model_load_duration.labels(
                        model_version=model_version
                    ).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    duration=duration,
                    error=str(e)
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def monitor_operation(operation_name: str, **labels):
    """Context manager to monitor operation duration.
    
    Args:
        operation_name: Name of the operation
        **labels: Additional labels for metrics
        
    Yields:
        Operation context
    """
    start_time = time.time()
    logger = StructuredLogger().get_logger("monitor")
    
    try:
        logger.info(f"Starting {operation_name}", **labels)
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name}", duration=duration, **labels)
        
        # Record specific metrics based on operation
        if operation_name == "fraud_detection":
            model_version = labels.get('model_version', 'unknown')
            fraud_detection_duration.labels(
                model_version=model_version
            ).observe(duration)
        elif operation_name == "feature_extraction":
            feature_type = labels.get('feature_type', 'unknown')
            feature_extraction_duration.labels(
                feature_type=feature_type
            ).observe(duration)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Failed {operation_name}",
            duration=duration,
            error=str(e),
            **labels
        )
        raise


def get_metrics_response() -> Response:
    """Get Prometheus metrics response.
    
    Returns:
        Prometheus metrics response
    """
    return PlainTextResponse(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


# Middleware for automatic request monitoring
class MetricsMiddleware:
    """Middleware for automatic metrics collection."""
    
    def __init__(self, app):
        self.app = app
        self.logger = StructuredLogger().get_logger("middleware")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Track active connections
        active_connections.inc()
        
        try:
            # Process request
            response_sent = False
            status_code = 500  # Default to error
            
            async def send_wrapper(message):
                nonlocal response_sent, status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                elif message["type"] == "http.response.body" and not message.get("more_body", False):
                    response_sent = True
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
            
            # Record metrics
            duration = time.time() - start_time
            endpoint = request.url.path
            method = request.method
            
            metrics_collector.record_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            endpoint = request.url.path
            method = request.method
            
            # Record error metrics
            metrics_collector.record_request(
                method=method,
                endpoint=endpoint,
                status_code=500,
                duration=duration
            )
            
            metrics_collector.record_api_error(
                error_type=type(e).__name__,
                endpoint=endpoint
            )
            
            self.logger.error(
                "Request processing failed",
                method=method,
                endpoint=endpoint,
                duration=duration,
                error=str(e)
            )
            
            raise
        
        finally:
            # Update active connections
            active_connections.dec()


# Initialize monitoring
def initialize_monitoring():
    """Initialize monitoring and logging."""
    # Setup structured logging
    structured_logger = StructuredLogger()
    
    # Setup system info
    setup_system_info()
    
    # Log initialization
    logger = structured_logger.get_logger("monitoring")
    logger.info(
        "Monitoring initialized",
        log_level=settings.log_level.value,
        environment=settings.environment.value
    )
    
    return structured_logger


# Export commonly used items
__all__ = [
    "metrics_collector",
    "StructuredLogger",
    "MetricsCollector",
    "MetricsMiddleware",
    "monitor_execution_time",
    "monitor_operation",
    "get_metrics_response",
    "initialize_monitoring",
    "registry"
]