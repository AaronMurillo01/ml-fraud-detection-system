"""Distributed tracing system for fraud detection API."""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource

from config import settings

logger = logging.getLogger(__name__)

# Context variables for request correlation
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


@dataclass
class TraceContext:
    """Trace context information."""
    correlation_id: str
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    api_key_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TracingManager:
    """Manager for distributed tracing and correlation."""
    
    def __init__(self):
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer = None
        self.initialized = False
    
    def initialize(self):
        """Initialize distributed tracing."""
        if self.initialized:
            return
        
        try:
            # Create resource with service information
            resource = Resource.create({
                "service.name": "fraud-detection-api",
                "service.version": settings.app_version,
                "service.environment": settings.environment.value,
            })
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self.tracer_provider)
            
            # Configure Jaeger exporter if enabled
            if settings.enable_tracing and settings.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                    collector_endpoint=settings.jaeger_endpoint,
                )
                
                span_processor = BatchSpanProcessor(jaeger_exporter)
                self.tracer_provider.add_span_processor(span_processor)
                
                logger.info(f"Jaeger tracing initialized: {settings.jaeger_endpoint}")
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Instrument FastAPI
            FastAPIInstrumentor.instrument()
            
            # Instrument SQLAlchemy
            SQLAlchemyInstrumentor().instrument()
            
            # Instrument Redis
            RedisInstrumentor().instrument()
            
            self.initialized = True
            logger.info("Distributed tracing initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            self.initialized = False
    
    def create_span(self, name: str, **attributes) -> trace.Span:
        """Create a new span.
        
        Args:
            name: Span name
            **attributes: Span attributes
            
        Returns:
            New span
        """
        if not self.tracer:
            return trace.NonRecordingSpan(trace.SpanContext(0, 0, False))
        
        span = self.tracer.start_span(name)
        
        # Add correlation context
        correlation_id = correlation_id_var.get()
        if correlation_id:
            span.set_attribute("correlation.id", correlation_id)
        
        request_id = request_id_var.get()
        if request_id:
            span.set_attribute("request.id", request_id)
        
        user_id = user_id_var.get()
        if user_id:
            span.set_attribute("user.id", user_id)
        
        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
        
        return span
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations.
        
        Args:
            operation_name: Name of the operation
            **attributes: Additional span attributes
        """
        span = self.create_span(operation_name, **attributes)
        start_time = time.time()
        
        try:
            with trace.use_span(span):
                yield span
                
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            duration = time.time() - start_time
            span.set_attribute("duration.seconds", duration)
            span.end()
    
    def set_correlation_context(self, context: TraceContext):
        """Set correlation context for current request.
        
        Args:
            context: Trace context information
        """
        correlation_id_var.set(context.correlation_id)
        request_id_var.set(context.request_id)
        if context.user_id:
            user_id_var.set(context.user_id)
    
    def get_correlation_context(self) -> Dict[str, Any]:
        """Get current correlation context.
        
        Returns:
            Dictionary with correlation context
        """
        return {
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
        }
    
    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID.
        
        Returns:
            New correlation ID
        """
        return str(uuid.uuid4())
    
    def generate_request_id(self) -> str:
        """Generate a new request ID.
        
        Returns:
            New request ID
        """
        return str(uuid.uuid4())


class CorrelationMiddleware:
    """Middleware for request correlation and tracing."""
    
    def __init__(self, app):
        self.app = app
        self.tracing_manager = get_tracing_manager()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Extract or generate correlation IDs
        headers = dict(scope.get("headers", []))
        
        correlation_id = None
        request_id = None
        
        # Check for existing correlation ID in headers
        for header_name, header_value in headers.items():
            header_name = header_name.decode().lower()
            header_value = header_value.decode()
            
            if header_name == "x-correlation-id":
                correlation_id = header_value
            elif header_name == "x-request-id":
                request_id = header_value
        
        # Generate IDs if not provided
        if not correlation_id:
            correlation_id = self.tracing_manager.generate_correlation_id()
        
        if not request_id:
            request_id = self.tracing_manager.generate_request_id()
        
        # Create trace context
        context = TraceContext(
            correlation_id=correlation_id,
            request_id=request_id
        )
        
        # Set correlation context
        self.tracing_manager.set_correlation_context(context)
        
        # Add correlation headers to response
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-correlation-id", correlation_id.encode()),
                    (b"x-request-id", request_id.encode()),
                ])
                message["headers"] = headers
            await send(message)
        
        await self.app(scope, receive, send_wrapper)


class StructuredLogger:
    """Enhanced structured logger with correlation context."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.tracing_manager = get_tracing_manager()
    
    def _add_correlation_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation context to log extra data.
        
        Args:
            extra: Existing extra data
            
        Returns:
            Enhanced extra data with correlation context
        """
        context = self.tracing_manager.get_correlation_context()
        
        enhanced_extra = extra.copy() if extra else {}
        enhanced_extra.update({
            k: v for k, v in context.items() if v is not None
        })
        
        return enhanced_extra
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with correlation context."""
        enhanced_extra = self._add_correlation_context(extra)
        enhanced_extra.update(kwargs)
        self.logger.info(message, extra=enhanced_extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message with correlation context."""
        enhanced_extra = self._add_correlation_context(extra)
        enhanced_extra.update(kwargs)
        self.logger.warning(message, extra=enhanced_extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error message with correlation context."""
        enhanced_extra = self._add_correlation_context(extra)
        enhanced_extra.update(kwargs)
        self.logger.error(message, extra=enhanced_extra)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message with correlation context."""
        enhanced_extra = self._add_correlation_context(extra)
        enhanced_extra.update(kwargs)
        self.logger.debug(message, extra=enhanced_extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log critical message with correlation context."""
        enhanced_extra = self._add_correlation_context(extra)
        enhanced_extra.update(kwargs)
        self.logger.critical(message, extra=enhanced_extra)


# Decorators for automatic tracing
def trace_function(operation_name: Optional[str] = None):
    """Decorator to automatically trace function execution.
    
    Args:
        operation_name: Custom operation name (defaults to function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracing_manager = get_tracing_manager()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracing_manager.trace_operation(op_name) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                return func(*args, **kwargs)
        
        async def async_wrapper(*args, **kwargs):
            tracing_manager = get_tracing_manager()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracing_manager.trace_operation(op_name) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """Get global tracing manager instance.
    
    Returns:
        Tracing manager instance
    """
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def initialize_tracing():
    """Initialize distributed tracing system."""
    tracing_manager = get_tracing_manager()
    tracing_manager.initialize()


def get_structured_logger(name: str) -> StructuredLogger:
    """Get structured logger with correlation context.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return StructuredLogger(name)


# Utility functions
def get_current_correlation_id() -> Optional[str]:
    """Get current correlation ID.
    
    Returns:
        Current correlation ID or None
    """
    return correlation_id_var.get()


def get_current_request_id() -> Optional[str]:
    """Get current request ID.
    
    Returns:
        Current request ID or None
    """
    return request_id_var.get()


def get_current_user_id() -> Optional[str]:
    """Get current user ID.
    
    Returns:
        Current user ID or None
    """
    return user_id_var.get()
