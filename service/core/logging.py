"""Structured logging configuration for the fraud detection system."""

import logging
import logging.config
import sys
from datetime import datetime
from typing import Dict, Any, Optional

import structlog
from pythonjsonlogger import jsonlogger

from .config import Settings


def setup_logging(settings: Settings) -> None:
    """Setup structured logging with JSON format for production."""
    
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
            structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d %(funcName)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": settings.log_format if settings.log_format in ["json", "standard"] else "standard",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "sqlalchemy": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
            "kafka": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False,
            },
        },
    }
    
    # Add file handler if log file is specified
    if settings.log_file:
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.log_level,
            "formatter": settings.log_format if settings.log_format in ["json", "standard"] else "standard",
            "filename": settings.log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        }
        
        # Add file handler to all loggers
        for logger_config in logging_config["loggers"].values():
            logger_config["handlers"].append("file")
    
    logging.config.dictConfig(logging_config)
    
    # Set third-party library log levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Log startup message
    logger = get_logger("fraud_detection.startup")
    logger.info(
        "Logging configured",
        log_level=settings.log_level,
        log_format=settings.log_format,
        environment=settings.environment,
        app_version=settings.app_version
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestLoggingMiddleware:
    """Middleware for logging HTTP requests and responses."""
    
    def __init__(self, app, logger_name: str = "fraud_detection.api"):
        self.app = app
        self.logger = get_logger(logger_name)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = datetime.utcnow()
        request_id = scope.get("headers", {}).get("x-request-id", "unknown")
        
        # Log request
        self.logger.info(
            "Request started",
            method=scope["method"],
            path=scope["path"],
            query_string=scope.get("query_string", b"").decode(),
            client_ip=scope.get("client", ["unknown", None])[0],
            request_id=request_id,
            user_agent=self._get_header(scope, "user-agent"),
        )
        
        # Capture response
        response_data = {"status_code": None, "response_size": 0}
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_data["status_code"] = message["status"]
            elif message["type"] == "http.response.body":
                response_data["response_size"] += len(message.get("body", b""))
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Log error
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(
                "Request failed",
                method=scope["method"],
                path=scope["path"],
                request_id=request_id,
                duration_seconds=duration,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        else:
            # Log successful response
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(
                "Request completed",
                method=scope["method"],
                path=scope["path"],
                status_code=response_data["status_code"],
                response_size=response_data["response_size"],
                duration_seconds=duration,
                request_id=request_id,
            )
    
    def _get_header(self, scope: Dict[str, Any], header_name: str) -> Optional[str]:
        """Extract header value from ASGI scope."""
        headers = scope.get("headers", [])
        for name, value in headers:
            if name.decode().lower() == header_name.lower():
                return value.decode()
        return None


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""
    
    def filter(self, record):
        # This would integrate with request context
        # For now, just add a placeholder
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'unknown'
        return True


class SecurityAuditLogger:
    """Specialized logger for security events."""
    
    def __init__(self):
        self.logger = get_logger("fraud_detection.security")
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str):
        """Log authentication attempts."""
        self.logger.info(
            "Authentication attempt",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            event_type="authentication",
        )
    
    def log_fraud_detection(self, transaction_id: str, fraud_probability: float, 
                          action_taken: str, model_version: str):
        """Log fraud detection events."""
        self.logger.info(
            "Fraud detection",
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            action_taken=action_taken,
            model_version=model_version,
            event_type="fraud_detection",
        )
    
    def log_suspicious_activity(self, activity_type: str, details: Dict[str, Any]):
        """Log suspicious activities."""
        self.logger.warning(
            "Suspicious activity detected",
            activity_type=activity_type,
            event_type="suspicious_activity",
            **details
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str):
        """Log data access events."""
        self.logger.info(
            "Data access",
            user_id=user_id,
            resource=resource,
            action=action,
            event_type="data_access",
        )


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self):
        self.logger = get_logger("fraud_detection.performance")
    
    def log_inference_time(self, model_name: str, inference_time_ms: float, 
                          transaction_id: str):
        """Log ML inference performance."""
        self.logger.info(
            "ML inference completed",
            model_name=model_name,
            inference_time_ms=inference_time_ms,
            transaction_id=transaction_id,
            event_type="ml_inference",
        )
    
    def log_feature_computation_time(self, feature_set: str, computation_time_ms: float,
                                   transaction_id: str):
        """Log feature computation performance."""
        self.logger.info(
            "Feature computation completed",
            feature_set=feature_set,
            computation_time_ms=computation_time_ms,
            transaction_id=transaction_id,
            event_type="feature_computation",
        )
    
    def log_database_query_time(self, query_type: str, execution_time_ms: float,
                              record_count: int):
        """Log database query performance."""
        self.logger.info(
            "Database query completed",
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            record_count=record_count,
            event_type="database_query",
        )
    
    def log_api_latency(self, endpoint: str, method: str, latency_ms: float,
                       status_code: int):
        """Log API endpoint latency."""
        self.logger.info(
            "API request completed",
            endpoint=endpoint,
            method=method,
            latency_ms=latency_ms,
            status_code=status_code,
            event_type="api_latency",
        )


# Global logger instances
security_logger = SecurityAuditLogger()
performance_logger = PerformanceLogger()