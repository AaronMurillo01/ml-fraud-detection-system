"""Structured logging configuration for the fraud detection system.

This module provides centralized logging configuration with structured JSON formatting,
contextual information, and integration with monitoring systems.
"""

import json
import logging
import logging.config
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context and structured fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service information
        log_record['service'] = 'fraud-detection-api'
        log_record['version'] = '1.0.0'
        
        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_record['request_id'] = request_id
            
        user_id = user_id_var.get()
        if user_id:
            log_record['user_id'] = user_id
            
        session_id = session_id_var.get()
        if session_id:
            log_record['session_id'] = session_id
        
        # Add source location
        log_record['source'] = {
            'file': record.filename,
            'function': record.funcName,
            'line': record.lineno,
            'module': record.module
        }
        
        # Add process information
        log_record['process'] = {
            'pid': record.process,
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Handle exception information
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Ensure level is properly set
        log_record['level'] = record.levelname
        log_record['logger'] = record.name


class BusinessEventFormatter(StructuredFormatter):
    """Specialized formatter for business events and fraud detection logs."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add business-specific fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add business event category
        log_record['event_category'] = 'business'
        
        # Extract business metrics if present in the message
        if hasattr(record, 'transaction_id'):
            log_record['transaction_id'] = record.transaction_id
        
        if hasattr(record, 'fraud_score'):
            log_record['fraud_score'] = record.fraud_score
        
        if hasattr(record, 'model_version'):
            log_record['model_version'] = record.model_version
        
        if hasattr(record, 'processing_time_ms'):
            log_record['processing_time_ms'] = record.processing_time_ms
        
        if hasattr(record, 'features_count'):
            log_record['features_count'] = record.features_count


class SecurityEventFormatter(StructuredFormatter):
    """Specialized formatter for security events and audit logs."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add security-specific fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add security event category
        log_record['event_category'] = 'security'
        
        # Extract security-related information
        if hasattr(record, 'ip_address'):
            log_record['ip_address'] = record.ip_address
        
        if hasattr(record, 'user_agent'):
            log_record['user_agent'] = record.user_agent
        
        if hasattr(record, 'endpoint'):
            log_record['endpoint'] = record.endpoint
        
        if hasattr(record, 'method'):
            log_record['method'] = record.method
        
        if hasattr(record, 'status_code'):
            log_record['status_code'] = record.status_code
        
        if hasattr(record, 'response_time_ms'):
            log_record['response_time_ms'] = record.response_time_ms


class PerformanceEventFormatter(StructuredFormatter):
    """Specialized formatter for performance monitoring logs."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add performance-specific fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add performance event category
        log_record['event_category'] = 'performance'
        
        # Extract performance metrics
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'memory_usage_mb'):
            log_record['memory_usage_mb'] = record.memory_usage_mb
        
        if hasattr(record, 'cpu_usage_percent'):
            log_record['cpu_usage_percent'] = record.cpu_usage_percent
        
        if hasattr(record, 'operation'):
            log_record['operation'] = record.operation
        
        if hasattr(record, 'resource'):
            log_record['resource'] = record.resource


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path for file output
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Base logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'structured': {
                '()': StructuredFormatter,
                'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
            },
            'business': {
                '()': BusinessEventFormatter,
                'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
            },
            'security': {
                '()': SecurityEventFormatter,
                'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
            },
            'performance': {
                '()': PerformanceEventFormatter,
                'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
            },
            'console': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'structured',
                'stream': sys.stdout
            },
            'console_readable': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'console',
                'stream': sys.stdout
            }
        },
        'loggers': {
            # Root logger
            '': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            # Application loggers
            'fraud_detection': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'fraud_detection.api': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'fraud_detection.ml': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'fraud_detection.business': {
                'level': log_level,
                'handlers': ['console'],
                'formatter': 'business',
                'propagate': False
            },
            'fraud_detection.security': {
                'level': log_level,
                'handlers': ['console'],
                'formatter': 'security',
                'propagate': False
            },
            'fraud_detection.performance': {
                'level': log_level,
                'handlers': ['console'],
                'formatter': 'performance',
                'propagate': False
            },
            # Third-party loggers
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'fastapi': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'sqlalchemy': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'kafka': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            }
        }
    }
    
    # Add file handler if log file is specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'structured',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
        
        # Add file handler to all loggers
        for logger_config in config['loggers'].values():
            if 'handlers' in logger_config:
                logger_config['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_request_context(request_id: Optional[str] = None, 
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None) -> None:
    """Set request context for logging.
    
    Args:
        request_id: Unique request identifier
        user_id: User identifier
        session_id: Session identifier
    """
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set(None)
    user_id_var.set(None)
    session_id_var.set(None)


def log_business_event(logger: logging.Logger, 
                      message: str,
                      transaction_id: Optional[str] = None,
                      fraud_score: Optional[float] = None,
                      model_version: Optional[str] = None,
                      processing_time_ms: Optional[float] = None,
                      features_count: Optional[int] = None,
                      level: str = 'INFO') -> None:
    """Log a business event with structured data.
    
    Args:
        logger: Logger instance
        message: Log message
        transaction_id: Transaction identifier
        fraud_score: Fraud detection score
        model_version: ML model version
        processing_time_ms: Processing time in milliseconds
        features_count: Number of features processed
        level: Log level
    """
    extra = {}
    if transaction_id:
        extra['transaction_id'] = transaction_id
    if fraud_score is not None:
        extra['fraud_score'] = fraud_score
    if model_version:
        extra['model_version'] = model_version
    if processing_time_ms is not None:
        extra['processing_time_ms'] = processing_time_ms
    if features_count is not None:
        extra['features_count'] = features_count
    
    getattr(logger, level.lower())(message, extra=extra)


def log_security_event(logger: logging.Logger,
                      message: str,
                      ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      endpoint: Optional[str] = None,
                      method: Optional[str] = None,
                      status_code: Optional[int] = None,
                      response_time_ms: Optional[float] = None,
                      level: str = 'INFO') -> None:
    """Log a security event with structured data.
    
    Args:
        logger: Logger instance
        message: Log message
        ip_address: Client IP address
        user_agent: Client user agent
        endpoint: API endpoint
        method: HTTP method
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        level: Log level
    """
    extra = {}
    if ip_address:
        extra['ip_address'] = ip_address
    if user_agent:
        extra['user_agent'] = user_agent
    if endpoint:
        extra['endpoint'] = endpoint
    if method:
        extra['method'] = method
    if status_code is not None:
        extra['status_code'] = status_code
    if response_time_ms is not None:
        extra['response_time_ms'] = response_time_ms
    
    getattr(logger, level.lower())(message, extra=extra)


def log_performance_event(logger: logging.Logger,
                         message: str,
                         operation: Optional[str] = None,
                         resource: Optional[str] = None,
                         duration_ms: Optional[float] = None,
                         memory_usage_mb: Optional[float] = None,
                         cpu_usage_percent: Optional[float] = None,
                         level: str = 'INFO') -> None:
    """Log a performance event with structured data.
    
    Args:
        logger: Logger instance
        message: Log message
        operation: Operation name
        resource: Resource name
        duration_ms: Operation duration in milliseconds
        memory_usage_mb: Memory usage in MB
        cpu_usage_percent: CPU usage percentage
        level: Log level
    """
    extra = {}
    if operation:
        extra['operation'] = operation
    if resource:
        extra['resource'] = resource
    if duration_ms is not None:
        extra['duration_ms'] = duration_ms
    if memory_usage_mb is not None:
        extra['memory_usage_mb'] = memory_usage_mb
    if cpu_usage_percent is not None:
        extra['cpu_usage_percent'] = cpu_usage_percent
    
    getattr(logger, level.lower())(message, extra=extra)


# Initialize logging on module import
if __name__ != '__main__':
    setup_logging()