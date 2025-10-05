"""Logging configuration for the fraud detection system.

This module provides:
- Centralized logging configuration
- Custom log formatters and handlers
- Log level management
- Audit logging for fraud detection events
"""

import logging
import logging.config
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

import structlog
from pythonjsonlogger import jsonlogger

from . import settings


class CustomJSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record.
        
        Args:
            log_record: Log record dictionary
            record: Original log record
            message_dict: Message dictionary
        """
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service information
        log_record['service'] = 'fraud-detection-api'
        log_record['version'] = settings.app_version
        log_record['environment'] = settings.environment.value
        
        # Add log level
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


class AuditLogger:
    """Specialized logger for audit events."""
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
        self._setup_audit_logger()
    
    def _setup_audit_logger(self):
        """Setup audit logger with specific configuration."""
        # Create audit log handler
        if settings.is_production():
            # In production, log to file
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            handler = logging.FileHandler(
                log_dir / 'audit.log',
                encoding='utf-8'
            )
        else:
            # In development, log to console
            handler = logging.StreamHandler(sys.stdout)
        
        # Set formatter
        formatter = CustomJSONFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Configure logger
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def log_fraud_detection(self, 
                          transaction_id: str,
                          user_id: Optional[str],
                          prediction: str,
                          confidence: float,
                          model_version: str,
                          processing_time: float,
                          features_used: Dict[str, Any],
                          ip_address: Optional[str] = None):
        """Log fraud detection event.
        
        Args:
            transaction_id: Transaction ID
            user_id: User ID (if available)
            prediction: Fraud prediction result
            confidence: Confidence score
            model_version: Model version used
            processing_time: Processing time in seconds
            features_used: Features used for prediction
            ip_address: Client IP address
        """
        self.logger.info(
            "Fraud detection completed",
            extra={
                'event_type': 'fraud_detection',
                'transaction_id': transaction_id,
                'user_id': user_id,
                'prediction': prediction,
                'confidence': confidence,
                'model_version': model_version,
                'processing_time_seconds': processing_time,
                'features_count': len(features_used),
                'ip_address': ip_address,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_model_deployment(self,
                           model_version: str,
                           deployment_id: str,
                           deployment_status: str,
                           deployed_by: str,
                           deployment_strategy: str):
        """Log model deployment event.
        
        Args:
            model_version: Model version
            deployment_id: Deployment ID
            deployment_status: Deployment status
            deployed_by: User who deployed the model
            deployment_strategy: Deployment strategy used
        """
        self.logger.info(
            "Model deployment event",
            extra={
                'event_type': 'model_deployment',
                'model_version': model_version,
                'deployment_id': deployment_id,
                'deployment_status': deployment_status,
                'deployed_by': deployed_by,
                'deployment_strategy': deployment_strategy,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_api_access(self,
                      endpoint: str,
                      method: str,
                      user_id: Optional[str],
                      api_key_id: Optional[str],
                      status_code: int,
                      response_time: float,
                      ip_address: Optional[str] = None):
        """Log API access event.
        
        Args:
            endpoint: API endpoint accessed
            method: HTTP method
            user_id: User ID (if authenticated)
            api_key_id: API key ID used
            status_code: HTTP status code
            response_time: Response time in seconds
            ip_address: Client IP address
        """
        self.logger.info(
            "API access",
            extra={
                'event_type': 'api_access',
                'endpoint': endpoint,
                'method': method,
                'user_id': user_id,
                'api_key_id': api_key_id,
                'status_code': status_code,
                'response_time_seconds': response_time,
                'ip_address': ip_address,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def log_security_event(self,
                          event_type: str,
                          description: str,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          additional_data: Optional[Dict[str, Any]] = None):
        """Log security event.
        
        Args:
            event_type: Type of security event
            description: Event description
            user_id: User ID (if applicable)
            ip_address: Client IP address
            additional_data: Additional event data
        """
        extra_data = {
            'event_type': 'security_event',
            'security_event_type': event_type,
            'description': description,
            'user_id': user_id,
            'ip_address': ip_address,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if additional_data:
            extra_data.update(additional_data)
        
        self.logger.warning(
            f"Security event: {event_type}",
            extra=extra_data
        )


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
        self._setup_performance_logger()
    
    def _setup_performance_logger(self):
        """Setup performance logger."""
        handler = logging.StreamHandler(sys.stdout)
        formatter = CustomJSONFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def log_slow_request(self,
                        endpoint: str,
                        method: str,
                        duration: float,
                        threshold: float = 1.0):
        """Log slow request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration: Request duration in seconds
            threshold: Slow request threshold
        """
        if duration > threshold:
            self.logger.warning(
                "Slow request detected",
                extra={
                    'event_type': 'slow_request',
                    'endpoint': endpoint,
                    'method': method,
                    'duration_seconds': duration,
                    'threshold_seconds': threshold,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
    
    def log_high_memory_usage(self,
                             memory_usage_mb: float,
                             threshold_mb: float = 1000.0):
        """Log high memory usage.
        
        Args:
            memory_usage_mb: Memory usage in MB
            threshold_mb: Memory usage threshold in MB
        """
        if memory_usage_mb > threshold_mb:
            self.logger.warning(
                "High memory usage detected",
                extra={
                    'event_type': 'high_memory_usage',
                    'memory_usage_mb': memory_usage_mb,
                    'threshold_mb': threshold_mb,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
    
    def log_model_performance(self,
                            model_version: str,
                            prediction_time: float,
                            feature_extraction_time: float,
                            total_time: float):
        """Log model performance metrics.
        
        Args:
            model_version: Model version
            prediction_time: Model prediction time in seconds
            feature_extraction_time: Feature extraction time in seconds
            total_time: Total processing time in seconds
        """
        self.logger.info(
            "Model performance metrics",
            extra={
                'event_type': 'model_performance',
                'model_version': model_version,
                'prediction_time_seconds': prediction_time,
                'feature_extraction_time_seconds': feature_extraction_time,
                'total_time_seconds': total_time,
                'timestamp': datetime.utcnow().isoformat()
            }
        )


def setup_logging() -> Dict[str, Any]:
    """Setup logging configuration.
    
    Returns:
        Logging configuration dictionary
    """
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Determine log level
    log_level = getattr(logging, settings.log_level.value.upper(), logging.INFO)
    
    # Base logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': CustomJSONFormatter,
                'format': '%(timestamp)s %(level)s %(name)s %(message)s'
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'json' if settings.is_production() else 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            '': {  # Root logger
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'fastapi': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False
            }
        }
    }
    
    # Add file handlers for production
    if settings.is_production():
        config['handlers'].update({
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'json',
                'filename': str(log_dir / 'app.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.ERROR,
                'formatter': 'json',
                'filename': str(log_dir / 'error.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'encoding': 'utf-8'
            }
        })
        
        # Update loggers to use file handlers
        for logger_name in config['loggers']:
            config['loggers'][logger_name]['handlers'].extend(['file', 'error_file'])
    
    return config


def configure_structlog():
    """Configure structlog for structured logging."""
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
            structlog.processors.JSONRenderer() if settings.is_production() else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def initialize_logging():
    """Initialize logging system.
    
    Returns:
        Tuple of (audit_logger, performance_logger)
    """
    # Setup basic logging
    config = setup_logging()
    logging.config.dictConfig(config)
    
    # Configure structlog
    configure_structlog()
    
    # Create specialized loggers
    audit_logger = AuditLogger()
    performance_logger = PerformanceLogger()
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            'log_level': settings.log_level.value,
            'environment': settings.environment.value,
            'version': settings.app_version
        }
    )
    
    return audit_logger, performance_logger


# Global logger instances
audit_logger = None
performance_logger = None


def get_audit_logger() -> AuditLogger:
    """Get audit logger instance.
    
    Returns:
        Audit logger instance
    """
    global audit_logger
    if audit_logger is None:
        audit_logger = AuditLogger()
    return audit_logger


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance.
    
    Returns:
        Performance logger instance
    """
    global performance_logger
    if performance_logger is None:
        performance_logger = PerformanceLogger()
    return performance_logger


# Export commonly used items
__all__ = [
    "AuditLogger",
    "PerformanceLogger",
    "CustomJSONFormatter",
    "setup_logging",
    "configure_structlog",
    "initialize_logging",
    "get_audit_logger",
    "get_performance_logger"
]