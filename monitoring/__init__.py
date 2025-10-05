"""Monitoring package for fraud detection system."""

from .health_checks import (
    HealthChecker,
    HealthCheckResult,
    SystemHealth,
    HealthStatus,
    get_health_checker,
    run_health_checks,
    register_health_check
)

from .tracing import (
    TracingManager,
    TraceContext,
    CorrelationMiddleware,
    StructuredLogger,
    get_tracing_manager,
    initialize_tracing,
    get_structured_logger,
    get_current_correlation_id,
    get_current_request_id,
    get_current_user_id,
    trace_function
)

from .alerting import (
    AlertManager,
    Alert,
    AlertRule,
    AlertSeverity,
    AlertType,
    get_alert_manager,
    trigger_custom_alert
)

__all__ = [
    # Health checks
    "HealthChecker",
    "HealthCheckResult",
    "SystemHealth",
    "HealthStatus",
    "get_health_checker",
    "run_health_checks",
    "register_health_check",

    # Tracing
    "TracingManager",
    "TraceContext",
    "CorrelationMiddleware",
    "StructuredLogger",
    "get_tracing_manager",
    "initialize_tracing",
    "get_structured_logger",
    "get_current_correlation_id",
    "get_current_request_id",
    "get_current_user_id",
    "trace_function",

    # Alerting
    "AlertManager",
    "Alert",
    "AlertRule",
    "AlertSeverity",
    "AlertType",
    "get_alert_manager",
    "trigger_custom_alert"
]
