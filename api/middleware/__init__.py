"""Middleware package for the fraud detection API."""

def setup_middleware(app):
    """Setup all middleware for the application."""
    # Placeholder implementation
    pass

def get_performance_metrics():
    """Get performance metrics (placeholder implementation)."""
    return {"status": "ok", "metrics": {}}

# Placeholder classes for monitoring
class PrometheusMonitoringMiddleware:
    pass

class BusinessMetricsMiddleware:
    pass

class DatabaseMetricsMiddleware:
    pass

def setup_monitoring_middleware(app):
    """Setup monitoring middleware (placeholder implementation)."""
    pass

__all__ = [
    "PrometheusMonitoringMiddleware",
    "BusinessMetricsMiddleware", 
    "DatabaseMetricsMiddleware",
    "setup_monitoring_middleware",
    "setup_middleware",
    "get_performance_metrics"
]