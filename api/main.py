"""Main FastAPI application for fraud detection system.

This module sets up the FastAPI application with:
- API routing and endpoints
- Middleware configuration
- Exception handling
- CORS and security settings
- Application lifecycle events
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from config import settings, get_config_summary
from service.ml_inference import get_inference_service
from features.feature_pipeline import get_feature_pipeline
from .endpoints import (
    fraud_detection,
    health,
    model_management,
    auth,
    database,
    cache,
    rate_limiting,
    monitoring,
    websocket,
    batch,
    export,
    history
)
from .middleware import (
    setup_middleware,
    PrometheusMonitoringMiddleware,
    BusinessMetricsMiddleware,
    DatabaseMetricsMiddleware,
    setup_monitoring_middleware
)
from .middleware.security_enhanced import (
    SecurityHeadersMiddleware,
    CSRFProtectionMiddleware,
    InputSanitizationMiddleware,
    RateLimitMiddleware
)
from .utils.logging_config import setup_logging, get_logger
from .exceptions import setup_exception_handlers
from .rate_limiting import create_rate_limit_middleware
from monitoring import initialize_tracing, CorrelationMiddleware

# Initialize structured logging
setup_logging()
logger = get_logger(__name__)

# Application settings are imported from config module


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting fraud detection API service")

    try:
        # Initialize database connections
        try:
            from database.connection import initialize_database
            await initialize_database()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.warning(f"Database initialization failed (optional): {e}")

        # Initialize Redis cache
        try:
            from cache import initialize_redis
            await initialize_redis()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed (optional): {e}")

        # Initialize optimized model loader
        try:
            from service.model_loader import initialize_model_loader
            await initialize_model_loader()
            logger.info("Optimized model loader initialized successfully")
        except Exception as e:
            logger.warning(f"Model loader initialization failed (optional): {e}")

        # Initialize ML service
        try:
            ml_service = get_inference_service()
            logger.info("ML inference service initialized")
        except Exception as e:
            logger.warning(f"ML inference service initialization failed (optional): {e}")

        # Initialize feature pipeline
        try:
            feature_pipeline = get_feature_pipeline()
            logger.info("Feature pipeline initialized")
        except Exception as e:
            logger.warning(f"Feature pipeline initialization failed (optional): {e}")

        # Initialize distributed tracing
        try:
            initialize_tracing()
            logger.info("Distributed tracing initialized")
        except Exception as e:
            logger.warning(f"Distributed tracing initialization failed (optional): {e}")

        # Setup Prometheus metrics if enabled
        if settings.enable_metrics:
            instrumentator = Instrumentator(
                should_group_status_codes=False,
                should_ignore_untemplated=True,
                should_respect_env_var=True,
                should_instrument_requests_inprogress=True,
                excluded_handlers=["/health", "/metrics"],
                env_var_name="ENABLE_METRICS",
                inprogress_name="inprogress",
                inprogress_labels=True,
            )
            instrumentator.instrument(app).expose(app)
            logger.info("Prometheus metrics enabled")
        
        logger.info("Fraud detection API service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start fraud detection API service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down fraud detection API service")
    
    try:
        # Shutdown model loader
        try:
            from service.model_loader import shutdown_model_loader
            await shutdown_model_loader()
            logger.info("Model loader shutdown")
        except Exception as e:
            logger.warning(f"Model loader shutdown failed: {e}")

        # Shutdown Redis cache
        try:
            from cache import shutdown_redis
            await shutdown_redis()
            logger.info("Redis cache closed")
        except Exception as e:
            logger.warning(f"Redis cache shutdown failed: {e}")

        # Shutdown database connections
        try:
            from database.connection import shutdown_database
            await shutdown_database()
            logger.info("Database connections closed")
        except Exception as e:
            logger.warning(f"Database shutdown failed: {e}")

        logger.info("Fraud detection API service shut down successfully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection system with ML-powered risk assessment",
    version="1.0.0",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
    lifespan=lifespan
)

# Setup exception handlers
setup_exception_handlers(app)

# Add enhanced security middleware (order matters!)
# 1. Security headers (first to apply to all responses)
app.add_middleware(SecurityHeadersMiddleware)

# 2. Input sanitization
app.add_middleware(InputSanitizationMiddleware)

# 3. CSRF protection
app.add_middleware(
    CSRFProtectionMiddleware,
    secret_key=settings.secret_key,
    exempt_paths=["/docs", "/redoc", "/openapi.json", "/health", "/metrics", "/ws"]
)

# 4. Enhanced rate limiting
# app.add_middleware(
#     RateLimitMiddleware,
#     requests_per_minute=settings.rate_limit_per_minute,
#     burst_size=settings.rate_limit_burst
# )

# Add correlation middleware for tracing
app.add_middleware(CorrelationMiddleware)

# Add rate limiting middleware (skip for demo - causes issues)
# app.add_middleware(RateLimitMiddleware, enabled=settings.enable_rate_limiting)

# Setup middleware
setup_middleware(app)

# Setup monitoring middleware
setup_monitoring_middleware(app)

# Add CORS middleware with proper configuration
if settings.enable_cors:
    # In production, replace ["*"] with specific allowed origins
    allowed_origins = settings.cors_origins if settings.cors_origins != ["*"] else [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://yourdomain.com"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-CSRF-Token"],
        max_age=3600,
    )

# Add trusted host middleware for security
if settings.trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts
    )


# Exception handlers are now configured in setup_exception_handlers()


# Request/Response middleware for logging and monitoring
@app.middleware("http")
async def request_response_middleware(request: Request, call_next):
    """Log requests and responses, measure processing time."""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.4f}s"
        )
        raise


# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    fraud_detection.router,
    prefix="/api/v1/fraud",
    tags=["Fraud Detection"]
)

app.include_router(
    model_management.router,
    prefix="/api/v1/models",
    tags=["Model Management"]
)

app.include_router(
    database.router,
    prefix="/api/v1/database",
    tags=["Database"]
)

app.include_router(
    cache.router,
    prefix="/api/v1/cache",
    tags=["Cache"]
)

app.include_router(
    rate_limiting.router,
    prefix="/api/v1/rate-limit",
    tags=["Rate Limiting"]
)

app.include_router(
    monitoring.router,
    tags=["Monitoring"]
)

# New enhanced endpoints
app.include_router(
    websocket.router,
    tags=["WebSocket"]
)

app.include_router(
    batch.router,
    prefix="/api/v1",
    tags=["Batch Processing"]
)

app.include_router(
    export.router,
    prefix="/api/v1",
    tags=["Export"]
)

app.include_router(
    history.router,
    prefix="/api/v1",
    tags=["Transaction History"]
)


# Mount static files for web UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint - serve the enhanced web UI
@app.get("/", include_in_schema=False)
async def root():
    """Serve the enhanced web UI."""
    from fastapi.responses import FileResponse
    return FileResponse('static/index_enhanced.html')

# Legacy endpoint for original UI
@app.get("/legacy", include_in_schema=False)
async def legacy_ui():
    """Serve the original web UI."""
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')


# Convenience redirects for easier page access
@app.get("/dashboard", include_in_schema=False)
async def dashboard_redirect():
    """Redirect to dashboard page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/dashboard.html")


@app.get("/analytics", include_in_schema=False)
async def analytics_redirect():
    """Redirect to analytics page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/analytics_dashboard.html")


@app.get("/models", include_in_schema=False)
async def models_redirect():
    """Redirect to models page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/models.html")


@app.get("/monitoring", include_in_schema=False)
async def monitoring_redirect():
    """Redirect to monitoring page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/monitoring.html")


@app.get("/reports", include_in_schema=False)
async def reports_redirect():
    """Redirect to reports page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/reports.html")


@app.get("/settings", include_in_schema=False)
async def settings_redirect():
    """Redirect to settings page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/settings.html")


# API information endpoint
@app.get("/info", tags=["Information"])
async def api_info():
    """Get API information and configuration."""
    return {
        "service": {
            "name": "Fraud Detection API",
            "version": "1.0.0",
            "description": "Real-time credit card fraud detection system",
            "environment": settings.environment.value
        },
        "features": {
            "real_time_detection": True,
            "batch_processing": True,
            "model_management": True,
            "metrics_collection": settings.enable_metrics,
            "documentation": settings.enable_docs
        },
        "limits": {
            "max_request_size": "10MB",
            "rate_limit": "1000 requests/minute",
            "timeout": "30 seconds"
        },
        "timestamp": time.time()
    }


@app.get("/config", tags=["Information"])
async def config_info():
    """Get current configuration summary (safe for production)."""
    return get_config_summary()


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )