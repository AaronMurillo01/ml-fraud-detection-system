"""FastAPI application factory and configuration."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from .health import router as health_router
from .metrics import router as metrics_router
from .fraud import router as fraud_router
from service.core.config import get_settings
from service.core.logging import setup_logging

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting fraud detection service...")
    
    # Initialize ML models, database connections, etc.
    # This will be expanded when we add the ML inference service
    
    yield
    
    # Shutdown
    logger.info("Shutting down fraud detection service...")
    # Cleanup resources


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    settings = get_settings()
    setup_logging(settings.log_level)
    
    app = FastAPI(
        title="Credit Card Fraud Detection API",
        description="Real-time fraud detection service with ML-powered risk scoring",
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )
    
    # Security middleware
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Request logging and metrics middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Log requests and collect metrics."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Extract endpoint info
        method = request.method
        endpoint = request.url.path
        status_code = response.status_code
        
        # Update metrics
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Log request
        logger.info(
            f"{method} {endpoint} - {status_code} - {duration:.3f}s",
            extra={
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_seconds": duration,
                "user_agent": request.headers.get("user-agent"),
                "remote_addr": request.client.host if request.client else None,
            }
        )
        
        return response
    
    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(
            f"Unhandled exception: {exc}",
            extra={
                "method": request.method,
                "endpoint": request.url.path,
                "exception_type": type(exc).__name__,
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    # Include routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
    app.include_router(fraud_router, prefix="/api/v1", tags=["Fraud Detection"])
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint."""
        return {
            "service": "Credit Card Fraud Detection API",
            "version": "1.0.0",
            "status": "operational",
            "docs": "/docs" if settings.environment != "production" else "disabled"
        }
    
    return app


def run_server():
    """Run the FastAPI server."""
    settings = get_settings()
    
    uvicorn.run(
        "service.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    run_server()