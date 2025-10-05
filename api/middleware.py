"""Middleware for fraud detection API.

This module provides middleware for:
- Request/response logging
- Security headers
- Request validation
- Performance monitoring
- Error handling
"""

import logging
import time
import json
import uuid
from typing import Callable, Dict, Any

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and log details.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            f"Request started - ID: {request_id} - "
            f"Method: {request.method} - "
            f"URL: {request.url} - "
            f"Client: {client_ip} - "
            f"User-Agent: {user_agent}"
        )
        
        # Log request body for POST/PUT requests (if enabled and not too large)
        if settings.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= settings.max_log_body_size:
                    # Try to parse as JSON for better logging
                    try:
                        body_json = json.loads(body.decode())
                        logger.debug(f"Request body - ID: {request_id} - Body: {json.dumps(body_json, indent=2)}")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.debug(f"Request body - ID: {request_id} - Body: {body[:500]}...")
                else:
                    logger.debug(f"Request body - ID: {request_id} - Body too large ({len(body)} bytes)")
            except Exception as e:
                logger.warning(f"Failed to log request body - ID: {request_id} - Error: {e}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            # Log response
            logger.info(
                f"Request completed - ID: {request_id} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.4f}s"
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed - ID: {request_id} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.4f}s",
                exc_info=True
            )
            
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add security headers to response.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }
        
        # Add HSTS header for HTTPS
        if request.url.scheme == "https":
            security_headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Apply headers
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and sanitization."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Validate and sanitize requests.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_request_size:
            return StarletteResponse(
                content=json.dumps({
                    "error": {
                        "type": "request_too_large",
                        "message": f"Request size exceeds maximum allowed size of {settings.max_request_size} bytes",
                        "status_code": 413
                    }
                }),
                status_code=413,
                media_type="application/json"
            )
        
        # Check content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            allowed_types = ["application/json", "application/x-www-form-urlencoded", "multipart/form-data"]
            
            if not any(allowed_type in content_type for allowed_type in allowed_types):
                return StarletteResponse(
                    content=json.dumps({
                        "error": {
                            "type": "unsupported_media_type",
                            "message": f"Content type '{content_type}' not supported",
                            "status_code": 415
                        }
                    }),
                    status_code=415,
                    media_type="application/json"
                )
        
        # Validate user agent (basic bot detection)
        user_agent = request.headers.get("user-agent", "")
        suspicious_agents = ["bot", "crawler", "spider", "scraper"]
        
        if settings.block_suspicious_agents and any(agent in user_agent.lower() for agent in suspicious_agents):
            logger.warning(f"Blocked suspicious user agent: {user_agent}")
            return StarletteResponse(
                content=json.dumps({
                    "error": {
                        "type": "forbidden",
                        "message": "Access denied",
                        "status_code": 403
                    }
                }),
                status_code=403,
                media_type="application/json"
            )
        
        return await call_next(request)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics collection."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.request_count = 0
        self.total_time = 0.0
        self.slow_requests = []
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Monitor request performance.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate metrics
            process_time = time.time() - start_time
            self.request_count += 1
            self.total_time += process_time
            
            # Track slow requests
            if process_time > settings.slow_request_threshold:
                slow_request = {
                    "method": request.method,
                    "url": str(request.url),
                    "time": process_time,
                    "timestamp": time.time(),
                    "status_code": response.status_code
                }
                
                self.slow_requests.append(slow_request)
                
                # Keep only recent slow requests
                if len(self.slow_requests) > 100:
                    self.slow_requests = self.slow_requests[-50:]
                
                logger.warning(
                    f"Slow request detected - "
                    f"Method: {request.method} - "
                    f"URL: {request.url} - "
                    f"Time: {process_time:.4f}s"
                )
            
            # Add performance headers
            response.headers["X-Request-Count"] = str(self.request_count)
            response.headers["X-Average-Time"] = f"{self.total_time / self.request_count:.4f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request failed after {process_time:.4f}s: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Performance metrics
        """
        return {
            "request_count": self.request_count,
            "total_time": self.total_time,
            "average_time": self.total_time / self.request_count if self.request_count > 0 else 0,
            "slow_requests_count": len(self.slow_requests),
            "recent_slow_requests": self.slow_requests[-10:] if self.slow_requests else []
        }


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Handle errors and provide consistent error responses.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or endpoint
            
        Returns:
            Response object
        """
        try:
            return await call_next(request)
            
        except Exception as e:
            # Log error with context
            request_id = getattr(request.state, 'request_id', 'unknown')
            logger.error(
                f"Unhandled error in request {request_id} - "
                f"Method: {request.method} - "
                f"URL: {request.url} - "
                f"Error: {str(e)}",
                exc_info=True
            )
            
            # Return consistent error response
            error_response = {
                "error": {
                    "type": "internal_error",
                    "message": "An internal error occurred",
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "status_code": 500
                }
            }
            
            # Add error details in debug mode
            if settings.debug:
                error_response["error"]["details"] = str(e)
                error_response["error"]["type"] = type(e).__name__
            
            return StarletteResponse(
                content=json.dumps(error_response),
                status_code=500,
                media_type="application/json",
                headers={"X-Request-ID": request_id}
            )


def setup_middleware(app: FastAPI):
    """Setup all middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Add middleware in reverse order (last added is executed first)
    
    # Error handling (outermost)
    if settings.enable_error_middleware:
        app.add_middleware(ErrorHandlingMiddleware)
    
    # Performance monitoring
    if settings.enable_performance_monitoring:
        performance_middleware = PerformanceMonitoringMiddleware(app)
        app.add_middleware(PerformanceMonitoringMiddleware)
        
        # Store reference for metrics endpoint
        app.state.performance_middleware = performance_middleware
    
    # Security headers
    if settings.enable_security_headers:
        app.add_middleware(SecurityHeadersMiddleware)
    
    # Request validation
    if settings.enable_request_validation:
        app.add_middleware(RequestValidationMiddleware)
    
    # Request logging (innermost, closest to endpoints)
    if settings.enable_request_logging:
        app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Middleware setup completed")


def get_performance_metrics(app: FastAPI) -> Dict[str, Any]:
    """Get performance metrics from middleware.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Performance metrics
    """
    if hasattr(app.state, 'performance_middleware'):
        return app.state.performance_middleware.get_metrics()
    
    return {
        "error": "Performance monitoring not enabled"
    }