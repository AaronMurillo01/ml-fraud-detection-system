"""Enhanced security middleware for the fraud detection API.

This module provides comprehensive security features including:
- Content Security Policy (CSP)
- CSRF Protection
- Security Headers
- Input Sanitization
- XSS Protection
"""

import logging
import secrets
import hashlib
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add comprehensive security headers to all responses."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Default security headers
        self.security_headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # HSTS - Force HTTPS
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            
            # Content Security Policy
            "Content-Security-Policy": self._build_csp(),
        }
        
        # Override with custom config
        self.security_headers.update(self.config.get("headers", {}))
    
    def _build_csp(self) -> str:
        """Build Content Security Policy header."""
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://cdn.plot.ly",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",
            "img-src 'self' data: https:",
            "connect-src 'self' ws: wss: https://cdn.plot.ly",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        return "; ".join(csp_directives)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add all security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add custom headers
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", "unknown")
        
        return response


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """Middleware to protect against Cross-Site Request Forgery attacks."""
    
    def __init__(self, app, secret_key: str, exempt_paths: Optional[list] = None):
        super().__init__(app)
        self.secret_key = secret_key
        self.exempt_paths = exempt_paths or [
            "/docs", "/redoc", "/openapi.json", "/health", "/metrics"
        ]
        self.token_header = "X-CSRF-Token"
        self.cookie_name = "csrf_token"
        self.token_expiry = timedelta(hours=1)
        
        # In-memory token store (use Redis in production)
        self.tokens: Dict[str, datetime] = {}
    
    def generate_token(self) -> str:
        """Generate a new CSRF token."""
        token = secrets.token_urlsafe(32)
        self.tokens[token] = datetime.utcnow() + self.token_expiry
        return token
    
    def validate_token(self, token: str) -> bool:
        """Validate a CSRF token."""
        if not token or token not in self.tokens:
            return False
        
        # Check if token is expired
        if datetime.utcnow() > self.tokens[token]:
            del self.tokens[token]
            return False
        
        return True
    
    def is_exempt(self, path: str) -> bool:
        """Check if path is exempt from CSRF protection."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate CSRF token for state-changing requests."""
        
        # Skip CSRF check for exempt paths
        if self.is_exempt(request.url.path):
            return await call_next(request)
        
        # Skip CSRF check for safe methods
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            response = await call_next(request)
            
            # Generate and set CSRF token for GET requests
            csrf_token = self.generate_token()
            response.set_cookie(
                key=self.cookie_name,
                value=csrf_token,
                httponly=True,
                secure=True,
                samesite="strict",
                max_age=3600
            )
            response.headers[self.token_header] = csrf_token
            
            return response
        
        # Validate CSRF token for state-changing methods
        csrf_token = request.headers.get(self.token_header)
        
        if not csrf_token:
            # Try to get from cookie
            csrf_token = request.cookies.get(self.cookie_name)
        
        if not self.validate_token(csrf_token):
            logger.warning(
                f"CSRF validation failed for {request.method} {request.url.path}",
                extra={"ip": request.client.host if request.client else "unknown"}
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "CSRF validation failed",
                    "message": "Invalid or missing CSRF token"
                }
            )
        
        return await call_next(request)


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware to sanitize user input and prevent injection attacks."""
    
    def __init__(self, app, max_body_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_body_size = max_body_size
        
        # Dangerous patterns to detect
        self.sql_injection_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(--|\#|\/\*|\*\/)",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Sanitize request input."""
        
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request too large",
                    "message": f"Request body exceeds maximum size of {self.max_body_size} bytes"
                }
            )
        
        # Log suspicious patterns (actual sanitization happens in validation layer)
        user_agent = request.headers.get("user-agent", "")
        if any(pattern in user_agent.lower() for pattern in ["sqlmap", "nikto", "nmap"]):
            logger.warning(
                f"Suspicious user agent detected: {user_agent}",
                extra={
                    "ip": request.client.host if request.client else "unknown",
                    "path": request.url.path
                }
            )
        
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting middleware with Redis support."""
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        redis_client = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.redis_client = redis_client
        
        # In-memory fallback
        self.request_counts: Dict[str, list] = {}
    
    def get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Fall back to IP address
        if request.client:
            return f"ip:{request.client.host}"
        
        return "unknown"
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        if self.redis_client:
            # Use Redis for distributed rate limiting
            key = f"rate_limit:{client_id}"
            count = await self.redis_client.incr(key)
            
            if count == 1:
                await self.redis_client.expire(key, 60)
            
            return count <= self.requests_per_minute
        else:
            # Use in-memory rate limiting
            if client_id not in self.request_counts:
                self.request_counts[client_id] = []
            
            # Remove old requests
            self.request_counts[client_id] = [
                ts for ts in self.request_counts[client_id]
                if ts > window_start
            ]
            
            # Check limit
            if len(self.request_counts[client_id]) >= self.requests_per_minute:
                return False
            
            self.request_counts[client_id].append(now)
            return True
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting."""
        client_id = self.get_client_id(request)
        
        if not await self.check_rate_limit(client_id):
            logger.warning(
                f"Rate limit exceeded for {client_id}",
                extra={"path": request.url.path}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed"
                },
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)

