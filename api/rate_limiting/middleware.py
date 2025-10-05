"""Rate limiting middleware for FastAPI."""

import logging
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .rate_limiter import get_rate_limiter, RateLimitType, RateLimitResult
from api.auth import get_current_user_from_token, get_api_key_from_header
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""
    
    def __init__(self, app, enabled: bool = True):
        """Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self.enabled = enabled
        
        # Endpoints that bypass rate limiting
        self.bypass_endpoints = {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
        
        # Endpoints with special rate limiting
        self.special_endpoints = {
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/fraud/predict",
            "/api/v1/fraud/batch"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for bypass endpoints
        if any(request.url.path.startswith(endpoint) for endpoint in self.bypass_endpoints):
            return await call_next(request)
        
        try:
            # Perform rate limit checks
            rate_limit_result = await self._check_rate_limits(request)
            
            if not rate_limit_result.allowed:
                # Rate limit exceeded
                return self._create_rate_limit_response(rate_limit_result)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            for header, value in rate_limit_result.to_headers().items():
                response.headers[header] = value
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue processing if rate limiting fails
            return await call_next(request)
    
    async def _check_rate_limits(self, request: Request) -> RateLimitResult:
        """Check various rate limits for the request.
        
        Args:
            request: HTTP request
            
        Returns:
            Most restrictive rate limit result
        """
        rate_limiter = await get_rate_limiter()
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        
        # Check IP-based rate limit first (most basic protection)
        ip_result = await rate_limiter.check_rate_limit(
            RateLimitType.IP_ADDRESS,
            client_ip,
            endpoint
        )
        
        if not ip_result.allowed:
            logger.warning(f"IP rate limit exceeded for {client_ip} on {endpoint}")
            return ip_result
        
        # Check endpoint-specific rate limit
        endpoint_result = await rate_limiter.check_rate_limit(
            RateLimitType.ENDPOINT,
            endpoint,
            endpoint
        )
        
        if not endpoint_result.allowed:
            logger.warning(f"Endpoint rate limit exceeded for {endpoint}")
            return endpoint_result
        
        # Check user/API key specific rate limits
        user_result = await self._check_authenticated_rate_limits(request, rate_limiter, endpoint)
        
        if user_result and not user_result.allowed:
            return user_result
        
        # Return the most restrictive result (lowest remaining)
        results = [ip_result, endpoint_result]
        if user_result:
            results.append(user_result)
        
        return min(results, key=lambda r: r.remaining)
    
    async def _check_authenticated_rate_limits(
        self,
        request: Request,
        rate_limiter,
        endpoint: str
    ) -> Optional[RateLimitResult]:
        """Check rate limits for authenticated users.
        
        Args:
            request: HTTP request
            rate_limiter: Rate limiter instance
            endpoint: API endpoint
            
        Returns:
            Rate limit result or None if not authenticated
        """
        # Check for API key authentication
        api_key = get_api_key_from_header(request)
        if api_key:
            api_key_result = await rate_limiter.check_rate_limit(
                RateLimitType.API_KEY,
                api_key,
                endpoint
            )
            
            if not api_key_result.allowed:
                logger.warning(f"API key rate limit exceeded for key ending in ...{api_key[-4:]}")
            
            return api_key_result
        
        # Check for JWT token authentication
        try:
            user = await get_current_user_from_token(request)
            if user:
                user_result = await rate_limiter.check_rate_limit(
                    RateLimitType.USER,
                    str(user.id),
                    endpoint
                )
                
                if not user_result.allowed:
                    logger.warning(f"User rate limit exceeded for user {user.username}")
                
                return user_result
        except Exception:
            # Not authenticated or invalid token
            pass
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"
    
    def _create_rate_limit_response(self, rate_limit_result: RateLimitResult) -> JSONResponse:
        """Create rate limit exceeded response.
        
        Args:
            rate_limit_result: Rate limit check result
            
        Returns:
            JSON response with rate limit information
        """
        content = {
            "error": "Rate limit exceeded",
            "message": f"Too many requests. Limit: {rate_limit_result.limit} per window",
            "limit": rate_limit_result.limit,
            "remaining": rate_limit_result.remaining,
            "reset_time": rate_limit_result.reset_time.isoformat(),
            "retry_after": rate_limit_result.retry_after
        }
        
        headers = rate_limit_result.to_headers()
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=content,
            headers=headers
        )


def create_rate_limit_middleware(enabled: bool = None) -> RateLimitMiddleware:
    """Create rate limiting middleware.
    
    Args:
        enabled: Whether to enable rate limiting (uses settings default if None)
        
    Returns:
        Rate limiting middleware
    """
    if enabled is None:
        enabled = settings.enable_rate_limiting
    
    return RateLimitMiddleware(enabled=enabled)


# Rate limiting dependency for manual checks
async def check_rate_limit_dependency(
    request: Request,
    limit_type: RateLimitType = RateLimitType.USER,
    identifier: Optional[str] = None
) -> RateLimitResult:
    """Dependency for manual rate limit checks in endpoints.
    
    Args:
        request: HTTP request
        limit_type: Type of rate limit to check
        identifier: Custom identifier (auto-detected if None)
        
    Returns:
        Rate limit check result
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    rate_limiter = await get_rate_limiter()
    
    # Auto-detect identifier if not provided
    if not identifier:
        if limit_type == RateLimitType.IP_ADDRESS:
            identifier = request.client.host if request.client else "unknown"
        elif limit_type == RateLimitType.ENDPOINT:
            identifier = request.url.path
        else:
            # Try to get user/API key
            api_key = get_api_key_from_header(request)
            if api_key:
                identifier = api_key
                limit_type = RateLimitType.API_KEY
            else:
                try:
                    user = await get_current_user_from_token(request)
                    if user:
                        identifier = str(user.id)
                        limit_type = RateLimitType.USER
                    else:
                        identifier = request.client.host if request.client else "unknown"
                        limit_type = RateLimitType.IP_ADDRESS
                except Exception:
                    identifier = request.client.host if request.client else "unknown"
                    limit_type = RateLimitType.IP_ADDRESS
    
    result = await rate_limiter.check_rate_limit(
        limit_type,
        identifier,
        request.url.path
    )
    
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "limit": result.limit,
                "remaining": result.remaining,
                "reset_time": result.reset_time.isoformat(),
                "retry_after": result.retry_after
            },
            headers=result.to_headers()
        )
    
    return result
