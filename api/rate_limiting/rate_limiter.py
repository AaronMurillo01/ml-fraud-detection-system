"""Rate limiting service for the fraud detection API."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from cache import get_redis_manager, RedisManager
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimitType(str, Enum):
    """Types of rate limits."""
    USER = "user"
    API_KEY = "api_key"
    IP_ADDRESS = "ip"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


class RateLimitWindow(str, Enum):
    """Rate limit time windows."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class RateLimitResult:
    """Result of a rate limit check."""
    
    def __init__(
        self,
        allowed: bool,
        limit: int,
        remaining: int,
        reset_time: datetime,
        retry_after: Optional[int] = None
    ):
        self.allowed = allowed
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time
        self.retry_after = retry_after  # Seconds until next allowed request
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_time.timestamp()))
        }
        
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        
        return headers


class RateLimitConfig:
    """Configuration for rate limiting."""
    
    def __init__(
        self,
        limit: int,
        window: RateLimitWindow,
        burst_limit: Optional[int] = None,
        burst_window: Optional[RateLimitWindow] = None
    ):
        self.limit = limit
        self.window = window
        self.burst_limit = burst_limit or limit * 2
        self.burst_window = burst_window or RateLimitWindow.MINUTE
    
    def get_window_seconds(self, window: RateLimitWindow) -> int:
        """Get window duration in seconds."""
        window_map = {
            RateLimitWindow.SECOND: 1,
            RateLimitWindow.MINUTE: 60,
            RateLimitWindow.HOUR: 3600,
            RateLimitWindow.DAY: 86400
        }
        return window_map[window]


class RateLimiter:
    """Redis-based rate limiter with sliding window algorithm."""
    
    def __init__(self, redis_manager: Optional[RedisManager] = None):
        """Initialize rate limiter.
        
        Args:
            redis_manager: Redis manager instance
        """
        self.redis_manager = redis_manager
        self.key_prefix = "fraud:rate_limit:"
        
        # Default rate limit configurations
        self.default_configs = {
            RateLimitType.USER: RateLimitConfig(100, RateLimitWindow.MINUTE),
            RateLimitType.API_KEY: RateLimitConfig(1000, RateLimitWindow.MINUTE),
            RateLimitType.IP_ADDRESS: RateLimitConfig(60, RateLimitWindow.MINUTE),
            RateLimitType.ENDPOINT: RateLimitConfig(500, RateLimitWindow.MINUTE),
            RateLimitType.GLOBAL: RateLimitConfig(10000, RateLimitWindow.MINUTE)
        }
        
        # Endpoint-specific configurations
        self.endpoint_configs = {
            "/api/v1/fraud/predict": RateLimitConfig(50, RateLimitWindow.MINUTE),
            "/api/v1/fraud/batch": RateLimitConfig(10, RateLimitWindow.MINUTE),
            "/api/v1/auth/login": RateLimitConfig(5, RateLimitWindow.MINUTE),
            "/api/v1/auth/register": RateLimitConfig(3, RateLimitWindow.HOUR)
        }
    
    async def _ensure_redis(self):
        """Ensure Redis manager is available."""
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
    
    def _generate_key(self, limit_type: RateLimitType, identifier: str, window: RateLimitWindow) -> str:
        """Generate Redis key for rate limiting.
        
        Args:
            limit_type: Type of rate limit
            identifier: Unique identifier (user_id, api_key, ip, etc.)
            window: Time window
            
        Returns:
            Redis key
        """
        current_window = int(time.time() // self._get_window_seconds(window))
        return f"{self.key_prefix}{limit_type.value}:{identifier}:{window.value}:{current_window}"
    
    def _get_window_seconds(self, window: RateLimitWindow) -> int:
        """Get window duration in seconds."""
        window_map = {
            RateLimitWindow.SECOND: 1,
            RateLimitWindow.MINUTE: 60,
            RateLimitWindow.HOUR: 3600,
            RateLimitWindow.DAY: 86400
        }
        return window_map[window]
    
    async def check_rate_limit(
        self,
        limit_type: RateLimitType,
        identifier: str,
        endpoint: Optional[str] = None,
        custom_config: Optional[RateLimitConfig] = None
    ) -> RateLimitResult:
        """Check if request is within rate limits.
        
        Args:
            limit_type: Type of rate limit to check
            identifier: Unique identifier
            endpoint: API endpoint (for endpoint-specific limits)
            custom_config: Custom rate limit configuration
            
        Returns:
            Rate limit check result
        """
        try:
            await self._ensure_redis()
            
            # Get configuration
            config = custom_config
            if not config:
                if endpoint and endpoint in self.endpoint_configs:
                    config = self.endpoint_configs[endpoint]
                else:
                    config = self.default_configs.get(limit_type, self.default_configs[RateLimitType.USER])
            
            # Check main rate limit
            main_result = await self._check_sliding_window(
                limit_type, identifier, config.limit, config.window
            )
            
            # Check burst limit if configured
            if config.burst_limit and config.burst_window:
                burst_result = await self._check_sliding_window(
                    limit_type, identifier, config.burst_limit, config.burst_window
                )
                
                # Use the more restrictive result
                if not burst_result.allowed:
                    return burst_result
            
            return main_result
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {limit_type}:{identifier}: {e}")
            # Fail open - allow request if rate limiting fails
            return RateLimitResult(
                allowed=True,
                limit=1000,
                remaining=999,
                reset_time=datetime.utcnow() + timedelta(minutes=1)
            )
    
    async def _check_sliding_window(
        self,
        limit_type: RateLimitType,
        identifier: str,
        limit: int,
        window: RateLimitWindow
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm.
        
        Args:
            limit_type: Type of rate limit
            identifier: Unique identifier
            limit: Request limit
            window: Time window
            
        Returns:
            Rate limit result
        """
        window_seconds = self._get_window_seconds(window)
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Use sorted set to track requests in time window
        key = f"{self.key_prefix}{limit_type.value}:{identifier}:{window.value}"
        
        # Remove old entries
        await self.redis_manager._redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_count = await self.redis_manager._redis.zcard(key)
        
        # Calculate reset time (next window)
        reset_time = datetime.fromtimestamp(current_time + window_seconds)
        
        if current_count >= limit:
            # Rate limit exceeded
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=int(window_seconds)
            )
        
        # Add current request
        await self.redis_manager._redis.zadd(key, {str(current_time): current_time})
        
        # Set expiration for cleanup
        await self.redis_manager._redis.expire(key, window_seconds * 2)
        
        remaining = limit - current_count - 1
        
        return RateLimitResult(
            allowed=True,
            limit=limit,
            remaining=max(0, remaining),
            reset_time=reset_time
        )
    
    async def reset_rate_limit(
        self,
        limit_type: RateLimitType,
        identifier: str,
        window: Optional[RateLimitWindow] = None
    ) -> bool:
        """Reset rate limit for a specific identifier.
        
        Args:
            limit_type: Type of rate limit
            identifier: Unique identifier
            window: Specific window to reset (all if None)
            
        Returns:
            True if reset successful
        """
        try:
            await self._ensure_redis()
            
            if window:
                # Reset specific window
                key = f"{self.key_prefix}{limit_type.value}:{identifier}:{window.value}"
                await self.redis_manager._redis.delete(key)
            else:
                # Reset all windows for this identifier
                pattern = f"{self.key_prefix}{limit_type.value}:{identifier}:*"
                await self.redis_manager.clear_pattern(pattern)
            
            logger.info(f"Reset rate limit for {limit_type}:{identifier}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit for {limit_type}:{identifier}: {e}")
            return False
    
    async def get_rate_limit_status(
        self,
        limit_type: RateLimitType,
        identifier: str
    ) -> Dict[str, Any]:
        """Get current rate limit status for an identifier.
        
        Args:
            limit_type: Type of rate limit
            identifier: Unique identifier
            
        Returns:
            Rate limit status information
        """
        try:
            await self._ensure_redis()
            
            status = {}
            
            for window in RateLimitWindow:
                key = f"{self.key_prefix}{limit_type.value}:{identifier}:{window.value}"
                current_count = await self.redis_manager._redis.zcard(key)
                
                config = self.default_configs.get(limit_type, self.default_configs[RateLimitType.USER])
                limit = config.limit if config.window == window else 0
                
                status[window.value] = {
                    "current_count": current_count,
                    "limit": limit,
                    "remaining": max(0, limit - current_count) if limit > 0 else 0
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get rate limit status for {limit_type}:{identifier}: {e}")
            return {}


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance.
    
    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        redis_manager = await get_redis_manager()
        _rate_limiter = RateLimiter(redis_manager)
    
    return _rate_limiter
