"""Redis cache manager for the fraud detection system."""

import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import asyncio

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import RedisError, ConnectionError

from config import get_settings
from api.exceptions import DatabaseException, ErrorCode

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisManager:
    """Manages Redis connections and caching operations."""
    
    def __init__(self, redis_url: str = None, max_connections: int = None):
        """Initialize Redis manager.
        
        Args:
            redis_url: Redis connection URL
            max_connections: Maximum number of connections in pool
        """
        self.redis_url = redis_url or settings.redis_url
        self.max_connections = max_connections or settings.redis_max_connections
        self._pool = None
        self._redis = None
        self._is_connected = False
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.prediction_ttl = 1800  # 30 minutes
        self.feature_ttl = 7200  # 2 hours
        self.model_metadata_ttl = 86400  # 24 hours
        
        # Key prefixes for different data types
        self.key_prefixes = {
            'prediction': 'fraud:prediction:',
            'features': 'fraud:features:',
            'model_metadata': 'fraud:model:',
            'user_profile': 'fraud:user:',
            'merchant_profile': 'fraud:merchant:',
            'rate_limit': 'fraud:rate_limit:',
            'session': 'fraud:session:'
        }
    
    async def initialize(self):
        """Initialize Redis connection pool."""
        try:
            # Parse Redis URL and create connection pool
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30
            )
            
            # Create Redis client
            self._redis = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._redis.ping()
            self._is_connected = True
            
            logger.info(f"Redis connection initialized: {self.redis_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            self._is_connected = False
            raise DatabaseException(
                message="Redis initialization failed",
                error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
                details=str(e)
            )
    
    async def close(self):
        """Close Redis connections."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        self._is_connected = False
        logger.info("Redis connections closed")
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix
            *args: Arguments to include in key
            
        Returns:
            Generated cache key
        """
        # Create a hash of the arguments for consistent key generation
        key_data = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}{key_hash}"
    
    async def health_check(self) -> bool:
        """Check Redis health.
        
        Returns:
            True if Redis is healthy, False otherwise
        """
        try:
            if not self._redis:
                return False
            
            await self._redis.ping()
            return True
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get Redis connection information.
        
        Returns:
            Connection information dictionary
        """
        try:
            info = await self._redis.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'is_connected': self._is_connected
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {'is_connected': False, 'error': str(e)}
    
    async def set_json(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set a JSON value in Redis.
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_connected:
                return False
            
            json_value = json.dumps(value, default=str)
            ttl = ttl or self.default_ttl
            
            await self._redis.setex(key, ttl, json_value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set JSON value for key {key}: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get a JSON value from Redis.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if not self._is_connected:
                return None
            
            value = await self._redis.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get JSON value for key {key}: {e}")
            return None
    
    async def set_pickle(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set a pickled value in Redis.
        
        Args:
            key: Cache key
            value: Value to cache (will be pickled)
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_connected:
                return False
            
            pickled_value = pickle.dumps(value)
            ttl = ttl or self.default_ttl
            
            await self._redis.setex(key, ttl, pickled_value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set pickled value for key {key}: {e}")
            return False
    
    async def get_pickle(self, key: str) -> Optional[Any]:
        """Get a pickled value from Redis.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if not self._is_connected:
                return None
            
            value = await self._redis.get(key)
            if value:
                return pickle.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get pickled value for key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key from Redis.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_connected:
                return False
            
            result = await self._redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            if not self._is_connected:
                return False
            
            result = await self._redis.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._is_connected:
                return False
            
            result = await self._redis.expire(key, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Failed to set expiration for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., 'fraud:prediction:*')
            
        Returns:
            Number of keys deleted
        """
        try:
            if not self._is_connected:
                return 0
            
            keys = await self._redis.keys(pattern)
            if keys:
                deleted = await self._redis.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Failed to clear keys with pattern {pattern}: {e}")
            return 0


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None


async def get_redis_manager() -> RedisManager:
    """Get the global Redis manager instance.
    
    Returns:
        RedisManager instance
    """
    global _redis_manager
    
    if _redis_manager is None:
        _redis_manager = RedisManager()
        await _redis_manager.initialize()
    
    return _redis_manager


async def initialize_redis():
    """Initialize Redis connections."""
    try:
        redis_manager = await get_redis_manager()
        is_healthy = await redis_manager.health_check()
        
        if not is_healthy:
            raise DatabaseException(
                message="Redis initialization failed",
                error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
                details="Health check failed"
            )
        
        logger.info("Redis initialized successfully")
        return redis_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        raise


async def shutdown_redis():
    """Shutdown Redis connections."""
    global _redis_manager
    
    if _redis_manager:
        await _redis_manager.close()
        _redis_manager = None
        logger.info("Redis connections shutdown")
