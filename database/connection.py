"""Database connection management with optimized pooling and retry logic."""

import logging
import asyncio
import time
from typing import Dict, Any, AsyncGenerator, Optional, Callable
from contextlib import asynccontextmanager
from functools import wraps

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError
from sqlalchemy.pool import QueuePool
from sqlalchemy import event, text

from config.settings import get_settings
from api.exceptions import DatabaseException, ErrorCode

logger = logging.getLogger(__name__)
settings = get_settings()


def retry_on_database_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry database operations on connection errors.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (DisconnectionError, OperationalError, ConnectionError) as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Database operation failed after {max_retries} retries: {e}")
                        break

                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    # Don't retry on non-connection errors
                    logger.error(f"Database operation failed with non-retryable error: {e}")
                    raise

            raise DatabaseException(
                message="Database operation failed after retries",
                error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
                details=str(last_exception)
            )
        return wrapper
    return decorator


class DatabaseManager:
    """Manages database connections and sessions with optimized pooling."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize database manager with configuration."""
        self.config = config
        self._connection_string = self._build_connection_string()
        self._engine = None
        self._session_factory = None
        self._health_check_interval = config.get('health_check_interval', 30)
        self._last_health_check = 0
        self._is_healthy = True

        # Initialize engine and session factory
        self._initialize_engine()

        # Setup connection pool event listeners
        self._setup_pool_events()

    def _initialize_engine(self):
        """Initialize the database engine with optimized settings."""
        engine_config = {
            'echo': self.config.get('echo', False),
            'echo_pool': self.config.get('echo_pool', False),

            # Connection pool settings
            'poolclass': QueuePool,
            'pool_size': self.config.get('pool_size', 20),
            'max_overflow': self.config.get('max_overflow', 30),
            'pool_timeout': self.config.get('pool_timeout', 30),
            'pool_recycle': self.config.get('pool_recycle', 3600),  # 1 hour
            'pool_pre_ping': self.config.get('pool_pre_ping', True),

            # Connection settings
            'connect_args': {
                'server_settings': {
                    'application_name': 'fraud_detection_api',
                    'jit': 'off',  # Disable JIT for faster connection
                },
                'command_timeout': 60,
                'prepared_statement_cache_size': 0,  # Disable prepared statement cache
            },

            # Performance settings
            'execution_options': {
                'isolation_level': 'READ_COMMITTED',
                'autocommit': False,
            }
        }

        self._engine = create_async_engine(self._connection_string, **engine_config)

        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )

        logger.info(f"Database engine initialized with pool_size={engine_config['pool_size']}, "
                   f"max_overflow={engine_config['max_overflow']}")

    def _setup_pool_events(self):
        """Setup connection pool event listeners for monitoring."""

        @event.listens_for(self._engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set connection-level settings for PostgreSQL."""
            if hasattr(dbapi_connection, 'set_session'):
                # Set session-level settings for better performance
                dbapi_connection.set_session(autocommit=False)

        @event.listens_for(self._engine.sync_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout for monitoring."""
            logger.debug("Connection checked out from pool")

        @event.listens_for(self._engine.sync_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin for monitoring."""
            logger.debug("Connection checked in to pool")

        @event.listens_for(self._engine.sync_engine, "invalidate")
        def receive_invalidate(dbapi_connection, connection_record, exception):
            """Log connection invalidation."""
            logger.warning(f"Connection invalidated: {exception}")

    @property
    def engine(self):
        """Get the database engine."""
        return self._engine

    @property
    def session_factory(self):
        """Get the session factory."""
        return self._session_factory
    
    def _build_connection_string(self) -> str:
        """Build database connection string from config."""
        host = self.config.get('host', 'localhost')
        port = self.config.get('port', 5432)
        database = self.config.get('database', 'fraud_detection')
        username = self.config.get('username', 'postgres')
        password = self.config.get('password', '')

        # Use asyncpg driver for better async performance
        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    
    @retry_on_database_error(max_retries=3, delay=0.5)
    async def health_check(self) -> bool:
        """Check database health and connectivity.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()
                self._is_healthy = True
                self._last_health_check = time.time()
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            self._is_healthy = False
            return False

    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status information.

        Returns:
            Dictionary with pool statistics
        """
        pool = self._engine.pool
        return {
            'pool_size': pool.size(),
            'checked_in_connections': pool.checkedin(),
            'checked_out_connections': pool.checkedout(),
            'overflow_connections': pool.overflow(),
            'invalid_connections': pool.invalid(),
            'is_healthy': self._is_healthy,
            'last_health_check': self._last_health_check
        }

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup and health checking.

        Yields:
            AsyncSession: Database session

        Raises:
            DatabaseException: If session creation fails
        """
        # Perform periodic health checks
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            await self.health_check()

        if not self._is_healthy:
            raise DatabaseException(
                message="Database is not healthy",
                error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
                details="Health check failed"
            )

        session = None
        try:
            session = self._session_factory()
            yield session
            await session.commit()
        except Exception as e:
            if session:
                await session.rollback()

            # Check if it's a connection error
            if isinstance(e, (DisconnectionError, OperationalError)):
                self._is_healthy = False
                raise DatabaseException(
                    message="Database connection error",
                    error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
                    details=str(e)
                )
            else:
                raise DatabaseException(
                    message="Database operation failed",
                    error_code=ErrorCode.DATABASE_QUERY_FAILED,
                    details=str(e)
                )
        finally:
            if session:
                await session.close()

    @retry_on_database_error(max_retries=2, delay=0.1)
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute a raw SQL query with retry logic.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query result

        Raises:
            DatabaseException: If query execution fails
        """
        async with self.get_session() as session:
            try:
                result = await session.execute(text(query), params or {})
                return result
            except Exception as e:
                logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
                raise

    async def close(self):
        """Close database connections and cleanup resources."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connections closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance.

    Returns:
        DatabaseManager instance

    Raises:
        DatabaseException: If database manager is not initialized
    """
    global _database_manager

    if _database_manager is None:
        # Initialize with settings
        db_config = {
            'host': settings.database_host,
            'port': settings.database_port,
            'database': settings.database_name,
            'username': settings.database_user,
            'password': settings.database_password,
            'pool_size': settings.database_pool_size,
            'max_overflow': settings.database_max_overflow,
            'echo': settings.debug,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'health_check_interval': 30
        }

        _database_manager = DatabaseManager(db_config)
        logger.info("Database manager initialized")

    return _database_manager


async def initialize_database():
    """Initialize database connections and perform startup checks."""
    try:
        db_manager = get_database_manager()

        # Perform health check
        is_healthy = await db_manager.health_check()
        if not is_healthy:
            raise DatabaseException(
                message="Database initialization failed",
                error_code=ErrorCode.DATABASE_CONNECTION_FAILED,
                details="Initial health check failed"
            )

        # Log pool status
        pool_status = await db_manager.get_pool_status()
        logger.info(f"Database initialized successfully. Pool status: {pool_status}")

        return db_manager

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def shutdown_database():
    """Shutdown database connections gracefully."""
    global _database_manager

    if _database_manager:
        await _database_manager.close()
        _database_manager = None
        logger.info("Database connections shutdown")


# Dependency function for FastAPI
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get database session.

    Yields:
        AsyncSession: Database session
    """
    db_manager = get_database_manager()
    async with db_manager.get_session() as session:
        yield session


# Utility functions for common database operations
@retry_on_database_error(max_retries=3)
async def execute_with_retry(session: AsyncSession, query, params: Dict[str, Any] = None):
    """Execute a query with automatic retry on connection errors.

    Args:
        session: Database session
        query: SQLAlchemy query or raw SQL
        params: Query parameters

    Returns:
        Query result
    """
    try:
        if isinstance(query, str):
            result = await session.execute(text(query), params or {})
        else:
            result = await session.execute(query, params or {})
        return result
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise


async def check_database_connectivity() -> Dict[str, Any]:
    """Check database connectivity and return status information.

    Returns:
        Dictionary with connectivity status and metrics
    """
    try:
        db_manager = get_database_manager()

        # Perform health check
        start_time = time.time()
        is_healthy = await db_manager.health_check()
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Get pool status
        pool_status = await db_manager.get_pool_status()

        return {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'response_time_ms': round(response_time, 2),
            'pool_status': pool_status,
            'timestamp': time.time()
        }

    except Exception as e:
        logger.error(f"Database connectivity check failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and connectivity."""
        start_time = time.time()
        try:
            async with self.engine.connect() as conn:
                await conn.execute("SELECT 1")
            
            response_time = (time.time() - start_time) * 1000
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2)
            }
        except SQLAlchemyError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    async def close(self):
        """Close database engine."""
        await self.engine.dispose()