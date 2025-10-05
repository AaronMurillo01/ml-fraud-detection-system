"""Security configuration and utilities."""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import lru_cache

from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from .config import Settings
from .logging import get_logger, security_logger

logger = get_logger("fraud_detection.security")


class SecurityConfig(BaseModel):
    """Security configuration settings."""
    
    # Password hashing
    password_schemes: List[str] = ["bcrypt"]
    bcrypt_rounds: int = 12
    
    # JWT settings
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    jwt_refresh_expire_days: int = 7
    
    # API Key settings
    api_key_length: int = 32
    api_key_prefix: str = "fd_"
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # Security headers
    security_headers: Dict[str, str] = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }
    
    # CORS settings
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = [
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "X-Request-ID",
        "X-API-Key"
    ]
    cors_max_age: int = 86400  # 24 hours


class TokenData(BaseModel):
    """JWT token data structure."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    scopes: List[str] = []
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    jti: Optional[str] = None  # JWT ID for token revocation


class APIKeyData(BaseModel):
    """API key data structure."""
    key_id: str
    name: str
    scopes: List[str] = []
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class SecurityManager:
    """Central security manager for authentication and authorization."""
    
    def __init__(self, settings: Settings, security_config: SecurityConfig):
        self.settings = settings
        self.config = security_config
        
        # Initialize password context
        self.pwd_context = CryptContext(
            schemes=security_config.password_schemes,
            deprecated="auto",
            bcrypt__rounds=security_config.bcrypt_rounds
        )
        
        # JWT settings
        self.secret_key = settings.secret_key
        self.algorithm = security_config.jwt_algorithm
        
        # In-memory stores (in production, use Redis or database)
        self.revoked_tokens: set = set()
        self.api_keys: Dict[str, APIKeyData] = {}
        
        logger.info("Security manager initialized")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.config.jwt_expire_minutes
            )
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        })
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.secret_key, 
            algorithm=self.algorithm
        )
        
        security_logger.log_authentication_attempt(
            user_id=data.get("sub", "unknown"),
            success=True,
            ip_address="unknown"  # This would come from request context
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a JWT refresh token."""
        expire = datetime.utcnow() + timedelta(
            days=self.config.jwt_refresh_expire_days
        )
        
        to_encode = {
            "sub": user_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)
        }
        
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self.revoked_tokens:
                logger.warning(f"Revoked token used: {jti}")
                return None
            
            # Extract token data
            token_data = TokenData(
                user_id=payload.get("sub"),
                username=payload.get("username"),
                scopes=payload.get("scopes", []),
                exp=datetime.fromtimestamp(payload.get("exp", 0)),
                iat=datetime.fromtimestamp(payload.get("iat", 0)),
                jti=jti
            )
            
            return token_data
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Allow expired tokens for revocation
            )
            
            jti = payload.get("jti")
            if jti:
                self.revoked_tokens.add(jti)
                logger.info(f"Token revoked: {jti}")
                return True
            
        except JWTError as e:
            logger.error(f"Failed to revoke token: {e}")
        
        return False
    
    def generate_api_key(self, name: str, scopes: List[str] = None) -> tuple[str, APIKeyData]:
        """Generate a new API key."""
        key_id = secrets.token_urlsafe(16)
        api_key = f"{self.config.api_key_prefix}{secrets.token_urlsafe(self.config.api_key_length)}"
        
        key_data = APIKeyData(
            key_id=key_id,
            name=name,
            scopes=scopes or [],
            created_at=datetime.utcnow()
        )
        
        self.api_keys[api_key] = key_data
        
        logger.info(f"API key generated: {key_id} for {name}")
        return api_key, key_data
    
    def verify_api_key(self, api_key: str) -> Optional[APIKeyData]:
        """Verify an API key."""
        key_data = self.api_keys.get(api_key)
        
        if not key_data:
            return None
        
        if not key_data.is_active:
            return None
        
        if key_data.expires_at and key_data.expires_at < datetime.utcnow():
            return None
        
        # Update last used timestamp
        key_data.last_used_at = datetime.utcnow()
        
        return key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_data = self.api_keys.get(api_key)
        if key_data:
            key_data.is_active = False
            logger.info(f"API key revoked: {key_data.key_id}")
            return True
        return False
    
    def check_permissions(self, required_scopes: List[str], user_scopes: List[str]) -> bool:
        """Check if user has required permissions."""
        if not required_scopes:
            return True
        
        # Check if user has all required scopes
        return all(scope in user_scopes for scope in required_scopes)
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return self.config.security_headers.copy()


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests: int = 100, window_seconds: int = 60):
        self.requests = requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Get or create client record
        if client_id not in self.clients:
            self.clients[client_id] = []
        
        client_requests = self.clients[client_id]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests if req_time > window_start]
        
        # Check if within limits
        if len(client_requests) >= self.requests:
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        if client_id not in self.clients:
            return self.requests
        
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        client_requests = self.clients[client_id]
        current_requests = [req_time for req_time in client_requests if req_time > window_start]
        
        return max(0, self.requests - len(current_requests))


@lru_cache()
def get_security_config() -> SecurityConfig:
    """Get cached security configuration."""
    return SecurityConfig()


@lru_cache()
def get_security_manager() -> SecurityManager:
    """Get cached security manager instance."""
    from .config import get_settings
    settings = get_settings()
    security_config = get_security_config()
    return SecurityManager(settings, security_config)


# Predefined scopes for the fraud detection system
class Scopes:
    """Predefined permission scopes."""
    
    # Transaction operations
    TRANSACTION_READ = "transaction:read"
    TRANSACTION_WRITE = "transaction:write"
    TRANSACTION_SCORE = "transaction:score"
    
    # Model operations
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DEPLOY = "model:deploy"
    
    # Admin operations
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    ADMIN_USERS = "admin:users"
    
    # Monitoring
    METRICS_READ = "metrics:read"
    LOGS_READ = "logs:read"
    
    # System
    HEALTH_CHECK = "system:health"
    
    @classmethod
    def get_all_scopes(cls) -> List[str]:
        """Get all available scopes."""
        return [
            cls.TRANSACTION_READ,
            cls.TRANSACTION_WRITE,
            cls.TRANSACTION_SCORE,
            cls.MODEL_READ,
            cls.MODEL_WRITE,
            cls.MODEL_DEPLOY,
            cls.ADMIN_READ,
            cls.ADMIN_WRITE,
            cls.ADMIN_USERS,
            cls.METRICS_READ,
            cls.LOGS_READ,
            cls.HEALTH_CHECK,
        ]