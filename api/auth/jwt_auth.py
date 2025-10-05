"""JWT-based authentication for the fraud detection API."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config import get_settings
from .models import User, UserRole, TokenData
from database.connection import get_database_session

logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )


def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Token data
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        role: str = payload.get("role")
        scopes: list = payload.get("scopes", [])
        
        if username is None:
            raise credentials_exception
            
        token_data = TokenData(
            username=username,
            user_id=user_id,
            role=role,
            scopes=scopes
        )
        return token_data
        
    except jwt.PyJWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise credentials_exception


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_database_session)
) -> User:
    """Get the current authenticated user.
    
    Args:
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If user is not authenticated
    """
    token_data = verify_token(credentials.credentials)
    
    # In a real implementation, you would query the database here
    # For now, return a mock user based on token data
    user = User(
        id=token_data.user_id,
        username=token_data.username,
        email=f"{token_data.username}@example.com",
        role=UserRole(token_data.role) if token_data.role else UserRole.VIEWER,
        is_active=True,
        is_verified=True
    )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        Active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_role(required_role: UserRole):
    """Decorator to require a specific user role.
    
    Args:
        required_role: Required user role
        
    Returns:
        Dependency function
    """
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires {required_role.value} role"
            )
        return current_user
    
    return role_checker


def require_any_role(*roles: UserRole):
    """Decorator to require any of the specified roles.
    
    Args:
        roles: Allowed user roles
        
    Returns:
        Dependency function
    """
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in roles and current_user.role != UserRole.ADMIN:
            role_names = [role.value for role in roles]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of these roles: {', '.join(role_names)}"
            )
        return current_user
    
    return role_checker


# Helper functions for middleware
async def get_current_user_from_token(request) -> Optional[User]:
    """Extract and verify user from JWT token in request.

    Args:
        request: FastAPI request object

    Returns:
        User if token is valid, None otherwise
    """
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]
        token_data = verify_token(token)
        if not token_data:
            return None

        # In a real implementation, fetch user from database
        # For now, return a mock user
        return User(
            id=token_data.user_id,
            username=token_data.username,
            email=f"{token_data.username}@example.com",
            role=UserRole.API_USER,
            is_active=True
        )
    except Exception:
        return None


def get_api_key_from_header(request) -> Optional[str]:
    """Extract API key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        API key if present, None otherwise
    """
    return request.headers.get("X-API-Key")


# Authentication utilities
def authenticate_user(username: str, password: str, db: Session) -> Optional[User]:
    """Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Password
        db: Database session
        
    Returns:
        User if authentication successful, None otherwise
    """
    # In a real implementation, query the database for the user
    # For now, return None to indicate authentication failed
    return None
