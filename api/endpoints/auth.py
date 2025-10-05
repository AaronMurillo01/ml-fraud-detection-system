"""Authentication endpoints for the fraud detection API."""

import logging
from datetime import timedelta
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from config.settings import get_settings
from api.auth import (
    User,
    UserCreate,
    UserUpdate,
    Token,
    APIKey,
    create_access_token,
    authenticate_user,
    get_current_active_user,
    get_password_hash,
    create_api_key,
    revoke_api_key,
    require_role,
    UserRole
)
from api.dependencies import get_database
from api.auth.api_key_manager import get_api_key_manager
from api.exceptions import AuthenticationException, ValidationException

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_database)
):
    """Authenticate user and return access token.
    
    Args:
        form_data: Login form data (username/password)
        db: Database session
        
    Returns:
        JWT access token
        
    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "role": user.role.value,
            "scopes": ["read", "write"] if user.role in [UserRole.ANALYST, UserRole.ADMIN] else ["read"]
        },
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/register", response_model=User)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_database),
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Register a new user (admin only).
    
    Args:
        user_data: User creation data
        db: Database session
        current_user: Current admin user
        
    Returns:
        Created user
        
    Raises:
        HTTPException: If user creation fails
    """
    # In a real implementation, check if user already exists and create in database
    hashed_password = get_password_hash(user_data.password)
    
    # Mock user creation
    new_user = User(
        id=999,  # Mock ID
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        is_active=True,
        is_verified=False
    )
    
    logger.info(f"User registered: {new_user.username} by admin {current_user.username}")
    
    return new_user


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user data
    """
    return current_user


@router.put("/me", response_model=User)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Update current user information.
    
    Args:
        user_update: User update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated user data
    """
    # In a real implementation, update user in database
    updated_user = current_user.copy(update=user_update.dict(exclude_unset=True))
    
    logger.info(f"User updated: {current_user.username}")
    
    return updated_user


@router.post("/api-keys", response_model=Dict[str, Any])
async def create_user_api_key(
    name: str,
    expires_days: int = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Create a new API key for the current user.
    
    Args:
        name: API key name
        expires_days: Days until expiration
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        API key information (including the key itself - only shown once)
    """
    api_key_string, api_key_model = create_api_key(
        user_id=current_user.id,
        name=name,
        scopes=["fraud_detection:read", "fraud_detection:write"],
        expires_days=expires_days,
        db=db
    )
    
    logger.info(f"API key created: {name} for user {current_user.username}")
    
    return {
        "api_key": api_key_string,  # Only shown once!
        "key_id": api_key_model.key_id,
        "name": api_key_model.name,
        "created_at": api_key_model.created_at,
        "expires_at": api_key_model.expires_at,
        "scopes": api_key_model.scopes,
        "warning": "This API key will only be shown once. Please save it securely."
    }


@router.delete("/api-keys/{key_id}")
async def revoke_user_api_key(
    key_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Revoke an API key.
    
    Args:
        key_id: API key identifier
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
    """
    try:
        manager = get_api_key_manager()
        success = manager.revoke_api_key(key_id, current_user.id)

        if not success:
            raise ValidationException("API key not found or unauthorized")

        logger.info(f"API key revoked: {key_id} by user {current_user.username}")

        return {"message": "API key revoked successfully"}

    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Failed to revoke API key {key_id}: {e}")
        raise ValidationException("Failed to revoke API key")


@router.get("/api-keys")
async def list_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """List all API keys for the current user.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of API keys (without the actual key values)
    """
    try:
        manager = get_api_key_manager()
        api_keys = manager.list_user_api_keys(current_user.id)

        return {"api_keys": api_keys}

    except Exception as e:
        logger.error(f"Failed to list API keys for user {current_user.id}: {e}")
        raise ValidationException("Failed to retrieve API keys")


@router.post("/api-keys/{key_id}/rotate")
async def rotate_api_key(
    key_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)
):
    """Rotate an API key (create new key, deactivate old one).

    Args:
        key_id: Current API key identifier
        current_user: Current authenticated user
        db: Database session

    Returns:
        New API key information

    Raises:
        ValidationException: If API key not found or unauthorized
    """
    try:
        manager = get_api_key_manager()
        new_api_key, new_api_key_model = manager.rotate_api_key(key_id, current_user.id)

        logger.info(f"API key rotated: {key_id} -> {new_api_key_model.key_id} by user {current_user.username}")

        return {
            "message": "API key rotated successfully",
            "api_key": new_api_key,
            "key_info": new_api_key_model,
            "warning": "This API key will only be shown once. Please save it securely."
        }

    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Failed to rotate API key {key_id}: {e}")
        raise ValidationException("Failed to rotate API key")


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """Logout current user.
    
    Note: With JWT tokens, logout is typically handled client-side by discarding the token.
    In a production system, you might maintain a blacklist of revoked tokens.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Success message
    """
    logger.info(f"User logged out: {current_user.username}")
    
    return {"message": "Logged out successfully"}


@router.get("/health")
async def auth_health_check():
    """Authentication service health check.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "authentication",
        "features": {
            "jwt_auth": True,
            "api_key_auth": True,
            "role_based_access": True
        }
    }
