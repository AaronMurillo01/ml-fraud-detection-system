"""Authentication and authorization module for fraud detection API."""

from .jwt_auth import (
    create_access_token,
    verify_token,
    get_current_user,
    get_current_active_user,
    require_role,
    require_any_role,
    authenticate_user,
    get_password_hash,
    get_current_user_from_token,
    get_api_key_from_header
)
from .api_key_auth import (
    create_api_key,
    verify_api_key,
    get_api_key_user,
    revoke_api_key
)
from .models import (
    User,
    UserRole,
    APIKey,
    Token,
    TokenData,
    UserCreate,
    UserUpdate
)

__all__ = [
    "create_access_token",
    "verify_token",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "require_any_role",
    "authenticate_user",
    "get_password_hash",
    "get_current_user_from_token",
    "get_api_key_from_header",
    "create_api_key",
    "verify_api_key",
    "get_api_key_user",
    "revoke_api_key",
    "User",
    "UserRole",
    "APIKey",
    "Token",
    "TokenData",
    "UserCreate",
    "UserUpdate"
]
