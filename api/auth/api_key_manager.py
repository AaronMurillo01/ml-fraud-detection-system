"""API key management system for the fraud detection API."""

import logging
import secrets
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from config import get_settings
from database.connection import get_database_session
from .models import APIKeyDB, UserDB, APIKey, User
from api.exceptions import AuthenticationException, ValidationException, ErrorCode

logger = logging.getLogger(__name__)
settings = get_settings()


class APIKeyManager:
    """Manages API key creation, validation, and lifecycle."""
    
    def __init__(self, db_session: Session = None):
        self.db = db_session
    
    def generate_api_key(self) -> tuple[str, str, str]:
        """Generate a new API key with ID and hash.
        
        Returns:
            Tuple of (api_key, key_id, key_hash)
        """
        # Generate a secure random API key
        api_key = f"fd_{secrets.token_urlsafe(32)}"
        
        # Generate a unique key ID
        key_id = f"key_{secrets.token_hex(8)}"
        
        # Create hash of the API key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        return api_key, key_id, key_hash
    
    def create_api_key(
        self,
        user_id: int,
        name: str,
        scopes: List[str] = None,
        expires_days: Optional[int] = None,
        description: str = None
    ) -> tuple[str, APIKey]:
        """Create a new API key for a user.
        
        Args:
            user_id: User ID
            name: API key name
            scopes: List of permissions/scopes
            expires_days: Days until expiration (None for no expiration)
            description: API key description
            
        Returns:
            Tuple of (api_key_string, api_key_model)
            
        Raises:
            ValidationException: If validation fails
        """
        if not name or len(name.strip()) == 0:
            raise ValidationException("API key name is required")
        
        if len(name) > 100:
            raise ValidationException("API key name cannot exceed 100 characters")
        
        # Check if user exists (in a real implementation)
        # user = self.db.query(UserDB).filter(UserDB.id == user_id).first()
        # if not user:
        #     raise ValidationException(f"User {user_id} not found")
        
        # Generate API key components
        api_key, key_id, key_hash = self.generate_api_key()
        
        # Set expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        # Create API key record
        api_key_db = APIKeyDB(
            key_id=key_id,
            key_hash=key_hash,
            name=name.strip(),
            user_id=user_id,
            is_active=True,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            scopes=json.dumps(scopes or [])
        )
        
        # In a real implementation, save to database
        # self.db.add(api_key_db)
        # self.db.commit()
        # self.db.refresh(api_key_db)
        
        # Create response model
        api_key_model = APIKey(
            id=1,  # Mock ID
            key_id=key_id,
            key_hash=key_hash,
            name=name.strip(),
            user_id=user_id,
            is_active=True,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            scopes=scopes or []
        )
        
        logger.info(f"Created API key '{name}' for user {user_id}")
        
        return api_key, api_key_model
    
    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Verify an API key and return associated key info.
        
        Args:
            api_key: API key to verify
            
        Returns:
            APIKey model if valid, None otherwise
        """
        if not api_key or not api_key.startswith("fd_"):
            return None
        
        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # In a real implementation, query database
        # api_key_db = self.db.query(APIKeyDB).filter(
        #     and_(
        #         APIKeyDB.key_hash == key_hash,
        #         APIKeyDB.is_active == True,
        #         or_(APIKeyDB.expires_at.is_(None), APIKeyDB.expires_at > datetime.utcnow())
        #     )
        # ).first()
        
        # For now, return a mock API key for valid-looking keys
        if len(api_key) > 10:
            # Update last used timestamp
            # api_key_db.last_used = datetime.utcnow()
            # self.db.commit()
            
            return APIKey(
                id=1,
                key_id="key_mock123",
                key_hash=key_hash,
                name="Mock API Key",
                user_id=1,
                is_active=True,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                scopes=["fraud_detection:read", "fraud_detection:write"]
            )
        
        return None
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user associated with an API key.
        
        Args:
            api_key: API key string
            
        Returns:
            User if API key is valid, None otherwise
        """
        api_key_model = self.verify_api_key(api_key)
        if not api_key_model:
            return None
        
        # In a real implementation, query database for user
        # user_db = self.db.query(UserDB).filter(UserDB.id == api_key_model.user_id).first()
        # if not user_db:
        #     return None
        
        # For now, return a mock user
        return User(
            id=api_key_model.user_id,
            username="api_user",
            email="api@example.com",
            role="api_user",
            is_active=True,
            is_verified=True
        )
    
    def revoke_api_key(self, key_id: str, user_id: int = None) -> bool:
        """Revoke an API key.
        
        Args:
            key_id: API key identifier
            user_id: User ID (for authorization check)
            
        Returns:
            True if API key was revoked successfully
        """
        # In a real implementation, update database
        # query = self.db.query(APIKeyDB).filter(APIKeyDB.key_id == key_id)
        # if user_id:
        #     query = query.filter(APIKeyDB.user_id == user_id)
        
        # api_key_db = query.first()
        # if not api_key_db:
        #     return False
        
        # api_key_db.is_active = False
        # self.db.commit()
        
        logger.info(f"API key revoked: {key_id}")
        return True
    
    def list_user_api_keys(self, user_id: int) -> List[APIKey]:
        """List all API keys for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of API keys (without the actual key values)
        """
        # In a real implementation, query database
        # api_keys_db = self.db.query(APIKeyDB).filter(APIKeyDB.user_id == user_id).all()
        
        # For now, return mock data
        return [
            APIKey(
                id=1,
                key_id="key_mock123",
                key_hash="hash123",
                name="Production API Key",
                user_id=user_id,
                is_active=True,
                created_at=datetime.utcnow() - timedelta(days=30),
                last_used=datetime.utcnow() - timedelta(hours=2),
                scopes=["fraud_detection:read", "fraud_detection:write"]
            )
        ]
    
    def rotate_api_key(self, key_id: str, user_id: int) -> tuple[str, APIKey]:
        """Rotate an API key (create new key, deactivate old one).
        
        Args:
            key_id: Current API key identifier
            user_id: User ID
            
        Returns:
            Tuple of (new_api_key, new_api_key_model)
        """
        # In a real implementation, get the old key
        # old_key = self.db.query(APIKeyDB).filter(
        #     and_(APIKeyDB.key_id == key_id, APIKeyDB.user_id == user_id)
        # ).first()
        
        # if not old_key:
        #     raise ValidationException("API key not found")
        
        # Create new key with same properties
        new_api_key, new_api_key_model = self.create_api_key(
            user_id=user_id,
            name=f"Rotated key (was {key_id})",
            scopes=["fraud_detection:read", "fraud_detection:write"],
            expires_days=None
        )
        
        # Deactivate old key
        self.revoke_api_key(key_id, user_id)
        
        logger.info(f"API key rotated: {key_id} -> {new_api_key_model.key_id}")
        
        return new_api_key, new_api_key_model
    
    def check_api_key_scopes(self, api_key_model: APIKey, required_scopes: List[str]) -> bool:
        """Check if API key has required scopes.
        
        Args:
            api_key_model: API key model
            required_scopes: List of required scopes
            
        Returns:
            True if API key has all required scopes
        """
        if not required_scopes:
            return True
        
        api_key_scopes = set(api_key_model.scopes)
        required_scopes_set = set(required_scopes)
        
        return required_scopes_set.issubset(api_key_scopes)


# Global API key manager instance
_api_key_manager = None


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance.
    
    Returns:
        APIKeyManager instance
    """
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
