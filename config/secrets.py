"""Secrets management utilities for secure configuration handling."""

import os
import json
import logging
import secrets
import base64
from typing import Dict, Any, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manager for handling encrypted secrets and secure configuration."""
    
    def __init__(self, master_key: Optional[str] = None, secrets_file: str = "secrets.enc"):
        """Initialize secrets manager.
        
        Args:
            master_key: Master key for encryption. If None, will try to load from environment.
            secrets_file: Path to encrypted secrets file.
        """
        self.secrets_file = Path(secrets_file)
        self._fernet = None
        self._secrets_cache: Dict[str, Any] = {}
        
        if master_key:
            self._init_encryption(master_key)
        else:
            # Try to load master key from environment
            env_key = os.getenv("SECRETS_MASTER_KEY")
            if env_key:
                self._init_encryption(env_key)
    
    def _init_encryption(self, master_key: str) -> None:
        """Initialize encryption with master key.
        
        Args:
            master_key: Master key for encryption
        """
        try:
            # Derive encryption key from master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'fraud_detection_salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
            self._fernet = Fernet(key)
            logger.info("Encryption initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            Encrypted value as base64 string
        """
        if not self._fernet:
            raise ValueError("Encryption not initialized")
        
        encrypted = self._fernet.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt a secret value.
        
        Args:
            encrypted_value: Encrypted value as base64 string
            
        Returns:
            Decrypted value
        """
        if not self._fernet:
            raise ValueError("Encryption not initialized")
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self._fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise
    
    def store_secret(self, key: str, value: str) -> None:
        """Store an encrypted secret.
        
        Args:
            key: Secret key/name
            value: Secret value to encrypt and store
        """
        if not self._fernet:
            raise ValueError("Encryption not initialized")
        
        # Load existing secrets
        secrets_data = self._load_secrets_file()
        
        # Encrypt and store new secret
        encrypted_value = self.encrypt_secret(value)
        secrets_data[key] = encrypted_value
        
        # Save back to file
        self._save_secrets_file(secrets_data)
        
        # Update cache
        self._secrets_cache[key] = value
        
        logger.info(f"Secret '{key}' stored successfully")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve and decrypt a secret.
        
        Args:
            key: Secret key/name
            default: Default value if secret not found
            
        Returns:
            Decrypted secret value or default
        """
        # Check cache first
        if key in self._secrets_cache:
            return self._secrets_cache[key]
        
        # Load from file
        secrets_data = self._load_secrets_file()
        
        if key not in secrets_data:
            return default
        
        try:
            decrypted_value = self.decrypt_secret(secrets_data[key])
            self._secrets_cache[key] = decrypted_value
            return decrypted_value
        except Exception as e:
            logger.error(f"Failed to decrypt secret '{key}': {e}")
            return default
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret.
        
        Args:
            key: Secret key/name to delete
            
        Returns:
            True if secret was deleted, False if not found
        """
        secrets_data = self._load_secrets_file()
        
        if key not in secrets_data:
            return False
        
        del secrets_data[key]
        self._save_secrets_file(secrets_data)
        
        # Remove from cache
        self._secrets_cache.pop(key, None)
        
        logger.info(f"Secret '{key}' deleted successfully")
        return True
    
    def list_secrets(self) -> list:
        """List all secret keys (not values).
        
        Returns:
            List of secret keys
        """
        secrets_data = self._load_secrets_file()
        return list(secrets_data.keys())
    
    def _load_secrets_file(self) -> Dict[str, str]:
        """Load secrets from encrypted file.
        
        Returns:
            Dictionary of encrypted secrets
        """
        if not self.secrets_file.exists():
            return {}
        
        try:
            with open(self.secrets_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load secrets file: {e}")
            return {}
    
    def _save_secrets_file(self, secrets_data: Dict[str, str]) -> None:
        """Save secrets to encrypted file.
        
        Args:
            secrets_data: Dictionary of encrypted secrets
        """
        try:
            # Ensure directory exists
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.secrets_file, 'w', encoding='utf-8') as f:
                json.dump(secrets_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save secrets file: {e}")
            raise


def generate_secret_key(length: int = 32) -> str:
    """Generate a secure random secret key.
    
    Args:
        length: Length of the secret key
        
    Returns:
        Secure random secret key
    """
    return secrets.token_urlsafe(length)


def generate_master_key() -> str:
    """Generate a master key for secrets encryption.
    
    Returns:
        Master key for encryption
    """
    return secrets.token_urlsafe(64)


def get_env_or_secret(
    key: str, 
    secrets_manager: Optional[SecretsManager] = None,
    default: Optional[str] = None
) -> Optional[str]:
    """Get value from environment variable or secrets manager.
    
    Args:
        key: Key to look up
        secrets_manager: Secrets manager instance
        default: Default value if not found
        
    Returns:
        Value from environment or secrets, or default
    """
    # First try environment variable
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    # Then try secrets manager
    if secrets_manager:
        secret_value = secrets_manager.get_secret(key)
        if secret_value:
            return secret_value
    
    return default


def setup_production_secrets(secrets_manager: SecretsManager) -> None:
    """Setup production secrets with secure defaults.
    
    Args:
        secrets_manager: Secrets manager instance
    """
    # Generate secure secret key if not exists
    if not secrets_manager.get_secret("SECRET_KEY"):
        secret_key = generate_secret_key(64)
        secrets_manager.store_secret("SECRET_KEY", secret_key)
        logger.info("Generated new SECRET_KEY")
    
    # Generate JWT signing key if not exists
    if not secrets_manager.get_secret("JWT_SIGNING_KEY"):
        jwt_key = generate_secret_key(64)
        secrets_manager.store_secret("JWT_SIGNING_KEY", jwt_key)
        logger.info("Generated new JWT_SIGNING_KEY")
    
    # Generate API key encryption key if not exists
    if not secrets_manager.get_secret("API_KEY_ENCRYPTION_KEY"):
        api_key = generate_secret_key(32)
        secrets_manager.store_secret("API_KEY_ENCRYPTION_KEY", api_key)
        logger.info("Generated new API_KEY_ENCRYPTION_KEY")
    
    logger.info("Production secrets setup completed")


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> Optional[SecretsManager]:
    """Get global secrets manager instance.
    
    Returns:
        Secrets manager instance or None if not initialized
    """
    return _secrets_manager


def initialize_secrets_manager(master_key: Optional[str] = None) -> SecretsManager:
    """Initialize global secrets manager.
    
    Args:
        master_key: Master key for encryption
        
    Returns:
        Initialized secrets manager
    """
    global _secrets_manager
    _secrets_manager = SecretsManager(master_key)
    return _secrets_manager
