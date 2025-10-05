"""Authentication models for the fraud detection system."""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class User(BaseModel):
    """User model for authentication."""
    
    id: Optional[int] = Field(None, description="User ID")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    role: UserRole = Field(default=UserRole.VIEWER, description="User role")
    is_active: bool = Field(default=True, description="Whether user is active")
    is_verified: bool = Field(default=False, description="Whether user email is verified")
    created_at: Optional[datetime] = Field(None, description="User creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "role": "analyst",
                "is_active": True
            }
        }


class UserCreate(BaseModel):
    """User creation model."""
    
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = Field(default=UserRole.VIEWER)


class UserUpdate(BaseModel):
    """User update model."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class Token(BaseModel):
    """JWT token model."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class TokenData(BaseModel):
    """Token data model for JWT payload."""
    
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)


class APIKey(BaseModel):
    """API key model."""
    
    id: Optional[int] = Field(None, description="API key ID")
    key_id: str = Field(..., description="API key identifier")
    key_hash: str = Field(..., description="Hashed API key")
    name: str = Field(..., max_length=100, description="API key name")
    user_id: int = Field(..., description="Owner user ID")
    is_active: bool = Field(default=True, description="Whether API key is active")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    last_used: Optional[datetime] = Field(None, description="Last used timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    scopes: List[str] = Field(default_factory=list, description="API key permissions")
    
    class Config:
        from_attributes = True


# SQLAlchemy Models
class UserDB(Base):
    """User database model."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(SQLEnum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)


class APIKeyDB(Base):
    """API key database model."""
    
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(String(50), unique=True, index=True, nullable=False)
    key_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    user_id = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used = Column(DateTime)
    expires_at = Column(DateTime)
    scopes = Column(Text)  # JSON string of scopes
