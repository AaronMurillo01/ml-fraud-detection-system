"""Base Pydantic models with common functionality."""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict


class BaseModel(PydanticBaseModel):
    """Base model with common configuration."""
    
    model_config = ConfigDict(
        # Enable validation on assignment
        validate_assignment=True,
        # Use enum values instead of names
        use_enum_values=True,
        # Validate default values
        validate_default=True,
        # Allow extra fields for flexibility
        extra="forbid",
        # Serialize by alias
        populate_by_name=True,
        # JSON schema generation
        json_schema_serialization_defaults_required=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp tracking."""
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was created",
        json_schema_extra={"example": "2024-01-15T10:30:00Z"}
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the record was last updated",
        json_schema_extra={"example": "2024-01-15T11:45:00Z"}
    )


class UUIDMixin(BaseModel):
    """Mixin for models that need UUID primary keys."""
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the record",
        json_schema_extra={"example": "123e4567-e89b-12d3-a456-426614174000"}
    )


class EntityModel(UUIDMixin, TimestampMixin):
    """Base model for entities with ID and timestamps."""
    pass