"""Exception handlers for the fraud detection API."""

import logging
import traceback
import uuid
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

from config.settings import get_settings
from .models import (
    ErrorResponse,
    ValidationErrorDetail,
    ErrorCode,
    RateLimitErrorResponse,
    ModelErrorResponse,
    DatabaseErrorResponse,
    BusinessLogicErrorResponse
)

logger = logging.getLogger(__name__)
settings = get_settings()


# Custom Exception Classes
class FraudDetectionException(Exception):
    """Base exception for fraud detection system."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: str = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_code = error_code
        self.details = details
        self.status_code = status_code
        super().__init__(message)


class ValidationException(FraudDetectionException):
    """Validation error exception."""
    
    def __init__(
        self,
        message: str,
        validation_errors: List[ValidationErrorDetail] = None,
        details: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        self.validation_errors = validation_errors or []


class AuthenticationException(FraudDetectionException):
    """Authentication error exception."""
    
    def __init__(self, message: str = "Authentication failed", details: str = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            details=details,
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationException(FraudDetectionException):
    """Authorization error exception."""
    
    def __init__(self, message: str = "Authorization failed", details: str = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            details=details,
            status_code=status.HTTP_403_FORBIDDEN
        )


class ModelException(FraudDetectionException):
    """ML model error exception."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.PREDICTION_FAILED,
        model_name: str = None,
        model_version: str = None,
        details: str = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        self.model_name = model_name
        self.model_version = model_version


class DatabaseException(FraudDetectionException):
    """Database error exception."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATABASE_QUERY_FAILED,
        operation: str = None,
        table: str = None,
        details: str = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        self.operation = operation
        self.table = table


class RateLimitException(FraudDetectionException):
    """Rate limit exceeded exception."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        limit: int = 100,
        window: int = 3600,
        remaining: int = 0,
        details: str = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            details=details,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )
        self.retry_after = retry_after
        self.limit = limit
        self.window = window
        self.remaining = remaining


# Exception Handlers
def create_error_response(
    request: Request,
    error_code: ErrorCode,
    message: str,
    details: str = None,
    validation_errors: List[ValidationErrorDetail] = None,
    debug_info: Dict[str, Any] = None
) -> ErrorResponse:
    """Create a standardized error response.
    
    Args:
        request: FastAPI request object
        error_code: Error code
        message: Error message
        details: Additional details
        validation_errors: Validation error details
        debug_info: Debug information (only in development)
        
    Returns:
        Error response model
    """
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        validation_errors=validation_errors,
        timestamp=datetime.utcnow(),
        request_id=request_id,
        path=str(request.url.path),
        method=request.method,
        debug_info=debug_info if settings.debug else None
    )


async def fraud_detection_exception_handler(
    request: Request, 
    exc: FraudDetectionException
) -> JSONResponse:
    """Handle custom fraud detection exceptions."""
    
    logger.error(
        f"FraudDetectionException: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # Create specific error response based on exception type
    if isinstance(exc, ValidationException):
        error_response = create_error_response(
            request=request,
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            validation_errors=exc.validation_errors
        )
    elif isinstance(exc, ModelException):
        error_response = ModelErrorResponse(
            **create_error_response(
                request=request,
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details
            ).dict(),
            model_name=exc.model_name,
            model_version=exc.model_version
        )
    elif isinstance(exc, DatabaseException):
        error_response = DatabaseErrorResponse(
            **create_error_response(
                request=request,
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details
            ).dict(),
            operation=exc.operation,
            table=exc.table
        )
    elif isinstance(exc, RateLimitException):
        error_response = RateLimitErrorResponse(
            **create_error_response(
                request=request,
                error_code=exc.error_code,
                message=exc.message,
                details=exc.details
            ).dict(),
            retry_after=exc.retry_after,
            limit=exc.limit,
            window=exc.window,
            remaining=exc.remaining
        )
    else:
        error_response = create_error_response(
            request=request,
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(exclude_none=True, mode='json')
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""

    logger.warning(
        f"HTTP Exception: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method
        }
    )

    # Map HTTP status codes to error codes
    error_code_map = {
        404: ErrorCode.NOT_FOUND,
        401: ErrorCode.AUTHENTICATION_FAILED,
        403: ErrorCode.AUTHORIZATION_FAILED,
        409: ErrorCode.CONFLICT,
        429: ErrorCode.RATE_LIMIT_EXCEEDED
    }

    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    error_response = create_error_response(
        request=request,
        error_code=error_code,
        message=str(exc.detail),
        details=f"HTTP {exc.status_code} error"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(exclude_none=True, mode='json')
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""

    logger.warning(
        f"Validation Error: {len(exc.errors())} validation errors",
        extra={
            "errors": exc.errors(),
            "path": request.url.path,
            "method": request.method
        }
    )

    # Convert Pydantic validation errors to our format
    validation_errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"][1:])  # Skip 'body' prefix
        validation_errors.append(
            ValidationErrorDetail(
                field=field,
                message=error["msg"],
                invalid_value=error.get("input")
            )
        )

    error_response = create_error_response(
        request=request,
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details=f"{len(validation_errors)} validation error(s) found",
        validation_errors=validation_errors
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(exclude_none=True, mode='json')
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general unhandled exceptions."""

    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))

    logger.error(
        f"Unhandled Exception: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )

    debug_info = None
    if settings.debug:
        debug_info = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }

    error_response = create_error_response(
        request=request,
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An internal server error occurred",
        details="Please contact support if this problem persists",
        debug_info=debug_info
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(exclude_none=True, mode='json')
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup all exception handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """

    # Custom exception handlers
    app.add_exception_handler(FraudDetectionException, fraud_detection_exception_handler)

    # Standard HTTP exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Validation exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)

    # General exception handler (catch-all)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Exception handlers configured successfully")
