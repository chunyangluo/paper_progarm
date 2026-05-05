from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AppException(Exception):
    def __init__(self, error: str, detail: Optional[str] = None, status_code: int = 500):
        self.error = error
        self.detail = detail
        self.status_code = status_code
        super().__init__(error)


class NotFoundException(AppException):
    def __init__(self, error: str = "Resource not found", detail: Optional[str] = None):
        super().__init__(error, detail, 404)


class ValidationError(AppException):
    def __init__(self, error: str = "Validation error", detail: Optional[str] = None):
        super().__init__(error, detail, 422)


class ModelNotLoadedException(AppException):
    def __init__(self, model_type: str = ""):
        super().__init__(
            "Model not loaded",
            f"Model '{model_type}' is not available. Please check model configuration.",
            503
        )


class InferenceException(AppException):
    def __init__(self, detail: str = "Inference failed"):
        super().__init__("Inference error", detail, 500)


class RateLimitException(AppException):
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__("Rate limit exceeded", detail, 429)


class DatabaseException(AppException):
    def __init__(self, detail: str = "Database error"):
        super().__init__("Database error", detail, 500)


async def app_exception_handler(request: Request, exc: AppException):
    logger.error(f"AppException: {exc.error} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error,
            "detail": exc.detail,
            "status_code": exc.status_code,
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if logger.isEnabledFor(logging.DEBUG) else None,
            "status_code": 500,
        }
    )
