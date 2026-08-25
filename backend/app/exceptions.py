"""Custom application exceptions and their FastAPI handlers.

Keeping exceptions explicit (rather than letting raw boto3 / qdrant / OpenAI
errors bubble up) means the API always returns a predictable, well-formed
JSON error body instead of leaking internal stack traces to callers.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("app")


class AppError(Exception):
    """Base class for all application-level errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class InvalidFileTypeError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_file_type"


class FileTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    error_code = "file_too_large"


class EmptyFileError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "empty_file"


class DocumentNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "document_not_found"


class DocumentProcessingError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_code = "document_processing_failed"


class InvalidQuestionError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "invalid_question"


class StorageError(AppError):
    """Raised when S3 (or another storage backend) operations fail."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "storage_error"


class VectorStoreError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "vector_store_error"


class LLMError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "llm_error"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so every AppError -> clean JSON, and unexpected errors
    are logged + returned as a generic 500 instead of crashing the process."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("AppError on %s %s: %s", request.method, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error_code, "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )
