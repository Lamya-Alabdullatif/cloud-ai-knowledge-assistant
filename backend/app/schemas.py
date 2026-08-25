"""Pydantic request/response models used by the FastAPI layer.

Keeping these separate from the internal service/domain objects gives us a
stable public API contract even if internal representations change.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Public representation of a document tracked by the system."""

    document_id: str
    document_name: str
    status: DocumentStatus
    upload_date: datetime
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    status: DocumentStatus
    message: str = "File accepted and processing has started."


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    message: str = "Document deleted successfully."


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional filter to restrict retrieval to specific documents. "
        "When omitted, the whole knowledge base is searched.",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty or whitespace-only")
        return cleaned


class SourceChunk(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    score: float
    text_snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    grounded: bool = Field(
        description="True when the answer was generated from retrieved document context; "
        "False when no sufficiently relevant content was found."
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    environment: str
    version: str


class ErrorResponse(BaseModel):
    error: str
    message: str
