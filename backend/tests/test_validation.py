"""Tests for input validation on file upload: wrong type, oversized, empty."""
from __future__ import annotations

import pytest

from app.exceptions import EmptyFileError, FileTooLargeError, InvalidFileTypeError


def test_upload_rejects_non_pdf_extension(client, sample_pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_file_type"


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "empty_file"


def test_upload_rejects_file_over_size_limit(client, test_settings):
    # test_settings.MAX_FILE_SIZE_MB == 1 -> generate something bigger than 1MB
    oversized = b"%PDF-1.4\n" + b"0" * (test_settings.max_file_size_bytes + 1024)
    response = client.post(
        "/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "file_too_large"


def test_upload_rejects_pdf_with_no_extractable_text(client, empty_pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("blank.pdf", empty_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "document_processing_failed"


def test_validate_file_directly_raises_for_missing_extension(document_service):
    with pytest.raises(InvalidFileTypeError):
        document_service.validate_file("noextension", b"%PDF-1.4 some bytes")


def test_validate_file_directly_raises_for_empty_bytes(document_service):
    with pytest.raises(EmptyFileError):
        document_service.validate_file("file.pdf", b"")


def test_validate_file_directly_raises_for_oversized(document_service, test_settings):
    oversized = b"0" * (test_settings.max_file_size_bytes + 1)
    with pytest.raises(FileTooLargeError):
        document_service.validate_file("file.pdf", oversized)
