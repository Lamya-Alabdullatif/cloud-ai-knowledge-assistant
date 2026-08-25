"""Tests for PDF text extraction and chunking (the core of correct sourcing)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.chunking import chunk_pages
from app.exceptions import DocumentProcessingError
from app.pdf_processor import extract_text_by_page


def test_extract_text_by_page_returns_one_entry_per_page(sample_pdf_bytes):
    pages = extract_text_by_page(sample_pdf_bytes)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2
    assert "overview" in pages[0]["text"].lower()
    assert "architecture" in pages[1]["text"].lower()


def test_extract_text_by_page_raises_on_garbage_bytes():
    with pytest.raises(DocumentProcessingError):
        extract_text_by_page(b"this is not a real pdf file at all")


def test_extract_text_by_page_raises_when_no_text_found(empty_pdf_bytes):
    with pytest.raises(DocumentProcessingError):
        extract_text_by_page(empty_pdf_bytes)


def test_chunk_pages_preserves_page_number_metadata(sample_pdf_bytes):
    pages = extract_text_by_page(sample_pdf_bytes)
    chunks = chunk_pages(
        pages=pages,
        document_id="doc-1",
        document_name="report.pdf",
        upload_date=datetime.now(timezone.utc),
        chunk_size=80,
        chunk_overlap=10,
    )

    assert len(chunks) > 0
    assert {c["page_number"] for c in chunks} == {1, 2}
    for chunk in chunks:
        assert chunk["document_id"] == "doc-1"
        assert chunk["document_name"] == "report.pdf"
        assert chunk["chunk_id"]
        assert chunk["text"].strip() != ""


def test_chunk_pages_skips_empty_pages():
    pages = [
        {"page_number": 1, "text": "Real content that should be chunked into pieces."},
        {"page_number": 2, "text": ""},
    ]
    chunks = chunk_pages(
        pages=pages,
        document_id="doc-1",
        document_name="doc.pdf",
        upload_date=datetime.now(timezone.utc),
    )
    assert all(c["page_number"] == 1 for c in chunks)


def test_document_status_becomes_failed_when_processing_raises(document_service, fake_s3, monkeypatch):
    def _boom(*args, **kwargs):
        raise DocumentProcessingError("simulated extraction failure")

    monkeypatch.setattr("app.document_service.extract_text_by_page", _boom)

    with pytest.raises(DocumentProcessingError):
        document_service.upload_and_process("broken.pdf", b"%PDF-1.4 fake bytes for this test")

    documents = document_service.list_documents()
    assert len(documents) == 1
    assert documents[0].status.value == "failed"
    assert documents[0].error_message == "simulated extraction failure"
