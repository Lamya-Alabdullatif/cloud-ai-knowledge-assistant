"""Tests for the file upload flow: POST /upload."""
from __future__ import annotations

from app.schemas import DocumentStatus


def test_upload_valid_pdf_returns_201_and_processed_status(client, sample_pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["document_name"] == "report.pdf"
    assert body["status"] == DocumentStatus.PROCESSED.value
    assert body["document_id"]


def test_upload_registers_document_with_page_and_chunk_counts(client, sample_pdf_bytes):
    response = client.post(
        "/upload",
        files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")},
    )
    document_id = response.json()["document_id"]

    detail = client.get(f"/documents/{document_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["page_count"] == 2
    assert body["chunk_count"] > 0


def test_upload_stores_original_file_in_fake_s3(client, sample_pdf_bytes, fake_s3):
    response = client.post(
        "/upload",
        files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")},
    )
    document_id = response.json()["document_id"]
    key = fake_s3.object_key(document_id, "report.pdf")
    assert key in fake_s3.store
    assert fake_s3.store[key] == sample_pdf_bytes


def test_upload_indexes_chunks_in_vector_store(client, sample_pdf_bytes, fake_vector_store):
    client.post("/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")})
    assert len(fake_vector_store.points) > 0
    assert all(p["document_name"] == "report.pdf" for p in fake_vector_store.points)
