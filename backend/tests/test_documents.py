"""Tests for document listing, retrieval, and deletion endpoints, plus error cases."""
from __future__ import annotations


def test_list_documents_empty_initially(client):
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == []
    assert body["total"] == 0


def test_list_documents_after_upload(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")})
    response = client.get("/documents")
    body = response.json()
    assert body["total"] == 1
    assert body["documents"][0]["document_name"] == "report.pdf"


def test_get_document_by_id(client, sample_pdf_bytes):
    upload = client.post(
        "/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")}
    ).json()

    response = client.get(f"/documents/{upload['document_id']}")
    assert response.status_code == 200
    assert response.json()["document_id"] == upload["document_id"]


def test_get_document_not_found_returns_404(client):
    response = client.get("/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "document_not_found"


def test_delete_document_removes_it_from_listing(client, sample_pdf_bytes):
    upload = client.post(
        "/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")}
    ).json()

    delete_response = client.delete(f"/documents/{upload['document_id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["document_id"] == upload["document_id"]

    listing = client.get("/documents").json()
    assert listing["total"] == 0


def test_delete_nonexistent_document_returns_404(client):
    response = client.delete("/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "document_not_found"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
