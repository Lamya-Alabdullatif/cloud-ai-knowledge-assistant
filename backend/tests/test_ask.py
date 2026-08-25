"""Tests for POST /ask, including the "no relevant information found" case."""
from __future__ import annotations


def test_ask_without_any_uploaded_documents_returns_ungrounded_answer(client):
    response = client.post("/ask", json={"question": "What is in the documents?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["sources"] == []
    assert "could not find" in body["answer"].lower() or "لم أجد" in body["answer"]


def test_ask_after_upload_returns_grounded_answer_with_sources(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")})

    response = client.post("/ask", json={"question": "What technologies does the architecture use?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["sources"]) > 0
    assert body["sources"][0]["document_name"] == "report.pdf"
    assert body["sources"][0]["page_number"] in (1, 2)
    assert body["answer"]


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422  # pydantic min_length/validator failure


def test_ask_rejects_missing_question_field(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_can_restrict_to_specific_document_ids(client, sample_pdf_bytes):
    first = client.post(
        "/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")}
    ).json()

    response = client.post(
        "/ask",
        json={"question": "architecture", "document_ids": [first["document_id"]]},
    )
    assert response.status_code == 200
    body = response.json()
    for source in body["sources"]:
        assert source["document_id"] == first["document_id"]


def test_ask_with_unknown_document_id_filter_returns_ungrounded_answer(client, sample_pdf_bytes):
    client.post("/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")})

    response = client.post(
        "/ask",
        json={"question": "architecture", "document_ids": ["does-not-exist"]},
    )
    assert response.status_code == 200
    assert response.json()["grounded"] is False
