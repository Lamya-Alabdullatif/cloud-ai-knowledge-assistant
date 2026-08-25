"""Tests for the retrieval half of the RAG pipeline (vector search + grounding)."""
from __future__ import annotations

from datetime import datetime, timezone


def _index_sample_document(document_service, sample_pdf_bytes, filename="report.pdf"):
    result = document_service.upload_and_process(filename, sample_pdf_bytes)
    return result.document_id


def test_search_returns_matches_ranked_by_similarity(fake_vector_store, fake_embeddings):
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "document_name": "doc.pdf",
            "page_number": 1,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "text": "The quarterly revenue grew by 12 percent.",
        },
        {
            "chunk_id": "c2",
            "document_id": "d1",
            "document_name": "doc.pdf",
            "page_number": 2,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "text": "Completely unrelated text about cooking pasta.",
        },
    ]
    vectors = fake_embeddings.embed_texts([c["text"] for c in chunks])
    fake_vector_store.upsert_chunks(chunks, vectors)

    query_vector = fake_embeddings.embed_query("The quarterly revenue grew by 12 percent.")
    results = fake_vector_store.search(query_vector, top_k=2)

    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] >= results[1]["score"]


def test_search_can_filter_by_document_ids(fake_vector_store, fake_embeddings):
    for doc_id in ("doc-a", "doc-b"):
        chunk = {
            "chunk_id": f"chunk-{doc_id}",
            "document_id": doc_id,
            "document_name": f"{doc_id}.pdf",
            "page_number": 1,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "text": f"content belonging to {doc_id}",
        }
        vector = fake_embeddings.embed_texts([chunk["text"]])
        fake_vector_store.upsert_chunks([chunk], vector)

    query_vector = fake_embeddings.embed_query("content")
    results = fake_vector_store.search(query_vector, top_k=5, document_ids=["doc-a"])

    assert all(r["document_id"] == "doc-a" for r in results)


def test_rag_pipeline_ask_returns_grounded_answer_with_sources(
    document_service, rag_pipeline, sample_pdf_bytes
):
    _index_sample_document(document_service, sample_pdf_bytes)

    response = rag_pipeline.ask("What does the architecture use?")

    assert response.grounded is True
    assert response.sources, "expected at least one cited source"
    assert response.sources[0].document_name == "report.pdf"
    assert response.sources[0].page_number in (1, 2)


def test_rag_pipeline_delete_removes_vectors_for_document(document_service, fake_vector_store, sample_pdf_bytes):
    document_id = _index_sample_document(document_service, sample_pdf_bytes)
    assert any(p["document_id"] == document_id for p in fake_vector_store.points)

    document_service.delete_document(document_id)

    assert not any(p["document_id"] == document_id for p in fake_vector_store.points)
