"""Shared pytest fixtures.

Nothing in this test suite talks to real AWS, Qdrant, or an LLM API - every
external collaborator (S3Client, VectorStore, EmbeddingClient, LLMClient) is
replaced with a small in-memory fake that implements the same interface. The
DocumentRegistry is real (it's just a local JSON file), pointed at a pytest
tmp_path so tests never touch the developer's real data/ directory.
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.config import Settings
from app.dependencies import (
    get_document_service,
    get_embedding_dep,
    get_llm_dep,
    get_rag_pipeline,
    get_registry_dep,
    get_s3_dep,
    get_vector_store_dep,
)
from app.document_registry import DocumentRegistry
from app.document_service import DocumentService
from app.main import app
from app.rag_pipeline import RAGPipeline

EMBEDDING_DIM = 32


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeS3Client:
    """In-memory stand-in for S3Client."""

    def __init__(self):
        self.store: Dict[str, bytes] = {}

    def object_key(self, document_id: str, filename: str) -> str:
        return f"documents/{document_id}/{filename}"

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        self.store[key] = data

    def get_bytes(self, key: str) -> bytes:
        return self.store[key]

    def delete_object(self, key: str) -> None:
        self.store.pop(key, None)

    def object_exists(self, key: str) -> bool:
        return key in self.store


def _fake_vector(text: str) -> List[float]:
    """Deterministic pseudo-embedding: hash the text into a fixed-size vector
    so that similar/identical text yields high cosine similarity, without
    calling any real embedding API."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:EMBEDDING_DIM]]


class FakeEmbeddingClient:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [_fake_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return _fake_vector(text)


class FakeVectorStore:
    """In-memory vector store using cosine similarity over the fake embeddings."""

    def __init__(self):
        self.points: List[dict] = []

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def upsert_chunks(self, chunks, vectors) -> None:
        for chunk, vector in zip(chunks, vectors):
            self.points.append({**chunk, "vector": vector})

    def search(self, query_vector, top_k: int = 5, document_ids: Optional[List[str]] = None):
        candidates = self.points
        if document_ids:
            candidates = [p for p in candidates if p["document_id"] in document_ids]
        scored = [
            {
                "chunk_id": p["chunk_id"],
                "document_id": p["document_id"],
                "document_name": p["document_name"],
                "page_number": p["page_number"],
                "upload_date": p["upload_date"],
                "text": p["text"],
                "score": self._cosine(query_vector, p["vector"]),
            }
            for p in candidates
        ]
        scored.sort(key=lambda m: m["score"], reverse=True)
        return scored[:top_k]

    def delete_by_document_id(self, document_id: str) -> None:
        self.points = [p for p in self.points if p["document_id"] != document_id]


class FakeLLMClient:
    """Canned LLM: just echoes back that it received context, so tests can
    assert on RAG plumbing without depending on real model output."""

    def generate_answer(self, question: str, context_blocks: List[str]) -> str:
        return f"[fake answer based on {len(context_blocks)} context block(s)] {question}"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def test_settings(tmp_path) -> Settings:
    return Settings(
        AWS_ACCESS_KEY_ID="test",
        AWS_SECRET_ACCESS_KEY="test",
        AWS_REGION="us-east-1",
        S3_BUCKET_NAME="test-bucket",
        QDRANT_URL="http://localhost:6333",
        QDRANT_API_KEY="test",
        OPENAI_API_KEY="test",
        EMBEDDING_DIMENSIONS=EMBEDDING_DIM,
        DOCUMENTS_REGISTRY_PATH=str(tmp_path / "documents_registry.json"),
        MAX_FILE_SIZE_MB=1,
        SIMILARITY_SCORE_THRESHOLD=0.0,  # deterministic fake vectors don't need tuning in tests
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=20,
    )


@pytest.fixture()
def fake_s3() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture()
def fake_embeddings() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture()
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture()
def registry(test_settings) -> DocumentRegistry:
    return DocumentRegistry(test_settings.DOCUMENTS_REGISTRY_PATH)


@pytest.fixture()
def document_service(test_settings, fake_s3, fake_embeddings, fake_vector_store, registry) -> DocumentService:
    return DocumentService(
        settings=test_settings,
        s3_client=fake_s3,
        embedding_client=fake_embeddings,
        vector_store=fake_vector_store,
        registry=registry,
    )


@pytest.fixture()
def rag_pipeline(test_settings, fake_embeddings, fake_vector_store, fake_llm) -> RAGPipeline:
    return RAGPipeline(
        settings=test_settings,
        embedding_client=fake_embeddings,
        vector_store=fake_vector_store,
        llm_client=fake_llm,
    )


@pytest.fixture()
def client(test_settings, document_service, rag_pipeline, fake_s3, fake_embeddings, fake_vector_store, fake_llm, registry):
    """A TestClient with every external dependency swapped for an in-memory fake."""

    app.dependency_overrides[get_document_service] = lambda: document_service
    app.dependency_overrides[get_rag_pipeline] = lambda: rag_pipeline
    app.dependency_overrides[get_s3_dep] = lambda: fake_s3
    app.dependency_overrides[get_embedding_dep] = lambda: fake_embeddings
    app.dependency_overrides[get_vector_store_dep] = lambda: fake_vector_store
    app.dependency_overrides[get_llm_dep] = lambda: fake_llm
    app.dependency_overrides[get_registry_dep] = lambda: registry

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _make_pdf_bytes(pages_text: List[str]) -> bytes:
    """Build a small, real, multi-page PDF in memory using reportlab, so
    extraction/chunking/embedding are exercised against a genuine PDF file
    rather than a hand-rolled fake."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages_text:
        pdf.setFont("Helvetica", 12)
        y = 800
        for line in text.split("\n"):
            pdf.drawString(72, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    return _make_pdf_bytes(
        [
            "Cloud AI Knowledge Assistant Test Document\n"
            "This project uses Retrieval-Augmented Generation.\n"
            "Page one covers the system overview and goals.",
            "Page two covers the architecture in more detail.\n"
            "It uses FastAPI, AWS S3, Qdrant Cloud, and an LLM.\n"
            "Sources are always cited with document name and page number.",
        ]
    )


@pytest.fixture()
def empty_pdf_bytes() -> bytes:
    """A structurally valid PDF with a blank page (no extractable text)."""
    return _make_pdf_bytes([""])
