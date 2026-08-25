"""FastAPI dependency providers.

Centralizing these makes it trivial for tests to override any single
collaborator (e.g. swap the real S3Client for a mock) via
`app.dependency_overrides[...]` without touching route code.
"""
from __future__ import annotations

from fastapi import Depends

from app.config import Settings, get_settings
from app.document_registry import DocumentRegistry, get_document_registry
from app.document_service import DocumentService
from app.embeddings import EmbeddingClient, get_embedding_client
from app.llm_client import LLMClient, get_llm_client
from app.rag_pipeline import RAGPipeline
from app.s3_client import S3Client, get_s3_client
from app.vector_store import VectorStore, get_vector_store


def get_s3_dep(settings: Settings = Depends(get_settings)) -> S3Client:
    return get_s3_client(settings)


def get_embedding_dep(settings: Settings = Depends(get_settings)) -> EmbeddingClient:
    return get_embedding_client(settings)


def get_vector_store_dep(settings: Settings = Depends(get_settings)) -> VectorStore:
    return get_vector_store(settings)


def get_llm_dep(settings: Settings = Depends(get_settings)) -> LLMClient:
    return get_llm_client(settings)


def get_registry_dep(settings: Settings = Depends(get_settings)) -> DocumentRegistry:
    return get_document_registry(settings.DOCUMENTS_REGISTRY_PATH)


def get_document_service(
    settings: Settings = Depends(get_settings),
    s3_client: S3Client = Depends(get_s3_dep),
    embedding_client: EmbeddingClient = Depends(get_embedding_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
    registry: DocumentRegistry = Depends(get_registry_dep),
) -> DocumentService:
    return DocumentService(
        settings=settings,
        s3_client=s3_client,
        embedding_client=embedding_client,
        vector_store=vector_store,
        registry=registry,
    )


def get_rag_pipeline(
    settings: Settings = Depends(get_settings),
    embedding_client: EmbeddingClient = Depends(get_embedding_dep),
    vector_store: VectorStore = Depends(get_vector_store_dep),
    llm_client: LLMClient = Depends(get_llm_dep),
) -> RAGPipeline:
    return RAGPipeline(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
        llm_client=llm_client,
    )
