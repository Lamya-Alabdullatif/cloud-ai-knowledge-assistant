"""Qdrant Cloud wrapper: collection management, upsert, similarity search, delete.

Every point stored carries a payload with the metadata needed to show a
grounded source back to the user: document_id, document_name, page_number,
chunk_id, upload_date, plus the chunk text itself (so retrieval doesn't need
a second round-trip to fetch text for the LLM context).
"""
from __future__ import annotations

import logging
from typing import List, Optional, TypedDict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.chunking import ChunkRecord
from app.config import Settings
from app.exceptions import VectorStoreError

logger = logging.getLogger("app")


class SearchMatch(TypedDict):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    upload_date: str
    text: str
    score: float


class VectorStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._collection = settings.QDRANT_COLLECTION_NAME
        self._client = QdrantClient(
            url=settings.QDRANT_URL or None,
            api_key=settings.QDRANT_API_KEY or None,
        )
        self._ensure_collection(settings.EMBEDDING_DIMENSIONS)

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if self._collection not in existing:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size, distance=qmodels.Distance.COSINE
                    ),
                )
                logger.info("Created Qdrant collection '%s'", self._collection)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not ensure Qdrant collection exists: %s", exc)
            raise VectorStoreError(f"Failed to initialize vector store collection: {exc}") from exc

    def upsert_chunks(self, chunks: List[ChunkRecord], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunk/vector count mismatch during upsert.")
        if not chunks:
            return
        points = [
            qmodels.PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload={
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "upload_date": chunk["upload_date"],
                    "text": chunk["text"],
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            self._client.upsert(collection_name=self._collection, points=points)
            logger.info("Upserted %d chunks into Qdrant collection '%s'", len(points), self._collection)
        except UnexpectedResponse as exc:
            logger.error("Qdrant upsert failed: %s", exc)
            raise VectorStoreError(f"Failed to store document vectors: {exc}") from exc

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchMatch]:
        query_filter = None
        if document_ids:
            query_filter = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="document_id", match=qmodels.MatchValue(value=doc_id)
                    )
                    for doc_id in document_ids
                ]
            )
        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            logger.error("Qdrant search failed: %s", exc)
            raise VectorStoreError(f"Failed to search vector store: {exc}") from exc

        matches: List[SearchMatch] = []
        for point in results:
            payload = point.payload or {}
            matches.append(
                {
                    "chunk_id": str(point.id),
                    "document_id": payload.get("document_id", ""),
                    "document_name": payload.get("document_name", ""),
                    "page_number": payload.get("page_number", 0),
                    "upload_date": payload.get("upload_date", ""),
                    "text": payload.get("text", ""),
                    "score": float(point.score),
                }
            )
        return matches

    def delete_by_document_id(self, document_id: str) -> None:
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )
            logger.info("Deleted vectors for document_id=%s from Qdrant", document_id)
        except UnexpectedResponse as exc:
            logger.error("Qdrant delete failed for document_id=%s: %s", document_id, exc)
            raise VectorStoreError(f"Failed to delete document vectors: {exc}") from exc


_vector_store_singleton: Optional[VectorStore] = None


def get_vector_store(settings: Settings) -> VectorStore:
    global _vector_store_singleton
    if _vector_store_singleton is None:
        _vector_store_singleton = VectorStore(settings)
    return _vector_store_singleton
