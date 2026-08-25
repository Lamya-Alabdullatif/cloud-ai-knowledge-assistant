"""Orchestrates the end-to-end document lifecycle:

upload -> validate -> store in S3 -> extract text -> chunk -> embed ->
store in Qdrant -> update registry status; plus listing and deletion
(which cleans up S3, Qdrant, and the registry together).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List

from app.chunking import chunk_pages
from app.config import Settings
from app.document_registry import DocumentRegistry
from app.embeddings import EmbeddingClient
from app.exceptions import (
    DocumentProcessingError,
    EmptyFileError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from app.pdf_processor import extract_text_by_page
from app.s3_client import S3Client
from app.schemas import DocumentMetadata, DocumentStatus
from app.vector_store import VectorStore

logger = logging.getLogger("app")


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        s3_client: S3Client,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        registry: DocumentRegistry,
    ):
        self._settings = settings
        self._s3 = s3_client
        self._embeddings = embedding_client
        self._vector_store = vector_store
        self._registry = registry

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate_file(self, filename: str, file_bytes: bytes) -> None:
        if not filename or "." not in filename:
            raise InvalidFileTypeError("The uploaded file must have a valid name and extension.")

        extension = "." + filename.rsplit(".", 1)[-1].lower()
        if extension not in self._settings.allowed_extensions_list:
            raise InvalidFileTypeError(
                f"File type '{extension}' is not supported. "
                f"Allowed types: {', '.join(self._settings.allowed_extensions_list)}."
            )

        if len(file_bytes) == 0:
            raise EmptyFileError("The uploaded file is empty.")

        if len(file_bytes) > self._settings.max_file_size_bytes:
            raise FileTooLargeError(
                f"File exceeds the maximum allowed size of {self._settings.MAX_FILE_SIZE_MB} MB."
            )

    # ------------------------------------------------------------------ #
    # Upload + processing pipeline
    # ------------------------------------------------------------------ #
    def upload_and_process(self, filename: str, file_bytes: bytes) -> DocumentMetadata:
        self.validate_file(filename, file_bytes)

        document_id = str(uuid.uuid4())
        s3_key = self._s3.object_key(document_id, filename)
        upload_date = datetime.now(timezone.utc)

        # 1. Persist the original file in S3 first, so it is never lost even
        #    if downstream processing (extraction/embeddings) fails.
        self._s3.upload_bytes(s3_key, file_bytes)

        record = self._registry.create(
            document_id=document_id,
            document_name=filename,
            file_size_bytes=len(file_bytes),
            s3_key=s3_key,
        )

        try:
            self._registry.update_status(document_id, DocumentStatus.PROCESSING)

            pages = extract_text_by_page(file_bytes)
            chunks = chunk_pages(
                pages=pages,
                document_id=document_id,
                document_name=filename,
                upload_date=upload_date,
                chunk_size=self._settings.CHUNK_SIZE,
                chunk_overlap=self._settings.CHUNK_OVERLAP,
            )

            if not chunks:
                raise DocumentProcessingError("No text could be chunked from this document.")

            vectors = self._embeddings.embed_texts([c["text"] for c in chunks])
            self._vector_store.upsert_chunks(chunks, vectors)

            record = self._registry.update_status(
                document_id,
                DocumentStatus.PROCESSED,
                page_count=len(pages),
                chunk_count=len(chunks),
            )
            logger.info(
                "Document %s processed successfully: %d pages, %d chunks",
                document_id,
                len(pages),
                len(chunks),
            )
            return record

        except Exception as exc:  # noqa: BLE001
            logger.error("Processing failed for document %s: %s", document_id, exc)
            self._registry.update_status(
                document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
            raise

    # ------------------------------------------------------------------ #
    # Listing / retrieval
    # ------------------------------------------------------------------ #
    def list_documents(self) -> List[DocumentMetadata]:
        return self._registry.list_all()

    def get_document(self, document_id: str) -> DocumentMetadata:
        return self._registry.get(document_id)

    # ------------------------------------------------------------------ #
    # Deletion (S3 + Qdrant + registry, best-effort but logged)
    # ------------------------------------------------------------------ #
    def delete_document(self, document_id: str) -> None:
        s3_key = self._registry.get_s3_key(document_id)

        self._vector_store.delete_by_document_id(document_id)
        self._s3.delete_object(s3_key)
        self._registry.delete(document_id)

        logger.info("Document %s fully deleted (S3 + Qdrant + registry)", document_id)
