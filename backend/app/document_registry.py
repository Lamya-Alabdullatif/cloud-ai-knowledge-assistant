"""Lightweight, thread-safe document metadata registry.

The project intentionally has no relational database - document status
(uploaded / processing / processed / failed), page counts, and chunk counts
are small enough to track in a local JSON file. This keeps the project
simple to run (no extra infra to stand up) while still surviving process
restarts, and it is trivially swappable for DynamoDB/Postgres later (see
README -> Future Improvements) since it's isolated behind this one class.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.exceptions import DocumentNotFoundError
from app.schemas import DocumentMetadata, DocumentStatus

logger = logging.getLogger("app")


class DocumentRegistry:
    def __init__(self, storage_path: str):
        self._path = Path(storage_path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({})

    # -- persistence helpers ------------------------------------------------
    def _read(self) -> Dict[str, dict]:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: Dict[str, dict]) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp_path, self._path)

    # -- public API -----------------------------------------------------------
    def create(
        self,
        document_id: str,
        document_name: str,
        file_size_bytes: int,
        s3_key: str,
    ) -> DocumentMetadata:
        with self._lock:
            data = self._read()
            record = {
                "document_id": document_id,
                "document_name": document_name,
                "status": DocumentStatus.UPLOADED.value,
                "upload_date": datetime.now(timezone.utc).isoformat(),
                "page_count": None,
                "chunk_count": None,
                "file_size_bytes": file_size_bytes,
                "s3_key": s3_key,
                "error_message": None,
            }
            data[document_id] = record
            self._write(data)
            logger.info("Registered new document %s (%s)", document_id, document_name)
            return self._to_model(record)

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        page_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> DocumentMetadata:
        with self._lock:
            data = self._read()
            if document_id not in data:
                raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
            record = data[document_id]
            record["status"] = status.value
            if page_count is not None:
                record["page_count"] = page_count
            if chunk_count is not None:
                record["chunk_count"] = chunk_count
            record["error_message"] = error_message
            data[document_id] = record
            self._write(data)
            return self._to_model(record)

    def get(self, document_id: str) -> DocumentMetadata:
        data = self._read()
        record = data.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        return self._to_model(record)

    def get_s3_key(self, document_id: str) -> str:
        data = self._read()
        record = data.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        return record["s3_key"]

    def list_all(self) -> List[DocumentMetadata]:
        data = self._read()
        records = sorted(data.values(), key=lambda r: r["upload_date"], reverse=True)
        return [self._to_model(r) for r in records]

    def delete(self, document_id: str) -> None:
        with self._lock:
            data = self._read()
            if document_id not in data:
                raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
            del data[document_id]
            self._write(data)
            logger.info("Removed document %s from registry", document_id)

    def exists(self, document_id: str) -> bool:
        return document_id in self._read()

    @staticmethod
    def _to_model(record: dict) -> DocumentMetadata:
        return DocumentMetadata(
            document_id=record["document_id"],
            document_name=record["document_name"],
            status=DocumentStatus(record["status"]),
            upload_date=record["upload_date"],
            page_count=record.get("page_count"),
            chunk_count=record.get("chunk_count"),
            file_size_bytes=record.get("file_size_bytes"),
            error_message=record.get("error_message"),
        )


_registry_singleton: Optional[DocumentRegistry] = None


def get_document_registry(storage_path: str) -> DocumentRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = DocumentRegistry(storage_path)
    return _registry_singleton
