"""Split extracted page text into overlapping chunks, preserving metadata.

Each chunk keeps a back-reference to the page it came from so the RAG
pipeline can cite an exact page number, plus a stable chunk_id so the same
chunk's vector, payload, and (if ever needed) raw text can all be
cross-referenced.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.pdf_processor import PageText


class ChunkRecord(TypedDict):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    upload_date: str
    text: str


def chunk_pages(
    pages: List[PageText],
    document_id: str,
    document_name: str,
    upload_date: datetime,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[ChunkRecord]:
    """Turn per-page text into overlapping, metadata-tagged chunks.

    Splitting is done independently per page (rather than on the whole
    document joined together) precisely so every resulting chunk can be
    attributed to a single, correct page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "؟", "!", " ", ""],
    )

    upload_date_iso = upload_date.isoformat()
    chunks: List[ChunkRecord] = []

    for page in pages:
        if not page["text"]:
            continue
        for piece in splitter.split_text(page["text"]):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "document_name": document_name,
                    "page_number": page["page_number"],
                    "upload_date": upload_date_iso,
                    "text": piece,
                }
            )

    return chunks
