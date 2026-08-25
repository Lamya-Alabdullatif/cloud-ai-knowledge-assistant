"""PDF text extraction, page by page.

Keeping extraction page-aware (rather than concatenating the whole document
into one blob) is what lets the rest of the pipeline attach an accurate
`page_number` to every chunk, which is required to show sources like
"invoice.pdf, page 3" back to the user.
"""
from __future__ import annotations

import io
import logging
from typing import List, TypedDict

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.exceptions import DocumentProcessingError

logger = logging.getLogger("app")


class PageText(TypedDict):
    page_number: int
    text: str


def extract_text_by_page(file_bytes: bytes) -> List[PageText]:
    """Extract text from every page of a PDF.

    Returns a list of {"page_number": 1-indexed int, "text": str}. Pages with
    no extractable text (e.g. scanned images) are still included with an
    empty string so page numbering stays accurate; empty pages are filtered
    out downstream at chunking time.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except PdfReadError as exc:
        raise DocumentProcessingError(f"The file is not a valid or readable PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise DocumentProcessingError(f"Could not open PDF file: {exc}") from exc

    if reader.is_encrypted:
        try:
            # Some PDFs are "encrypted" with an empty owner password and can
            # still be read; attempt that before giving up.
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise DocumentProcessingError(
                "This PDF is password-protected and cannot be processed."
            ) from exc

    pages: List[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract text from page %d: %s", index, exc)
            text = ""
        pages.append({"page_number": index, "text": text.strip()})

    if not pages:
        raise DocumentProcessingError("The PDF has no pages.")

    non_empty_pages = [p for p in pages if p["text"]]
    if not non_empty_pages:
        raise DocumentProcessingError(
            "No extractable text was found in this PDF. Scanned/image-only PDFs "
            "are not supported yet (see README -> Future Improvements)."
        )

    return pages
