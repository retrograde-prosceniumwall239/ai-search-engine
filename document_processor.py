"""
document_processor.py
======================
Extracts plain text from uploaded documents.

Supported formats
------------------
  * .pdf  -- extracted page-by-page using pypdf
  * .txt  -- read directly as UTF-8 text
  * .md   -- read directly as UTF-8 text (Markdown syntax is left intact;
             the LLM and embedding model both handle Markdown fine)

Adding a new format only requires adding a new branch to
``extract_text`` and registering the extension in ``SUPPORTED_EXTENSIONS``.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension is not supported for text extraction."""


class DocumentExtractionError(Exception):
    """Raised when text extraction fails for a supported file type."""


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def extract_text(file_path: str, filename: str) -> str:
    """Extract plain text from a file on disk.

    Args:
        file_path: Path to the file on disk.
        filename: Original filename (used to determine the extension).

    Returns:
        The extracted text content.

    Raises:
        UnsupportedFileTypeError: If the extension is not supported.
        DocumentExtractionError: If extraction fails for any reason.
    """
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if extension == ".pdf":
            return _extract_pdf(file_path)
        # .txt, .md, .markdown are all plain-text formats.
        return _extract_plain_text(file_path)
    except (UnsupportedFileTypeError, DocumentExtractionError):
        raise
    except Exception as exc:  # noqa: BLE001 - convert any failure into our own error type
        logger.exception("Failed to extract text from %s", filename)
        raise DocumentExtractionError(f"Failed to extract text from '{filename}': {exc}") from exc


def _extract_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)
        else:
            logger.debug("Page %d of %s produced no extractable text", page_number, file_path)

    full_text = "\n\n".join(pages_text)
    if not full_text.strip():
        raise DocumentExtractionError(
            "No extractable text found in PDF (it may be a scanned/image-only PDF)."
        )
    return full_text


def _extract_plain_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
