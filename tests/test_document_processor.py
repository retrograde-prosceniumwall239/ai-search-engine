"""Unit tests for document_processor.py"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import document_processor as dp


def test_is_supported():
    assert dp.is_supported("notes.txt") is True
    assert dp.is_supported("report.pdf") is True
    assert dp.is_supported("readme.md") is True
    assert dp.is_supported("archive.zip") is False


def test_extract_text_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello, world!", encoding="utf-8")

    text = dp.extract_text(str(file_path), "sample.txt")
    assert text == "Hello, world!"


def test_extract_text_markdown(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Heading\n\nSome body text.", encoding="utf-8")

    text = dp.extract_text(str(file_path), "sample.md")
    assert "Heading" in text
    assert "Some body text." in text


def test_extract_text_unsupported_extension(tmp_path):
    file_path = tmp_path / "sample.zip"
    file_path.write_bytes(b"not a real zip")

    with pytest.raises(dp.UnsupportedFileTypeError):
        dp.extract_text(str(file_path), "sample.zip")
