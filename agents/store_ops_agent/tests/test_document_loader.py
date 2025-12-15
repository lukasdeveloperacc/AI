"""Tests for document loaders."""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from app.indexing.document_loader import (
    DocumentLoaderFactory,
    JSONLoader,
    LoadedDocument,
    MarkdownLoader,
    PDFLoader,
)


class TestLoadedDocument:
    """Tests for LoadedDocument dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        doc = LoadedDocument(
            doc_id="doc_001",
            title="Test Document",
            content="Test content",
            source_path="/path/to/file.json",
            category="refund",
            store_type="cafe",
            version="1.0",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            language="ko",
        )

        result = doc.to_dict()

        assert result["doc_id"] == "doc_001"
        assert result["title"] == "Test Document"
        assert result["content"] == "Test content"
        assert result["source_path"] == "/path/to/file.json"
        assert result["category"] == "refund"
        assert result["store_type"] == "cafe"
        assert result["version"] == "1.0"
        assert result["valid_from"] == "2024-01-01"
        assert result["valid_to"] == "2024-12-31"
        assert result["language"] == "ko"

    def test_to_dict_with_none_values(self):
        """Test conversion with None values."""
        doc = LoadedDocument(
            doc_id="doc_001",
            title="Test",
            content="Content",
            source_path="/path",
        )

        result = doc.to_dict()

        assert result["valid_from"] is None
        assert result["valid_to"] is None
        assert result["category"] is None


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_supports_md_extension(self):
        """Test that .md files are supported."""
        loader = MarkdownLoader()
        assert loader.supports(Path("test.md"))
        assert loader.supports(Path("test.markdown"))
        assert not loader.supports(Path("test.txt"))
        assert not loader.supports(Path("test.pdf"))

    def test_load_markdown_file(self):
        """Test loading a Markdown file."""
        loader = MarkdownLoader()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Test Document\n\nThis is test content.\n\n## Section 1\n\nMore content here.")
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)

            assert doc.doc_id == f"doc_{temp_path.stem}"
            assert doc.title == "Test Document"
            assert "This is test content" in doc.content
            assert doc.source_path == str(temp_path)
        finally:
            temp_path.unlink()

    def test_load_markdown_without_h1(self):
        """Test loading Markdown without H1 heading."""
        loader = MarkdownLoader()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("This is content without a heading.\n\nJust plain text.")
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)
            # Should use filename as title
            assert doc.title == temp_path.stem
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        loader = MarkdownLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.md"))


class TestJSONLoader:
    """Tests for JSONLoader."""

    def test_supports_json_extension(self):
        """Test that .json files are supported."""
        loader = JSONLoader()
        assert loader.supports(Path("test.json"))
        assert not loader.supports(Path("test.md"))
        assert loader.supports(Path("test.JSON"))  # Case insensitive

    def test_load_json_file_with_metadata(self):
        """Test loading JSON file with full metadata."""
        loader = JSONLoader()

        data = {
            "doc_id": "custom_id",
            "title": "Policy Document",
            "content": "This is the policy content.",
            "category": "refund",
            "store_type": "cafe",
            "version": "2.0",
            "valid_from": "2024-01-01",
            "valid_to": "2024-12-31",
            "language": "ko",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)

            assert doc.doc_id == "custom_id"
            assert doc.title == "Policy Document"
            assert doc.content == "This is the policy content."
            assert doc.category == "refund"
            assert doc.store_type == "cafe"
            assert doc.version == "2.0"
            assert doc.valid_from == date(2024, 1, 1)
            assert doc.valid_to == date(2024, 12, 31)
            assert doc.language == "ko"
        finally:
            temp_path.unlink()

    def test_load_json_minimal(self):
        """Test loading JSON with only required content field."""
        loader = JSONLoader()

        data = {"content": "Minimal content"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            doc = loader.load(temp_path)

            assert doc.content == "Minimal content"
            assert doc.doc_id == f"doc_{temp_path.stem}"
            assert doc.title == temp_path.stem
            assert doc.category is None
        finally:
            temp_path.unlink()

    def test_load_json_missing_content(self):
        """Test loading JSON without content field."""
        loader = JSONLoader()

        data = {"title": "No content"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="must contain 'content' field"):
                loader.load(temp_path)
        finally:
            temp_path.unlink()

    def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        loader = JSONLoader()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                loader.load(temp_path)
        finally:
            temp_path.unlink()


class TestPDFLoader:
    """Tests for PDFLoader."""

    def test_supports_pdf_extension(self):
        """Test that .pdf files are supported."""
        loader = PDFLoader()
        assert loader.supports(Path("test.pdf"))
        assert loader.supports(Path("test.PDF"))
        assert not loader.supports(Path("test.txt"))

    def test_load_nonexistent_pdf(self):
        """Test loading nonexistent PDF."""
        loader = PDFLoader()

        with pytest.raises(FileNotFoundError):
            loader.load(Path("/nonexistent/file.pdf"))


class TestDocumentLoaderFactory:
    """Tests for DocumentLoaderFactory."""

    def test_get_loader_for_md(self):
        """Test getting loader for Markdown."""
        loader = DocumentLoaderFactory.get_loader(Path("test.md"))
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_for_json(self):
        """Test getting loader for JSON."""
        loader = DocumentLoaderFactory.get_loader(Path("test.json"))
        assert isinstance(loader, JSONLoader)

    def test_get_loader_for_pdf(self):
        """Test getting loader for PDF."""
        loader = DocumentLoaderFactory.get_loader(Path("test.pdf"))
        assert isinstance(loader, PDFLoader)

    def test_get_loader_for_unsupported(self):
        """Test getting loader for unsupported file."""
        loader = DocumentLoaderFactory.get_loader(Path("test.txt"))
        assert loader is None

    def test_load_unsupported_file(self):
        """Test loading unsupported file type."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            DocumentLoaderFactory.load(Path("test.txt"))

    def test_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = DocumentLoaderFactory.supported_extensions()

        assert ".pdf" in extensions
        assert ".md" in extensions
        assert ".markdown" in extensions
        assert ".json" in extensions
