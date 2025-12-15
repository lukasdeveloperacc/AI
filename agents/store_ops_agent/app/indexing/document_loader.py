"""Document loaders for various file formats.

This module provides loaders for:
- PDF files
- Markdown files
- JSON files
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass
class LoadedDocument:
    """Represents a loaded document with its content and metadata.

    Attributes:
        doc_id: Unique document identifier.
        title: Document title.
        content: Extracted text content.
        source_path: Path to the source file.
        category: Document category (e.g., "refund", "promo").
        store_type: Store type (e.g., "cafe", "convenience").
        version: Document version.
        valid_from: Start date of validity.
        valid_to: End date of validity.
        language: Document language code.
    """

    doc_id: str
    title: str
    content: str
    source_path: str
    category: Optional[str] = None
    store_type: Optional[str] = None
    version: Optional[str] = None
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    language: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source_path": self.source_path,
            "category": self.category,
            "store_type": self.store_type,
            "version": self.version,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "language": self.language,
        }


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> LoadedDocument:
        """Load a document from file.

        Args:
            file_path: Path to the document file.

        Returns:
            LoadedDocument with extracted content and metadata.
        """
        pass

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        """Check if this loader supports the given file type.

        Args:
            file_path: Path to the file.

        Returns:
            True if this loader can handle the file.
        """
        pass

    @staticmethod
    def _generate_doc_id(file_path: Path) -> str:
        """Generate a document ID from file path.

        Args:
            file_path: Path to the document file.

        Returns:
            Unique document identifier.
        """
        # Use stem (filename without extension) and parent folder
        return f"doc_{file_path.stem}"

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[date]:
        """Parse date string to date object.

        Args:
            date_str: Date string in YYYY-MM-DD format.

        Returns:
            date object or None if parsing fails.
        """
        if not date_str:
            return None
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            return None


class PDFLoader(BaseDocumentLoader):
    """Loader for PDF documents."""

    def supports(self, file_path: Path) -> bool:
        """Check if file is a PDF."""
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> LoadedDocument:
        """Load a PDF document.

        Args:
            file_path: Path to the PDF file.

        Returns:
            LoadedDocument with extracted text.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file cannot be parsed.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            text_parts = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            content = "\n\n".join(text_parts)

            # Extract title from PDF metadata or filename
            title = file_path.stem
            if reader.metadata and reader.metadata.title:
                title = reader.metadata.title

            return LoadedDocument(
                doc_id=self._generate_doc_id(file_path),
                title=title,
                content=content,
                source_path=str(file_path),
            )
        except Exception as e:
            raise ValueError(f"Failed to parse PDF file: {e}")


class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown documents."""

    def supports(self, file_path: Path) -> bool:
        """Check if file is a Markdown file."""
        return file_path.suffix.lower() in (".md", ".markdown")

    def load(self, file_path: Path) -> LoadedDocument:
        """Load a Markdown document.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            LoadedDocument with content.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")

        # Extract title from first H1 heading or filename
        title = file_path.stem
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return LoadedDocument(
            doc_id=self._generate_doc_id(file_path),
            title=title,
            content=content,
            source_path=str(file_path),
        )


class JSONLoader(BaseDocumentLoader):
    """Loader for JSON documents with metadata.

    Expected JSON structure:
    {
        "doc_id": "optional_id",
        "title": "Document Title",
        "content": "Document content...",
        "category": "refund",
        "store_type": "cafe",
        "version": "1.0",
        "valid_from": "2024-01-01",
        "valid_to": "2024-12-31",
        "language": "ko"
    }
    """

    def supports(self, file_path: Path) -> bool:
        """Check if file is a JSON file."""
        return file_path.suffix.lower() == ".json"

    def load(self, file_path: Path) -> LoadedDocument:
        """Load a JSON document with metadata.

        Args:
            file_path: Path to the JSON file.

        Returns:
            LoadedDocument with content and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If JSON is invalid or missing required fields.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        # Validate required fields
        if "content" not in data:
            raise ValueError("JSON document must contain 'content' field")

        return LoadedDocument(
            doc_id=data.get("doc_id", self._generate_doc_id(file_path)),
            title=data.get("title", file_path.stem),
            content=data["content"],
            source_path=str(file_path),
            category=data.get("category"),
            store_type=data.get("store_type"),
            version=data.get("version"),
            valid_from=self._parse_date(data.get("valid_from")),
            valid_to=self._parse_date(data.get("valid_to")),
            language=data.get("language"),
        )


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders."""

    _loaders: list[BaseDocumentLoader] = [
        PDFLoader(),
        MarkdownLoader(),
        JSONLoader(),
    ]

    @classmethod
    def get_loader(cls, file_path: Path) -> Optional[BaseDocumentLoader]:
        """Get appropriate loader for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Appropriate loader or None if not supported.
        """
        for loader in cls._loaders:
            if loader.supports(file_path):
                return loader
        return None

    @classmethod
    def load(cls, file_path: Path) -> LoadedDocument:
        """Load a document using appropriate loader.

        Args:
            file_path: Path to the document file.

        Returns:
            LoadedDocument with content and metadata.

        Raises:
            ValueError: If file type is not supported.
        """
        loader = cls.get_loader(file_path)
        if loader is None:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        return loader.load(file_path)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported extensions (e.g., [".pdf", ".md", ".json"]).
        """
        return [".pdf", ".md", ".markdown", ".json"]
