"""Document store for metadata management.

This module provides:
- DocStore: JSON-based document metadata storage
- Document CRUD operations
- Version tracking
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Document metadata record for DocStore.

    Attributes:
        doc_id: Unique document identifier.
        title: Document title.
        category: Document category (e.g., "refund", "promo").
        store_type: Store type (e.g., "cafe", "convenience").
        version: Document version.
        valid_from: Start date of validity.
        valid_to: End date of validity.
        language: Document language code.
        source_path: Original file path.
        chunk_count: Number of chunks created.
        indexed_at: Timestamp when document was indexed.
        updated_at: Timestamp of last update.
    """

    doc_id: str
    title: str
    category: Optional[str] = None
    store_type: Optional[str] = None
    version: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    language: Optional[str] = None
    source_path: Optional[str] = None
    chunk_count: int = 0
    indexed_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRecord":
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            DocumentRecord instance.
        """
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            category=data.get("category"),
            store_type=data.get("store_type"),
            version=data.get("version"),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            language=data.get("language"),
            source_path=data.get("source_path"),
            chunk_count=data.get("chunk_count", 0),
            indexed_at=data.get("indexed_at"),
            updated_at=data.get("updated_at"),
        )


class DocStore:
    """JSON-based document metadata store.

    Stores document-level metadata separately from the vector store.
    Supports version tracking and efficient document lookup.
    """

    def __init__(self, store_path: Optional[Path] = None) -> None:
        """Initialize the DocStore.

        Args:
            store_path: Path to the JSON store file. If None, operates in memory.
        """
        self.store_path = Path(store_path) if store_path else None
        self._documents: dict[str, DocumentRecord] = {}
        self._version_history: dict[str, list[str]] = {}  # doc_id -> list of versions

        if self.store_path and self.store_path.exists():
            self._load()

    def _load(self) -> None:
        """Load store from disk."""
        if not self.store_path or not self.store_path.exists():
            return

        try:
            with open(self.store_path, encoding="utf-8") as f:
                data = json.load(f)

            self._documents = {
                doc_id: DocumentRecord.from_dict(doc_data)
                for doc_id, doc_data in data.get("documents", {}).items()
            }
            self._version_history = data.get("version_history", {})

            logger.info(f"Loaded {len(self._documents)} documents from {self.store_path}")

        except Exception as e:
            logger.error(f"Failed to load DocStore: {e}")
            raise

    def _save(self) -> None:
        """Save store to disk."""
        if not self.store_path:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": {
                doc_id: record.to_dict()
                for doc_id, record in self._documents.items()
            },
            "version_history": self._version_history,
        }

        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved DocStore to {self.store_path}")

    def add(self, record: DocumentRecord) -> DocumentRecord:
        """Add or update a document record.

        Args:
            record: DocumentRecord to add.

        Returns:
            The added/updated record with timestamps set.
        """
        now = datetime.now().isoformat()

        # Check if this is an update
        existing = self._documents.get(record.doc_id)
        if existing:
            # Track version history
            if existing.version and existing.version != record.version:
                if record.doc_id not in self._version_history:
                    self._version_history[record.doc_id] = []
                self._version_history[record.doc_id].append(existing.version)

            record.indexed_at = existing.indexed_at
            record.updated_at = now
        else:
            record.indexed_at = now
            record.updated_at = now

        self._documents[record.doc_id] = record
        self._save()

        logger.info(f"{'Updated' if existing else 'Added'} document: {record.doc_id}")
        return record

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get a document record by ID.

        Args:
            doc_id: Document identifier.

        Returns:
            DocumentRecord or None if not found.
        """
        return self._documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Delete a document record.

        Args:
            doc_id: Document identifier.

        Returns:
            True if deleted, False if not found.
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._version_history.pop(doc_id, None)
            self._save()
            logger.info(f"Deleted document: {doc_id}")
            return True
        return False

    def list_all(self) -> list[DocumentRecord]:
        """List all document records.

        Returns:
            List of all DocumentRecords.
        """
        return list(self._documents.values())

    def find_by_category(self, category: str) -> list[DocumentRecord]:
        """Find documents by category.

        Args:
            category: Category to filter by.

        Returns:
            List of matching DocumentRecords.
        """
        return [
            record
            for record in self._documents.values()
            if record.category and record.category.lower() == category.lower()
        ]

    def find_by_store_type(self, store_type: str) -> list[DocumentRecord]:
        """Find documents by store type.

        Args:
            store_type: Store type to filter by.

        Returns:
            List of matching DocumentRecords.
        """
        return [
            record
            for record in self._documents.values()
            if record.store_type and record.store_type.lower() == store_type.lower()
        ]

    def find_valid_on_date(self, target_date: date) -> list[DocumentRecord]:
        """Find documents valid on a specific date.

        Args:
            target_date: Date to check validity against.

        Returns:
            List of documents valid on the target date.
        """
        results = []
        target_str = target_date.isoformat()

        for record in self._documents.values():
            # If no validity dates, consider always valid
            if record.valid_from is None and record.valid_to is None:
                results.append(record)
                continue

            # Check date range
            valid = True
            if record.valid_from and target_str < record.valid_from:
                valid = False
            if record.valid_to and target_str > record.valid_to:
                valid = False

            if valid:
                results.append(record)

        return results

    def get_version_history(self, doc_id: str) -> list[str]:
        """Get version history for a document.

        Args:
            doc_id: Document identifier.

        Returns:
            List of previous versions (not including current).
        """
        return self._version_history.get(doc_id, [])

    def count(self) -> int:
        """Get total number of documents.

        Returns:
            Number of documents in store.
        """
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents from store."""
        self._documents.clear()
        self._version_history.clear()
        self._save()
        logger.info("Cleared DocStore")

    def get_metadata_for_retrieval(self) -> dict[str, dict]:
        """Get metadata dict formatted for FilteredRetriever.

        Returns:
            Dict mapping doc_id to metadata for filtering.
        """
        from datetime import date as date_type

        result = {}
        for doc_id, record in self._documents.items():
            result[doc_id] = {
                "doc_id": doc_id,
                "store_type": record.store_type,
                "category": record.category,
                "language": record.language,
                "valid_from": (
                    date_type.fromisoformat(record.valid_from)
                    if record.valid_from
                    else None
                ),
                "valid_to": (
                    date_type.fromisoformat(record.valid_to)
                    if record.valid_to
                    else None
                ),
            }
        return result
