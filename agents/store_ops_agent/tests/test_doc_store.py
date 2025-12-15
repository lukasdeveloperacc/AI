"""Tests for DocStore."""

import tempfile
from datetime import date
from pathlib import Path

import pytest

from app.indexing.doc_store import DocStore, DocumentRecord


@pytest.fixture
def doc_store() -> DocStore:
    """Create an in-memory DocStore for testing."""
    return DocStore()


@pytest.fixture
def sample_record() -> DocumentRecord:
    """Create a sample document record."""
    return DocumentRecord(
        doc_id="doc_001",
        title="Test Policy",
        category="refund",
        store_type="cafe",
        version="1.0",
        valid_from="2024-01-01",
        valid_to="2024-12-31",
        language="ko",
        source_path="/test/policy.md",
        chunk_count=10,
    )


class TestDocumentRecord:
    """Tests for DocumentRecord dataclass."""

    def test_to_dict(self, sample_record):
        """Test conversion to dictionary."""
        result = sample_record.to_dict()

        assert result["doc_id"] == "doc_001"
        assert result["title"] == "Test Policy"
        assert result["category"] == "refund"
        assert result["store_type"] == "cafe"
        assert result["version"] == "1.0"
        assert result["valid_from"] == "2024-01-01"
        assert result["valid_to"] == "2024-12-31"
        assert result["language"] == "ko"
        assert result["chunk_count"] == 10

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "doc_id": "doc_002",
            "title": "Another Policy",
            "category": "promo",
            "store_type": "convenience",
            "version": "2.0",
            "valid_from": "2024-06-01",
            "chunk_count": 5,
        }

        record = DocumentRecord.from_dict(data)

        assert record.doc_id == "doc_002"
        assert record.title == "Another Policy"
        assert record.category == "promo"
        assert record.store_type == "convenience"
        assert record.version == "2.0"
        assert record.valid_from == "2024-06-01"
        assert record.valid_to is None
        assert record.chunk_count == 5


class TestDocStore:
    """Tests for DocStore."""

    def test_init_in_memory(self):
        """Test in-memory initialization."""
        store = DocStore()

        assert store.count() == 0

    def test_init_with_path(self):
        """Test initialization with file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "doc_store.json"
            store = DocStore(store_path=store_path)

            assert store.store_path == store_path
            assert store.count() == 0

    def test_add_new_record(self, doc_store, sample_record):
        """Test adding a new record."""
        result = doc_store.add(sample_record)

        assert result.doc_id == "doc_001"
        assert result.indexed_at is not None
        assert result.updated_at is not None
        assert doc_store.count() == 1

    def test_add_update_record(self, doc_store, sample_record):
        """Test updating an existing record."""
        # Add initial record
        doc_store.add(sample_record)
        initial_indexed_at = doc_store.get("doc_001").indexed_at

        # Update record
        updated = DocumentRecord(
            doc_id="doc_001",
            title="Updated Policy",
            category="refund",
            version="2.0",
        )
        result = doc_store.add(updated)

        assert result.title == "Updated Policy"
        assert result.version == "2.0"
        assert result.indexed_at == initial_indexed_at  # Preserved
        assert result.updated_at != initial_indexed_at  # Changed
        assert doc_store.count() == 1

    def test_add_tracks_version_history(self, doc_store):
        """Test that version history is tracked."""
        # Add v1
        v1 = DocumentRecord(doc_id="doc_001", title="Policy", version="1.0")
        doc_store.add(v1)

        # Update to v2
        v2 = DocumentRecord(doc_id="doc_001", title="Policy", version="2.0")
        doc_store.add(v2)

        # Update to v3
        v3 = DocumentRecord(doc_id="doc_001", title="Policy", version="3.0")
        doc_store.add(v3)

        history = doc_store.get_version_history("doc_001")

        assert "1.0" in history
        assert "2.0" in history
        assert "3.0" not in history  # Current version not in history

    def test_get_existing(self, doc_store, sample_record):
        """Test getting an existing record."""
        doc_store.add(sample_record)

        result = doc_store.get("doc_001")

        assert result is not None
        assert result.title == "Test Policy"

    def test_get_nonexistent(self, doc_store):
        """Test getting a nonexistent record."""
        result = doc_store.get("nonexistent")

        assert result is None

    def test_delete_existing(self, doc_store, sample_record):
        """Test deleting an existing record."""
        doc_store.add(sample_record)

        result = doc_store.delete("doc_001")

        assert result is True
        assert doc_store.count() == 0
        assert doc_store.get("doc_001") is None

    def test_delete_nonexistent(self, doc_store):
        """Test deleting a nonexistent record."""
        result = doc_store.delete("nonexistent")

        assert result is False

    def test_list_all(self, doc_store):
        """Test listing all records."""
        for i in range(5):
            record = DocumentRecord(
                doc_id=f"doc_{i:03d}",
                title=f"Document {i}",
            )
            doc_store.add(record)

        result = doc_store.list_all()

        assert len(result) == 5
        doc_ids = {r.doc_id for r in result}
        assert "doc_000" in doc_ids
        assert "doc_004" in doc_ids

    def test_find_by_category(self, doc_store):
        """Test finding documents by category."""
        categories = ["refund", "promo", "refund", "inventory", "refund"]
        for i, cat in enumerate(categories):
            record = DocumentRecord(
                doc_id=f"doc_{i:03d}",
                title=f"Document {i}",
                category=cat,
            )
            doc_store.add(record)

        result = doc_store.find_by_category("refund")

        assert len(result) == 3

    def test_find_by_category_case_insensitive(self, doc_store):
        """Test that category search is case-insensitive."""
        record = DocumentRecord(
            doc_id="doc_001",
            title="Test",
            category="Refund",
        )
        doc_store.add(record)

        result = doc_store.find_by_category("refund")

        assert len(result) == 1

    def test_find_by_store_type(self, doc_store):
        """Test finding documents by store type."""
        types = ["cafe", "convenience", "cafe", "apparel"]
        for i, st in enumerate(types):
            record = DocumentRecord(
                doc_id=f"doc_{i:03d}",
                title=f"Document {i}",
                store_type=st,
            )
            doc_store.add(record)

        result = doc_store.find_by_store_type("cafe")

        assert len(result) == 2

    def test_find_valid_on_date(self, doc_store):
        """Test finding documents valid on a specific date."""
        # Doc 1: valid Jan-Jun 2024
        doc1 = DocumentRecord(
            doc_id="doc_001",
            title="Doc 1",
            valid_from="2024-01-01",
            valid_to="2024-06-30",
        )
        # Doc 2: valid Jul-Dec 2024
        doc2 = DocumentRecord(
            doc_id="doc_002",
            title="Doc 2",
            valid_from="2024-07-01",
            valid_to="2024-12-31",
        )
        # Doc 3: always valid (no dates)
        doc3 = DocumentRecord(
            doc_id="doc_003",
            title="Doc 3",
        )

        doc_store.add(doc1)
        doc_store.add(doc2)
        doc_store.add(doc3)

        # March 2024 - should match doc1 and doc3
        result = doc_store.find_valid_on_date(date(2024, 3, 15))
        doc_ids = {r.doc_id for r in result}

        assert len(result) == 2
        assert "doc_001" in doc_ids
        assert "doc_003" in doc_ids

        # September 2024 - should match doc2 and doc3
        result = doc_store.find_valid_on_date(date(2024, 9, 15))
        doc_ids = {r.doc_id for r in result}

        assert len(result) == 2
        assert "doc_002" in doc_ids
        assert "doc_003" in doc_ids

    def test_clear(self, doc_store, sample_record):
        """Test clearing all records."""
        doc_store.add(sample_record)
        assert doc_store.count() == 1

        doc_store.clear()

        assert doc_store.count() == 0

    def test_persistence(self):
        """Test that data persists to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "doc_store.json"

            # Create and populate store
            store1 = DocStore(store_path=store_path)
            store1.add(
                DocumentRecord(
                    doc_id="doc_001",
                    title="Persistent Doc",
                    category="test",
                )
            )

            # Create new store from same path
            store2 = DocStore(store_path=store_path)

            assert store2.count() == 1
            assert store2.get("doc_001").title == "Persistent Doc"

    def test_get_metadata_for_retrieval(self, doc_store):
        """Test getting metadata in FilteredRetriever format."""
        doc_store.add(
            DocumentRecord(
                doc_id="doc_001",
                title="Test",
                store_type="cafe",
                category="refund",
                language="ko",
                valid_from="2024-01-01",
                valid_to="2024-12-31",
            )
        )

        result = doc_store.get_metadata_for_retrieval()

        assert "doc_001" in result
        meta = result["doc_001"]
        assert meta["store_type"] == "cafe"
        assert meta["category"] == "refund"
        assert meta["language"] == "ko"
        assert meta["valid_from"] == date(2024, 1, 1)
        assert meta["valid_to"] == date(2024, 12, 31)
