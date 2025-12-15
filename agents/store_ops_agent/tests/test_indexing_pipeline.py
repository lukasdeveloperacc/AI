"""Tests for the indexing pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from app.indexing.document_loader import LoadedDocument
from app.indexing.pipeline import (
    IndexingConfig,
    IndexingPipeline,
    IndexingResult,
    create_pipeline,
)


@pytest.fixture
def mock_pipeline() -> IndexingPipeline:
    """Create a pipeline with mock embeddings."""
    config = IndexingConfig(
        chunk_size=200,
        chunk_overlap=20,
        use_mock_embeddings=True,
    )
    return IndexingPipeline(config)


@pytest.fixture
def temp_document_dir():
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a markdown file
        md_file = tmpdir / "policy.md"
        md_file.write_text(
            "# Store Policy\n\n"
            "This is the main policy document.\n\n"
            "## Refund Rules\n\n"
            "Customers can request refunds within 30 days.\n"
            "The product must be in original condition.\n"
            "Receipt is required for all refunds.\n",
            encoding="utf-8",
        )

        # Create a JSON file with metadata
        json_file = tmpdir / "promo.json"
        json.dump(
            {
                "doc_id": "promo_001",
                "title": "Summer Promotion",
                "content": (
                    "Summer sale starting June 1st. "
                    "All items 20% off. "
                    "Valid until August 31st. "
                    "Cannot be combined with other offers."
                ),
                "category": "promo",
                "store_type": "cafe",
                "version": "1.0",
                "valid_from": "2024-06-01",
                "valid_to": "2024-08-31",
                "language": "en",
            },
            json_file.open("w", encoding="utf-8"),
            ensure_ascii=False,
        )

        yield tmpdir


class TestIndexingConfig:
    """Tests for IndexingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = IndexingConfig()

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.chunker_type == "recursive"
        assert config.embedding_batch_size == 32
        assert config.use_mock_embeddings is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = IndexingConfig(
            chunk_size=300,
            chunk_overlap=30,
            chunker_type="sentence",
            use_mock_embeddings=True,
        )

        assert config.chunk_size == 300
        assert config.chunk_overlap == 30
        assert config.chunker_type == "sentence"
        assert config.use_mock_embeddings is True


class TestIndexingResult:
    """Tests for IndexingResult."""

    def test_success_result(self):
        """Test successful result creation."""
        result = IndexingResult(
            doc_id="doc_001",
            title="Test",
            chunk_count=10,
            success=True,
            duration_ms=100.5,
        )

        assert result.doc_id == "doc_001"
        assert result.chunk_count == 10
        assert result.success is True
        assert result.error is None
        assert result.duration_ms == 100.5

    def test_failure_result(self):
        """Test failure result creation."""
        result = IndexingResult(
            doc_id="doc_001",
            title="Test",
            chunk_count=0,
            success=False,
            error="File not found",
        )

        assert result.success is False
        assert result.error == "File not found"


class TestIndexingPipeline:
    """Tests for IndexingPipeline."""

    def test_init_with_mock_embeddings(self, mock_pipeline):
        """Test initialization with mock embeddings."""
        assert mock_pipeline.config.use_mock_embeddings is True
        assert len(mock_pipeline.vector_store) == 0
        assert mock_pipeline.doc_store.count() == 0

    def test_init_with_sentence_chunker(self):
        """Test initialization with sentence chunker."""
        config = IndexingConfig(
            chunker_type="sentence",
            use_mock_embeddings=True,
        )
        pipeline = IndexingPipeline(config)

        # Should not raise
        assert pipeline.chunker is not None

    def test_index_document(self, mock_pipeline):
        """Test indexing a loaded document."""
        doc = LoadedDocument(
            doc_id="doc_test",
            title="Test Document",
            content=(
                "This is a test document with some content. "
                "It has multiple sentences for testing. "
                "The pipeline should chunk and index it properly."
            ),
            source_path="/test/doc.md",
            category="test",
            store_type="cafe",
        )

        result = mock_pipeline.index_document(doc)

        assert result.success is True
        assert result.doc_id == "doc_test"
        assert result.chunk_count > 0
        assert result.duration_ms > 0

        # Check vector store
        assert len(mock_pipeline.vector_store) > 0

        # Check doc store
        assert mock_pipeline.doc_store.count() == 1
        stored = mock_pipeline.doc_store.get("doc_test")
        assert stored.title == "Test Document"
        assert stored.category == "test"

    def test_index_empty_document(self, mock_pipeline):
        """Test indexing an empty document."""
        doc = LoadedDocument(
            doc_id="doc_empty",
            title="Empty",
            content="",
            source_path="/test/empty.md",
        )

        result = mock_pipeline.index_document(doc)

        assert result.success is True
        assert result.chunk_count == 0

    def test_index_file_markdown(self, mock_pipeline, temp_document_dir):
        """Test indexing a markdown file."""
        md_file = temp_document_dir / "policy.md"

        result = mock_pipeline.index_file(md_file)

        assert result.success is True
        assert result.title == "Store Policy"
        assert result.chunk_count > 0

    def test_index_file_json(self, mock_pipeline, temp_document_dir):
        """Test indexing a JSON file."""
        json_file = temp_document_dir / "promo.json"

        result = mock_pipeline.index_file(json_file)

        assert result.success is True
        assert result.doc_id == "promo_001"
        assert result.title == "Summer Promotion"

        # Check metadata is preserved
        stored = mock_pipeline.doc_store.get("promo_001")
        assert stored.category == "promo"
        assert stored.store_type == "cafe"
        assert stored.version == "1.0"

    def test_index_nonexistent_file(self, mock_pipeline):
        """Test indexing a nonexistent file."""
        result = mock_pipeline.index_file(Path("/nonexistent/file.md"))

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "no such file" in result.error.lower()

    def test_index_unsupported_file(self, mock_pipeline):
        """Test indexing an unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Some text content")
            temp_path = Path(f.name)

        try:
            result = mock_pipeline.index_file(temp_path)

            assert result.success is False
            assert "unsupported" in result.error.lower()
        finally:
            temp_path.unlink()

    def test_index_directory(self, mock_pipeline, temp_document_dir):
        """Test indexing a directory."""
        results = mock_pipeline.index_directory(temp_document_dir)

        assert len(results) == 2
        success_count = sum(1 for r in results if r.success)
        assert success_count == 2

    def test_remove_document(self, mock_pipeline):
        """Test removing a document."""
        # First, index a document
        doc = LoadedDocument(
            doc_id="doc_remove",
            title="To Remove",
            content="Content to be removed from the index.",
            source_path="/test/remove.md",
        )
        mock_pipeline.index_document(doc)

        assert mock_pipeline.doc_store.count() == 1

        # Remove it
        result = mock_pipeline.remove_document("doc_remove")

        assert result is True
        assert mock_pipeline.doc_store.count() == 0

    def test_remove_nonexistent_document(self, mock_pipeline):
        """Test removing a nonexistent document."""
        result = mock_pipeline.remove_document("nonexistent")

        assert result is False

    def test_get_stats(self, mock_pipeline):
        """Test getting pipeline statistics."""
        # Index some documents
        doc = LoadedDocument(
            doc_id="doc_stats",
            title="Stats Test",
            content="Content for statistics test. " * 10,
            source_path="/test/stats.md",
        )
        mock_pipeline.index_document(doc)

        stats = mock_pipeline.get_stats()

        assert stats["document_count"] == 1
        assert stats["vector_count"] > 0
        assert stats["embedding_dim"] > 0
        assert stats["chunk_size"] == 200
        assert stats["chunk_overlap"] == 20

    def test_persistence(self, temp_document_dir):
        """Test that index persists across pipeline instances."""
        with tempfile.TemporaryDirectory() as index_dir:
            index_dir = Path(index_dir)

            # Create pipeline with persistence
            config1 = IndexingConfig(
                use_mock_embeddings=True,
                vector_store_path=index_dir / "vectors",
                doc_store_path=index_dir / "docs.json",
            )
            pipeline1 = IndexingPipeline(config1)

            # Index a document
            pipeline1.index_file(temp_document_dir / "policy.md")
            initial_count = len(pipeline1.vector_store)

            # Create new pipeline from same paths
            config2 = IndexingConfig(
                use_mock_embeddings=True,
                vector_store_path=index_dir / "vectors",
                doc_store_path=index_dir / "docs.json",
            )
            pipeline2 = IndexingPipeline(config2)

            assert len(pipeline2.vector_store) == initial_count
            assert pipeline2.doc_store.count() == 1


class TestCreatePipeline:
    """Tests for create_pipeline factory function."""

    def test_create_default(self):
        """Test creating pipeline with defaults."""
        pipeline = create_pipeline(use_mock=True)

        assert pipeline is not None
        assert pipeline.config.use_mock_embeddings is True

    def test_create_with_paths(self):
        """Test creating pipeline with custom paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            pipeline = create_pipeline(
                vector_store_path=tmpdir / "vectors",
                doc_store_path=tmpdir / "docs.json",
                use_mock=True,
            )

            assert pipeline.config.vector_store_path == tmpdir / "vectors"
            assert pipeline.config.doc_store_path == tmpdir / "docs.json"
