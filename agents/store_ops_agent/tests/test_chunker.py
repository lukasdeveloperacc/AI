"""Tests for text chunking strategies."""

from datetime import date

import pytest

from app.indexing.chunker import (
    Chunk,
    ChunkerFactory,
    ChunkResult,
    RecursiveChunker,
    SentenceChunker,
)
from app.indexing.document_loader import LoadedDocument


@pytest.fixture
def sample_document() -> LoadedDocument:
    """Create a sample document for testing."""
    return LoadedDocument(
        doc_id="doc_test",
        title="Test Document",
        content=(
            "This is the first paragraph. It contains multiple sentences. "
            "Each sentence provides some information.\n\n"
            "This is the second paragraph. It also has sentences. "
            "They are separated by periods.\n\n"
            "Third paragraph here. More content follows. "
            "This should be chunked properly."
        ),
        source_path="/test/path.md",
        category="test",
        store_type="cafe",
        valid_from=date(2024, 1, 1),
        language="ko",
    )


@pytest.fixture
def long_document() -> LoadedDocument:
    """Create a longer document for chunking tests."""
    # Create content longer than default chunk size
    paragraphs = []
    for i in range(10):
        para = f"This is paragraph {i}. " * 20
        paragraphs.append(para)

    return LoadedDocument(
        doc_id="doc_long",
        title="Long Document",
        content="\n\n".join(paragraphs),
        source_path="/test/long.md",
    )


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        chunk = Chunk(
            chunk_id="doc_test_chunk_0000",
            doc_id="doc_test",
            content="Test content",
            index=0,
            start_char=0,
            end_char=12,
        )

        result = chunk.to_dict()

        assert result["chunk_id"] == "doc_test_chunk_0000"
        assert result["doc_id"] == "doc_test"
        assert result["content"] == "Test content"
        assert result["index"] == 0
        assert result["start_char"] == 0
        assert result["end_char"] == 12


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        chunker = RecursiveChunker()

        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)

        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20

    def test_init_invalid_chunk_size(self):
        """Test initialization with invalid chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveChunker(chunk_size=-100)

    def test_init_invalid_overlap(self):
        """Test initialization with invalid overlap."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            RecursiveChunker(chunk_overlap=-1)

        with pytest.raises(ValueError, match="chunk_overlap must be less than"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_short_document(self, sample_document):
        """Test chunking a short document that fits in one chunk."""
        chunker = RecursiveChunker(chunk_size=1000)

        result = chunker.chunk(sample_document)

        assert isinstance(result, ChunkResult)
        assert result.doc_id == "doc_test"
        assert result.title == "Test Document"
        assert len(result.chunks) == 1
        assert result.chunks[0].doc_id == "doc_test"

    def test_chunk_long_document(self, long_document):
        """Test chunking a long document into multiple chunks."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)

        result = chunker.chunk(long_document)

        assert len(result.chunks) > 1

        # Check chunk IDs are sequential
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_id == f"doc_long_chunk_{i:04d}"
            assert chunk.index == i

        # Check all content is preserved (approximately)
        total_content = " ".join(c.content for c in result.chunks)
        # Due to overlap, total may be longer
        assert len(total_content) >= len(long_document.content) * 0.8

    def test_chunk_preserves_metadata(self, sample_document):
        """Test that chunking preserves document metadata."""
        chunker = RecursiveChunker()

        result = chunker.chunk(sample_document)

        assert result.metadata["source_path"] == "/test/path.md"
        assert result.metadata["category"] == "test"
        assert result.metadata["store_type"] == "cafe"
        assert result.metadata["valid_from"] == "2024-01-01"
        assert result.metadata["language"] == "ko"

    def test_chunk_empty_document(self):
        """Test chunking an empty document."""
        doc = LoadedDocument(
            doc_id="doc_empty",
            title="Empty",
            content="",
            source_path="/test/empty.md",
        )
        chunker = RecursiveChunker()

        result = chunker.chunk(doc)

        assert len(result.chunks) == 0

    def test_chunk_whitespace_document(self):
        """Test chunking a whitespace-only document."""
        doc = LoadedDocument(
            doc_id="doc_ws",
            title="Whitespace",
            content="   \n\n   \t   ",
            source_path="/test/ws.md",
        )
        chunker = RecursiveChunker()

        result = chunker.chunk(doc)

        assert len(result.chunks) == 0


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        chunker = SentenceChunker()

        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_init_invalid_values(self):
        """Test initialization with invalid values."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SentenceChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            SentenceChunker(chunk_overlap=-1)

    def test_chunk_preserves_sentences(self):
        """Test that sentence boundaries are preserved."""
        doc = LoadedDocument(
            doc_id="doc_sent",
            title="Sentence Test",
            content=(
                "First sentence. Second sentence. Third sentence. "
                "Fourth sentence. Fifth sentence."
            ),
            source_path="/test/sent.md",
        )
        chunker = SentenceChunker(chunk_size=50, chunk_overlap=10)

        result = chunker.chunk(doc)

        # Each chunk should end at sentence boundary
        for chunk in result.chunks:
            # Should not end with partial word (unless very short)
            if len(chunk.content) > 10:
                assert chunk.content.strip().endswith((".", "!", "?", "sentence"))

    def test_chunk_long_document(self, long_document):
        """Test sentence chunking on long document."""
        chunker = SentenceChunker(chunk_size=200, chunk_overlap=20)

        result = chunker.chunk(long_document)

        assert len(result.chunks) > 1

        # Check chunk IDs
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_id == f"doc_long_chunk_{i:04d}"


class TestChunkerFactory:
    """Tests for ChunkerFactory."""

    def test_create_recursive_chunker(self):
        """Test creating recursive chunker."""
        chunker = ChunkerFactory.create_recursive(chunk_size=300, chunk_overlap=30)

        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunk_size == 300
        assert chunker.chunk_overlap == 30

    def test_create_sentence_chunker(self):
        """Test creating sentence chunker."""
        chunker = ChunkerFactory.create_sentence(chunk_size=400, chunk_overlap=40)

        assert isinstance(chunker, SentenceChunker)
        assert chunker.chunk_size == 400
        assert chunker.chunk_overlap == 40
