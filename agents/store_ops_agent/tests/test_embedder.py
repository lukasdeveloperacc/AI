"""Tests for embedding generation."""

import numpy as np
import pytest

from app.indexing.chunker import Chunk, ChunkResult
from app.indexing.embedder import (
    EmbeddedChunk,
    EmbeddingResult,
    MockEmbeddingGenerator,
)


@pytest.fixture
def sample_chunk_result() -> ChunkResult:
    """Create a sample ChunkResult for testing."""
    chunks = [
        Chunk(
            chunk_id="doc_test_chunk_0000",
            doc_id="doc_test",
            content="This is the first chunk content.",
            index=0,
            start_char=0,
            end_char=33,
        ),
        Chunk(
            chunk_id="doc_test_chunk_0001",
            doc_id="doc_test",
            content="Second chunk has different text.",
            index=1,
            start_char=34,
            end_char=66,
        ),
        Chunk(
            chunk_id="doc_test_chunk_0002",
            doc_id="doc_test",
            content="Third chunk completes the document.",
            index=2,
            start_char=67,
            end_char=103,
        ),
    ]

    return ChunkResult(
        doc_id="doc_test",
        title="Test Document",
        chunks=chunks,
        metadata={
            "source_path": "/test/path.md",
            "category": "test",
            "store_type": "cafe",
        },
    )


class TestMockEmbeddingGenerator:
    """Tests for MockEmbeddingGenerator."""

    def test_init_default_dim(self):
        """Test initialization with default dimension."""
        generator = MockEmbeddingGenerator()
        assert generator.embedding_dim == 384

    def test_init_custom_dim(self):
        """Test initialization with custom dimension."""
        generator = MockEmbeddingGenerator(embedding_dim=512)
        assert generator.embedding_dim == 512

    def test_embed_text_returns_correct_dim(self):
        """Test that single text embedding has correct dimension."""
        generator = MockEmbeddingGenerator(embedding_dim=256)

        embedding = generator.embed_text("Test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (256,)

    def test_embed_text_is_normalized(self):
        """Test that embeddings are normalized."""
        generator = MockEmbeddingGenerator()

        embedding = generator.embed_text("Test text")

        # L2 norm should be approximately 1
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, rtol=1e-5)

    def test_embed_text_is_deterministic(self):
        """Test that same text produces same embedding."""
        generator = MockEmbeddingGenerator()

        embedding1 = generator.embed_text("Test text")
        embedding2 = generator.embed_text("Test text")

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_embed_text_different_texts_differ(self):
        """Test that different texts produce different embeddings."""
        generator = MockEmbeddingGenerator()

        embedding1 = generator.embed_text("First text")
        embedding2 = generator.embed_text("Second text")

        assert not np.allclose(embedding1, embedding2)

    def test_embed_texts_batch(self):
        """Test batch text embedding."""
        generator = MockEmbeddingGenerator(embedding_dim=128)
        texts = ["Text one", "Text two", "Text three"]

        embeddings = generator.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 128)

    def test_embed_chunks(self, sample_chunk_result):
        """Test embedding chunks from ChunkResult."""
        generator = MockEmbeddingGenerator(embedding_dim=128)

        result = generator.embed_chunks(sample_chunk_result)

        assert isinstance(result, EmbeddingResult)
        assert result.doc_id == "doc_test"
        assert result.title == "Test Document"
        assert result.embedding_model == "mock-embedding-model"
        assert result.embedding_dim == 128
        assert len(result.embedded_chunks) == 3

        # Check each embedded chunk
        for i, ec in enumerate(result.embedded_chunks):
            assert isinstance(ec, EmbeddedChunk)
            assert ec.chunk.chunk_id == f"doc_test_chunk_{i:04d}"
            assert ec.embedding.shape == (128,)

    def test_embed_chunks_empty(self):
        """Test embedding empty chunk result."""
        generator = MockEmbeddingGenerator()
        empty_result = ChunkResult(
            doc_id="doc_empty",
            title="Empty",
            chunks=[],
            metadata={},
        )

        result = generator.embed_chunks(empty_result)

        assert len(result.embedded_chunks) == 0
        assert result.doc_id == "doc_empty"

    def test_embed_chunks_preserves_metadata(self, sample_chunk_result):
        """Test that metadata is preserved through embedding."""
        generator = MockEmbeddingGenerator()

        result = generator.embed_chunks(sample_chunk_result)

        assert result.metadata["source_path"] == "/test/path.md"
        assert result.metadata["category"] == "test"
        assert result.metadata["store_type"] == "cafe"


class TestEmbeddedChunk:
    """Tests for EmbeddedChunk dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        chunk = Chunk(
            chunk_id="test_chunk",
            doc_id="doc_test",
            content="Content",
            index=0,
            start_char=0,
            end_char=7,
        )
        embedding = np.zeros(384, dtype=np.float32)

        embedded = EmbeddedChunk(chunk=chunk, embedding=embedding)
        result = embedded.to_dict()

        assert result["chunk_id"] == "test_chunk"
        assert result["doc_id"] == "doc_test"
        assert result["content"] == "Content"
        assert result["embedding_dim"] == 384
