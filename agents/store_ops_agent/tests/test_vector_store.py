"""Tests for FAISS vector store."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.indexing.chunker import Chunk
from app.indexing.embedder import EmbeddedChunk, EmbeddingResult
from app.indexing.vector_store import FAISSVectorStore


@pytest.fixture
def vector_store() -> FAISSVectorStore:
    """Create a fresh vector store for testing."""
    return FAISSVectorStore(embedding_dim=128)


@pytest.fixture
def sample_embedding_result() -> EmbeddingResult:
    """Create a sample EmbeddingResult for testing."""
    chunks = []
    embeddings = []

    for i in range(5):
        chunk = Chunk(
            chunk_id=f"doc_test_chunk_{i:04d}",
            doc_id="doc_test",
            content=f"This is chunk {i} content.",
            index=i,
            start_char=i * 30,
            end_char=(i + 1) * 30,
        )
        chunks.append(chunk)

        # Create deterministic embedding
        rng = np.random.default_rng(seed=i)
        embedding = rng.random(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)

    embedded_chunks = [
        EmbeddedChunk(chunk=c, embedding=e)
        for c, e in zip(chunks, embeddings, strict=True)
    ]

    return EmbeddingResult(
        doc_id="doc_test",
        title="Test Document",
        embedded_chunks=embedded_chunks,
        metadata={
            "source_path": "/test/doc.md",
            "category": "test",
            "store_type": "cafe",
            "version": "1.0",
        },
        embedding_model="mock-model",
        embedding_dim=128,
    )


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def test_init_default(self):
        """Test initialization with defaults."""
        store = FAISSVectorStore()

        assert store.embedding_dim == 384
        assert store.index_type == "flat"
        assert len(store) == 0

    def test_init_custom_dim(self):
        """Test initialization with custom dimension."""
        store = FAISSVectorStore(embedding_dim=512)

        assert store.embedding_dim == 512
        assert len(store) == 0

    def test_init_invalid_index_type(self):
        """Test initialization with invalid index type."""
        with pytest.raises(ValueError, match="Unknown index type"):
            FAISSVectorStore(index_type="invalid")

    def test_add_embedding_result(self, vector_store, sample_embedding_result):
        """Test adding embeddings from EmbeddingResult."""
        count = vector_store.add(sample_embedding_result)

        assert count == 5
        assert len(vector_store) == 5

    def test_add_empty_result(self, vector_store):
        """Test adding empty embedding result."""
        empty_result = EmbeddingResult(
            doc_id="doc_empty",
            title="Empty",
            embedded_chunks=[],
            metadata={},
            embedding_model="mock",
            embedding_dim=128,
        )

        count = vector_store.add(empty_result)

        assert count == 0
        assert len(vector_store) == 0

    def test_search_by_embedding(self, vector_store, sample_embedding_result):
        """Test searching by embedding vector."""
        vector_store.add(sample_embedding_result)

        # Use first chunk's embedding as query
        query_embedding = sample_embedding_result.embedded_chunks[0].embedding

        results = vector_store.search_by_embedding(query_embedding, topk=3)

        assert len(results) == 3

        # First result should match the query chunk
        assert results[0].chunk_id == "doc_test_chunk_0000"
        assert results[0].doc_id == "doc_test"
        assert results[0].score > 0.5  # Should have high similarity

    def test_search_empty_store(self, vector_store):
        """Test searching empty store."""
        query = np.random.random(128).astype(np.float32)

        results = vector_store.search_by_embedding(query, topk=5)

        assert results == []

    def test_search_respects_topk(self, vector_store, sample_embedding_result):
        """Test that search respects topk limit."""
        vector_store.add(sample_embedding_result)

        query = np.random.random(128).astype(np.float32)

        results = vector_store.search_by_embedding(query, topk=2)

        assert len(results) == 2

    def test_search_result_contains_metadata(self, vector_store, sample_embedding_result):
        """Test that search results contain metadata."""
        vector_store.add(sample_embedding_result)

        query = sample_embedding_result.embedded_chunks[0].embedding
        results = vector_store.search_by_embedding(query, topk=1)

        assert len(results) == 1
        citation = results[0]

        assert citation.doc_id == "doc_test"
        assert citation.title == "Test Document"
        assert citation.version == "1.0"
        assert "chunk" in citation.chunk_id.lower()

    def test_save_and_load(self, sample_embedding_result):
        """Test saving and loading vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "index"

            # Create and populate store
            store1 = FAISSVectorStore(embedding_dim=128)
            store1.add(sample_embedding_result)
            store1.save(save_path)

            # Load into new store
            store2 = FAISSVectorStore.load(save_path)

            assert len(store2) == len(store1)
            assert store2.embedding_dim == store1.embedding_dim

            # Search should work on loaded store
            query = sample_embedding_result.embedded_chunks[0].embedding
            results = store2.search_by_embedding(query, topk=3)

            assert len(results) == 3
            assert results[0].chunk_id == "doc_test_chunk_0000"

    def test_clear(self, vector_store, sample_embedding_result):
        """Test clearing the vector store."""
        vector_store.add(sample_embedding_result)
        assert len(vector_store) == 5

        vector_store.clear()

        assert len(vector_store) == 0
        assert len(vector_store.id_to_metadata) == 0
        assert len(vector_store.chunk_id_to_idx) == 0

    def test_remove_document(self, vector_store, sample_embedding_result):
        """Test removing a document."""
        vector_store.add(sample_embedding_result)

        removed = vector_store.remove_document("doc_test")

        assert removed == 5
        # Note: FAISS doesn't actually remove vectors, just metadata
        assert len(vector_store.id_to_metadata) == 0

    def test_remove_nonexistent_document(self, vector_store, sample_embedding_result):
        """Test removing a nonexistent document."""
        vector_store.add(sample_embedding_result)

        removed = vector_store.remove_document("doc_nonexistent")

        assert removed == 0

    def test_multiple_documents(self, vector_store):
        """Test adding multiple documents."""
        # Create two different embedding results
        for doc_num in range(2):
            chunks = []
            for i in range(3):
                chunk = Chunk(
                    chunk_id=f"doc_{doc_num}_chunk_{i:04d}",
                    doc_id=f"doc_{doc_num}",
                    content=f"Doc {doc_num}, chunk {i}",
                    index=i,
                    start_char=0,
                    end_char=20,
                )
                embedding = np.random.random(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

            result = EmbeddingResult(
                doc_id=f"doc_{doc_num}",
                title=f"Document {doc_num}",
                embedded_chunks=chunks,
                metadata={"category": f"cat_{doc_num}"},
                embedding_model="mock",
                embedding_dim=128,
            )
            vector_store.add(result)

        assert len(vector_store) == 6

        # Remove one document
        vector_store.remove_document("doc_0")

        # Only metadata is removed
        assert len(vector_store.id_to_metadata) == 3
