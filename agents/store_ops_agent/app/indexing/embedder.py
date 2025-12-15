"""Embedding generation for text chunks.

This module provides:
- EmbeddingGenerator: Generate embeddings using sentence-transformers
- Batch processing for efficient embedding generation
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.indexing.chunker import Chunk, ChunkResult

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """Chunk with its embedding vector.

    Attributes:
        chunk: Original chunk data.
        embedding: Embedding vector.
    """

    chunk: Chunk
    embedding: np.ndarray

    def to_dict(self) -> dict:
        """Convert to dictionary (without embedding for serialization)."""
        return {
            **self.chunk.to_dict(),
            "embedding_dim": len(self.embedding),
        }


@dataclass
class EmbeddingResult:
    """Result of embedding generation.

    Attributes:
        doc_id: Source document ID.
        title: Document title.
        embedded_chunks: List of chunks with embeddings.
        metadata: Document metadata.
        embedding_model: Name of the embedding model used.
        embedding_dim: Dimension of embeddings.
    """

    doc_id: str
    title: str
    embedded_chunks: list[EmbeddedChunk]
    metadata: dict
    embedding_model: str
    embedding_dim: int


class EmbeddingGenerator:
    """Generate embeddings for text chunks using sentence-transformers.

    Uses a pre-trained model to convert text into dense vector representations.
    Supports batch processing for efficiency.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model.
            batch_size: Batch size for encoding.
            device: Device to use (cpu, cuda, mps). Auto-detected if None.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._embedding_dim: Optional[int] = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(
                    f"Loaded embedding model: {self.model_name} "
                    f"(dim={self._embedding_dim})"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            # Force model load to get dimension
            _ = self.model
        return self._embedding_dim

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of embeddings with shape (num_texts, embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings

    def embed_chunks(self, chunk_result: ChunkResult) -> EmbeddingResult:
        """Generate embeddings for all chunks in a ChunkResult.

        Args:
            chunk_result: ChunkResult containing chunks to embed.

        Returns:
            EmbeddingResult with embedded chunks.
        """
        if not chunk_result.chunks:
            return EmbeddingResult(
                doc_id=chunk_result.doc_id,
                title=chunk_result.title,
                embedded_chunks=[],
                metadata=chunk_result.metadata,
                embedding_model=self.model_name,
                embedding_dim=self.embedding_dim,
            )

        # Extract texts from chunks
        texts = [chunk.content for chunk in chunk_result.chunks]

        # Generate embeddings in batch
        embeddings = self.embed_texts(texts)

        # Create EmbeddedChunk objects
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunk_result.chunks, embeddings, strict=True)
        ]

        return EmbeddingResult(
            doc_id=chunk_result.doc_id,
            title=chunk_result.title,
            embedded_chunks=embedded_chunks,
            metadata=chunk_result.metadata,
            embedding_model=self.model_name,
            embedding_dim=self.embedding_dim,
        )


class MockEmbeddingGenerator:
    """Mock embedding generator for testing.

    Generates random embeddings instead of using a real model.
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        """Initialize the mock generator.

        Args:
            embedding_dim: Dimension of generated embeddings.
        """
        self._embedding_dim = embedding_dim
        self.model_name = "mock-embedding-model"

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a random embedding for text.

        Args:
            text: Text to embed (ignored, random generated).

        Returns:
            Random embedding vector.
        """
        # Use hash of text as seed for reproducibility
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.random(self._embedding_dim).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate random embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            Array of random embeddings.
        """
        embeddings = np.array([self.embed_text(text) for text in texts])
        return embeddings

    def embed_chunks(self, chunk_result: ChunkResult) -> EmbeddingResult:
        """Generate embeddings for all chunks.

        Args:
            chunk_result: ChunkResult containing chunks to embed.

        Returns:
            EmbeddingResult with embedded chunks.
        """
        if not chunk_result.chunks:
            return EmbeddingResult(
                doc_id=chunk_result.doc_id,
                title=chunk_result.title,
                embedded_chunks=[],
                metadata=chunk_result.metadata,
                embedding_model=self.model_name,
                embedding_dim=self._embedding_dim,
            )

        texts = [chunk.content for chunk in chunk_result.chunks]
        embeddings = self.embed_texts(texts)

        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunk_result.chunks, embeddings, strict=True)
        ]

        return EmbeddingResult(
            doc_id=chunk_result.doc_id,
            title=chunk_result.title,
            embedded_chunks=embedded_chunks,
            metadata=chunk_result.metadata,
            embedding_model=self.model_name,
            embedding_dim=self._embedding_dim,
        )
