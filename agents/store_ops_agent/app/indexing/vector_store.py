"""FAISS vector store for similarity search.

This module provides:
- FAISSVectorStore: Store and search embeddings using FAISS
- Persistence support for saving/loading index
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.indexing.embedder import EmbeddedChunk, EmbeddingResult
from app.models.schemas import Citation

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for similarity search.

    Stores embeddings in a FAISS index and supports:
    - Adding embeddings in batch
    - Similarity search with score normalization
    - Persistence (save/load)
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "flat",
    ) -> None:
        """Initialize the FAISS vector store.

        Args:
            embedding_dim: Dimension of embedding vectors.
            index_type: Type of FAISS index ("flat" or "ivf").
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Mapping from FAISS index position to chunk metadata
        self.id_to_metadata: dict[int, dict] = {}
        self.chunk_id_to_idx: dict[str, int] = {}
        self._next_idx = 0

    def add(self, embedding_result: EmbeddingResult) -> int:
        """Add embeddings from an EmbeddingResult to the index.

        Args:
            embedding_result: EmbeddingResult with embedded chunks.

        Returns:
            Number of chunks added.
        """
        if not embedding_result.embedded_chunks:
            return 0

        # Prepare embeddings matrix
        embeddings = np.array(
            [ec.embedding for ec in embedding_result.embedded_chunks],
            dtype=np.float32,
        )

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata
        for embedded_chunk in embedding_result.embedded_chunks:
            metadata = {
                "chunk_id": embedded_chunk.chunk.chunk_id,
                "doc_id": embedded_chunk.chunk.doc_id,
                "content": embedded_chunk.chunk.content,
                "index": embedded_chunk.chunk.index,
                "title": embedding_result.title,
                **embedding_result.metadata,
            }
            self.id_to_metadata[self._next_idx] = metadata
            self.chunk_id_to_idx[embedded_chunk.chunk.chunk_id] = self._next_idx
            self._next_idx += 1

        logger.info(
            f"Added {len(embedding_result.embedded_chunks)} chunks from "
            f"document '{embedding_result.doc_id}' to index"
        )

        return len(embedding_result.embedded_chunks)

    def add_chunks(
        self,
        embedded_chunks: list[EmbeddedChunk],
        doc_metadata: dict,
    ) -> int:
        """Add embedded chunks directly to the index.

        Args:
            embedded_chunks: List of EmbeddedChunk objects.
            doc_metadata: Metadata to associate with all chunks.

        Returns:
            Number of chunks added.
        """
        if not embedded_chunks:
            return 0

        embeddings = np.array(
            [ec.embedding for ec in embedded_chunks],
            dtype=np.float32,
        )

        self.index.add(embeddings)

        for embedded_chunk in embedded_chunks:
            metadata = {
                "chunk_id": embedded_chunk.chunk.chunk_id,
                "doc_id": embedded_chunk.chunk.doc_id,
                "content": embedded_chunk.chunk.content,
                "index": embedded_chunk.chunk.index,
                **doc_metadata,
            }
            self.id_to_metadata[self._next_idx] = metadata
            self.chunk_id_to_idx[embedded_chunk.chunk.chunk_id] = self._next_idx
            self._next_idx += 1

        return len(embedded_chunks)

    async def search(
        self,
        query: str,
        topk: int,
        query_embedding: Optional[np.ndarray] = None,
    ) -> list[Citation]:
        """Search for similar documents.

        Note: This method requires a query_embedding to be passed.
        In practice, the caller should generate the embedding.

        Args:
            query: Search query (for reference, not used directly).
            topk: Number of results to return.
            query_embedding: Pre-computed query embedding.

        Returns:
            List of Citation objects.
        """
        if query_embedding is None:
            raise ValueError("query_embedding must be provided for FAISS search")

        return self.search_by_embedding(query_embedding, topk)

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        topk: int,
    ) -> list[Citation]:
        """Search using a pre-computed embedding.

        Args:
            query_embedding: Query embedding vector.
            topk: Number of results to return.

        Returns:
            List of Citation objects sorted by score.
        """
        if self.index.ntotal == 0:
            return []

        # Ensure proper shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Search
        k = min(topk, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        # Convert to Citations
        citations = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            metadata = self.id_to_metadata.get(int(idx))
            if metadata is None:
                continue

            # Normalize score to 0-1 range (cosine similarity is already -1 to 1)
            normalized_score = max(0.0, min(1.0, (float(score) + 1) / 2))

            citation = Citation(
                doc_id=metadata["doc_id"],
                title=metadata.get("title", "Unknown"),
                chunk_id=metadata["chunk_id"],
                snippet=metadata["content"],
                score=normalized_score,
                effective_date=metadata.get("valid_from"),
                version=metadata.get("version"),
            )
            citations.append(citation)

        return citations

    def save(self, path: Path) -> None:
        """Save the index and metadata to disk.

        Args:
            path: Directory path to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type,
                    "next_idx": self._next_idx,
                    "id_to_metadata": {
                        str(k): v for k, v in self.id_to_metadata.items()
                    },
                    "chunk_id_to_idx": self.chunk_id_to_idx,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Saved vector store to {path}")

    @classmethod
    def load(cls, path: Path) -> "FAISSVectorStore":
        """Load an index from disk.

        Args:
            path: Directory path to load from.

        Returns:
            Loaded FAISSVectorStore instance.
        """
        path = Path(path)

        # Load metadata first
        metadata_path = path / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)

        # Create instance
        store = cls(
            embedding_dim=data["embedding_dim"],
            index_type=data["index_type"],
        )

        # Load FAISS index
        index_path = path / "faiss.index"
        store.index = faiss.read_index(str(index_path))

        # Restore metadata
        store._next_idx = data["next_idx"]
        store.id_to_metadata = {int(k): v for k, v in data["id_to_metadata"].items()}
        store.chunk_id_to_idx = data["chunk_id_to_idx"]

        logger.info(f"Loaded vector store from {path} ({store.index.ntotal} vectors)")

        return store

    def __len__(self) -> int:
        """Return number of vectors in the index."""
        return self.index.ntotal

    def clear(self) -> None:
        """Clear all data from the index."""
        # Recreate the index
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)

        self.id_to_metadata.clear()
        self.chunk_id_to_idx.clear()
        self._next_idx = 0

        logger.info("Cleared vector store")

    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks for a document.

        Note: FAISS doesn't support efficient deletion, so this marks
        entries as deleted in metadata but doesn't remove from index.
        For production, consider rebuilding the index periodically.

        Args:
            doc_id: Document ID to remove.

        Returns:
            Number of chunks marked as removed.
        """
        removed = 0
        chunks_to_remove = []

        for chunk_id, idx in self.chunk_id_to_idx.items():
            metadata = self.id_to_metadata.get(idx)
            if metadata and metadata.get("doc_id") == doc_id:
                chunks_to_remove.append((chunk_id, idx))

        for chunk_id, idx in chunks_to_remove:
            self.id_to_metadata.pop(idx, None)
            self.chunk_id_to_idx.pop(chunk_id, None)
            removed += 1

        if removed > 0:
            logger.info(f"Marked {removed} chunks from document '{doc_id}' as removed")

        return removed
