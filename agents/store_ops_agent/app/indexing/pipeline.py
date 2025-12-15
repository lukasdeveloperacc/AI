"""Document indexing pipeline.

This module provides:
- IndexingPipeline: End-to-end document indexing
- Orchestrates loading, chunking, embedding, and storage
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from app.indexing.chunker import ChunkerFactory, RecursiveChunker, SentenceChunker
from app.indexing.doc_store import DocStore, DocumentRecord
from app.indexing.document_loader import DocumentLoaderFactory, LoadedDocument
from app.indexing.embedder import EmbeddingGenerator, EmbeddingResult, MockEmbeddingGenerator
from app.indexing.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexingConfig:
    """Configuration for the indexing pipeline.

    Attributes:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks.
        chunker_type: Type of chunker ("recursive" or "sentence").
        embedding_model: Sentence-transformers model name.
        embedding_batch_size: Batch size for embedding.
        vector_store_path: Path to store FAISS index.
        doc_store_path: Path to store document metadata.
        use_mock_embeddings: Use mock embeddings for testing.
    """

    chunk_size: int = 500
    chunk_overlap: int = 50
    chunker_type: str = "recursive"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    vector_store_path: Optional[Path] = None
    doc_store_path: Optional[Path] = None
    use_mock_embeddings: bool = False


@dataclass
class IndexingResult:
    """Result of indexing a document.

    Attributes:
        doc_id: Document identifier.
        title: Document title.
        chunk_count: Number of chunks created.
        success: Whether indexing succeeded.
        error: Error message if failed.
        duration_ms: Processing time in milliseconds.
    """

    doc_id: str
    title: str
    chunk_count: int
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0


class IndexingPipeline:
    """End-to-end document indexing pipeline.

    Orchestrates the complete indexing process:
    1. Load document
    2. Chunk text
    3. Generate embeddings
    4. Store in vector store
    5. Save metadata to DocStore
    """

    def __init__(self, config: Optional[IndexingConfig] = None) -> None:
        """Initialize the indexing pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or IndexingConfig()

        # Initialize chunker
        if self.config.chunker_type == "recursive":
            self.chunker: Union[RecursiveChunker, SentenceChunker] = (
                ChunkerFactory.create_recursive(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            )
        else:
            self.chunker = ChunkerFactory.create_sentence(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

        # Initialize embedder
        if self.config.use_mock_embeddings:
            self.embedder: Union[EmbeddingGenerator, MockEmbeddingGenerator] = (
                MockEmbeddingGenerator()
            )
        else:
            self.embedder = EmbeddingGenerator(
                model_name=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )

        # Initialize vector store
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedder.embedding_dim,
        )

        # Load existing index if path specified
        if self.config.vector_store_path and self.config.vector_store_path.exists():
            self.vector_store = FAISSVectorStore.load(self.config.vector_store_path)

        # Initialize doc store
        self.doc_store = DocStore(store_path=self.config.doc_store_path)

    def index_file(self, file_path: Path) -> IndexingResult:
        """Index a single file.

        Args:
            file_path: Path to the file to index.

        Returns:
            IndexingResult with status and metrics.
        """
        start_time = datetime.now()
        file_path = Path(file_path)

        try:
            # Step 1: Load document
            logger.info(f"Loading document: {file_path}")
            document = DocumentLoaderFactory.load(file_path)

            # Step 2: Index the loaded document
            result = self.index_document(document)

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            result.duration_ms = duration_ms

            return result

        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return IndexingResult(
                doc_id=f"doc_{file_path.stem}",
                title=file_path.stem,
                chunk_count=0,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def index_document(self, document: LoadedDocument) -> IndexingResult:
        """Index a loaded document.

        Args:
            document: LoadedDocument to index.

        Returns:
            IndexingResult with status and metrics.
        """
        start_time = datetime.now()

        try:
            # Step 1: Chunk the document
            logger.info(f"Chunking document: {document.doc_id}")
            chunk_result = self.chunker.chunk(document)

            if not chunk_result.chunks:
                logger.warning(f"No chunks generated for {document.doc_id}")
                return IndexingResult(
                    doc_id=document.doc_id,
                    title=document.title,
                    chunk_count=0,
                    success=True,
                    duration_ms=0,
                )

            # Step 2: Generate embeddings
            logger.info(
                f"Generating embeddings for {len(chunk_result.chunks)} chunks"
            )
            embedding_result = self.embedder.embed_chunks(chunk_result)

            # Step 3: Add to vector store
            logger.info(f"Adding to vector store")
            self.vector_store.add(embedding_result)

            # Step 4: Save to DocStore
            doc_record = DocumentRecord(
                doc_id=document.doc_id,
                title=document.title,
                category=document.category,
                store_type=document.store_type,
                version=document.version,
                valid_from=(
                    document.valid_from.isoformat() if document.valid_from else None
                ),
                valid_to=(
                    document.valid_to.isoformat() if document.valid_to else None
                ),
                language=document.language,
                source_path=document.source_path,
                chunk_count=len(chunk_result.chunks),
            )
            self.doc_store.add(doc_record)

            # Step 5: Persist if paths configured
            if self.config.vector_store_path:
                self.vector_store.save(self.config.vector_store_path)

            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(
                f"Successfully indexed {document.doc_id}: "
                f"{len(chunk_result.chunks)} chunks in {duration_ms:.1f}ms"
            )

            return IndexingResult(
                doc_id=document.doc_id,
                title=document.title,
                chunk_count=len(chunk_result.chunks),
                success=True,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.error(f"Failed to index document {document.doc_id}: {e}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return IndexingResult(
                doc_id=document.doc_id,
                title=document.title,
                chunk_count=0,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )

    def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[IndexingResult]:
        """Index all supported files in a directory.

        Args:
            directory: Directory path to index.
            recursive: Whether to search subdirectories.

        Returns:
            List of IndexingResults for each file.
        """
        directory = Path(directory)
        results = []

        # Get supported extensions
        extensions = DocumentLoaderFactory.supported_extensions()

        # Find all matching files
        files = []
        for ext in extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files.extend(directory.glob(pattern))

        logger.info(f"Found {len(files)} files to index in {directory}")

        for file_path in files:
            result = self.index_file(file_path)
            results.append(result)

        # Summary
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Indexed {success_count}/{len(results)} files successfully"
        )

        return results

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index.

        Args:
            doc_id: Document identifier.

        Returns:
            True if removed, False if not found.
        """
        # Remove from vector store
        removed_chunks = self.vector_store.remove_document(doc_id)

        # Remove from doc store
        removed_doc = self.doc_store.delete(doc_id)

        # Persist
        if self.config.vector_store_path:
            self.vector_store.save(self.config.vector_store_path)

        return removed_doc or removed_chunks > 0

    def get_stats(self) -> dict:
        """Get indexing statistics.

        Returns:
            Dictionary with index statistics.
        """
        return {
            "document_count": self.doc_store.count(),
            "vector_count": len(self.vector_store),
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.embedding_dim,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }


def create_pipeline(
    vector_store_path: Optional[Path] = None,
    doc_store_path: Optional[Path] = None,
    use_mock: bool = False,
) -> IndexingPipeline:
    """Create a configured indexing pipeline.

    Args:
        vector_store_path: Path to store FAISS index.
        doc_store_path: Path to store document metadata.
        use_mock: Use mock embeddings for testing.

    Returns:
        Configured IndexingPipeline.
    """
    config = IndexingConfig(
        vector_store_path=vector_store_path,
        doc_store_path=doc_store_path,
        use_mock_embeddings=use_mock,
    )
    return IndexingPipeline(config)
