"""Document indexing and vector store package.

This package provides:
- Document loaders (PDF, Markdown, JSON)
- Text chunking strategies
- Embedding generation
- FAISS vector store
- Document metadata store
- Indexing pipeline
"""

from app.indexing.chunker import (
    Chunk,
    ChunkerFactory,
    ChunkResult,
    RecursiveChunker,
    SentenceChunker,
)
from app.indexing.doc_store import DocStore, DocumentRecord
from app.indexing.document_loader import (
    BaseDocumentLoader,
    DocumentLoaderFactory,
    JSONLoader,
    LoadedDocument,
    MarkdownLoader,
    PDFLoader,
)
from app.indexing.embedder import (
    EmbeddedChunk,
    EmbeddingGenerator,
    EmbeddingResult,
    MockEmbeddingGenerator,
)
from app.indexing.pipeline import (
    IndexingConfig,
    IndexingPipeline,
    IndexingResult,
    create_pipeline,
)
from app.indexing.vector_store import FAISSVectorStore

__all__ = [
    # Document loaders
    "BaseDocumentLoader",
    "DocumentLoaderFactory",
    "JSONLoader",
    "LoadedDocument",
    "MarkdownLoader",
    "PDFLoader",
    # Chunking
    "Chunk",
    "ChunkerFactory",
    "ChunkResult",
    "RecursiveChunker",
    "SentenceChunker",
    # Embedding
    "EmbeddedChunk",
    "EmbeddingGenerator",
    "EmbeddingResult",
    "MockEmbeddingGenerator",
    # Vector store
    "FAISSVectorStore",
    # Document store
    "DocStore",
    "DocumentRecord",
    # Pipeline
    "IndexingConfig",
    "IndexingPipeline",
    "IndexingResult",
    "create_pipeline",
]
