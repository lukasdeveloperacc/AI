"""Retrieve Node for StoreOps Agent.

This node:
1. Takes normalized query and filters from state
2. Searches FAISS vector store with topK (default 8-10)
3. Applies metadata filtering (post-retrieval for FAISS)
4. Limits to max_chunks_per_doc=2
5. Returns candidates with scores
"""

import logging
import time
from collections import defaultdict
from typing import Any, Optional

from app.agents.state import (
    AgentCounters,
    AgentFilters,
    AgentState,
    RetrievalResult,
)
from app.core.metadata_filter import (
    FilterRelaxationStrategy,
    MetadataFilter,
    MetadataFilterParser,
)
from app.indexing.embedder import EmbeddingGenerator
from app.indexing.vector_store import FAISSVectorStore
from app.models.schemas import Citation

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TOPK = 8
MAX_CHUNKS_PER_DOC = 2
OVERFETCH_MULTIPLIER = 3


class VectorStoreManager:
    """Singleton manager for vector store access."""

    _instance: Optional["VectorStoreManager"] = None
    _vector_store: Optional[FAISSVectorStore] = None
    _embedder: Optional[EmbeddingGenerator] = None

    @classmethod
    def get_instance(cls) -> "VectorStoreManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_vector_store(cls, vector_store: FAISSVectorStore) -> None:
        """Set the vector store to use."""
        cls._vector_store = vector_store

    @classmethod
    def set_embedder(cls, embedder: EmbeddingGenerator) -> None:
        """Set the embedder to use."""
        cls._embedder = embedder

    @classmethod
    def get_vector_store(cls) -> Optional[FAISSVectorStore]:
        """Get the vector store."""
        return cls._vector_store

    @classmethod
    def get_embedder(cls) -> Optional[EmbeddingGenerator]:
        """Get the embedder."""
        return cls._embedder


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant documents from vector store.

    Args:
        state: Current agent state with parsed query and filters.

    Returns:
        Dictionary with retrieval results.
    """
    start_time = time.time()

    # Get query from parsed state or fall back to original question
    parsed = state.get("parsed")
    query = parsed.normalized_query if parsed else state["question"]
    topk = state.get("topk", DEFAULT_TOPK)
    filters = state.get("filters", AgentFilters())
    counters = state.get("counters", AgentCounters())

    # Increment retrieval attempts
    counters.retrieval_attempts += 1

    try:
        # Get vector store and embedder
        vector_store = VectorStoreManager.get_vector_store()
        embedder = VectorStoreManager.get_embedder()

        if vector_store is None or embedder is None:
            logger.warning("Vector store or embedder not initialized")
            return _create_empty_result(state, counters, start_time)

        # Generate query embedding
        query_embedding = await _get_query_embedding(query, embedder)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return _create_empty_result(state, counters, start_time)

        # Search with overfetch for filtering
        overfetch_count = topk * OVERFETCH_MULTIPLIER
        all_citations = vector_store.search_by_embedding(query_embedding, overfetch_count)

        # Apply metadata filters
        filtered_citations = _apply_filters(all_citations, filters)

        # If no results and we have filters, try relaxation
        filter_applied = not _is_filter_empty(filters)
        if not filtered_citations and filter_applied and counters.retrieval_attempts < 2:
            logger.info("No results with filters, attempting relaxation")
            filtered_citations, filters = _relax_and_retry(all_citations, filters)

        # Apply max_chunks_per_doc limit
        limited_citations = _limit_chunks_per_doc(filtered_citations, MAX_CHUNKS_PER_DOC)

        # Take top-k after filtering
        final_citations = limited_citations[:topk]

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Retrieve completed: {len(final_citations)} results "
            f"(from {len(all_citations)} total), latency={latency_ms:.1f}ms"
        )

        # Update metrics
        current_metrics = state.get("metrics")
        if current_metrics:
            current_metrics.retrieve_latency_ms = latency_ms

        return {
            "retrieval": RetrievalResult(
                candidates=final_citations,
                query_embedding=query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding),
                total_retrieved=len(all_citations),
                filter_applied=filter_applied,
            ),
            "filters": filters,
            "counters": counters,
            "metrics": current_metrics,
        }

    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        errors = state.get("errors", [])
        errors.append(f"Retrieve error: {e}")
        return {
            "retrieval": RetrievalResult(
                candidates=[],
                total_retrieved=0,
                filter_applied=False,
            ),
            "counters": counters,
            "errors": errors,
        }


async def _get_query_embedding(query: str, embedder: EmbeddingGenerator) -> Any:
    """Get embedding for query.

    Args:
        query: Query string.
        embedder: EmbeddingGenerator instance.

    Returns:
        Query embedding vector.
    """
    try:
        # EmbeddingGenerator.embed_text returns numpy array
        embedding = embedder.embed_text(query)
        return embedding
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return None


def _apply_filters(
    citations: list[Citation], filters: AgentFilters
) -> list[Citation]:
    """Apply metadata filters to citations.

    Since FAISS doesn't support native filtering, we filter post-retrieval.

    Args:
        citations: List of retrieved citations.
        filters: Filters to apply.

    Returns:
        Filtered list of citations.
    """
    if _is_filter_empty(filters):
        return citations

    filtered = []
    for citation in citations:
        # Check category match (from doc_id or metadata in snippet)
        if filters.category:
            # Simple heuristic: check if category appears in doc_id or snippet
            if not _matches_category(citation, filters.category):
                continue

        # Check store_type match
        if filters.store_type:
            if not _matches_store_type(citation, filters.store_type):
                continue

        # Check effective_date
        if filters.effective_date and citation.effective_date:
            try:
                from datetime import datetime

                doc_date = datetime.strptime(citation.effective_date, "%Y-%m-%d").date()
                if doc_date > filters.effective_date:
                    continue
            except (ValueError, TypeError):
                pass

        filtered.append(citation)

    return filtered


def _matches_category(citation: Citation, category: str) -> bool:
    """Check if citation matches category filter."""
    category_lower = category.lower()
    # Check in doc_id, title, or snippet
    if category_lower in citation.doc_id.lower():
        return True
    if category_lower in citation.title.lower():
        return True
    return False


def _matches_store_type(citation: Citation, store_type: str) -> bool:
    """Check if citation matches store_type filter."""
    store_type_lower = store_type.lower()
    if store_type_lower in citation.doc_id.lower():
        return True
    if store_type_lower in citation.title.lower():
        return True
    return False


def _is_filter_empty(filters: AgentFilters) -> bool:
    """Check if all filters are empty."""
    return all(
        v is None
        for v in [filters.category, filters.store_type, filters.effective_date, filters.language]
    )


def _relax_and_retry(
    all_citations: list[Citation], filters: AgentFilters
) -> tuple[list[Citation], AgentFilters]:
    """Relax filters and retry filtering.

    Args:
        all_citations: All retrieved citations before filtering.
        filters: Current filters.

    Returns:
        Tuple of (filtered citations, relaxed filters).
    """
    # Convert AgentFilters to MetadataFilter for relaxation
    try:
        metadata_filter = MetadataFilterParser.parse(
            store_type=filters.store_type,
            category=filters.category,
            effective_date=filters.effective_date.isoformat() if filters.effective_date else None,
            language=filters.language,
        )

        result = FilterRelaxationStrategy.relax_once(metadata_filter)
        if result.relaxed_filter is None:
            return [], filters

        # Convert back to AgentFilters
        relaxed_filters = AgentFilters(
            store_type=result.relaxed_filter.store_type.value if result.relaxed_filter.store_type else None,
            category=result.relaxed_filter.category.value if result.relaxed_filter.category else None,
            effective_date=result.relaxed_filter.effective_date,
            language=result.relaxed_filter.language.value if result.relaxed_filter.language else None,
        )

        # Retry filtering with relaxed filters
        filtered = _apply_filters(all_citations, relaxed_filters)
        logger.info(f"Relaxed filter (removed {result.removed_field}), got {len(filtered)} results")

        return filtered, relaxed_filters

    except Exception as e:
        logger.warning(f"Filter relaxation failed: {e}")
        return [], filters


def _limit_chunks_per_doc(
    citations: list[Citation], max_chunks: int
) -> list[Citation]:
    """Limit number of chunks per document.

    Args:
        citations: List of citations.
        max_chunks: Maximum chunks per document.

    Returns:
        Citations with limit applied.
    """
    doc_counts: dict[str, int] = defaultdict(int)
    limited = []

    for citation in citations:
        if doc_counts[citation.doc_id] < max_chunks:
            limited.append(citation)
            doc_counts[citation.doc_id] += 1

    return limited


def _create_empty_result(
    state: AgentState, counters: AgentCounters, start_time: float
) -> dict[str, Any]:
    """Create empty retrieval result."""
    latency_ms = (time.time() - start_time) * 1000

    current_metrics = state.get("metrics")
    if current_metrics:
        current_metrics.retrieve_latency_ms = latency_ms

    return {
        "retrieval": RetrievalResult(
            candidates=[],
            total_retrieved=0,
            filter_applied=False,
        ),
        "counters": counters,
        "metrics": current_metrics,
    }
