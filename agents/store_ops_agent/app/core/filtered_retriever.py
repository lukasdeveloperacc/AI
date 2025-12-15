"""Filtered document retriever with metadata-based search refinement.

This module provides:
- FilteredRetriever: Retriever that applies metadata filters to search results
- Post-retrieval filtering for FAISS (which doesn't support native metadata filtering)
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional, Protocol

from app.core.metadata_filter import (
    FilterRelaxationStrategy,
    MetadataFilter,
    RelaxationResult,
)
from app.models.schemas import Citation


class VectorStoreProtocol(Protocol):
    """Protocol for vector store operations."""

    async def search(
        self,
        query: str,
        topk: int,
    ) -> list[Citation]:
        """Search for similar documents.

        Args:
            query: Search query.
            topk: Number of results to return.

        Returns:
            List of citations.
        """
        ...


@dataclass
class DocumentMetadata:
    """Metadata associated with a document for filtering.

    Attributes:
        doc_id: Document identifier.
        store_type: Type of store (e.g., "cafe", "convenience").
        category: Document category (e.g., "refund", "promo").
        valid_from: Start date of document validity.
        valid_to: End date of document validity.
        language: Document language code.
    """

    doc_id: str
    store_type: Optional[str] = None
    category: Optional[str] = None
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    language: Optional[str] = None


@dataclass
class FilteredRetrievalResult:
    """Result of filtered retrieval operation.

    Attributes:
        citations: Retrieved citations after filtering.
        original_count: Number of citations before filtering.
        filter_applied: The filter that was applied.
        was_relaxed: Whether filter relaxation occurred.
        relaxation_message: Message about filter relaxation if any.
    """

    citations: list[Citation]
    original_count: int
    filter_applied: Optional[MetadataFilter]
    was_relaxed: bool = False
    relaxation_message: Optional[str] = None


class FilteredRetriever:
    """Document retriever with metadata filtering support.

    Since FAISS doesn't support native metadata filtering, this class
    implements post-retrieval filtering:
    1. Retrieve more documents than requested (overfetch)
    2. Apply metadata filters
    3. Return top-k filtered results

    If no results match the filters, it can optionally relax filters
    and retry once.
    """

    def __init__(
        self,
        metadata_store: Optional[dict[str, DocumentMetadata]] = None,
        overfetch_multiplier: int = 3,
        enable_relaxation: bool = True,
    ) -> None:
        """Initialize the filtered retriever.

        Args:
            metadata_store: Dictionary mapping doc_id to DocumentMetadata.
            overfetch_multiplier: Multiplier for overfetching documents.
            enable_relaxation: Whether to enable filter relaxation.
        """
        self.metadata_store = metadata_store or {}
        self.overfetch_multiplier = overfetch_multiplier
        self.enable_relaxation = enable_relaxation

    def set_metadata_store(self, metadata_store: dict[str, DocumentMetadata]) -> None:
        """Set the metadata store.

        Args:
            metadata_store: Dictionary mapping doc_id to DocumentMetadata.
        """
        self.metadata_store = metadata_store

    async def retrieve_with_filter(
        self,
        query: str,
        topk: int,
        filter_obj: Optional[MetadataFilter],
        vector_store: VectorStoreProtocol,
    ) -> FilteredRetrievalResult:
        """Retrieve documents with metadata filtering.

        Args:
            query: Search query.
            topk: Number of results to return.
            filter_obj: Metadata filter to apply.
            vector_store: Vector store for similarity search.

        Returns:
            FilteredRetrievalResult with citations and metadata.
        """
        # If no filter, just do normal retrieval
        if filter_obj is None or filter_obj.is_empty():
            citations = await vector_store.search(query, topk)
            return FilteredRetrievalResult(
                citations=citations,
                original_count=len(citations),
                filter_applied=None,
            )

        # Overfetch to have more candidates for filtering
        overfetch_count = topk * self.overfetch_multiplier
        all_citations = await vector_store.search(query, overfetch_count)

        # Apply filter
        filtered = self._apply_filter(all_citations, filter_obj)

        # If we have results, return them
        if filtered:
            return FilteredRetrievalResult(
                citations=filtered[:topk],
                original_count=len(all_citations),
                filter_applied=filter_obj,
            )

        # No results - try relaxation if enabled
        if self.enable_relaxation and not filter_obj.is_empty():
            return await self._retrieve_with_relaxation(
                all_citations,
                topk,
                filter_obj,
            )

        # Return empty result
        return FilteredRetrievalResult(
            citations=[],
            original_count=len(all_citations),
            filter_applied=filter_obj,
        )

    async def _retrieve_with_relaxation(
        self,
        all_citations: list[Citation],
        topk: int,
        original_filter: MetadataFilter,
    ) -> FilteredRetrievalResult:
        """Try retrieval with relaxed filter.

        Only performs one relaxation attempt as per spec.

        Args:
            all_citations: Already fetched citations.
            topk: Number of results to return.
            original_filter: Original filter that yielded no results.

        Returns:
            FilteredRetrievalResult with relaxation metadata.
        """
        relaxation_result = FilterRelaxationStrategy.relax_once(original_filter)

        if relaxation_result.relaxed_filter is None:
            return FilteredRetrievalResult(
                citations=[],
                original_count=len(all_citations),
                filter_applied=original_filter,
                was_relaxed=True,
                relaxation_message="필터를 더 이상 완화할 수 없습니다.",
            )

        # Apply relaxed filter
        filtered = self._apply_filter(all_citations, relaxation_result.relaxed_filter)
        relaxation_message = FilterRelaxationStrategy.get_relaxation_message(relaxation_result)

        return FilteredRetrievalResult(
            citations=filtered[:topk],
            original_count=len(all_citations),
            filter_applied=relaxation_result.relaxed_filter,
            was_relaxed=True,
            relaxation_message=relaxation_message,
        )

    def _apply_filter(
        self,
        citations: list[Citation],
        filter_obj: MetadataFilter,
    ) -> list[Citation]:
        """Apply metadata filter to citations.

        Args:
            citations: List of citations to filter.
            filter_obj: Filter to apply.

        Returns:
            Filtered list of citations.
        """
        result = []
        for citation in citations:
            metadata = self.metadata_store.get(citation.doc_id)
            if metadata is None:
                # If no metadata, include by default
                # (could be changed to exclude)
                result.append(citation)
                continue

            if self._matches_filter(metadata, filter_obj):
                result.append(citation)

        return result

    def _matches_filter(
        self,
        metadata: DocumentMetadata,
        filter_obj: MetadataFilter,
    ) -> bool:
        """Check if document metadata matches the filter.

        Args:
            metadata: Document metadata.
            filter_obj: Filter to match against.

        Returns:
            True if metadata matches all filter criteria.
        """
        # Check store_type
        if filter_obj.store_type is not None:
            if metadata.store_type is None:
                return False
            if metadata.store_type.lower() != filter_obj.store_type.value:
                return False

        # Check category
        if filter_obj.category is not None:
            if metadata.category is None:
                return False
            if metadata.category.lower() != filter_obj.category.value:
                return False

        # Check effective_date (valid_from <= effective_date <= valid_to)
        if filter_obj.effective_date is not None:
            if not self._is_date_valid(
                filter_obj.effective_date,
                metadata.valid_from,
                metadata.valid_to,
            ):
                return False

        # Check language
        if filter_obj.language is not None:
            if metadata.language is None:
                return False
            if metadata.language.lower() != filter_obj.language.value:
                return False

        return True

    @staticmethod
    def _is_date_valid(
        effective_date: date,
        valid_from: Optional[date],
        valid_to: Optional[date],
    ) -> bool:
        """Check if effective_date falls within valid range.

        Args:
            effective_date: Date to check.
            valid_from: Start of validity period (inclusive).
            valid_to: End of validity period (inclusive).

        Returns:
            True if date is within valid range.
        """
        # If no validity dates specified, consider always valid
        if valid_from is None and valid_to is None:
            return True

        # Check lower bound
        if valid_from is not None and effective_date < valid_from:
            return False

        # Check upper bound
        if valid_to is not None and effective_date > valid_to:
            return False

        return True
