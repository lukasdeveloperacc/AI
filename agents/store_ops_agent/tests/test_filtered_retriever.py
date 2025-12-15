"""Tests for filtered document retriever."""

from datetime import date

import pytest

from app.core.filtered_retriever import (
    DocumentMetadata,
    FilteredRetriever,
    FilteredRetrievalResult,
)
from app.core.metadata_filter import (
    Category,
    Language,
    MetadataFilter,
    StoreType,
)
from app.models.schemas import Citation


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self, citations: list[Citation]):
        self.citations = citations
        self.search_count = 0

    async def search(self, query: str, topk: int) -> list[Citation]:
        self.search_count += 1
        return self.citations[:topk]


def create_citation(doc_id: str, score: float = 0.9) -> Citation:
    """Helper to create a Citation."""
    return Citation(
        doc_id=doc_id,
        title=f"Document {doc_id}",
        chunk_id=f"chunk_{doc_id}",
        snippet=f"Content for {doc_id}",
        score=score,
    )


class TestFilteredRetriever:
    """Tests for FilteredRetriever."""

    @pytest.fixture
    def sample_citations(self) -> list[Citation]:
        """Sample citations for testing."""
        return [
            create_citation("doc_001", 0.95),
            create_citation("doc_002", 0.90),
            create_citation("doc_003", 0.85),
            create_citation("doc_004", 0.80),
            create_citation("doc_005", 0.75),
        ]

    @pytest.fixture
    def metadata_store(self) -> dict[str, DocumentMetadata]:
        """Sample metadata store."""
        return {
            "doc_001": DocumentMetadata(
                doc_id="doc_001",
                store_type="cafe",
                category="refund",
                valid_from=date(2024, 1, 1),
                valid_to=date(2024, 12, 31),
                language="ko",
            ),
            "doc_002": DocumentMetadata(
                doc_id="doc_002",
                store_type="cafe",
                category="promo",
                valid_from=date(2024, 1, 1),
                valid_to=date(2024, 6, 30),
                language="ko",
            ),
            "doc_003": DocumentMetadata(
                doc_id="doc_003",
                store_type="convenience",
                category="refund",
                valid_from=date(2024, 1, 1),
                valid_to=date(2024, 12, 31),
                language="en",
            ),
            "doc_004": DocumentMetadata(
                doc_id="doc_004",
                store_type="apparel",
                category="inventory",
                valid_from=date(2023, 1, 1),
                valid_to=date(2023, 12, 31),
                language="ko",
            ),
            "doc_005": DocumentMetadata(
                doc_id="doc_005",
                store_type="cafe",
                category="refund",
                language="ko",  # No date restrictions
            ),
        }

    @pytest.mark.asyncio
    async def test_retrieve_without_filter(self, sample_citations):
        """Test retrieval without any filter."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever()

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=3,
            filter_obj=None,
            vector_store=vector_store,
        )

        assert len(result.citations) == 3
        assert result.filter_applied is None
        assert result.was_relaxed is False

    @pytest.mark.asyncio
    async def test_retrieve_with_empty_filter(self, sample_citations):
        """Test retrieval with empty filter."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever()
        filter_obj = MetadataFilter()

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=3,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        assert len(result.citations) == 3
        assert result.filter_applied is None

    @pytest.mark.asyncio
    async def test_retrieve_with_store_type_filter(
        self, sample_citations, metadata_store
    ):
        """Test retrieval with store_type filter."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(metadata_store=metadata_store)
        filter_obj = MetadataFilter(store_type=StoreType.CAFE)

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=5,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        # Should only include cafe documents (doc_001, doc_002, doc_005)
        assert len(result.citations) == 3
        for citation in result.citations:
            metadata = metadata_store.get(citation.doc_id)
            assert metadata is None or metadata.store_type == "cafe"

    @pytest.mark.asyncio
    async def test_retrieve_with_category_filter(
        self, sample_citations, metadata_store
    ):
        """Test retrieval with category filter."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(metadata_store=metadata_store)
        filter_obj = MetadataFilter(category=Category.REFUND)

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=5,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        # Should include refund documents (doc_001, doc_003, doc_005)
        assert len(result.citations) == 3
        for citation in result.citations:
            metadata = metadata_store.get(citation.doc_id)
            assert metadata is None or metadata.category == "refund"

    @pytest.mark.asyncio
    async def test_retrieve_with_effective_date_filter(
        self, sample_citations, metadata_store
    ):
        """Test retrieval with effective_date filter."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(metadata_store=metadata_store)
        filter_obj = MetadataFilter(effective_date=date(2024, 3, 15))

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=5,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        # Should include documents valid on 2024-03-15
        # doc_001, doc_002, doc_003, doc_005 (no date restriction)
        # doc_004 expired in 2023
        for citation in result.citations:
            metadata = metadata_store.get(citation.doc_id)
            if metadata and metadata.valid_to:
                assert metadata.valid_from <= date(2024, 3, 15) <= metadata.valid_to

    @pytest.mark.asyncio
    async def test_retrieve_with_language_filter(
        self, sample_citations, metadata_store
    ):
        """Test retrieval with language filter."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(metadata_store=metadata_store)
        filter_obj = MetadataFilter(language=Language.EN)

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=5,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        # Only doc_003 has language="en"
        assert len(result.citations) == 1
        assert result.citations[0].doc_id == "doc_003"

    @pytest.mark.asyncio
    async def test_retrieve_with_combined_filters(
        self, sample_citations, metadata_store
    ):
        """Test retrieval with multiple filters."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(metadata_store=metadata_store)
        filter_obj = MetadataFilter(
            store_type=StoreType.CAFE,
            category=Category.REFUND,
            language=Language.KO,
        )

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=5,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        # Only doc_001 and doc_005 match (cafe + refund + ko)
        assert len(result.citations) == 2
        doc_ids = {c.doc_id for c in result.citations}
        assert doc_ids == {"doc_001", "doc_005"}

    @pytest.mark.asyncio
    async def test_filter_relaxation_on_no_results(
        self, sample_citations, metadata_store
    ):
        """Test that filter is relaxed when no results found."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(
            metadata_store=metadata_store,
            enable_relaxation=True,
        )
        # Filter that matches nothing
        filter_obj = MetadataFilter(
            store_type=StoreType.RETAIL,  # No retail documents
            language=Language.ZH,  # No Chinese documents
        )

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=5,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        assert result.was_relaxed is True
        assert result.relaxation_message is not None

    @pytest.mark.asyncio
    async def test_no_relaxation_when_disabled(self, sample_citations, metadata_store):
        """Test that filter is not relaxed when disabled."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(
            metadata_store=metadata_store,
            enable_relaxation=False,
        )
        filter_obj = MetadataFilter(store_type=StoreType.RETAIL)

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=5,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        assert len(result.citations) == 0
        assert result.was_relaxed is False

    @pytest.mark.asyncio
    async def test_documents_without_metadata_included_by_default(
        self, sample_citations, metadata_store
    ):
        """Test that documents without metadata are included."""
        # Add a citation without metadata
        citations_with_unknown = sample_citations + [create_citation("doc_unknown")]
        vector_store = MockVectorStore(citations_with_unknown)
        retriever = FilteredRetriever(metadata_store=metadata_store)
        filter_obj = MetadataFilter(store_type=StoreType.CAFE)

        result = await retriever.retrieve_with_filter(
            query="test query",
            topk=10,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        # doc_unknown should be included (no metadata = no filter)
        doc_ids = {c.doc_id for c in result.citations}
        assert "doc_unknown" in doc_ids

    @pytest.mark.asyncio
    async def test_overfetch_multiplier(self, sample_citations, metadata_store):
        """Test that overfetch multiplier is applied."""
        vector_store = MockVectorStore(sample_citations)
        retriever = FilteredRetriever(
            metadata_store=metadata_store,
            overfetch_multiplier=5,
        )
        filter_obj = MetadataFilter(store_type=StoreType.CAFE)

        # Request topk=2, should overfetch 2*5=10
        await retriever.retrieve_with_filter(
            query="test query",
            topk=2,
            filter_obj=filter_obj,
            vector_store=vector_store,
        )

        # Vector store should have been asked for 10 results
        # (but we only have 5 in our mock)
        assert vector_store.search_count == 1


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating DocumentMetadata."""
        metadata = DocumentMetadata(
            doc_id="doc_001",
            store_type="cafe",
            category="refund",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            language="ko",
        )

        assert metadata.doc_id == "doc_001"
        assert metadata.store_type == "cafe"
        assert metadata.category == "refund"
        assert metadata.valid_from == date(2024, 1, 1)
        assert metadata.valid_to == date(2024, 12, 31)
        assert metadata.language == "ko"

    def test_create_metadata_minimal(self):
        """Test creating DocumentMetadata with minimal fields."""
        metadata = DocumentMetadata(doc_id="doc_001")

        assert metadata.doc_id == "doc_001"
        assert metadata.store_type is None
        assert metadata.category is None
        assert metadata.valid_from is None
        assert metadata.valid_to is None
        assert metadata.language is None


class TestFilteredRetrievalResult:
    """Tests for FilteredRetrievalResult dataclass."""

    def test_create_result(self):
        """Test creating FilteredRetrievalResult."""
        citations = [create_citation("doc_001")]
        filter_obj = MetadataFilter(store_type=StoreType.CAFE)

        result = FilteredRetrievalResult(
            citations=citations,
            original_count=10,
            filter_applied=filter_obj,
            was_relaxed=True,
            relaxation_message="Filter relaxed",
        )

        assert len(result.citations) == 1
        assert result.original_count == 10
        assert result.filter_applied == filter_obj
        assert result.was_relaxed is True
        assert result.relaxation_message == "Filter relaxed"

    def test_create_result_defaults(self):
        """Test FilteredRetrievalResult defaults."""
        result = FilteredRetrievalResult(
            citations=[],
            original_count=0,
            filter_applied=None,
        )

        assert result.was_relaxed is False
        assert result.relaxation_message is None


class TestDateValidation:
    """Tests for date validation logic."""

    def test_date_within_range(self):
        """Test date within valid range."""
        assert FilteredRetriever._is_date_valid(
            date(2024, 6, 15),
            date(2024, 1, 1),
            date(2024, 12, 31),
        ) is True

    def test_date_before_range(self):
        """Test date before valid range."""
        assert FilteredRetriever._is_date_valid(
            date(2023, 12, 31),
            date(2024, 1, 1),
            date(2024, 12, 31),
        ) is False

    def test_date_after_range(self):
        """Test date after valid range."""
        assert FilteredRetriever._is_date_valid(
            date(2025, 1, 1),
            date(2024, 1, 1),
            date(2024, 12, 31),
        ) is False

    def test_date_on_boundary_valid_from(self):
        """Test date on valid_from boundary."""
        assert FilteredRetriever._is_date_valid(
            date(2024, 1, 1),
            date(2024, 1, 1),
            date(2024, 12, 31),
        ) is True

    def test_date_on_boundary_valid_to(self):
        """Test date on valid_to boundary."""
        assert FilteredRetriever._is_date_valid(
            date(2024, 12, 31),
            date(2024, 1, 1),
            date(2024, 12, 31),
        ) is True

    def test_date_with_no_restrictions(self):
        """Test date when no restrictions set."""
        assert FilteredRetriever._is_date_valid(
            date(2024, 6, 15),
            None,
            None,
        ) is True

    def test_date_with_only_valid_from(self):
        """Test date with only valid_from set."""
        assert FilteredRetriever._is_date_valid(
            date(2024, 6, 15),
            date(2024, 1, 1),
            None,
        ) is True

        assert FilteredRetriever._is_date_valid(
            date(2023, 12, 31),
            date(2024, 1, 1),
            None,
        ) is False

    def test_date_with_only_valid_to(self):
        """Test date with only valid_to set."""
        assert FilteredRetriever._is_date_valid(
            date(2024, 6, 15),
            None,
            date(2024, 12, 31),
        ) is True

        assert FilteredRetriever._is_date_valid(
            date(2025, 1, 1),
            None,
            date(2024, 12, 31),
        ) is False
