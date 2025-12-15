"""Tests for CitationFormatter."""

from app.core.citation_formatter import CitationFormatter
from app.models.schemas import Citation


class TestCitationFormatter:
    """Tests for CitationFormatter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = CitationFormatter()
        self.citation = Citation(
            doc_id="doc_001",
            title="Test Document",
            chunk_id="chunk_001",
            snippet="This is a test snippet.",
            score=0.85,
        )
        self.citations = [
            Citation(
                doc_id="doc_001",
                title="Doc A",
                chunk_id="chunk_001",
                snippet="Snippet A",
                score=0.9,
            ),
            Citation(
                doc_id="doc_002",
                title="Doc B",
                chunk_id="chunk_002",
                snippet="Snippet B",
                score=0.7,
            ),
            Citation(
                doc_id="doc_003",
                title="Doc C",
                chunk_id="chunk_003",
                snippet="Snippet C",
                score=0.4,
            ),
        ]

    def test_format_inline(self):
        """Test inline citation formatting."""
        result = self.formatter.format_inline(self.citation)
        assert result == "[Test Document] - This is a test snippet."

    def test_format_reference(self):
        """Test reference citation formatting."""
        result = self.formatter.format_reference(self.citation, 1)
        assert "[1]" in result
        assert "Test Document" in result
        assert "85%" in result

    def test_format_markdown(self):
        """Test markdown citation formatting."""
        result = self.formatter.format_markdown(self.citation)
        assert "**Test Document**" in result
        assert "0.85" in result
        assert "> This is a test snippet." in result

    def test_format_citations_block(self):
        """Test formatting multiple citations as a block."""
        result = self.formatter.format_citations_block(self.citations[:2])
        assert "**Sources:**" in result
        assert "Doc A" in result
        assert "Doc B" in result

    def test_format_citations_block_empty(self):
        """Test formatting empty citations list."""
        result = self.formatter.format_citations_block([])
        assert result == "No citations available."

    def test_create_citation(self):
        """Test creating citation from fields."""
        citation = self.formatter.create_citation(
            doc_id="doc_test",
            title="Test Title",
            chunk_id="chunk_test",
            snippet="Test snippet",
            score=0.75,
        )
        assert citation.doc_id == "doc_test"
        assert citation.title == "Test Title"
        assert citation.score == 0.75

    def test_create_citation_clamps_score(self):
        """Test that create_citation clamps score to valid range."""
        citation_high = self.formatter.create_citation(
            doc_id="doc_test",
            title="Test",
            chunk_id="chunk_test",
            snippet="Test",
            score=1.5,
        )
        assert citation_high.score == 1.0

        citation_low = self.formatter.create_citation(
            doc_id="doc_test",
            title="Test",
            chunk_id="chunk_test",
            snippet="Test",
            score=-0.5,
        )
        assert citation_low.score == 0.0

    def test_filter_by_score(self):
        """Test filtering citations by minimum score."""
        filtered = self.formatter.filter_by_score(self.citations, min_score=0.5)
        assert len(filtered) == 2
        assert all(c.score >= 0.5 for c in filtered)

    def test_filter_by_score_default(self):
        """Test default score filter (0.5)."""
        filtered = self.formatter.filter_by_score(self.citations)
        assert len(filtered) == 2

    def test_sort_by_score_descending(self):
        """Test sorting citations by score (descending)."""
        sorted_citations = self.formatter.sort_by_score(self.citations)
        assert sorted_citations[0].score == 0.9
        assert sorted_citations[-1].score == 0.4

    def test_sort_by_score_ascending(self):
        """Test sorting citations by score (ascending)."""
        sorted_citations = self.formatter.sort_by_score(
            self.citations, descending=False
        )
        assert sorted_citations[0].score == 0.4
        assert sorted_citations[-1].score == 0.9

    def test_deduplicate(self):
        """Test deduplicating citations."""
        citations_with_dups = [
            Citation(
                doc_id="doc_001",
                title="Doc A",
                chunk_id="chunk_001",
                snippet="Snippet 1",
                score=0.9,
            ),
            Citation(
                doc_id="doc_001",
                title="Doc A",
                chunk_id="chunk_001",
                snippet="Snippet 1 duplicate",
                score=0.85,
            ),
            Citation(
                doc_id="doc_002",
                title="Doc B",
                chunk_id="chunk_002",
                snippet="Snippet 2",
                score=0.7,
            ),
        ]
        unique = self.formatter.deduplicate(citations_with_dups)
        assert len(unique) == 2
        # First occurrence is kept
        assert unique[0].snippet == "Snippet 1"
