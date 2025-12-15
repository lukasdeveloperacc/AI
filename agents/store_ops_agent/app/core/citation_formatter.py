"""Citation formatter for formatting document references."""

from app.models.schemas import Citation


class CitationFormatter:
    """Formats citations for display in responses."""

    @staticmethod
    def format_inline(citation: Citation) -> str:
        """Format a citation for inline display.

        Args:
            citation: The citation to format.

        Returns:
            Formatted string like "[doc_title] - snippet"
        """
        return f"[{citation.title}] - {citation.snippet}"

    @staticmethod
    def format_reference(citation: Citation, index: int) -> str:
        """Format a citation as a numbered reference.

        Args:
            citation: The citation to format.
            index: The reference number (1-based).

        Returns:
            Formatted reference string.
        """
        return f"[{index}] {citation.title} (relevance: {citation.score:.0%})\n    \"{citation.snippet}\""

    @staticmethod
    def format_markdown(citation: Citation) -> str:
        """Format a citation in markdown format.

        Args:
            citation: The citation to format.

        Returns:
            Markdown formatted citation.
        """
        return f"**{citation.title}** (score: {citation.score:.2f})\n> {citation.snippet}"

    @staticmethod
    def format_citations_block(citations: list[Citation]) -> str:
        """Format multiple citations as a block.

        Args:
            citations: List of citations to format.

        Returns:
            Formatted block of all citations.
        """
        if not citations:
            return "No citations available."

        lines = ["**Sources:**"]
        for i, citation in enumerate(citations, 1):
            lines.append(f"\n{i}. **{citation.title}**")
            lines.append(f"   > {citation.snippet}")
            lines.append(f"   (relevance: {citation.score:.0%})")
        return "\n".join(lines)

    @staticmethod
    def create_citation(
        doc_id: str,
        title: str,
        chunk_id: str,
        snippet: str,
        score: float,
    ) -> Citation:
        """Create a Citation object from individual fields.

        Args:
            doc_id: Unique document identifier.
            title: Document title.
            chunk_id: Chunk identifier within document.
            snippet: Relevant text excerpt.
            score: Relevance score (0.0 to 1.0).

        Returns:
            Citation object.
        """
        return Citation(
            doc_id=doc_id,
            title=title,
            chunk_id=chunk_id,
            snippet=snippet,
            score=min(max(score, 0.0), 1.0),  # Clamp to valid range
        )

    @staticmethod
    def filter_by_score(
        citations: list[Citation],
        min_score: float = 0.5,
    ) -> list[Citation]:
        """Filter citations by minimum relevance score.

        Args:
            citations: List of citations to filter.
            min_score: Minimum score threshold (default 0.5).

        Returns:
            Filtered list of citations.
        """
        return [c for c in citations if c.score >= min_score]

    @staticmethod
    def sort_by_score(
        citations: list[Citation],
        descending: bool = True,
    ) -> list[Citation]:
        """Sort citations by relevance score.

        Args:
            citations: List of citations to sort.
            descending: Sort in descending order (default True).

        Returns:
            Sorted list of citations.
        """
        return sorted(citations, key=lambda c: c.score, reverse=descending)

    @staticmethod
    def deduplicate(citations: list[Citation]) -> list[Citation]:
        """Remove duplicate citations based on doc_id and chunk_id.

        Args:
            citations: List of citations to deduplicate.

        Returns:
            Deduplicated list of citations.
        """
        seen = set()
        unique = []
        for citation in citations:
            key = (citation.doc_id, citation.chunk_id)
            if key not in seen:
                seen.add(key)
                unique.append(citation)
        return unique
