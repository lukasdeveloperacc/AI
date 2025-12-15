"""Text chunking strategies for document processing.

This module provides:
- RecursiveChunker: Chunk text recursively by separators
- SentenceChunker: Chunk by sentence boundaries
- ChunkResult: Container for chunks with metadata
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from app.indexing.document_loader import LoadedDocument


@dataclass
class Chunk:
    """Represents a single chunk of text.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        doc_id: Parent document ID.
        content: Text content of the chunk.
        index: Position index within the document.
        start_char: Starting character position in original document.
        end_char: Ending character position in original document.
    """

    chunk_id: str
    doc_id: str
    content: str
    index: int
    start_char: int
    end_char: int

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class ChunkResult:
    """Result of chunking operation.

    Attributes:
        doc_id: Source document ID.
        title: Document title.
        chunks: List of generated chunks.
        metadata: Document metadata for storage.
    """

    doc_id: str
    title: str
    chunks: list[Chunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """Recursive text chunker that splits by multiple separators.

    Attempts to split text by larger semantic units first (paragraphs),
    then falls back to smaller units (sentences, words) as needed.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ) -> None:
        """Initialize the recursive chunker.

        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            separators: List of separators to use (ordered by preference).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: LoadedDocument) -> ChunkResult:
        """Chunk a document into smaller pieces.

        Args:
            document: LoadedDocument to chunk.

        Returns:
            ChunkResult with generated chunks.
        """
        text = document.content
        chunks = self._split_text(text, self.separators)

        result = ChunkResult(
            doc_id=document.doc_id,
            title=document.title,
            metadata={
                "source_path": document.source_path,
                "category": document.category,
                "store_type": document.store_type,
                "version": document.version,
                "valid_from": (
                    document.valid_from.isoformat() if document.valid_from else None
                ),
                "valid_to": (
                    document.valid_to.isoformat() if document.valid_to else None
                ),
                "language": document.language,
            },
        )

        # Create Chunk objects
        char_offset = 0
        for idx, chunk_text in enumerate(chunks):
            # Find actual position in original text
            start_pos = text.find(chunk_text, char_offset)
            if start_pos == -1:
                start_pos = char_offset

            chunk = Chunk(
                chunk_id=f"{document.doc_id}_chunk_{idx:04d}",
                doc_id=document.doc_id,
                content=chunk_text,
                index=idx,
                start_char=start_pos,
                end_char=start_pos + len(chunk_text),
            )
            result.chunks.append(chunk)

            # Update offset for next search (accounting for overlap)
            char_offset = start_pos + len(chunk_text) - self.chunk_overlap
            if char_offset < 0:
                char_offset = 0

        return result

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split.
            separators: List of separators to use.

        Returns:
            List of text chunks.
        """
        if not text.strip():
            return []

        # If text is already small enough, return it
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # If no more separators, force split
        if not separators:
            return self._force_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        splits = text.split(separator)

        # Merge small splits and recursively process large ones
        chunks = []
        current_chunk = ""

        for split in splits:
            # Add separator back (except for first split)
            if current_chunk and separator != " ":
                candidate = current_chunk + separator + split
            elif current_chunk:
                candidate = current_chunk + " " + split
            else:
                candidate = split

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Save current chunk if non-empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Process split with remaining separators if too large
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_text(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Apply overlap
        if self.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _force_split(self, text: str) -> list[str]:
        """Force split text into chunk_size pieces.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to find a word boundary
            if end < len(text):
                # Look for space within the last 20% of chunk
                search_start = max(start, end - int(self.chunk_size * 0.2))
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start <= 0 and end < len(text):
                start = end  # Prevent infinite loop

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between chunks.

        Args:
            chunks: List of chunks without overlap.

        Returns:
            List of chunks with overlap applied.
        """
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap :]

                # Find a good boundary for overlap
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1 :]

                new_chunk = overlap_text + " " + chunk if overlap_text else chunk
                result.append(new_chunk.strip())

        return result


class SentenceChunker:
    """Sentence-based chunker that preserves sentence boundaries.

    Chunks text by grouping complete sentences up to the target size.
    """

    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize the sentence chunker.

        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Approximate overlap in characters.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: LoadedDocument) -> ChunkResult:
        """Chunk a document by sentence boundaries.

        Args:
            document: LoadedDocument to chunk.

        Returns:
            ChunkResult with generated chunks.
        """
        text = document.content
        sentences = self.SENTENCE_ENDINGS.split(text)

        result = ChunkResult(
            doc_id=document.doc_id,
            title=document.title,
            metadata={
                "source_path": document.source_path,
                "category": document.category,
                "store_type": document.store_type,
                "version": document.version,
                "valid_from": (
                    document.valid_from.isoformat() if document.valid_from else None
                ),
                "valid_to": (
                    document.valid_to.isoformat() if document.valid_to else None
                ),
                "language": document.language,
            },
        )

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Keep last sentence(s) for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

        # Create Chunk objects
        char_offset = 0
        for idx, chunk_text in enumerate(chunks):
            start_pos = text.find(chunk_text[:50], char_offset)
            if start_pos == -1:
                start_pos = char_offset

            chunk = Chunk(
                chunk_id=f"{document.doc_id}_chunk_{idx:04d}",
                doc_id=document.doc_id,
                content=chunk_text,
                index=idx,
                start_char=start_pos,
                end_char=start_pos + len(chunk_text),
            )
            result.chunks.append(chunk)
            char_offset = start_pos + len(chunk_text) - self.chunk_overlap

        return result


class ChunkerFactory:
    """Factory for creating chunkers."""

    @staticmethod
    def create_recursive(
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> RecursiveChunker:
        """Create a recursive chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.

        Returns:
            Configured RecursiveChunker.
        """
        return RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def create_sentence(
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> SentenceChunker:
        """Create a sentence chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.

        Returns:
            Configured SentenceChunker.
        """
        return SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
