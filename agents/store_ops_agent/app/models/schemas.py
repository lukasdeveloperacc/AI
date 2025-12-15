"""Pydantic schemas for API request/response models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    """Answer verdict indicating confidence level.

    Values:
        ANSWERED: Full answer with sufficient evidence.
        PARTIAL: Partial answer with limited evidence.
        NOT_FOUND: No relevant documents found.
        INSUFFICIENT: Evidence found but insufficient to support a definitive answer.
        CONFLICT: Conflicting information found in documents.
    """

    ANSWERED = "answered"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    INSUFFICIENT = "insufficient"
    CONFLICT = "conflict"


class Citation(BaseModel):
    """Citation model for document references.

    Contains information about the source document chunk used to generate the answer.
    """

    doc_id: str = Field(..., description="Unique identifier of the source document")
    title: str = Field(..., description="Title of the source document")
    chunk_id: str = Field(..., description="Identifier of the specific chunk within the document")
    snippet: str = Field(..., description="Relevant text excerpt from the document")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0.0 to 1.0)")
    effective_date: Optional[str] = Field(
        default=None, description="Effective date of the document (for conflict resolution)"
    )
    version: Optional[str] = Field(
        default=None, description="Version of the document (for conflict resolution)"
    )


class ConflictInfo(BaseModel):
    """Information about conflicting citations when verdict is CONFLICT."""

    conflicting_citations: list[str] = Field(
        ..., description="List of citation doc_ids that conflict"
    )
    resolution_basis: Optional[str] = Field(
        default=None, description="Basis used for conflict resolution (e.g., 'effective_date', 'version')"
    )
    recommended_citation_id: Optional[str] = Field(
        default=None, description="Doc_id of the recommended citation based on resolution"
    )


class Meta(BaseModel):
    """Metadata about the answer generation process.

    Contains tracing and performance information for debugging and monitoring.
    """

    trace_id: str = Field(..., description="Unique identifier for request tracing")
    topk: int = Field(..., ge=1, description="Number of top documents retrieved")
    retrieval_attempts: int = Field(
        ..., ge=1, description="Number of retrieval attempts made"
    )
    latency_ms: float = Field(..., ge=0.0, description="Total processing time in milliseconds")
    verdict: Verdict = Field(..., description="Confidence level of the answer")
    conflict_info: Optional[ConflictInfo] = Field(
        default=None, description="Details about conflicting information when verdict is CONFLICT"
    )


class QuestionRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    topk: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")


class AnswerResponse(BaseModel):
    """Response model containing the answer with citations.

    Provides a summarized answer along with source citations and metadata.
    When evidence is insufficient or conflicting, the answer may be withheld
    and follow-up questions are provided instead.
    """

    answer: str = Field(..., description="Summarized answer (3-5 sentences)")
    citations: list[Citation] = Field(
        ..., min_length=0, description="List of source citations supporting the answer"
    )
    follow_up_questions: Optional[list[str]] = Field(
        default=None, description="Suggested follow-up questions"
    )
    withheld: bool = Field(
        default=False,
        description="True if the answer was withheld due to insufficient evidence or conflict",
    )
    withheld_reason: Optional[str] = Field(
        default=None,
        description="Reason for withholding the answer (when withheld=True)",
    )
    meta: Meta = Field(..., description="Metadata about the answer generation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "매장 영업시간은 오전 9시부터 오후 10시까지입니다. 주말과 공휴일에도 동일한 시간에 운영됩니다.",
                "citations": [
                    {
                        "doc_id": "doc_001",
                        "title": "매장 운영 매뉴얼",
                        "chunk_id": "chunk_003",
                        "snippet": "영업시간: 오전 9시 - 오후 10시 (연중무휴)",
                        "score": 0.95,
                    },
                    {
                        "doc_id": "doc_002",
                        "title": "직원 근무 규정",
                        "chunk_id": "chunk_012",
                        "snippet": "매장 운영 시간은 09:00-22:00이며, 주말/공휴일 포함 동일 적용",
                        "score": 0.87,
                    },
                ],
                "follow_up_questions": [
                    "야간 근무 수당은 어떻게 계산되나요?",
                    "휴무일 지정은 어떻게 하나요?",
                ],
                "meta": {
                    "trace_id": "tr_abc123def456",
                    "topk": 5,
                    "retrieval_attempts": 1,
                    "latency_ms": 245.5,
                    "verdict": "answered",
                },
            }
        }
    )
