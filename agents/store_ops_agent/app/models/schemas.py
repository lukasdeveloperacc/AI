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


class FilterInfo(BaseModel):
    """Information about applied search filters."""

    store_type: Optional[str] = Field(default=None, description="Applied store type filter")
    category: Optional[str] = Field(default=None, description="Applied category filter")
    effective_date: Optional[str] = Field(default=None, description="Applied effective date filter")
    language: Optional[str] = Field(default=None, description="Applied language filter")
    was_relaxed: bool = Field(default=False, description="Whether the filter was relaxed")
    relaxation_message: Optional[str] = Field(
        default=None, description="Message about filter relaxation"
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
    generation_attempts: int = Field(
        default=1, ge=1, description="Number of generation attempts made"
    )
    latency_ms: float = Field(..., ge=0.0, description="Total processing time in milliseconds")
    verdict: Verdict = Field(..., description="Confidence level of the answer")
    conflict_info: Optional[ConflictInfo] = Field(
        default=None, description="Details about conflicting information when verdict is CONFLICT"
    )
    filter_info: Optional[FilterInfo] = Field(
        default=None, description="Information about applied search filters"
    )


class QuestionRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    topk: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    store_type: Optional[str] = Field(
        default=None,
        description="Store type filter (e.g., 'cafe', 'convenience', 'apparel', 'restaurant', 'retail')",
    )
    category: Optional[str] = Field(
        default=None,
        description="Document category filter (e.g., 'refund', 'promo', 'inventory', 'cs', 'operation', 'hr')",
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Filter by documents valid on this date (YYYY-MM-DD format)",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language filter (e.g., 'ko', 'en', 'ja', 'zh')",
    )


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
                    "generation_attempts": 1,
                    "latency_ms": 245.5,
                    "verdict": "answered",
                },
            }
        }
    )


class ChatRequest(BaseModel):
    """Request model for POST /chat endpoint.

    Follows the PRD Section 10 API specification.
    """

    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    store_type: Optional[str] = Field(
        default=None,
        description="Store type filter (e.g., 'cafe', 'convenience', 'apparel', 'restaurant', 'retail')",
    )
    category: Optional[str] = Field(
        default=None,
        description="Document category filter (e.g., 'refund', 'promo', 'inventory', 'cs', 'operation', 'hr')",
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Filter by documents valid on this date (YYYY-MM-DD format)",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language filter (e.g., 'ko', 'en', 'ja', 'zh')",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "환불 정책이 어떻게 되나요?",
                "store_type": "cafe",
                "category": "refund",
                "language": "ko",
            }
        }
    )


class ChatResponse(BaseModel):
    """Response model for POST /chat endpoint.

    Follows the PRD Section 10 API specification with:
    - answer: Generated answer text
    - citations: List of source citations
    - follow_up_question: Single follow-up question (not a list)
    - meta: Metadata about the generation process
    """

    answer: str = Field(..., description="Generated answer (3-5 sentences)")
    citations: list[Citation] = Field(
        default_factory=list, description="List of source citations supporting the answer"
    )
    follow_up_question: Optional[str] = Field(
        default=None, description="Suggested follow-up question for clarification"
    )
    meta: Meta = Field(..., description="Metadata about the answer generation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "환불은 구매 후 7일 이내에 가능하며, 영수증 지참이 필요합니다. 단, 식품류의 경우 개봉 전에만 환불이 가능합니다.",
                "citations": [
                    {
                        "doc_id": "doc_refund_001",
                        "title": "환불 정책 가이드",
                        "chunk_id": "chunk_005",
                        "snippet": "구매 후 7일 이내 영수증 지참 시 환불 가능",
                        "score": 0.95,
                    }
                ],
                "follow_up_question": "특정 상품에 대한 환불 조건이 궁금하신가요?",
                "meta": {
                    "trace_id": "tr_abc123def456",
                    "topk": 8,
                    "retrieval_attempts": 1,
                    "generation_attempts": 1,
                    "latency_ms": 523.4,
                    "verdict": "answered",
                },
            }
        }
    )


class ErrorDetail(BaseModel):
    """Error detail model for error responses."""

    code: str = Field(..., description="Error code for client handling")
    message: str = Field(..., description="Human-readable error message")
    trace_id: Optional[str] = Field(
        default=None, description="Trace ID for debugging (if available)"
    )
    details: Optional[dict] = Field(
        default=None, description="Additional error details"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid date format. Expected YYYY-MM-DD.",
                "trace_id": "tr_abc123def456",
                "details": {"field": "effective_date", "received": "2024/01/01"},
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: ErrorDetail = Field(..., description="Error details")
