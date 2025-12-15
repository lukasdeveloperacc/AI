"""Follow-up question generator for insufficient evidence scenarios.

This module generates context-aware follow-up questions when the system
cannot provide a definitive answer due to insufficient or conflicting evidence.
"""

from typing import Optional

from app.models.schemas import Citation, Verdict


class FollowUpGenerator:
    """Generates follow-up questions based on context and verdict.

    When evidence is insufficient or conflicting, this generator creates
    relevant follow-up questions to help users refine their queries.
    """

    # Template follow-up questions for different scenarios
    TEMPLATES = {
        Verdict.NOT_FOUND: [
            "어떤 종류의 문서에서 정보를 찾고 계신가요? (예: 매뉴얼, 규정, 가이드)",
            "질문하신 내용과 관련된 특정 키워드가 있으신가요?",
            "언제 적용되는 정보인지 알려주시겠어요? (예: 현재, 특정 날짜)",
        ],
        Verdict.INSUFFICIENT: [
            "조금 더 구체적으로 어떤 상황에 대해 알고 싶으신가요?",
            "특정 매장이나 부서에 관한 질문인가요?",
            "관련 규정이나 매뉴얼 이름을 알고 계신가요?",
        ],
        Verdict.CONFLICT: [
            "어느 시점의 정보가 필요하신가요? (최신 vs 특정 시점)",
            "특정 버전의 문서를 찾고 계신가요?",
            "담당 부서에서 확인받으신 내용이 있으신가요?",
        ],
        Verdict.PARTIAL: [
            "추가로 알고 싶으신 부분이 있으신가요?",
            "다른 관련 정보도 필요하신가요?",
        ],
    }

    def __init__(self, max_questions: int = 3) -> None:
        """Initialize the follow-up generator.

        Args:
            max_questions: Maximum number of follow-up questions to generate.
        """
        self.max_questions = max_questions

    def generate(
        self,
        verdict: Verdict,
        question: str,
        citations: Optional[list[Citation]] = None,
    ) -> list[str]:
        """Generate follow-up questions based on verdict and context.

        Args:
            verdict: The grounding check verdict.
            question: The original user question.
            citations: Available citations (may be empty or partial).

        Returns:
            List of follow-up questions.
        """
        questions = []

        # Add context-specific question based on the original question
        context_question = self._generate_context_question(question, verdict)
        if context_question:
            questions.append(context_question)

        # Add citation-based questions if available
        if citations:
            citation_questions = self._generate_citation_questions(citations, verdict)
            questions.extend(citation_questions)

        # Add template questions for the verdict type
        template_questions = self.TEMPLATES.get(verdict, [])
        for tq in template_questions:
            if tq not in questions:
                questions.append(tq)
                if len(questions) >= self.max_questions:
                    break

        return questions[: self.max_questions]

    def _generate_context_question(
        self,
        question: str,
        verdict: Verdict,
    ) -> Optional[str]:
        """Generate a context-aware question based on the original question.

        Args:
            question: The original user question.
            verdict: The grounding check verdict.

        Returns:
            A context-specific follow-up question, or None.
        """
        # Extract key terms from the question for personalized follow-up
        question_lower = question.lower()

        # Check for common question patterns and generate relevant follow-ups
        if "언제" in question or "시간" in question:
            if verdict == Verdict.INSUFFICIENT:
                return "특정 요일이나 기간에 대해 질문하시는 건가요?"

        if "어떻게" in question or "방법" in question:
            if verdict == Verdict.INSUFFICIENT:
                return "어떤 상황에서의 방법을 알고 싶으신가요?"

        if "누가" in question or "담당" in question:
            if verdict == Verdict.INSUFFICIENT:
                return "어떤 업무나 상황의 담당자를 찾으시는 건가요?"

        if "왜" in question or "이유" in question:
            if verdict == Verdict.INSUFFICIENT:
                return "특정 정책이나 절차의 이유를 알고 싶으신가요?"

        if verdict == Verdict.NOT_FOUND:
            return f"'{self._extract_key_term(question)}'에 대해 다른 표현으로 검색해 볼까요?"

        return None

    def _generate_citation_questions(
        self,
        citations: list[Citation],
        verdict: Verdict,
    ) -> list[str]:
        """Generate questions based on available citations.

        Args:
            citations: Available citations.
            verdict: The grounding check verdict.

        Returns:
            List of citation-based follow-up questions.
        """
        questions = []

        if not citations:
            return questions

        # If there are low-score citations, ask if they want related info
        low_score_citations = [c for c in citations if c.score < 0.6]
        if low_score_citations and verdict == Verdict.INSUFFICIENT:
            doc_titles = set(c.title for c in low_score_citations[:2])
            if doc_titles:
                titles_str = ", ".join(f"'{t}'" for t in doc_titles)
                questions.append(f"{titles_str} 문서에서 관련 정보를 찾아볼까요?")

        # If there are conflicting citations, ask for clarification
        if verdict == Verdict.CONFLICT and len(citations) > 1:
            # Get unique document titles
            doc_titles = list(set(c.title for c in citations[:3]))
            if len(doc_titles) > 1:
                questions.append(
                    f"'{doc_titles[0]}'과 '{doc_titles[1]}' 중 어느 문서의 정보가 필요하신가요?"
                )

        return questions

    @staticmethod
    def _extract_key_term(question: str) -> str:
        """Extract the main keyword from a question.

        This is a simplified extraction that removes common question words.

        Args:
            question: The user's question.

        Returns:
            Extracted key term or the first few words.
        """
        # Remove common question prefixes and suffixes
        stop_words = [
            "이", "가", "은", "는", "을", "를", "에", "의", "로", "으로",
            "어떻게", "무엇", "언제", "어디", "누가", "왜",
            "하나요", "인가요", "있나요", "할까요", "될까요",
            "해주세요", "알려주세요", "설명해주세요",
        ]

        words = question.split()
        key_words = []

        for word in words:
            is_stop = False
            for stop in stop_words:
                if word.endswith(stop) or word == stop:
                    # Keep the word if it's meaningful after removing stop word
                    cleaned = word.replace(stop, "").strip()
                    if cleaned and len(cleaned) > 1:
                        key_words.append(cleaned)
                    is_stop = True
                    break
            if not is_stop and len(word) > 1:
                key_words.append(word)

        if key_words:
            return " ".join(key_words[:3])

        # Fallback: return first few words
        return " ".join(words[:3]) if words else question
