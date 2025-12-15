# PRD: StoreOps Agentic RAG MVP (FAISS + LangGraph)

## 1) 배경 / 문제정의
오프라인 매장 운영(환불/프로모션/재고/CS)은 규정이 복잡하고 빈번히 변경된다. 담당자(점장/스태프)는 즉시 정확한 답을 얻기 어렵고, LLM 단독 응답은 근거 부족/환각(Hallucination) 위험이 있다.  
따라서 **근거 문서 기반(RAG) + 안전장치(근거 인용 강제, 모르면 보류) + 작업 흐름(Agent)**가 필요하다.

---

## 2) 목표(Goals)
- 매장 운영 관련 질문에 대해 **근거 문서 기반 답변**을 제공한다.
- **LangGraph** 기반으로 `분류 → 검색 → 생성 → 근거검증 → 확정` 흐름을 구성한다.
- 최소한의 운영 요소(로그/캐시)와 평가 지표를 포함해 **“운영 가능한 시스템”** 형태로 제시한다.

---

## 3) 비목표(Non-goals)
- 멀티테넌시, 권한/SSO, 대규모 분산 인덱스/샤딩
- POS/재고 시스템 등 외부 시스템 실연동(도구 호출은 mock 수준)
- 고급 Reranker/Cross-Encoder 최적화(시간 남으면 옵션)

---

## 4) 타겟 사용자 / 시나리오
- **매장 스태프/점장**: “환불 가능한 기간은?”, “이번 주 프로모션 적용 조건은?”
- **운영 관리자**: “정책 버전별 차이”, “최신 규정이 무엇인지”

---

## 5) 핵심 사용자 스토리(User Stories)
1. 사용자는 질문을 입력하면 **정확한 답변 + 근거(문서명/발췌)**를 받는다.
2. 근거가 부족하면 시스템은 **추가 질문을 하거나 답변을 보류**한다.
3. 사용자는 `store_type`, `category`, `effective_date` 등 컨텍스트로 검색 범위를 좁힐 수 있다.

---

## 6) 성공 기준(Acceptance Criteria / KPI)
- **Citation rate**: 전체 응답 중 95% 이상이 근거 인용 포함
- **Retrieval hit@5**: 골든셋 기준 0.75 이상(튜닝 후 0.85 목표)
- **Latency (p95)**: 2초 이내(로컬 기준, 측정 기준은 README에 명시)
- **Fail-safe**: 근거 부족 시 단정 금지(추가 질문/보류) 100%

---

## 7) UX / 응답 정책
### 응답 포맷
- 요약 답변(3~5문장)
- 근거 인용 2~3개: `[doc_title] - snippet`
- 필요 시 follow-up 질문 1개

### 정책
- **근거 없으면 단정 금지**
- 근거가 상충하면 “상충”을 명시하고 `effective_date/버전` 기준으로 최신 우선

---

## 8) 시스템 설계(High-level)

### 8.1 구성요소
- API: **FastAPI**
- Indexing: 문서 로더 → Chunking → Embedding → **FAISS**
- Agent Orchestration: **LangGraph**
- Storage:
  - FAISS index(벡터)
  - DocStore(JSON/SQLite): chunk text + metadata + version

### 8.2 문서/메타데이터 스키마(예시)
- `doc_id`, `title`, `category` (refund/promo/inventory/cs)
- `store_type` (cafe/convenience/apparel)
- `version`, `valid_from`, `valid_to`
- `chunk_id`, `chunk_text`, `source_path`

### 8.3 Retrieval 파이프라인(MVP)
1) query normalize  
2) metadata filter(optional): store_type/category/effective_date  
3) FAISS similarity topK(기본 8)  
4) (옵션) light rerank(간단 스코어링) 후 top 5 사용  

---

## 9) LangGraph 설계(Agent 그래프)

### 9.1 State 정의(예시)
- `question: str`
- `filters: {store_type?: str, category?: str, effective_date?: str}`
- `retrieved_chunks: list`
- `draft_answer: str`
- `citations: list`
- `final_answer: str`
- `need_clarification: bool`
- `errors: list`

### 9.2 Nodes (MVP 최소 5개)
1. **Parse/Classify Node**
   - 질문에서 `category/store_type/effective_date` 추론(없으면 null)
2. **Retrieve Node (Tool)**
   - FAISS 검색 + 메타필터 적용
3. **Generate Node**
   - 답변 초안 생성(근거 인용 포맷 강제)
4. **Grounding Check Node**
   - 인용이 충분한지/내용 불일치가 없는지 간단 검증
   - 실패 시: `need_clarification=True` 또는 재검색 루프(1회 제한)
5. **Finalize Node**
   - 최종 응답 + follow-up 질문(필요 시)

> LangGraph 채택 이유: RAG는 단발 호출보다 **상태/분기/루프(재검색, 추가 질문)**가 핵심이라 그래프가 적합

---

## 10) API 스펙(MVP)

### POST `/chat`
#### Request
```json
{
  "question": "환불 규정 알려줘",
  "store_type": "cafe",
  "category": "refund",
  "effective_date": "2025-12-01"
}
