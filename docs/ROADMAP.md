# Project Roadmap & Future Improvements

This document tracks planned enhancements, technical debt, and ideas for improving the Trump Rally Speeches NLP Chatbot. Items are categorized by priority and complexity to help with implementation planning.

> **For Interview Prep:** Focus on understanding *why* each improvement matters and *how* it would be implemented

---

## 🎯 High Priority Improvements

### ~~1. Semantic Chunking for RAG~~ ✅ Completed

> Implemented with custom sentence-level embedding similarity approach (no `langchain_experimental` dependency). See Completed section below for details. Files modified: `src/services/rag/document_loader.py`, `src/config/settings.py`, `src/services/rag_service.py`, `src/main.py`, all config YAMLs.

---

### 2. Query Rewriting with LangChain Prompt Templates

**Current State:**

- User queries are passed directly to the search engine without modification
- Typos, vague language, or ambiguous phrasing can hurt retrieval quality

**Why This Matters:**
Users don't always phrase questions optimally for retrieval. Query rewriting reformulates the input to be more search-friendly:

- Fix typos and grammatical errors
- Expand abbreviations and acronyms
- Add synonyms to broaden search coverage
- Decompose complex multi-part questions

**How It Works:**

1. User submits query → LLM rewrites to search-optimized version
2. Optionally generate multiple query variants for broader recall
3. Search with improved query → better chunk retrieval

**Implementation:**

```python
from langchain.prompts import PromptTemplate

query_rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are a search query optimizer. Rewrite the following question 
to be more effective for semantic search over political speech transcripts.

Original question: {question}

Rewritten query (fix typos, expand abbreviations, add relevant terms):"""
)
```

**Benefits:** 10-20% improvement in retrieval recall for poorly-phrased queries

**Effort:** Small (2-4 hours)  
**Files:** `src/services/rag_service.py`, new `src/services/rag/query_rewriter.py`

---

### 3. HyDE (Hypothetical Document Embeddings)

**Current State:**

- Query embedding is generated directly from user question
- Short questions may not embed close to relevant long-form document chunks

**Why This Matters:**
HyDE bridges the gap between short queries and long documents by generating a hypothetical answer first, then searching with *that* embedding:

- User question: "What about the wall?" (5 tokens)
- Hypothetical doc: "Trump discussed building a border wall to prevent illegal immigration..." (30+ tokens)
- The hypothetical doc embedding is much closer to actual speech chunks than the short query

**How HyDE Works:**

1. User asks question
2. LLM generates a *hypothetical* document that would answer the question
3. Embed the hypothetical document (not the question)
4. Search using that embedding → finds more relevant chunks
5. Ground the final answer in the *real* retrieved chunks

**Implementation:**

```python
def hyde_search(question: str, llm, embedding_model, collection) -> List[Chunk]:
    # Step 1: Generate hypothetical answer
    hypothetical = llm.generate(f"Write a paragraph answering: {question}")

    # Step 2: Embed the hypothetical document
    hyde_embedding = embedding_model.encode(hypothetical)

    # Step 3: Search with hypothetical embedding
    results = collection.query(query_embeddings=[hyde_embedding], n_results=5)

    return results
```

**Benefits:** Especially effective for short, vague, or keyword-style queries

**Effort:** Medium (4-8 hours)  
**Files:** `src/services/rag/search_engine.py`

---

### ~~4. Cosine Similarity Threshold Filtering~~ ✅ Completed

> Implemented as part of the three-layer RAG Guardrails system. See Completed section below for details. Files modified: `src/services/rag/models.py`, `src/services/rag/guardrails.py`, `src/services/rag_service.py`, `src/config/settings.py`, all config YAMLs.

---

### ~~5. RAG Guardrails (Context Grounding + Safety)~~ ✅ Completed

> Implemented as a three-layer guardrails pipeline (pre-retrieval validation → post-retrieval relevance filtering → post-generation grounding verification). See Completed section below for details. Files modified: `src/services/rag/guardrails.py` (new), `src/services/rag_service.py`, `src/services/llm/base.py`, `src/models/schemas.py`, `src/config/settings.py`, all config YAMLs.

---

### ~~6. Extended Chunk Metadata~~ ✅ Completed

> Implemented filename-based metadata extraction. See Completed section below for details. Files modified: `src/services/rag/document_loader.py`, `src/services/rag/models.py`, `src/services/rag_service.py`, `src/services/llm/gemini.py`.

---

## 🚀 Performance & Scalability (Big Hitters)

| Improvement | Impact | Effort | Notes |
| ------------ | -------- | -------- | ------- |
| **Model Quantization** | 4x memory reduction | Medium | INT8/ONNX for FinBERT/RoBERTa |
| **GPU Acceleration** | 5-10x faster inference | Medium | CUDA support, Docker GPU |
| **Response Caching** | Reduced LLM costs | Medium | Redis for common queries |
| **Async Processing** | Non-blocking API | Large | Celery for heavy analytics |

---

## 🧠 Advanced NLP (Big Hitters)

| Improvement | Impact | Effort | Notes |
| ------------ | -------- | -------- | ------- |
| **Enhanced NER** | Better entity extraction | Medium | spaCy or HuggingFace NER |
| **Topic Modeling** | Auto-discover themes | Medium | BERTopic over speeches |

---

## 🛡️ Production Readiness (Big Hitters)

| Improvement | Impact | Effort | Notes |
| ------------ | -------- | -------- | ------- |
| **API Auth + Rate Limiting** | Security | Medium | API keys, per-IP limits |
| **Observability** | Debugging | Medium | Prometheus + Grafana |
| **CI/CD Pipeline** | Automation | Medium | GitHub Actions |

---

## 🔬 Research & Experimentation

| Improvement | Impact | Effort | Notes |
| ------------ | -------- | -------- | ------- |
| **Alternative Embeddings** | Retrieval quality | Medium | Test OpenAI vs MPNet |
| **Fine-tuned Embeddings** | Domain accuracy | Large | Political speech domain |
| **Prompt Engineering** | Answer quality | Small | Few-shot, chain-of-thought |

---

## ✅ Completed (For Reference)

Already implemented:

- ✅ **Semantic Chunking for RAG**: Custom implementation using NLTK sentence tokenization + embedding-based cosine similarity breakpoint detection. Configurable via `chunking_strategy` ("semantic" or "fixed"), `semantic_breakpoint_percentile`, `semantic_min_chunk_size`, and `semantic_similarity_threshold`. Falls back to `RecursiveCharacterTextSplitter` for oversized groups. Produces ~2354 semantically coherent chunks from 35 speeches (vs ~1082 with fixed chunking).
- ✅ **Cosine Similarity Threshold Filtering + RAG Guardrails**: Three-layer guardrails pipeline integrated into the RAG service. **Layer 1 — Pre-retrieval validation:** rejects empty/too-short queries before any search. **Layer 2 — Post-retrieval relevance filtering:** sigmoid-normalized relevance scores (cross-encoder logits → 0-1 probability) with configurable threshold (default 0.4); fetches 2× candidates for filtering headroom, returns "no relevant info" if all results are below threshold. **Layer 3 — Post-generation grounding verification:** token-overlap heuristic between answer content words and retrieved context (stop-word filtered, configurable threshold 0.3); appends a caveat if grounding fails. Additionally strengthened the RAG prompt with explicit anti-hallucination instructions. Response schema extended with `guardrails` metadata (enabled, triggered, relevance_filtered, grounding_score, grounding_passed). Fully configurable per environment via `similarity_threshold`, `grounding_threshold`, `guardrails_enabled`. 32 dedicated tests.
- ✅ **Extended Chunk Metadata**: Filename-based metadata extraction (`extract_speech_metadata()`) parses `{Location}{MonthDay}_{Year}.txt` filenames into structured fields: `location` (CamelCase→spaced, hyphen-preserved), `year`, `month`, `day`, `date` (ISO format). Enriched metadata flows through the full pipeline — stored in ChromaDB, propagated via `ContextChunk.from_search_result()`, surfaced in LLM source labels and API response context. Handles all 35 speech filenames including edge cases (multi-word cities, hyphenated names). 13 dedicated tests.
- ✅ **Cross-Encoder Re-ranking**: Using `ms-marco-MiniLM-L-6-v2` for precision optimization
- ✅ **Hybrid Search (ANN + BM25)**: ChromaDB HNSW + BM25Okapi with 70/30 weighting
- ✅ **Basic Chunk Metadata**: source, chunk_index, total_chunks
- ✅ **LLM Provider Abstraction**: Pluggable Gemini/OpenAI/Anthropic support
- ✅ **Modular RAG Architecture**: Separated components (search, confidence, entities)
- ✅ **Component Testing**: 65%+ test coverage
- ✅ **Type Safety**: Pydantic models for all data structures
- ✅ **Production Logging**: JSON + colored output for different environments

---

## 📝 Notes for Interview Prep

**When discussing improvements, mention:**

1. **Why it matters** (business value, user experience, accuracy)
2. **Technical approach** (specific tools, algorithms, architectures)
3. **Trade-offs** (complexity vs. benefit, cost vs. performance)
4. **Measurable impact** (% improvement in accuracy, latency reduction, etc.)

**Example:**
> "I implemented semantic chunking to replace fixed-size character splitting. The approach uses NLTK sentence tokenization, embeds each sentence with our existing MPNet model, computes cosine similarity between consecutive sentences, and detects topic boundaries via percentile-based breakpoints. This produced ~2354 semantically coherent chunks from 35 speeches (vs ~1082 fixed chunks), with each chunk preserving complete ideas. The implementation avoids the `langchain_experimental` dependency by using a custom algorithm with configurable threshold and fallback to `RecursiveCharacterTextSplitter` for oversized groups."

---

## 🏷️ Contribution Guidelines

When adding items to this roadmap:

1. **Add context**: Why is this needed? What problem does it solve?
2. **Technical details**: How would it work? What tools/libraries?
3. **Effort estimate**: Small (hours), Medium (days), Large (weeks)
4. **Dependencies**: What needs to be done first?
5. **Success metrics**: How do we know it worked?

**Template:**

```markdown
### Feature Name

**Current State:** [What we have now]

**Why This Matters:** [Problem being solved]

**How It Works:** [Technical approach]

**Implementation:** [Code/tools/steps]

**Benefits:** [Concrete improvements]

**Effort:** [Time estimate]
**Files:** [What changes]
**Dependencies:** [Prerequisites]
```

---

*This roadmap is a living document. Feel free to add ideas, update priorities, or move items to "Completed" as work progresses!*
