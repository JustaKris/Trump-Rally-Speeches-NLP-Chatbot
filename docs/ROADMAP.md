# Project Roadmap & Future Improvements

This document tracks planned enhancements, technical debt, and ideas for improving the Trump Rally Speeches NLP Chatbot. Items are categorized by priority and complexity to help with implementation planning.

> **For Interview Prep:** Focus on understanding *why* each improvement matters and *how* it would be implemented

---

## 🎯 High Priority Improvements

### 1. Semantic Chunking for RAG

**Current State:**

- Using `RecursiveCharacterTextSplitter` with fixed-size chunks (2048 characters, 150 overlap)
- Splits text mechanically at character boundaries with fallback separators (`\n\n`, `\n`, `,` + ` `, etc.)
- No awareness of semantic boundaries or topic shifts

**Why This Matters:**
Semantic chunking improves RAG retrieval accuracy by creating chunks that preserve complete ideas and context. Fixed-size chunking can:

- Split sentences mid-thought, losing context
- Combine unrelated topics in one chunk, reducing precision
- Create chunks that don't align with how humans organize information
- Miss natural semantic boundaries in speeches (topic transitions, applause breaks, etc.)

**How Semantic Chunking Works:**

1. **Sentence-Level Embeddings**: Generate embeddings for each sentence in the document
2. **Similarity Analysis**: Calculate cosine similarity between consecutive sentences
3. **Breakpoint Detection**: Identify where similarity drops significantly (topic shift)
4. **Intelligent Grouping**: Group sentences into chunks at natural boundaries
5. **Size Constraints**: Still respect min/max chunk sizes, but prioritize semantic coherence

**Implementation Approach:**

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use the same embedding model as RAG for consistency
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95  # Top 5% similarity drops = breakpoints
)
```

**Benefits:**

- **Better Retrieval**: More relevant chunks matched to queries
- **Improved Context**: Chunks contain complete thoughts/topics
- **Higher Answer Quality**: LLM receives more coherent context
- **Speech-Specific**: Respects natural flow of political rally speeches

**Estimated Effort:** Medium (1-2 days)  
**Files to Modify:** `src/services/rag/document_loader.py`, config files  
**Testing Required:** Compare retrieval quality on sample queries before/after

**References:**

- [LangChain SemanticChunker](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker)

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

### 4. Cosine Similarity Threshold Filtering

**Current State:**

- Returns top-k results regardless of actual relevance scores
- Low-quality results can pollute the LLM context

**Why This Matters:**
If a query has no good matches, returning irrelevant chunks leads to:

- Hallucinated answers based on unrelated content
- Lower confidence that doesn't reflect actual uncertainty
- Wasted LLM tokens on useless context

**Implementation:**

```python
# In search_engine.py
SIMILARITY_THRESHOLD = 0.35  # Tuned based on evaluation

def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
    results = self._hybrid_search(query, top_k * 2)  # Get more candidates
    
    # Filter by threshold
    filtered = [r for r in results if r.similarity >= SIMILARITY_THRESHOLD]
    
    if not filtered:
        return []  # Return empty - triggers "no relevant info" response
    
    return filtered[:top_k]
```

**Benefits:** Prevents hallucination from irrelevant context, cleaner failure mode

**Effort:** Small (1-2 hours)  
**Files:** `src/services/rag/search_engine.py`, config

---

### 5. RAG Guardrails (Context Grounding + Safety)

**Current State:**

- LLM can potentially answer questions outside the provided context
- No explicit check for dangerous/harmful query intents
- Relies on LLM's own judgment for "I don't know" responses

**Why This Matters:**
Production RAG systems need:

1. **Context grounding** — Only answer from retrieved documents, not general knowledge
2. **Safety filtering** — Refuse dangerous requests (violence, illegal activities)
3. **Scope enforcement** — Stay within the domain (Trump speeches, not general politics)

**Implementation Approaches:**

```python
# 1. Strengthen the system prompt
GROUNDED_PROMPT = """Answer ONLY based on the provided context.
If the answer is not in the context, respond: 
"The available documents don't contain information about this topic."

NEVER use your general knowledge to answer.
NEVER make up quotes or statements not in the context."""

# 2. Add pre-flight query safety check
BLOCKED_INTENTS = ["how to make", "how to build weapon", "illegal"]

def is_safe_query(query: str) -> bool:
    query_lower = query.lower()
    return not any(blocked in query_lower for blocked in BLOCKED_INTENTS)

# 3. Post-generation verification
def verify_grounded(answer: str, chunks: List[str]) -> bool:
    """Check if answer content appears in the chunks."""
    # Use embedding similarity or substring matching
    ...
```

**Effort:** Medium (4-6 hours)  
**Files:** `src/services/llm/base.py`, `src/services/rag_service.py`, new guardrails module

---

### 6. Extended Chunk Metadata

**Current State:**

- Chunks have basic metadata: `source`, `chunk_index`, `total_chunks`
- Missing: date, location, temporal context

**Why This Matters:**
Rich metadata enables:

- **Temporal filtering**: "What did he say in 2020 vs 2019?"
- **Location-based queries**: "What topics in battleground states?"
- **Better source attribution**: Display dates in citations
- **Metadata filtering**: Pre-filter before semantic search

**Implementation:**

```python
# Parse filename: "BattleCreekDec19_2019.txt"
def extract_metadata(filename: str) -> dict:
    # Extract location and date from filename pattern
    match = re.match(r"(\w+)([A-Z][a-z]+\d+)_(\d{4})\.txt", filename)
    if match:
        return {
            "location": match.group(1),       # "BattleCreek"
            "date_str": match.group(2),        # "Dec19"
            "year": int(match.group(3)),       # 2019
            "source": filename
        }
```

**Benefits:** Enables metadata filtering, better citations, temporal analysis

**Effort:** Small (2-3 hours)  
**Files:** `src/services/rag/document_loader.py`

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
> "I'm planning to implement semantic chunking because our current fixed-size approach sometimes splits sentences mid-thought, which reduces RAG retrieval accuracy. By using sentence embeddings to detect topic boundaries, we can create more coherent chunks. I'd use LangChain's SemanticChunker with our existing MPNet model, and A/B test retrieval quality on a sample of 100 questions. Expected improvement: 10-15% better answer relevance based on similar implementations I've researched."

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
