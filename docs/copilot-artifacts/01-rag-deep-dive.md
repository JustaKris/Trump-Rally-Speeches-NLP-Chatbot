# RAG Deep Dive: Retrieval-Augmented Generation Explained

## Understanding Your Q&A System from First Principles

---

## What Is RAG? The Core Concept

### RAG = Retrieval-Augmented Generation

Think of RAG as giving an AI a textbook before asking it to answer questions. Instead of relying solely on what the AI learned during training, you first **retrieve** relevant information from a knowledge base, then use that information to **augment** the AI's **generation** of an answer.

### The Problem RAG Solves

**Problem with Traditional LLMs:**

- LLMs (like GPT, Gemini) have a **knowledge cutoff date**
- They can't access proprietary or private data
- They sometimes **hallucinate** (make up plausible-sounding but false information)
- Their knowledge becomes stale over time

**Example:**

Ask GPT-4 "What did Trump say about immigration in his September 2020 Ohio rally?"

- GPT-4's training data likely doesn't include that specific speech
- It might hallucinate an answer based on general knowledge
- It can't cite sources

**RAG Solution:**

1. **Store** your speeches in a searchable database
2. When user asks a question, **search** for relevant speech excerpts
3. **Provide** those excerpts as context to the LLM
4. LLM generates answer **grounded in actual data**
5. System can **cite sources** (which speech, which paragraph)

---

## How RAG Works: The Three-Step Process

### Step 1: Indexing (Done Once at Startup)

**Goal:** Transform your documents into a searchable format

**What Happens:**

1. **Load Documents** — Read all 35 speech files from disk
2. **Chunk Text** — Split each speech into manageable pieces
3. **Generate Embeddings** — Convert each chunk into a 768-number vector
4. **Store in Database** — Save vectors + metadata in ChromaDB

**Why Chunking?**

- Speeches are long (average 8,000 words)
- LLMs have token limits (can't process entire speech at once)
- Smaller chunks provide more precise retrieval
- Chunks preserve coherent context

**Your Settings:**

- **Chunk Size:** 2048 characters (~512-768 tokens)
- **Overlap:** 150 characters (prevents splitting sentences awkwardly)

**Example:**

```text
Original Speech (10,000 words):
├── Chunk 1: Characters 0-2048 (with metadata: source=OhioSep21_2020.txt, chunk_id=1)
├── Chunk 2: Characters 1898-3946 (150 char overlap with Chunk 1)
├── Chunk 3: Characters 3796-5844
└── ... (total ~1082 chunks across all 35 speeches)
```

### Step 2: Search (Happens on Every Query)

**Goal:** Find the most relevant chunks for a given question

**What Happens:**

1. **Embed Query** — Convert question into same 768d vector format
2. **Search Database** — Find chunks with similar vectors
3. **Rank Results** — Order by relevance (similarity score)
4. **Return Top K** — Get the 5 most relevant chunks (configurable)

**Example:**

```text
Question: "What was said about the economy?"

Embedding: [0.23, -0.45, 0.67, ..., 0.12] (768 numbers)

Search Results:
1. OhioSep21_2020.txt, chunk_23 (similarity: 0.91)
   "The economy is booming like never before..."
2. TulsaJun20_2020.txt, chunk_45 (similarity: 0.89)
   "We've created millions of jobs..."
3. BattleCreekDec19_2019.txt, chunk_12 (similarity: 0.87)
   "Our economic policies are working..."
```

### Step 3: Generation (Happens After Search)

**Goal:** Use retrieved context to generate an informed answer

**What Happens:**

1. **Format Context** — Combine top chunks into a single context string
2. **Create Prompt** — Build instruction for LLM with context
3. **Call LLM** — Send to Gemini/GPT/Claude
4. **Parse Response** — Extract generated answer
5. **Add Metadata** — Include confidence, sources, entity stats

**Example Prompt Sent to LLM:**

```text
You are an expert analyst answering questions about political speeches.

CONTEXT FROM SPEECHES:
[1] OhioSep21_2020.txt:
"The economy is booming like never before. We've added 11 million jobs..."

[2] TulsaJun20_2020.txt:
"Manufacturing jobs are returning to America. We're rebuilding our economy..."

[3] BattleCreekDec19_2019.txt:
"Our economic policies have created the best economy in history..."

QUESTION: What was said about the economy?

Provide a comprehensive answer based ONLY on the context above. Cite sources.
```

**LLM Response:**
"The speeches emphasize economic success, highlighting job creation (11 million jobs mentioned in Ohio speech), manufacturing growth (Tulsa speech), and claims of historical economic performance (Battle Creek speech). The messaging focuses on economic prosperity as a key achievement."

---

## The Vector Database: ChromaDB Explained

### What Is a Vector Database?

**Traditional Database:**
Stores data in rows and columns, searches using exact matches or SQL queries.

**Vector Database:**
Stores data as high-dimensional vectors (arrays of numbers), searches using mathematical similarity.

### Why Vectors?

**The Core Insight:**
Words with similar meanings should have similar vector representations.

**Example:**

```text
"king"    → [0.5, 0.8, 0.2, ...]
"queen"   → [0.48, 0.79, 0.19, ...] (very similar to "king")
"economy" → [0.1, -0.3, 0.9, ...] (very different from "king")
```

**Similarity Calculation:**
Use **cosine similarity** to measure how "close" two vectors are:

- 1.0 = identical
- 0.5 = somewhat similar
- 0.0 = completely different

### ChromaDB in Your Project

**Why ChromaDB?**

- **Easy to use** — Minimal setup, SQLite-backed
- **Fast** — Efficient HNSW indexing for approximate nearest neighbor search
- **Persistent** — Data survives restarts (saved to `data/chromadb/`)
- **Pythonic** — Native Python library, no separate server needed

**What's Stored:**

```python
{
  "id": "speech_35_chunk_23",
  "embedding": [0.23, -0.45, ..., 0.12],  # 768 numbers
  "metadata": {
    "source": "OhioSep21_2020.txt",
    "chunk_index": 23,
    "text": "The economy is booming..."
  }
}
```

**Key Operations:**

1. **Add Documents** — `collection.add(embeddings=..., metadatas=..., ids=...)`
2. **Search** — `collection.query(query_embeddings=..., n_results=5)`
3. **Get Stats** — `collection.count()` returns total indexed chunks

---

## Embeddings: Turning Text Into Numbers

### What Are Embeddings?

**Definition:**
A learned representation of text as a dense vector of real numbers that captures semantic meaning.

**Analogy:**
Imagine plotting every word in 768-dimensional space where:

- Nearby words have similar meanings
- Distance represents semantic difference
- Directions represent relationships (king - man + woman ≈ queen)

### MPNet: Your Embedding Model

**Model:** `sentence-transformers/all-mpnet-base-v2`

**Specifications:**

- **Dimensions:** 768 (each text becomes a list of 768 numbers)
- **Type:** Sentence transformer (optimized for semantic similarity)
- **Training:** Trained on 1 billion+ sentence pairs
- **Use Case:** General-purpose semantic search

**Why MPNet?**

- **Excellent performance** on semantic similarity tasks
- **Balanced** between speed and quality
- **Well-maintained** by sentence-transformers library
- **Standard choice** for many RAG systems

**How It Works:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")

# Single sentence
text = "The economy is strong"
embedding = model.encode(text)
print(embedding.shape)  # (768,)

# Batch processing (more efficient)
chunks = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(chunks)
print(embeddings.shape)  # (3, 768)
```

### Embedding Properties

**Semantic Similarity:**

```text
"economy" ≈ "financial system" ≈ "market"
"immigration" ≈ "border security" ≈ "asylum policy"
```

**Distance Preserving:**
Texts with similar meaning have similar vectors (low distance).

**Contextual:**

The same word in different contexts gets different embeddings:

- "apple" (fruit) vs "Apple" (company) have different embeddings

---

## Hybrid Search: Best of Both Worlds

### The Problem with Pure Semantic Search

**Scenario:**
User searches for "climate change policy"

**Semantic Search Alone:**

- Finds: "environmental regulations", "carbon emissions", "green energy"
- **Misses:** Exact phrase "climate change" if semantically similar terms are used

**Problem:** Sometimes users want **exact term matches**, not just semantic similarity.

### BM25: Keyword Search

**BM25 = Best Match 25** (a ranking function)

**How It Works:**

1. **Term Frequency (TF):** How often does term appear in document?
2. **Inverse Document Frequency (IDF):** How rare is the term across all documents?
3. **Length Normalization:** Adjust for document length

**Formula (simplified):**

```text
score = IDF(term) × (TF(term) × (k+1)) / (TF(term) + k)
```

**Example:**

Query: "climate change"

- "climate" appears in 5/35 speeches (moderately rare)
- "change" appears in 30/35 speeches (very common)
- BM25 weights "climate" higher than "change"

### Hybrid Search in Your System

**Combination Strategy:**

```python
final_score = (0.7 × semantic_score) + (0.3 × bm25_score)
```

**Why 70/30?**

- Semantic search captures meaning (primary)
- BM25 ensures exact matches aren't missed (secondary)
- Weights are configurable based on use case

**Example:**

```text
Query: "Biden immigration policy"

Chunk A:
  Semantic: 0.85 (high - discusses immigration)
  BM25: 0.60 (moderate - contains "immigration" but not "Biden")
  Final: (0.7 × 0.85) + (0.3 × 0.60) = 0.595 + 0.180 = 0.775

Chunk B:
  Semantic: 0.70 (moderate - discusses Biden and borders)
  BM25: 0.95 (very high - contains "Biden" and "immigration")
  Final: (0.7 × 0.70) + (0.3 × 0.95) = 0.490 + 0.285 = 0.775

Chunk C:
  Semantic: 0.90 (very high - discusses immigration policy in detail)
  BM25: 0.40 (low - uses different terminology)
  Final: (0.7 × 0.90) + (0.3 × 0.40) = 0.630 + 0.120 = 0.750
```

Result: Chunk A and B tie for top spot, Chunk C is close behind.

### Cross-Encoder Reranking (Optional)

**What It Does:**
After hybrid search, optionally run a final "re-ranking" pass for maximum precision.

**How It Works:**

1. Get top 20 results from hybrid search
2. For each result, run a specialized model that scores query-document pairs
3. Re-sort based on cross-encoder scores
4. Return top 5 from re-ranked results

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Why Optional?**

- **Slower** — Processes each candidate individually
- **More accurate** — Better at judging relevance than dot product similarity
- **Trade-off** — Use when precision is critical, skip for speed

**Your Configuration:**

```python
use_reranking=True  # Enable for best quality (default in your system)
```

---

## Confidence Scoring: How Confident Is the Answer?

### Why Confidence Matters

**The Challenge:**
Not all answers are equally reliable. Some queries have clear, strong evidence in the corpus. Others are speculative or poorly supported.

**User Need:**
Users should know if the answer is:

- **High confidence** — Strong evidence, trust it
- **Medium confidence** — Some evidence, verify if critical
- **Low confidence** — Weak evidence, treat with skepticism

### Multi-Factor Confidence Calculation

Your system computes confidence using **four weighted factors**:

#### 1. Retrieval Quality (40% weight)

**Measures:** How semantically similar are the retrieved chunks to the query?

**Calculation:**

```python
retrieval_score = mean([chunk1_similarity, chunk2_similarity, ...])
```

**Example:**

- Average similarity: 0.91 → Excellent retrieval
- Average similarity: 0.50 → Poor retrieval

**Why 40%?**
This is the most important signal—if we didn't retrieve relevant chunks, the answer will be bad.

#### 2. Consistency (25% weight)

**Measures:** How consistent are the similarity scores?

**Calculation:**

```python
variance = variance([similarity_scores])
consistency = 1 - variance  # Invert so lower variance = higher consistency
```

**Example:**

- Scores: [0.90, 0.89, 0.91, 0.88] → Low variance → High consistency → Confident
- Scores: [0.90, 0.45, 0.72, 0.38] → High variance → Low consistency → Not confident

**Why This Matters:**
If all retrieved chunks are similarly relevant, we're confident. If scores vary wildly, we're uncertain.

#### 3. Coverage (20% weight)

**Measures:** How many supporting chunks did we retrieve?

**Calculation:**

```python
coverage = min(chunk_count / 10, 1.0)  # Normalized to 0-1
```

**Example:**

- 5 chunks: coverage = 0.5
- 10 chunks: coverage = 1.0
- 15 chunks: coverage = 1.0 (capped)

**Why This Matters:**
More evidence = more confidence. But returns diminish beyond 10 chunks.

#### 4. Entity Coverage (15% weight)

**Measures:** For entity queries, how well-represented is the entity in retrieved chunks?

**Calculation:**

```python
entity_coverage = chunks_with_entity / total_chunks
```

**Example:**

Query: "What was said about Biden?"

- Entity "Biden" appears in 5/5 chunks → entity_coverage = 1.0 → High confidence
- Entity "Biden" appears in 1/5 chunks → entity_coverage = 0.2 → Low confidence

**Why This Matters:**
If asking about Biden, but Biden rarely appears in results, the answer probably won't be good.

### Final Score Calculation

```python
combined_score = (
    0.40 × retrieval_score +
    0.25 × consistency +
    0.20 × coverage +
    0.15 × entity_coverage  # (only for entity queries)
)

if combined_score >= 0.7:
    confidence_level = "high"
elif combined_score >= 0.4:
    confidence_level = "medium"
else:
    confidence_level = "low"
```

### Confidence Explanation

**Human-Readable Output:**

```text
"Overall confidence is HIGH (score: 0.87) based on excellent semantic match 
(similarity: 0.91), very consistent results (consistency: 0.93), 5 supporting 
context chunks, 'Biden' mentioned in all retrieved chunks."
```

**Why Provide Explanation?**

- Transparency builds trust
- Users understand *why* confidence is high or low
- Helps users calibrate trust in the system

---

## Entity Analytics: Going Beyond Basic Q&A

### What Are Entities?

**Definition:**
Named entities are specific people, organizations, locations, or concepts mentioned in text.

**Examples:**

- **People:** Biden, Obama, Hillary, Bernie
- **Places:** Minneapolis, California, China
- **Organizations:** CNN, FBI, Democrat Party

### Entity Detection in Your System

**Simple Heuristic Approach:**

```python
# Extract capitalized phrases (simplified)
entities = [word for word in text.split() if word[0].isupper() and len(word) > 2]
```

**Why Not spaCy/Hugging Face NER?**

- Your corpus is political—entities are well-formed (proper nouns capitalized)
- Simple heuristic works well for this domain
- Faster than running a neural NER model
- Can upgrade to spaCy later if needed

### Entity Statistics

When a query mentions an entity, your system provides rich analytics:

#### 1. Mention Count

**What:** Total times entity appears across entire corpus

**How:**

```python
mention_count = sum([
    chunk.count(entity) 
    for chunk in all_chunks
])
```

**Example:**
"Biden" appears 524 times across all speeches

#### 2. Speech Coverage

**What:** Which speeches mention the entity

**How:**

```python
speeches_with_entity = [
    source 
    for chunk in all_chunks 
    if entity in chunk.text
    for source in chunk.source
]
```

**Example:**
Biden mentioned in 30 of 35 speeches (85.7% of corpus)

#### 3. Sentiment Toward Entity

**What:** Average sentiment in chunks containing the entity

**How:**

1. Find all chunks mentioning entity (up to 50 for performance)
2. Run FinBERT sentiment analysis on each
3. Convert scores to -1 (negative) to +1 (positive)
4. Average across all chunks

**Example:**

```python
chunks_with_biden = find_chunks_with("Biden")  # Returns 50 chunks
sentiments = [analyze_sentiment(chunk) for chunk in chunks_with_biden]
# sentiments = [-0.72, -0.55, -0.68, ..., -0.50]
average = mean(sentiments)  # -0.61
classification = "Negative" if average < -0.2 else "Neutral" if average < 0.2 else "Positive"
```

Result: "Negative (-0.61)" — Biden is discussed negatively

#### 4. Co-Occurrence Analysis

**What:** Words frequently appearing near the entity

**How:**

1. Extract contexts around entity (±100 characters)
2. Tokenize and clean
3. Filter stopwords
4. Count frequencies
5. Return top 5

**Example:**

```python
contexts = [
    "...Biden's socialist agenda is destroying...",
    "...Biden and the radical left want...",
    "...Biden's failure on the border...",
]

terms = extract_terms(contexts)  # ["socialist", "radical", "destroying", "failure", "border"]
top_5 = most_common(terms, 5)  # ["socialism", "weakness", "failure", "china", "corrupt"]
```

Result: Biden is associated with negative policy terms

---

## Component Architecture: How It's Built

### Modular Design Philosophy

**Problem with Monolithic Code:**

```python
# Bad: Everything in one function
def ask(question):
    # 500 lines of search, confidence, entities, LLM logic
    # Hard to test, hard to maintain, hard to extend
```

#### Solution: Separation of Concerns

```python
# Good: Each component has a single responsibility
DocumentLoader → loads and chunks documents
SearchEngine → handles hybrid retrieval
ConfidenceCalculator → computes confidence
EntityAnalyzer → extracts and analyzes entities
RAGService → orchestrates everything
```

### Component Breakdown

#### DocumentLoader (`services/rag/document_loader.py`)

**Responsibility:** Load documents and chunk them intelligently

**Key Methods:**

```python
def load_documents(self, directory: str) -> List[Document]:
    """Load all .txt files from directory."""
    
def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
    """Split documents into overlapping chunks."""
```

**Why Separate?**

- Easy to change chunking strategy
- Testable independently
- Can swap for different data sources (PDF, web, database)

#### SearchEngine (`services/rag/search_engine.py`)

**Responsibility:** Perform hybrid search with optional reranking

**Key Methods:**

```python
def search(self, query: str, top_k: int) -> List[SearchResult]:
    """Hybrid search combining semantic + BM25."""
    
def _semantic_search(self, query: str, top_k: int) -> List[Result]:
    """Vector similarity search in ChromaDB."""
    
def _bm25_search(self, query: str, top_k: int) -> List[Result]:
    """Keyword-based search using BM25."""
    
def _combine_scores(self, semantic: List, bm25: List) -> List:
    """Merge and rank by weighted combination."""
    
def _rerank(self, query: str, results: List) -> List:
    """Optional cross-encoder reranking."""
```

**Why Separate?**

- Search is complex, deserves its own module
- Easy to A/B test different search strategies
- Can optimize search without touching other code

#### ConfidenceCalculator (`services/rag/confidence.py`)

**Responsibility:** Calculate multi-factor confidence scores

**Key Methods:**

```python
def calculate_confidence(
    self, 
    results: List[SearchResult],
    entities: List[str]
) -> Dict:
    """Compute confidence score and explanation."""
    
def _compute_retrieval_quality(self, results: List) -> float:
    """Average similarity score."""
    
def _compute_consistency(self, results: List) -> float:
    """Inverse of score variance."""
    
def _compute_coverage(self, results: List) -> float:
    """Normalized chunk count."""
    
def _compute_entity_coverage(self, results: List, entities: List) -> float:
    """Percentage of chunks with entity."""
```

**Why Separate?**

- Confidence logic is complex (4 factors, weighting, thresholds)
- Easy to tune weights without affecting search
- Testable with mock search results

#### EntityAnalyzer (`services/rag/entity_analyzer.py`)

**Responsibility:** Extract entities and provide analytics

**Key Methods:**

```python
def extract_entities(self, text: str) -> List[str]:
    """Detect entities using capitalization heuristic."""
    
def get_entity_statistics(
    self,
    entity: str,
    collection: ChromaCollection
) -> Dict:
    """Comprehensive entity analytics."""
    
def _analyze_entity_sentiment(self, chunks: List) -> Dict:
    """Sentiment analysis on entity contexts."""
    
def _extract_associations(self, chunks: List, entity: str) -> List:
    """Co-occurring terms."""
```

**Why Separate?**

- Entity analytics is optional (only for entity queries)
- Complex (sentiment analysis, co-occurrence)
- Can upgrade to spaCy NER without changing RAG logic

#### RAGService (`services/rag_service.py`)

**Responsibility:** Orchestrate all components

**Key Methods:**

```python
def ask(self, question: str, top_k: int) -> Dict:
    """Main entry point for Q&A."""
    # 1. Search
    results = self.search_engine.search(question, top_k)
    # 2. Extract entities
    entities = self.entity_analyzer.extract_entities(question)
    # 3. Calculate confidence
    confidence = self.confidence_calculator.calculate_confidence(results, entities)
    # 4. Get entity stats (if entities found)
    entity_stats = self.entity_analyzer.get_entity_statistics(entities[0])
    # 5. Generate answer (if LLM available)
    answer = self.llm.generate(question, results)
    # 6. Return comprehensive response
    return {
        "answer": answer,
        "context": results,
        "confidence": confidence,
        "entity_statistics": entity_stats
    }
```

**Why This Design?**

- RAGService is a thin orchestrator—no complex logic
- Delegates to specialized components
- Easy to test (mock each component)
- Easy to extend (add new components)

---

## Performance & Optimization

### First Request Latency

**What Happens:**

- Download models (~1.5 GB) from HuggingFace
- Initialize ChromaDB
- Load embeddings into memory
- Index documents

**Time:** 30-60 seconds
**Frequency:** Once per environment (models cached)

### Subsequent Request Latency

**Typical Query:**

- Embedding generation: ~50ms
- Vector search: ~100ms
- BM25 search: ~50ms
- Score combination: ~10ms
- Reranking (optional): ~200ms
- LLM generation: ~1-2 seconds
- **Total: 1.5-2.5 seconds**

**Entity Query:**

- Above + entity extraction: ~10ms
- sentiment analysis (50 chunks): ~500ms
- co-occurrence: ~100ms
- **Total: 2.1-3.1 seconds**

### Optimization Opportunities

**Caching:**

- Cache entity statistics (computed once)
- Cache frequent query embeddings
- Use Redis for distributed cache

**Async Processing:**

- Entity analytics can run in background
- Sentiment analysis can be parallelized
- LLM call doesn't block search

**Index Optimization:**

- Use HNSW parameters for speed/accuracy trade-off
- Pre-filter metadata before vector search
- Batch embedding generation

---

## Testing Strategy

### Unit Tests

**What:** Test individual components in isolation

**Examples:**

```python
def test_confidence_calculator_high_confidence():
    """Test high confidence scenario."""
    results = [
        {"similarity": 0.91},
        {"similarity": 0.90},
        {"similarity": 0.89}
    ]
    confidence = calculator.calculate_confidence(results, [])
    assert confidence["confidence"] == "high"
    assert confidence["confidence_score"] >= 0.7

def test_entity_extraction():
    """Test entity detection."""
    text = "Biden and Trump discussed immigration"
    entities = analyzer.extract_entities(text)
    assert "Biden" in entities
    assert "Trump" in entities
```

### Integration Tests

**What:** Test end-to-end workflows

**Examples:**

```python
def test_rag_ask_integration():
    """Test complete RAG pipeline."""
    rag = RAGService()
    result = rag.ask("What was said about the economy?", top_k=5)
    
    assert "answer" in result
    assert len(result["context"]) <= 5
    assert result["confidence"] in ["high", "medium", "low"]
    assert "confidence_score" in result
```

### Current Coverage: 65%+

**Well-Tested:**

- Search engine (hybrid search, reranking)
- Confidence calculator (all factors)
- Entity analyzer (extraction, statistics)

**Could Improve:**

- LLM integration (mocked in tests)
- Error handling edge cases
- Performance regression tests

---

## Common Issues & Solutions

### Issue: Poor Search Results

**Symptoms:** Irrelevant chunks returned for query

**Diagnosis:**

1. Check similarity scores—are they below 0.5?
2. Is BM25 dominating? (Adjust weights)
3. Are chunks too large? (Reduce chunk_size)

**Solutions:**

- Increase `top_k` to get more candidates
- Adjust hybrid search weights
- Enable reranking for precision
- Check if query is too vague

### Issue: Low Confidence Scores

**Symptoms:** All answers rated "low" confidence

**Diagnosis:**

1. Check retrieval quality—are similarity scores low?
2. Is variance high? (Inconsistent results)
3. Are there too few chunks?

**Solutions:**

- Improve query clarity
- Add more documents to corpus
- Tune confidence weights
- Adjust confidence thresholds

### Issue: Entity Not Detected

**Symptoms:** Entity statistics missing for valid entities

**Diagnosis:**

1. Is entity capitalized in query?
2. Does entity appear in corpus?

**Solutions:**

- Ensure proper capitalization
- Check corpus for entity presence
- Consider upgrading to spaCy NER
- Add entity aliases

---

## Next Steps

**Now that you understand RAG, continue to:**

- **`02-sentiment-analysis-deep-dive.md`** — Learn multi-model sentiment analysis
- **`05-llm-integration.md`** — Understand how LLMs are integrated
- **`06-concepts-glossary.md`** — Quick reference for all terms

**Practice:**

- Explain RAG to someone in 2 minutes
- Draw the RAG pipeline on a whiteboard
- Identify the trade-offs in your architecture

**Interview Prep:**

- Why RAG vs fine-tuning?
- How does hybrid search improve results?
- What's the purpose of confidence scoring?
- How would you scale to millions of documents?

---

You now have a deep understanding of RAG! This is the foundation of your Q&A system. 🚀
