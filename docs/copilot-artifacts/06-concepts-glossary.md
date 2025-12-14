# Concepts Glossary: Quick Reference Guide

## Your Complete Technical Dictionary for Portfolio Presentations

---

## Core AI/ML Concepts

### Artificial Intelligence (AI)

**Definition:** Computer systems that can perform tasks typically requiring human intelligence.

**In Your Project:** The overall category of technology you're using (LLMs, embeddings, clustering).

**Interview Talking Point:** "My project uses several AI techniques including natural language processing, machine learning clustering, and large language models."

---

### Natural Language Processing (NLP)

**Definition:** AI field focused on understanding and generating human language.

**In Your Project:**

- Q&A system understands questions
- Sentiment analysis interprets emotions
- Topic analysis extracts themes

**Interview Talking Point:** "I use NLP to process political speech transcripts and extract insights through question answering, sentiment analysis, and topic modeling."

---

### Machine Learning (ML)

**Definition:** AI systems that learn patterns from data without explicit programming.

**In Your Project:**

- FinBERT learned sentiment patterns from financial news
- RoBERTa learned emotion patterns from social media
- MPNet learned semantic similarity from text pairs
- KMeans learns cluster centers from embeddings

**Interview Talking Point:** "The project leverages pre-trained ML models like FinBERT and RoBERTa, which learned from millions of text examples."

---

### Deep Learning

**Definition:** ML using neural networks with multiple layers.

**In Your Project:** All transformer models (BERT, RoBERTa, MPNet, Gemini) are deep learning models.

**Technical Detail:** FinBERT has 110 million parameters across 12 transformer layers.

---

### Transformer

**Definition:** Neural network architecture using self-attention mechanisms, foundation of modern NLP.

**In Your Project:**

- MPNet (embeddings)
- FinBERT (sentiment)
- RoBERTa (emotions)
- Gemini (generation)

**Interview Talking Point:** "The project uses transformer-based models which revolutionized NLP by understanding context bidirectionally."

---

## RAG (Retrieval-Augmented Generation)

### RAG

**Definition:** AI pattern that retrieves relevant documents before generating answers.

**Why Important:** Grounds LLM responses in your data, reducing hallucinations.

**Your Implementation:**

1. User asks question
2. Search vector database for relevant speeches
3. LLM generates answer from retrieved context

**Interview Talking Point:** "I implemented RAG to ensure answers are grounded in actual speech transcripts rather than generic knowledge."

---

### Embeddings / Vector Embeddings

**Definition:** Dense numerical representations of text that capture semantic meaning.

**Technical:** 768-dimensional float arrays for your MPNet model.

**Example:**

```text
"economy" → [0.042, -0.13, 0.28, ..., 0.15]  (768 numbers)
"jobs"    → [0.038, -0.11, 0.31, ..., 0.18]  (similar values = similar meaning)
```

**Interview Talking Point:** "I use MPNet to convert text into 768-dimensional embeddings that represent semantic meaning in vector space."

---

### Vector Database

**Definition:** Database optimized for storing and searching high-dimensional vectors.

**Your Implementation:** ChromaDB with HNSW indexing.

**Why Not Regular Database:** Regular databases can't efficiently find "similar" vectors in 768-dimensional space.

**Interview Talking Point:** "ChromaDB enables fast semantic search by indexing 768-dimensional embeddings with HNSW algorithm."

---

### Semantic Search

**Definition:** Search based on meaning rather than exact keyword matches.

**Example:**

```text
Query: "job creation"
Matches: "employment growth", "hiring increases", "new positions"
```

**How It Works:** Convert query to embedding, find similar embeddings in vector database.

---

### BM25 (Best Matching 25)

**Definition:** Probabilistic keyword-based search algorithm.

**In Your Project:** Combined with semantic search in 70/30 weighted hybrid.

**Why Both:** Semantic finds conceptual matches, BM25 finds exact term matches. Best of both worlds.

**Interview Talking Point:** "I use hybrid search combining semantic similarity (70%) with BM25 keyword matching (30%) for robust retrieval."

---

### Cosine Similarity

**Definition:** Metric measuring similarity between vectors (0 = unrelated, 1 = identical).

**Formula:**

$$\text{similarity} = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \cdot ||\vec{B}||}$$

**In Your Project:** ChromaDB uses cosine similarity to rank search results.

**Example:**

```text
query_embedding · document_embedding = 0.85 → Very relevant
query_embedding · document_embedding = 0.32 → Less relevant
```

---

### HNSW (Hierarchical Navigable Small World)

**Definition:** Graph-based algorithm for approximate nearest neighbor search.

**Why Important:** Makes vector search fast even with thousands of documents.

**Technical:** Builds hierarchical graph connecting similar vectors for efficient traversal.

**Interview Talking Point:** "ChromaDB uses HNSW indexing to search 768-dimensional space in milliseconds rather than seconds."

---

### Chunking

**Definition:** Breaking long documents into smaller segments.

**Your Implementation:**

- RAG: Split speeches into paragraphs for granular retrieval
- Sentiment: 510 tokens per chunk for model limits

**Why Necessary:** Models have maximum input length (512 tokens for BERT-based models).

---

### Context Window

**Definition:** Maximum text length a model can process at once.

**Examples:**

- FinBERT: 512 tokens
- RoBERTa: 512 tokens
- Gemini 1.5 Flash: 1 million tokens

**Interview Talking Point:** "I chunk long speeches to fit within BERT's 512-token context window, then aggregate results."

---

## Sentiment Analysis

### Sentiment Analysis Overview

**Definition:** Determining emotional tone (positive/negative/neutral) of text.

**Your Approach:** Multi-model ensemble (FinBERT + RoBERTa + Gemini).

**Output Example:**

```text
Sentiment: Positive (85%)
Emotions: Joy (60%), Surprise (25%)
Interpretation: "Optimistic tone celebrating economic success"
```

---

### FinBERT

**Full Name:** Financial BERT

**Definition:** BERT model fine-tuned on financial news for sentiment analysis.

**Your Use Case:** Analyzes political/economic language effectively.

**Technical:**

- 110 million parameters
- Fine-tuned on 10,000+ financial texts
- 3 classes: positive, negative, neutral

**Interview Talking Point:** "I use FinBERT because political speeches often discuss economic topics, where FinBERT excels."

---

### RoBERTa

**Full Name:** Robustly Optimized BERT Pretraining Approach

**Definition:** Improved BERT variant with better training methodology.

**Your Model:** `SamLowe/roberta-base-go_emotions` (emotion classification)

**Technical:**

- 82 million parameters
- Distilled from larger model
- 7 emotion classes

**Emotions Detected:**

1. Joy
2. Anger
3. Sadness
4. Fear
5. Surprise
6. Disgust
7. Neutral

---

### Multi-Model Ensemble

**Definition:** Combining predictions from multiple models for better results.

**Your Implementation:**

1. FinBERT → Overall sentiment
2. RoBERTa → Specific emotions
3. Gemini → Human-readable interpretation

**Why Better:** Each model has different strengths; combination is more robust.

**Interview Talking Point:** "I use an ensemble approach: FinBERT for sentiment, RoBERTa for emotions, and Gemini for interpretation, giving comprehensive analysis."

---

## Topic Analysis

### Topic Modeling / Topic Analysis

**Definition:** Automatically discovering themes in text collections.

**Your Approach:** Semantic clustering with KMeans + LLM labeling.

**Output Example:**

```text
Topic 1: "Economic Policy" (120 mentions)
  Keywords: economy, jobs, market, growth
  Snippet: "The economy is booming..."

Topic 2: "Border Security" (85 mentions)
  Keywords: immigration, border, wall, security
  Snippet: "We're securing our border..."
```

---

### Clustering

**Definition:** Grouping similar items together without predefined labels.

**In Your Project:** Group semantically similar keywords into topic clusters.

**Algorithm:** KMeans (unsupervised learning).

---

### KMeans

**Definition:** Clustering algorithm that partitions data into K clusters by minimizing distance to cluster centers.

**How It Works:**

1. Initialize K random cluster centers
2. Assign each point to nearest center
3. Update centers to mean of assigned points
4. Repeat until convergence

**Your Configuration:** Auto-determine K (3-6 clusters) based on keyword count.

**Interview Talking Point:** "I use KMeans to cluster keyword embeddings, automatically discovering 3-6 main topics per speech."

---

### Unsupervised Learning

**Definition:** ML learning patterns without labeled training data.

**In Your Project:** KMeans finds topics without being told what topics exist.

**Contrast:** Supervised learning needs labels (e.g., "this text is about economy").

---

### MPNet (Multi-Perspective Network)

**Model:** `sentence-transformers/all-mpnet-base-v2`

**Definition:** State-of-the-art sentence embedding model.

**Technical:**

- 768-dimensional embeddings
- Trained on 1+ billion sentence pairs
- Best general-purpose embedding model (as of 2024)

**Why You Use It:** Captures semantic similarity better than older models (Word2Vec, GloVe).

**Interview Talking Point:** "I use MPNet for embeddings because it's currently the best general-purpose model for semantic similarity."

---

### Semantic Similarity

**Definition:** How close two pieces of text are in meaning (not just words).

**Example:**

```text
High Similarity:
- "job creation" ≈ "employment growth"
- "border security" ≈ "immigration control"

Low Similarity:
- "economy" ≠ "border"
```

**How Measured:** Cosine similarity of embeddings.

---

## Large Language Models (LLMs)

### LLM (Large Language Model)

**Definition:** Neural network trained on massive text datasets to understand and generate human language.

**Examples:**

- Gemini (Google)
- GPT-4 (OpenAI)
- Claude (Anthropic)

**Your Usage:**

1. Generate answers in Q&A
2. Interpret sentiment results
3. Label topic clusters
4. Summarize main themes

---

### Gemini

**Definition:** Google's multimodal LLM family.

**Your Models:**

- **gemini-1.5-flash**: Fast, cheap (development)
- **gemini-1.5-pro**: More capable (production)

**Technical:**

- 1 million token context window
- Multimodal (text, images, video)
- Fast inference

**Interview Talking Point:** "I use Gemini 1.5 Flash as my default LLM for cost-effective, high-quality text generation."

---

### Prompt Engineering

**Definition:** Crafting inputs to LLMs to get desired outputs.

**Your Techniques:**

1. **Clear instructions** ("Answer using ONLY the context")
2. **Role assignment** ("You are a helpful AI assistant")
3. **Few-shot examples** (show examples of desired output)
4. **Output constraints** ("2-3 sentences")

**Example:**

```python
prompt = f"""You are a topic labeling expert.

EXAMPLES:
Keywords: economy, jobs → Label: "Economic Policy"
Keywords: border, wall → Label: "Border Security"

YOUR TASK:
Keywords: {keywords}
Label:"""
```

---

### Temperature

**Definition:** LLM parameter controlling randomness (0 = deterministic, 1 = creative).

**Your Usage:**

- **0.2-0.3**: Topic labels, factual answers (want consistency)
- **0.5**: Q&A answers (balanced)
- **0.7**: Sentiment interpretations (want nuance)

**Interview Talking Point:** "I tune temperature based on task: low (0.3) for factual answers, higher (0.7) for interpretive analysis."

---

### Tokens

**Definition:** Pieces of text (words, subwords, or characters) that LLMs process.

**Rough Estimate:** 1 token ≈ 0.75 words (English)

**Example:**

```text
"Hello world" = 2 tokens
"Economy" = 2 tokens ("Econ" + "omy")
"AI" = 1 token
```

**Why Important:** API costs and limits based on token count.

---

### Hallucination

**Definition:** When LLM generates plausible-sounding but factually incorrect information.

**Your Solution:** RAG grounds answers in actual speech text, reducing hallucinations.

**Example Without RAG:**

```text
Q: "What did Trump say about Mars colonization?"
A: "He proposed a $100B Mars program" ← Hallucination (not in speeches)
```

**Example With RAG:**

```text
Q: "What did Trump say about Mars colonization?"
A: "I don't have information about Mars in these speeches" ← Grounded
```

---

## Software Engineering

### API (Application Programming Interface)

**Definition:** Interface for programs to communicate.

**Your Implementation:** FastAPI REST API

**Endpoints:**

- `POST /qa/ask` - Ask questions
- `POST /sentiment/analyze` - Analyze sentiment
- `POST /topics/analyze` - Extract topics

---

### REST (Representational State Transfer)

**Definition:** Architectural style for web APIs using HTTP methods.

**Your API:**

```text
POST /qa/ask
Request: {"question": "What about jobs?", "top_k": 5}
Response: {
  "answer": "...",
  "confidence": 0.85,
  "sources": [...]
}
```

---

### Async / Asynchronous

**Definition:** Non-blocking execution allowing multiple operations concurrently.

**Your Usage:**

- FastAPI async endpoints
- Parallel LLM calls
- Concurrent search operations

**Code Example:**

```python
# Sequential (slow)
result1 = llm.generate(prompt1)  # Wait 2s
result2 = llm.generate(prompt2)  # Wait 2s
# Total: 4s

# Async (fast)
results = await asyncio.gather(
    llm.generate(prompt1),  # Both start simultaneously
    llm.generate(prompt2)
)
# Total: 2s
```

---

### Dependency Injection

**Definition:** Design pattern where objects receive dependencies rather than creating them.

**Example:**

```python
# Bad: Creates own dependencies
class RAGService:
    def __init__(self):
        self.search_engine = SearchEngine()  # Tightly coupled

# Good: Receives dependencies
class RAGService:
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine  # Injected
```

**Benefits:** Easier testing (inject mocks), more flexible.

---

### Strategy Pattern

**Definition:** Design pattern allowing algorithms to be selected at runtime.

**Your Implementation:** Pluggable LLM providers

```python
# Strategy interface
class LLMProvider(ABC):
    def generate(self, prompt: str) -> str:
        pass

# Concrete strategies
class GeminiProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...

# Use any strategy
provider = GeminiProvider()  # Or OpenAIProvider()
llm_service = LLMService(provider=provider)
```

**Interview Talking Point:** "I use the strategy pattern for LLM providers, making it easy to swap Gemini for OpenAI without changing business logic."

---

### Factory Pattern

**Definition:** Design pattern for creating objects without specifying exact class.

**Your Implementation:**

```python
provider = LLMFactory.create_provider(
    provider_name="gemini",  # Or "openai", "claude"
    api_key=api_key
)
```

**Benefit:** Centralized creation logic, easy to add new providers.

---

### Repository Pattern

**Definition:** Abstraction layer for data access.

**Your Implementation:** `DocumentRepository` abstracts ChromaDB access.

**Benefit:** Business logic doesn't know data storage details.

---

### Docker

**Definition:** Platform for packaging applications in containers.

**Your Project:** `docker-compose.yml` for consistent deployment.

**Benefits:**

- Same environment everywhere
- Easy dependency management
- Isolated from host system

---

### CI/CD (Continuous Integration/Continuous Deployment)

**Definition:** Automated testing and deployment pipeline.

**Best Practice:** Run tests on every commit, auto-deploy on merge to main.

---

## Performance & Optimization

### Latency

**Definition:** Time between request and response.

**Your System:**

- Q&A: 2-4 seconds
- Sentiment: 3-5 seconds
- Topics: 3-5 seconds

**Breakdown:** Embedding (500ms) + Search (200ms) + LLM (1-2s) + Processing (500ms)

---

### Caching

**Definition:** Storing results to avoid recomputation.

**Your Opportunities:**

```python
@lru_cache(maxsize=128)
def embed_text(text: str):
    return model.encode(text)  # Cache embeddings
```

**Benefit:** Instant response for repeated queries.

---

### Lazy Loading

**Definition:** Loading resources only when needed.

**Your Implementation:**

```python
@property
def finbert(self):
    if self._finbert is None:
        self._finbert = load_finbert_model()  # Load on first use
    return self._finbert
```

**Benefit:** Faster startup, lower memory if model unused.

---

### Batch Processing

**Definition:** Processing multiple items together for efficiency.

**Example:**

```python
# Slow
embeddings = [model.encode(text) for text in texts]

# Fast
embeddings = model.encode(texts)  # Batch encode
```

**Benefit:** GPU can process multiple items in parallel.

---

## Data & Metrics

### Precision

**Definition:** Of retrieved documents, what fraction is relevant?

**Formula:**

$$\text{Precision} = \frac{\text{Relevant Retrieved}}{\text{Total Retrieved}}$$

**Example:** Retrieved 10 docs, 7 relevant → Precision = 70%

---

### Recall

**Definition:** Of all relevant documents, what fraction was retrieved?

**Formula:**
$$\text{Recall} = \frac{\text{Relevant Retrieved}}{\text{Total Relevant}}$$

**Example:** 20 relevant docs exist, retrieved 7 → Recall = 35%

---

### F1 Score

**Definition:** Harmonic mean of precision and recall.

**Formula:**
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Use:** Balances precision and recall.

---

### Confidence Score

**Definition:** How certain the system is about its answer.

**Your Calculation:** Weighted average of 4 factors

1. Similarity score (40%)
2. Keyword overlap (25%)
3. Answer completeness (15%)
4. Entity match (20%)

**Example:** 0.85 confidence = High quality answer

---

## Testing

### Unit Test

**Definition:** Test individual components in isolation.

**Example:**

```python
def test_confidence_calculation():
    calc = ConfidenceCalculator()
    score = calc.calculate({"similarity": 0.8})
    assert score == 0.8
```

---

### Integration Test

**Definition:** Test multiple components working together.

**Example:**

```python
def test_rag_pipeline():
    result = rag_service.answer_question("What about jobs?")
    assert "answer" in result
    assert result["confidence"] > 0
```

---

### Mock

**Definition:** Fake object used in testing to isolate components.

**Example:**

```python
mock_llm = Mock()
mock_llm.generate.return_value = "Test answer"
service = RAGService(llm_service=mock_llm)
```

**Benefit:** Test without hitting real APIs (faster, free, deterministic).

---

## Quick Reference: Acronyms

| Acronym | Full Name | What It Means |
|---------|-----------|---------------|
| **AI** | Artificial Intelligence | Computer systems mimicking human intelligence |
| **API** | Application Programming Interface | Interface for program communication |
| **BERT** | Bidirectional Encoder Representations from Transformers | Foundation transformer model |
| **BM25** | Best Matching 25 | Keyword search algorithm |
| **CI/CD** | Continuous Integration/Continuous Deployment | Automated testing/deployment |
| **HNSW** | Hierarchical Navigable Small World | Fast vector search algorithm |
| **LLM** | Large Language Model | Neural network for language understanding/generation |
| **ML** | Machine Learning | AI that learns from data |
| **MPNet** | Multi-Perspective Network | State-of-art embedding model |
| **NLP** | Natural Language Processing | AI field for human language |
| **RAG** | Retrieval-Augmented Generation | Retrieve then generate pattern |
| **REST** | Representational State Transfer | Web API architecture |
| **RoBERTa** | Robustly Optimized BERT | Improved BERT variant |

---

## Interview Cheat Sheet

### "Explain your tech stack in 30 seconds"

> "I built an NLP system using FastAPI for the backend, ChromaDB for vector storage, and transformer models like MPNet for embeddings and FinBERT for sentiment analysis. The system uses RAG to answer questions about political speeches, with Gemini LLM generating coherent responses from retrieved context. I implemented hybrid search combining semantic and keyword approaches, and use multi-model ensembles for robust sentiment analysis."

### "What's the most complex part?"

> "The RAG pipeline was most complex—balancing semantic vs keyword search weights, calculating confidence scores from multiple factors, and engineering prompts to prevent LLM hallucinations while generating helpful answers."

### "How do you ensure quality?"

> "Multi-layered approach: hybrid search for better retrieval, confidence scoring to flag uncertain answers, multi-model ensemble for sentiment, and RAG to ground LLM responses in actual data. Plus comprehensive testing with unit and integration tests."

### "What would you improve?"

> "I'd add: (1) caching for common queries, (2) A/B testing for search weight tuning, (3) user feedback loop to improve retrieval, (4) fine-tuned embeddings on political text, (5) streaming responses for better UX."

---

**You're now ready to confidently discuss any technical aspect of your project!** 🎓
