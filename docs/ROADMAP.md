# Roadmap

What's been built, what's next, and the reasoning behind each decision. This serves as both a technical changelog for the RAG pipeline and a decision log — the kind of thing I wish more open-source projects maintained.

---

## What's Been Done

The core RAG pipeline is significantly beyond tutorial-grade at this point. Here's what's implemented and why each piece exists.

### Semantic Chunking

Custom sentence-level chunking using NLTK tokenisation + embedding cosine similarity to detect topic boundaries. Not the LangChain off-the-shelf splitter — this one actually groups sentences by meaning.

- Configurable via `chunking_strategy`, `semantic_breakpoint_percentile`, `semantic_min_chunk_size`
- Falls back to `RecursiveCharacterTextSplitter` for any groups that exceed `chunk_size`
- Produces ~2,354 coherent chunks from 35 speeches (vs ~1,082 with naive fixed-size splitting)

**Why it matters:** Fixed-size chunking cuts mid-sentence and mid-thought. Semantic chunks preserve complete ideas, which directly improves embedding quality and retrieval relevance.

### Three-Layer RAG Guardrails

A pipeline that prevents the system from hallucinating or returning garbage:

1. **Pre-retrieval validation** — Rejects empty/trivially short queries before any compute is spent
2. **Post-retrieval relevance filtering** — Cross-encoder logits are sigmoid-normalised to 0–1 scores; results below a configurable threshold (default 0.01) are dropped. Fetches 2× candidates for filtering headroom
3. **Post-generation grounding verification** — Token-overlap heuristic between the generated answer and retrieved context. If the overlap is too low, a caveat is appended

Response metadata exposes everything: `guardrails.relevance_filtered`, `grounding_score`, `grounding_passed`. 32 dedicated tests.

### Query Rewriting

LLM-powered query cleaning that sits between guardrails Layer 1 and search. Fixes typos, corrects spelling, and expands abbreviations — but deliberately does *not* add synonyms or broaden scope (that was actually degrading retrieval quality for well-formed queries, so I dialled it back to conservative cleaning only).

- Uses the existing LLM provider at `temperature=0.0` for deterministic rewrites
- Safety guards: empty-query passthrough, error fallback to original, rejection of suspiciously long rewrites (>5× original length)
- The rewritten query drives search; the original is preserved for entity extraction and answer generation
- 16 dedicated tests

### Extended Chunk Metadata

Filename-based metadata extraction: `BattleCreekDec19_2019.txt` → `location: "Battle Creek"`, `date: "2019-12-19"`, `year: 2019`. Metadata flows through the full pipeline — stored in ChromaDB, propagated through search results, surfaced in LLM source labels and API responses. Handles CamelCase splitting, hyphenated names, all 35 filenames. 13 tests.

### Hybrid Search (Semantic + BM25 + Cross-Encoder)

Dense MPNet embeddings (768d) combined with BM25 keyword matching (70/30 weighting) and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`). The cross-encoder is the precision layer — it takes the top candidates and re-scores them with a more compute-intensive model.

### Pluggable LLM Providers

Factory pattern with lazy imports. Swap between Gemini, OpenAI, and Anthropic by changing `LLM_PROVIDER` in your env. Abstract base class ensures a consistent interface. Only the provider you're using gets imported.

### CI/CD Pipeline

Nine GitHub Actions workflows: `python-tests.yml`, `python-lint.yml`, `python-typecheck.yml`, `security-audit.yml`, `markdown-lint.yml`, `build-push-docker.yml`, `deploy-docs.yml`, `deploy-azure.yml`, `deploy-render.yml`. Tests, lint, and security are required gates; type-check is informational.

### Also Done

- **Modular RAG architecture** — Dedicated components for search, confidence, entity analysis, and document loading
- **Multi-factor confidence scoring** — Weighted combination of retrieval quality, consistency, coverage, and entity presence
- **Component testing** — 66%+ coverage, 237 tests
- **Type safety** — Pydantic models for all data structures
- **Production logging** — JSON output for prod, coloured pretty-print for dev

### Response Caching

Redis-backed caching with automatic fallback to in-memory cache when Redis is unavailable. Eliminates redundant LLM calls for repeated queries.

- **Redis primary** — `redis:7-alpine` container with persistence, 128MB max memory, LRU eviction
- **MemoryCache fallback** — Thread-safe in-memory cache for development or when Redis is unavailable
- **Deterministic cache keys** — SHA-256 hash of normalised query (lowercase, stripped whitespace) + top_k
- **TTL support** — Configurable expiration via `CACHE_TTL_SECONDS` (default: 1 hour)
- **Cache statistics** — Hit/miss tracking, hit rate, backend info exposed via `get_cache_stats()`
- **Response metadata** — Cached responses marked with `cached: true` and `cache_key` in API responses

**Configuration:**

```bash
CACHE_ENABLED=true
CACHE_REDIS_HOST=redis
CACHE_TTL_SECONDS=3600
```

The cache layer sits at the top of the RAG pipeline — before any search, reranking, or LLM calls. Cache invalidation is available via `clear_cache()` on the RAG service. 25 dedicated tests.

### Enhanced Named Entity Recognition (NER)

Replaced the original capitalisation-heuristic entity extraction with proper NER using spaCy `en_core_web_sm`.

- **spaCy `en_core_web_sm`** (~12 MB, CPU-only) — handles multi-word entities, entity-type classification, and edge cases the old word-split approach could not
- **Relevant labels** — filters to politically meaningful entity types: `PERSON`, `ORG`, `GPE`, `NORP`, `FAC`, `EVENT`, `LAW`, `PRODUCT`, `WORK_OF_ART`; drops noise like `DATE`, `CARDINAL`, `MONEY`
- **`EntityMatch` model** — new Pydantic model carrying `text` and `label` fields so callers can distinguish "Donald Trump (PERSON)" from "Washington (GPE)"
- **`extract_entities_with_types()`** — primary new method returning `List[EntityMatch]`; `extract_entities()` delegates to it and strips labels for backward compatibility
- **Graceful fallback** — if spaCy is not installed or the model is missing the analyzer automatically falls back to the original capitalisation heuristics, emitting `label="UNKNOWN"`. No configuration change required; the rest of the pipeline is unaffected
- **Lazy model loading** — spaCy pipeline loaded on first use; subsequent calls reuse the cached object
- **Optional dependency group** — `uv sync --group ner` installs spaCy. Production Docker image includes it via `--group ner` in `uv export`
- **17 new tests** covering the spaCy path, fallback path, `EntityMatch`, label filtering, and lazy-load mechanics (42 entity-analyzer tests total)

**Configuration:**

```yaml
# configs/production.yaml
models:
  ner_model_name: "en_core_web_sm"
  ner_enabled: true
```

```bash
# Environment override
NER_ENABLED=false  # disable NER, use heuristics only
```

---

## What's Next

### HyDE (Hypothetical Document Embeddings)

The next retrieval improvement. Instead of embedding the user's short query directly, generate a hypothetical answer first and embed *that*. The hypothetical document's embedding is much closer to actual speech chunks than a 5-word question.

```text
User: "What about the wall?"  →  5 tokens
HyDE: "Trump discussed building a border wall..."  →  30+ tokens, closer to actual chunks
```

The final answer is still grounded in *real* retrieved chunks — HyDE only affects the search vector.

**Effort:** Medium — touches `search_engine.py` and needs an LLM call per query.

### Enhanced NER

~~Current entity extraction uses capitalisation heuristics. Proper NER with spaCy or a HuggingFace model would catch multi-word entities, disambiguate, and handle edge cases.~~

**Done** — see the Enhanced NER section in *What's Been Done* above.

### Topic Modelling

BERTopic over the full speech corpus for automatic theme discovery. Different from the existing per-request topic extraction — this would be a corpus-level analysis.

### Prompt Engineering

Few-shot examples and chain-of-thought prompting in the RAG answer generation. Low effort, potentially high impact on answer quality.

---

## Considered and Removed

Some items were on the roadmap but removed after evaluation:

| Item | Why It Was Removed |
| --- | --- |
| **GPU Acceleration** | No CUDA available; project runs on Azure free tier. Not viable. |
| **Model Quantisation** | Not a bottleneck at this scale — 35 documents, CPU inference is fast enough |
| **Async Processing** | Over-engineered for the current corpus size |
| **Alternative Embeddings** | Adds API cost/external dependency for marginal gain over MPNet |
| **Fine-tuned Embeddings** | Requires significant compute and training data we don't have |

---

## Adding to This Roadmap

If you're contributing or have ideas:

1. **Context** — What problem does it solve?
2. **Technical approach** — What tools/libraries? Where in the codebase?
3. **Effort** — Small (hours), Medium (days), Large (weeks)
4. **Trade-offs** — What's the cost of adding this?
