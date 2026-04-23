# Changelog

All notable changes to the Trump Speeches NLP Chatbot.

## [0.6.0] — Unreleased

### Added — Enhanced NER Integration

**Label-aware entity confidence scoring (`confidence.py`):**

- Added `LABEL_SPECIFICITY` dict mapping each spaCy entity type to a 0.45–0.95 weight. `PERSON` / `NORP` entities (e.g. "Trump", "Democrats") appear in virtually every political-speech chunk and carry a lower weight; rarer, query-specific labels (`LAW`, `EVENT`, `FAC`) carry a higher weight.
- Added `_calculate_entity_coverage_typed()` — per-entity chunk coverage scaled by label specificity. Prevents generic queries from inflating the entity-coverage component of the confidence score.
- `ConfidenceCalculator.calculate()` now accepts an optional `entity_matches: List[EntityMatch]` parameter; typed scoring is used when provided, with graceful fallback to the existing heuristic path.

**Typed `EntityMatch` through the RAG pipeline (`service.py`, `responses.py`):**

- `RAGService.ask()` now calls `extract_entities_with_types()` instead of `extract_entities()`, threading `List[EntityMatch]` through the pipeline end-to-end.
- `_no_relevant_info_response()` serialises each match as `{"text": ..., "label": ...}` in the response body.
- `RAGAnswerResponse` gains an optional `entities` field (`List[Dict[str, str]]`) so callers can inspect which named entities were detected in the question alongside their spaCy type labels.
- Entity statistics are now always fetched (without sentiment) for faster pipeline execution; sentiment can still be requested separately via `include_sentiment=True`.

### Enhanced — Web UI Tab Improvements

**Sentiment tab:**

- Added four quick-load sample buttons (Economy & Trade, Immigration, Military & Veterans, Media & Democrats) pre-populated with real excerpts from the Cincinnati Aug 2019 speech, removing the blank-slate friction that caused most visitors to skip the tab.

**Topics tab:**

- Added three quick-load speech buttons (Cincinnati Aug 2019, Minneapolis Oct 2019, Dallas Oct 2019) pre-populated with full-length excerpts. Topics extraction needs substantial text for DBSCAN to produce meaningful clusters; the buttons provide that with one click.

**Dataset tab:**

- Stats now auto-load on first tab visit (`statsLoaded` one-shot flag); the manual "Load Dataset Stats" button is removed.
- Speech list table added below the stat boxes, fetched concurrently with stats via `Promise.all`. Displays location, month/year, and a proportional word-count progress bar for all 35 speeches sorted by year then month. Uses the existing `/analyze/speeches/list` endpoint — no backend changes required.

---

## [0.5.0] — Unreleased

### Added — Response Caching

**Module: `src/speech_nlp/services/cache/`**

Redis-backed caching layer with automatic in-memory fallback:

- **`base.py`**: Abstract `CacheBackend` interface and high-level `CacheService` with hit/miss metrics
- **`redis.py`**: `RedisCache` with automatic `MemoryCache` fallback when Redis is unavailable
- Deterministic cache keys using SHA-256 hash of normalised query + top_k
- Configurable TTL (default: 1 hour) via `CACHE_TTL_SECONDS`
- Cache statistics: hits, misses, hit rate, backend info
- Response metadata includes `cached: true` and `cache_key` for cached responses

**Docker Integration:**

- Added `redis:7-alpine` service to `docker-compose.yml` with health checks and persistence
- 128MB max memory with LRU eviction policy
- Environment variables: `CACHE_ENABLED`, `CACHE_REDIS_HOST`, `CACHE_REDIS_PORT`, `CACHE_TTL_SECONDS`

**Testing:** 25 dedicated tests covering cache operations, TTL expiration, fallback behaviour, and RAG integration.

---

## [0.4.0] — April 2026

### Added — RAG Guardrails

**Module: `src/speech_nlp/services/rag/guardrails.py`**

Three-layer input validation pipeline running before retrieval:

- **Layer 1 — Topic Filter**: LLM-based relevance check. Blocks questions outside the speech corpus scope.
- **Layer 2 — Safety Filter**: Screens for harmful, manipulative, or deceptive intent.
- **Layer 3 — Quality Filter**: Ensures queries are specific enough to produce a useful answer.

Rejected queries return a structured response with a clear reason rather than a hallucinated answer.

### Added — Query Rewriting

**Module: `src/speech_nlp/services/rag/rewriter.py`**

LLM-powered query cleaner running between guardrails and retrieval:

- Fixes spelling, grammar, and ambiguous phrasing
- Resolves pronouns and implicit references ("he said" → "Trump said")
- Conservative by design — no synonym expansion, no scope broadening
- Falls back to the original query if the rewrite is worse or fails

### Enhanced — Semantic Chunking

**Module: `src/speech_nlp/services/rag/chunking.py`**

Replaced fixed-size text splitting with semantic boundary detection:

- Splits at natural discourse boundaries (topic shifts) rather than arbitrary character counts
- Each chunk is a self-contained thought rather than a mid-sentence fragment
- Extended chunk metadata: `speech_date`, `filename`, `char_start`, `char_end`, `chunk_index`
- Metadata enables date-range filtering, source attribution, and improved confidence scoring

### Enhanced — Documentation 2

- README reduced from ~630 to ~250 lines — merged three redundant feature sections, added complete API endpoint table (including previously undocumented `/health`, `/config`, `/diagnostics`), fixed inaccurate project tree
- `docs/index.md` reduced from ~230 to ~45 lines — removed duplicated curl examples, tech stack lists, and link lists that mirrored the sidebar
- Architecture reference updated: added RAGGuardrails and QueryRewriter components, updated pipeline Mermaid diagram, updated workflow to 11 steps
- Fixed portfolio/showcase language across all docs
- Fixed 3 broken doc links

---

## [0.3.0] — December 2025

### Added — Pluggable LLM Provider Architecture

**Modules: `src/speech_nlp/services/llm/`**

Unified interface for multiple LLM providers:

- **`base.py`**: Abstract `LLMProvider` interface
- **`factory.py`**: Factory pattern with lazy imports for optional dependencies
- **`gemini.py`**: Google Gemini (default, always installed)
- **`openai.py`**: OpenAI GPT (optional: `uv sync --group llm-openai`)
- **`anthropic.py`**: Anthropic Claude (optional: `uv sync --group llm-anthropic`)

Switch providers via three environment variables — no code changes required:

```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk-your_openai_key
LLM_MODEL_NAME=gpt-4o-mini
```

### Enhanced — Documentation

- Added MIT `LICENSE` file
- Added `docs/reference/faq.md`
- Updated all docs to reference Ruff (replacing Black/flake8/isort)
- Fixed markdown linting issues (MD032, MD031, MD040)

### Enhanced — Dark Mode UI

- Increased `max_tokens` for sentiment interpretation (800 → 2000) — fixes response truncation
- Custom scrollbar styling for dark mode
- Improved card contrast

---

## [0.2.0] — November 2025

### Added — Multi-Model Sentiment Analysis

**Module: `src/speech_nlp/services/analysis/sentiment.py`**

Three-model ensemble replacing the previous single-model approach:

- **FinBERT**: Political/financial sentiment (positive/negative/neutral)
- **RoBERTa-Emotion**: Six-emotion detection (anger, joy, fear, sadness, surprise, disgust)
- **LLM Provider**: Contextual interpretation — explains *why* the text received those scores

Response now includes emotion probabilities, sentiment confidence, and an AI-generated 2-3 sentence interpretation.

### Added — AI Topic Analysis with Semantic Clustering

**Module: `src/speech_nlp/services/analysis/topics.py`**

Replaced keyword frequency counting with semantic clustering:

- MPNet embeddings (768d) group related keywords into coherent topics
- KMeans clustering (3–6 clusters, auto-determined by content)
- LLM generates human-readable labels ("border", "wall", "immigration" → "Border Security")
- Contextual snippets show keywords in actual speech passages (±100 char windows)
- LLM summary provides 2-3 sentence thematic analysis
- Smart filtering removes common verbs and drops low-relevance clusters

### Added — Modular RAG Architecture

**Modules: `src/speech_nlp/services/rag/`**

Refactored monolithic RAG service into independently testable components:

- `search.py` — Hybrid search (semantic + BM25 + cross-encoder reranking)
- `confidence.py` — Multi-factor confidence scoring (retrieval quality, consistency, coverage, entity mentions)
- `entities.py` — Entity extraction, sentiment, and co-occurrence analytics
- `chunking.py` — Chunking with metadata
- `models.py` — Pydantic models for type-safe RAG operations

### Added — Production Logging

**Module: `src/speech_nlp/config/logging.py`**

- JSON format in production (compatible with Azure Application Insights, CloudWatch, ELK)
- Coloured output in development
- Auto-detects environment via `ENVIRONMENT` setting
- Suppresses noisy third-party library logs

### Added — Configuration Management

**Module: `src/speech_nlp/config/settings.py`**

Pydantic Settings-based configuration system:

- Type-safe validation with clear error messages on startup
- Full `.env` support for local and cloud deployment
- Covers LLM provider, RAG parameters, model names, data paths, and API config

### Fixed — ChromaDB Duplicate Warnings

Implemented chunk deduplication on indexing:

- Checks existing chunk IDs before embedding
- Eliminates 1000+ `WARNING: Add of existing embedding ID` messages on every query
- Re-indexing skips already-embedded chunks

---

## [0.1.0] — October 2025

Initial release:

- RAG Q&A via ChromaDB + MPNet embeddings + Gemini LLM
- Hybrid search: semantic + BM25 keyword matching
- Cross-encoder reranking
- Multi-factor confidence scoring
- Entity analytics with sentiment analysis
- FastAPI backend with 12+ endpoints
- Single-page web interface
- Docker multi-stage build with health checks
- GitHub Actions CI/CD (tests, lint, type-check, security, Docker, docs deploy)
- pytest suite with 50%+ coverage

**ML Models**: Gemini 2.5 Flash · FinBERT · all-mpnet-base-v2 · ms-marco-MiniLM

---

**Repository**: [GitHub](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot)  
**Documentation**: [GitHub Pages](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)  
**Maintainer**: Kristiyan Bonev
