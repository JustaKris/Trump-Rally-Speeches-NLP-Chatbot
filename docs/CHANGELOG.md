# Changelog

All notable changes to the Trump Speeches NLP Chatbot.

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
