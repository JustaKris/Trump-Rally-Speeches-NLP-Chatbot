# Trump Speeches NLP Chatbot

[![Tests](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-tests.yml/badge.svg)](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-tests.yml)
[![Linting](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-lint.yml/badge.svg)](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-lint.yml)
[![Security](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/security-audit.yml/badge.svg)](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/security-audit.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full-stack NLP platform built from scratch — retrieval-augmented generation with hybrid search, multi-model sentiment analysis, and AI-powered topic extraction, all wrapped in a FastAPI service with pluggable LLM providers (Gemini, OpenAI, Claude), semantic document chunking, and automated deployment pipelines.

The code is the resume. If you want to see how I think about software, architecture, and ML engineering — you're in the right place.

## What's Inside

### RAG Pipeline

This isn't a LangChain tutorial copy-paste. It's a modular pipeline where each component (search, confidence scoring, entity analysis, document loading, guardrails, query rewriting) has its own responsibilities, tests, and clean interfaces.

- **Hybrid Search** — Dense MPNet embeddings (768d) + BM25 keyword matching (70/30 weighting) + cross-encoder reranking for precision
- **Three-Layer Guardrails** — Pre-retrieval query validation → post-retrieval relevance filtering (sigmoid-normalised cross-encoder scores) → post-generation grounding verification. If no chunks pass the relevance gate, it says "I don't know" instead of hallucinating
- **Query Rewriting** — LLM-powered query cleaning (typos, abbreviations) before search. Conservative by design — no synonym expansion, no scope broadening. Deterministic rewrites at temperature=0.0
- **Semantic Chunking** — Custom sentence-level embedding similarity chunker (not LangChain's off-the-shelf splitter). NLTK tokenisation + cosine similarity with configurable percentile-based breakpoints and tail-merging. Produces ~2,354 coherent chunks from 35 speeches
- **Smart Confidence Scoring** — Multi-factor calculation: retrieval quality (40%), consistency (25%), coverage (20%), entity presence (15%)
- **Entity Analytics** — Extraction with sentiment analysis, co-occurrence tracking, and speech coverage mapping

### Sentiment Analysis

Three models working together: FinBERT for polarity, RoBERTa for emotion detection, and an LLM for contextual interpretation. Not just "positive/negative" — actual nuanced analysis with explanations.

### Topic Extraction

DBSCAN semantic clustering with sentence-transformers, LLM-generated topic labels and summaries, contextual snippets with keyword highlighting.

### The Engineering

- **Response Caching** — Redis-backed with in-memory fallback. Eliminates redundant LLM calls for repeated queries with configurable TTL and cache statistics
- **Pluggable LLM Providers** — Factory pattern with lazy imports. Swap Gemini/OpenAI/Anthropic by changing one env var
- **FastAPI Backend** — 12+ endpoints, async handling, Pydantic validation, dependency injection
- **CI/CD** — 9 GitHub Actions workflows: tests, lint, type-check, security, Docker, docs, Azure/Render deployment
- **Testing** — 191 tests, 66%+ coverage, parametrised test cases
- **Modern Python** — uv, Ruff, mypy, structured logging (JSON for prod, pretty for dev), Docker multi-stage builds

## Try It Live

The API is deployed on Azure and ready to explore:

- **[Interactive Web App](https://trump-speeches-nlp-chatbot.azurewebsites.net)** — Try the RAG system, sentiment analysis, and topic extraction
- **[API Docs (Swagger)](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs)** — Interactive API playground
- **[API Docs (ReDoc)](https://trump-speeches-nlp-chatbot.azurewebsites.net/redoc)** — Clean, readable documentation
- **[Full Documentation](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)** — Complete guides, architecture diagrams, and API reference

> **⚠️ Important - Cold Start Notice:**
>
> The app runs on **Azure Free Tier** hosting. Due to the large ML models (~2GB) and containerized deployment:
>
> - **Initial load (cold start):** 1-5 minutes when idle. You may need to refresh the page several times.
> - **AI-generated responses:** 30 seconds to 2 minutes for complex queries (LLM processing + embeddings).
> - **Subsequent requests:** Fast once warmed up (~2-5 seconds).
>
> **Recommended workflow:** Open the link, refresh every 30 seconds for a few minutes until the page loads successfully. Once loaded, the app is responsive.

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/rag/ask` | POST | Ask questions — returns AI-generated answers with confidence scores and source attribution |
| `/rag/search` | POST | Semantic search over indexed documents |
| `/rag/stats` | GET | Vector database statistics |
| `/rag/index` | POST | Index/re-index documents |
| `/analyze/sentiment` | POST | Multi-model sentiment analysis (FinBERT + RoBERTa + LLM) |
| `/analyze/topics` | POST | AI-powered topic extraction with semantic clustering |
| `/analyze/words` | POST | Word frequency analysis |
| `/analyze/ngrams` | POST | N-gram analysis |
| `/health` | GET | System health and service status |
| `/config` | GET | Public runtime configuration |
| `/diagnostics` | GET | Detailed diagnostics for troubleshooting |

Interactive docs at `/docs` (Swagger) and `/redoc` (ReDoc). Web UI at `/`.

## The Dataset

35 rally speech transcripts (2019–2020), 300,000+ words, indexed as ~2,354 semantic chunks in ChromaDB. Real-world political text with nuanced language — a good stress test for NLP.

## Get Started

### What You Need

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (modern Python package manager)
- An LLM API key (grab a free one from [Google Gemini](https://ai.google.dev/) — it's the default provider)
  - Or use [OpenAI](https://platform.openai.com/api-keys) / [Anthropic](https://console.anthropic.com/) if you prefer

### Setup

1. **Install dependencies**

   ```powershell
   uv sync
   ```

2. **Configure LLM Provider**

   The project supports multiple LLM providers with a model-agnostic configuration approach.

   **Option A: Google Gemini (Default)**

   Create a `.env` file in the project root:

   ```bash
   # LLM Provider Configuration
   LLM_PROVIDER=gemini
   LLM_API_KEY=your_gemini_api_key_here
   LLM_MODEL_NAME=gemini-2.0-flash-exp
   
   # Optional: Adjust LLM parameters
   LLM_TEMPERATURE=0.7
   LLM_MAX_OUTPUT_TOKENS=2048
   ```

   **Option B: OpenAI**

   ```powershell
   # Install OpenAI support
   uv sync --group llm-openai
   ```

   Update `.env`:

   ```bash
   LLM_PROVIDER=openai
   LLM_API_KEY=sk-your_openai_api_key_here
   LLM_MODEL_NAME=gpt-4o-mini
   LLM_TEMPERATURE=0.7
   LLM_MAX_OUTPUT_TOKENS=2048
   ```

   **Option C: Anthropic (Claude)**

   ```powershell
   # Install Anthropic support
   uv sync --group llm-anthropic
   ```

   Update `.env`:

   ```bash
   LLM_PROVIDER=anthropic
   LLM_API_KEY=sk-ant-your_anthropic_api_key_here
   LLM_MODEL_NAME=claude-3-5-sonnet-20241022
   LLM_TEMPERATURE=0.7
   LLM_MAX_OUTPUT_TOKENS=2048
   ```

   **Install All Providers:**

   ```powershell
   uv sync --group llm-all
   ```

3. **Start the FastAPI server**

   ```powershell
   uv run uvicorn speech_nlp.app:app --reload
   ```

4. **Access the application**
   - **Local:** <http://localhost:8000> (instant, recommended for testing)
   - **Azure (deployed):** <https://trump-speeches-nlp-chatbot.azurewebsites.net> *(Cold start: 1-5min, refresh periodically)*
   - **API Docs:** <https://trump-speeches-nlp-chatbot.azurewebsites.net/docs>
   - **ReDoc:** <https://trump-speeches-nlp-chatbot.azurewebsites.net/redoc>

### Try the RAG System

**Web Interface:** Navigate to the RAG tab and ask a question

**API Example:**

```powershell
curl -X POST http://localhost:8000/rag/ask `
  -H "Content-Type: application/json" `
  -d '{"question": "What was said about the economy?", "top_k": 5}'
```

**Python Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/rag/ask",
    json={"question": "What economic policies were discussed?", "top_k": 5}
)
print(response.json()["answer"])
```

### Alternative: Docker

**Note:** Add your Gemini API key to the Dockerfile or pass it as an environment variable.

### Run with Docker

1. **Build the Docker image**

   ```powershell
   docker build -t trump-speeches-nlp-chatbot .
   ```

2. **Run the container**

   ```powershell
   docker run -it --rm -p 8000:8000 trump-speeches-nlp-chatbot
   ```

3. **Or use Docker Compose** (Recommended — includes Redis for caching)

   ```powershell
   docker-compose up -d
   ```

   This starts the API server with Redis for response caching. The cache significantly improves response times for repeated queries.

### View Documentation Site (Optional)

The project includes comprehensive documentation built with MkDocs:

```powershell
# Install documentation dependencies
uv sync --group docs

# Serve documentation site locally (with live reload)
uv run mkdocs serve
```

Then open <http://localhost:8001> to browse the documentation with search and navigation.

**Build static site:**

```powershell
uv run mkdocs build
```

This generates a `site/` folder with the complete static documentation website.

### Explore Analysis Notebooks (Optional)

```powershell
# Install notebook dependencies (includes matplotlib, seaborn, plotly, etc.)
uv sync --group notebooks
uv run jupyter lab
```

Navigate to `notebooks/` to explore statistical NLP analysis and visualizations.

## Testing & Code Quality

```powershell
uv sync --group dev        # Install dev dependencies
uv run pytest              # Run all tests with coverage
uv run ruff check src/     # Lint
uv run ruff format src/    # Format
uv run mypy src/           # Type check
uv run bandit -r src/ -c pyproject.toml  # Security scan
```

CI/CD runs 9 GitHub Actions workflows on every push: tests (Python 3.11 + 3.12), lint, type-check, security scanning, Docker build, docs deployment, and Azure/Render deployment.

### CI/CD Pipeline

The project uses modular GitHub Actions workflows for continuous integration:

- ✅ **Automated Testing** on Python 3.11, 3.12 ([`python-tests.yml`](.github/workflows/python-tests.yml))
- ✅ **Code Quality** — Ruff linting and formatting ([`python-lint.yml`](.github/workflows/python-lint.yml))
- ✅ **Type Checking** — Mypy static analysis ([`python-typecheck.yml`](.github/workflows/python-typecheck.yml))
- ✅ **Security Scanning** — Bandit and pip-audit ([`security-audit.yml`](.github/workflows/security-audit.yml))
- ✅ **Documentation Linting** — Markdownlint ([`markdown-lint.yml`](.github/workflows/markdown-lint.yml))
- ✅ **Documentation Deployment** — Auto-deploy to GitHub Pages ([`deploy-docs.yml`](.github/workflows/deploy-docs.yml))
- ✅ **Docker Build and Push** — Automated image build and push to DockerHub ([`build-push-docker.yml`](.github/workflows/build-push-docker.yml))
- ✅ **Azure Deployment** — Deploy to Azure on push to main ([`deploy-azure.yml`](.github/workflows/deploy-azure.yml))

For detailed testing documentation, see the [Testing Guide](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/development/testing/).

## Project Structure

```text
src/speech_nlp/
├── app.py                       # Application entry point
├── constants.py                 # Application-wide constants
├── exceptions.py                # Custom exception classes
├── security.py                  # API key + input sanitization
├── api/
│   ├── chatbot.py               # RAG endpoints
│   ├── analysis.py              # NLP analysis endpoints
│   ├── health.py                # Health, config, diagnostics
│   └── dependencies.py          # Dependency injection
├── config/
│   ├── settings.py              # Pydantic settings
│   └── logging.py               # Structured logging (JSON + color)
├── schemas/
│   ├── requests.py              # API request models
│   └── responses.py             # API response models
├── services/
│   ├── llm/                     # Pluggable LLM providers
│   │   ├── base.py              #   Abstract interface
│   │   ├── factory.py           #   Factory + lazy imports
│   │   ├── gemini.py            #   Google Gemini
│   │   ├── openai.py            #   OpenAI GPT (optional)
│   │   └── anthropic.py         #   Anthropic Claude (optional)
│   ├── rag/                     # RAG pipeline components
│   │   ├── service.py           #   RAG orchestrator
│   │   ├── search.py            #   Hybrid search
│   │   ├── guardrails.py        #   Three-layer guardrails
│   │   ├── rewriter.py          #   LLM query cleaning
│   │   ├── confidence.py        #   Multi-factor scoring
│   │   ├── entities.py          #   Entity extraction + analytics
│   │   ├── chunking.py          #   Semantic chunking + metadata
│   │   └── models.py            #   Internal domain models
│   └── analysis/
│       ├── sentiment.py         # Multi-model sentiment
│       ├── topics.py            # Semantic topic extraction
│       └── text.py              # Word frequency, n-grams
├── utils/
│   ├── embeddings.py            # Embedding utilities
│   ├── formatting.py            # Response formatting
│   ├── io.py                    # Data loading
│   └── text.py                  # Text cleaning
├── templates/index.html         # Web UI
└── static/                      # CSS + images

tests/                           # 191 tests, 66%+ coverage
data/                            # Speech transcripts + ChromaDB
configs/                         # YAML env configs (dev/staging/prod)
docs/                            # MkDocs documentation site
```

## Documentation

**[Full Documentation Site](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)** — Guides, architecture diagrams, and reference docs.

To run locally: `uv sync --group docs && uv run mkdocs serve --dev-addr localhost:8001`

## License

MIT License — see [LICENSE](LICENSE). Speech transcripts are publicly available data.

---

**Kristiyan Bonev** — [GitHub](https://github.com/JustaKris) · [LinkedIn](https://www.linkedin.com/in/kristiyan-bonev-profile/) · [k.s.bonev@gmail.com](mailto:k.s.bonev@gmail.com)
