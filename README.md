# Trump Speeches NLP Chatbot

[![Tests](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-tests.yml/badge.svg)](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-tests.yml)
[![Linting](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-lint.yml/badge.svg)](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/python-lint.yml)
[![Security](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/security-audit.yml/badge.svg)](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/actions/workflows/security-audit.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full-stack NLP application built from the ground up — combining retrieval-augmented generation, hybrid search, multi-model sentiment analysis, and AI-powered topic extraction into a production-ready FastAPI service. Features pluggable LLM providers (Gemini, OpenAI, Claude), comprehensive testing, and automated deployment pipelines.

## What's Inside

### The AI Stack

- **RAG Q&A System** — Natural language questions over 300,000+ words using ChromaDB vector storage, MPNet embeddings (768d), and hybrid search combining semantic similarity with BM25 keyword matching
- **Multi-Provider LLM Integration** — Pluggable architecture supporting Gemini, OpenAI GPT, and Anthropic Claude with a unified interface and lazy-loaded dependencies
- **Smart Confidence Scoring** — Multi-factor calculation weighing semantic similarity, answer consistency, context coverage, and entity presence
- **Entity Analytics Engine** — Extract entities with sentiment analysis, track co-occurrences, and map contextual associations across documents
- **Advanced Sentiment Analysis** — Ensemble approach combining FinBERT (sentiment), RoBERTa (emotion detection), and LLM-generated contextual interpretation
- **AI-Powered Topic Clustering** — DBSCAN semantic clustering with sentence-transformers + LLM-generated labels and summaries

### The Engineering Side

- **FastAPI Backend** — 12+ RESTful endpoints with async handling, dependency injection, Pydantic validation, and comprehensive error handling
- **Modular RAG Architecture** — Separated concerns with dedicated components for search, confidence calculation, entity analysis, and document loading
- **Production Deployment** — Multi-stage Docker builds, Azure Web App hosting, GitHub Actions CI/CD, automated testing and security scanning
- **Developer Experience** — Type hints throughout, structured logging (JSON for prod, pretty for dev), comprehensive documentation with MkDocs, and 65%+ test coverage
- **Modern Python Tooling** — uv for dependency management, Ruff for linting/formatting, pytest with parametrized tests, mypy for type checking

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

## How It Works

### RAG System Architecture

Built a modular question-answering system over 35 political speeches (300,000+ words) with these components:

**Core Services:**

- **`services/rag_service.py`** — Orchestrates RAG pipeline, manages ChromaDB, coordinates components
- **`services/llm/`** — Pluggable LLM abstraction layer supporting multiple providers (Gemini, OpenAI, Anthropic)
  - **`base.py`** — Abstract LLMProvider interface
  - **`factory.py`** — Factory pattern with lazy imports for optional providers
  - **`gemini.py`** — Google Gemini implementation
  - **`openai.py`** — OpenAI GPT models (optional dependency)
  - **`anthropic.py`** — Anthropic Claude models (optional dependency)

**Modular RAG Components** (`services/rag/`):

- **`search_engine.py`** — Hybrid search combining semantic (MPNet 768d), BM25 keyword, and cross-encoder reranking
- **`confidence.py`** — Multi-factor confidence scoring (retrieval quality, consistency, coverage, entity mentions)
- **`entity_analyzer.py`** — Entity extraction with sentiment analysis, speech coverage, and co-occurrence analytics
- **`document_loader.py`** — Document chunking (2048 chars, 150 overlap) with metadata tracking
- **`models.py`** — Pydantic data models for type-safe RAG operations

**RAG API Endpoints:**

- `POST /rag/ask` — Ask natural language questions with AI-generated answers
- `POST /rag/search` — Semantic search over indexed documents
- `GET /rag/stats` — Vector database statistics and health check
- `POST /rag/index` — Index/re-index documents

### 📝 Traditional NLP Endpoints

**API Layer** (`api/`):

- **`routes_chatbot.py`** — RAG question-answering endpoints
- **`routes_nlp.py`** — Traditional NLP analysis endpoints
- **`routes_health.py`** — Health checks and system status
- **`dependencies.py`** — Dependency injection for services

**Core Services:**

- **`services/nlp_service.py`** — Word frequency, n-gram analysis
- **`services/sentiment_service.py`** — Enhanced AI-powered sentiment analysis with emotion detection and contextual interpretation
- **`services/topic_service.py`** — AI-powered topic extraction with semantic clustering and LLM-generated summaries

**Additional Endpoints:**

- `POST /analyze/sentiment` — Multi-model sentiment analysis (FinBERT + RoBERTa emotions + Gemini interpretation)
- `POST /analyze/words` — Word frequency
- `POST /analyze/topics` — AI-powered topic extraction with semantic clustering and contextual analysis
- `POST /analyze/ngrams` — N-gram analysis

### 📊 Demo Dataset

35 political rally speech transcripts (2019-2020) totaling 300,000+ words — indexed in ChromaDB for RAG queries. The dataset demonstrates the system's ability to handle real-world political text with nuanced language.

### 🎨 Interactive Web Interface

Single-page application at the root (`/`) for testing all API features including the RAG Q&A system.

### 📓 Analysis Notebooks

Jupyter notebooks showcasing statistical NLP and exploratory data analysis techniques on the speech corpus.

## Technical Highlights

### AI/ML Engineering

- **RAG Systems**: End-to-end retrieval-augmented generation with ChromaDB vector database
- **LLM Integration**: Multi-provider abstraction layer with Gemini, OpenAI GPT, and Anthropic Claude support
- **Design Patterns**: Factory pattern with lazy imports, abstract base classes, dependency injection
- **Semantic Search**: Hybrid search combining dense embeddings (MPNet) and sparse retrieval (BM25)
- **Model Selection**: Cross-encoder reranking for precision optimization
- **Confidence Scoring**: Multi-factor confidence calculation for answer quality assessment
- **Transformer Models**: FinBERT sentiment analysis, sentence-transformers for embeddings
- **Entity Analytics**: NER-based entity extraction with sentiment and co-occurrence analysis

### Backend Engineering

- **API Development**: Production-grade FastAPI with 12+ RESTful endpoints, async request handling, modular route organization
- **Vector Databases**: ChromaDB with persistent storage, smart deduplication, efficient querying
- **Modular Architecture**: Separated concerns with dedicated services for search, confidence, entities, and document loading
- **Configuration Management**: Pydantic Settings with type validation and environment-based config
- **Production Logging**: JSON-formatted logs for cloud platforms, colored output for development
- **Error Handling**: Graceful fallbacks, comprehensive exception handling, structured error responses
- **Performance**: Efficient chunking (2048 chars), hybrid search, cross-encoder reranking
- **Type Safety**: Full Pydantic validation, Python 3.11+ type hints throughout
- **Dependency Injection**: Clean service initialization and testable architecture

### DevOps & Quality

- **Containerization**: Multi-stage Docker builds, non-root user, health checks
- **CI/CD**: GitHub Actions with automated testing, security scanning, code quality gates
- **Testing**: pytest with 65%+ coverage, unit and integration tests, parametrized test cases, modular component testing
- **Code Quality**: Ruff (linting + formatting), mypy (type checking) enforced via CI
- **Security**: pip-audit for vulnerability scanning, bandit for security analysis
- **Documentation**: Comprehensive MkDocs site, API docs via Swagger/ReDoc, inline docstrings
- **Observability**: Structured logging, health endpoints, startup configuration display

## What You Can Ask

The RAG system handles complex queries like:

- *"What economic policies were discussed in the speeches?"*
- *"How many times was Biden mentioned and in what context?"*
- *"What did the speaker say about immigration?"*
- *"Compare the themes between 2019 and 2020 speeches"*

It retrieves relevant chunks via hybrid search, analyzes entities and sentiment, calculates multi-factor confidence scores, and generates coherent answers with source attribution.

## What's New

### LLM Provider Abstraction (Recent)

- **Multi-Provider Support**: Pluggable architecture supporting Gemini, OpenAI GPT, and Anthropic Claude
- **Model-Agnostic Configuration**: Single configuration interface for all providers (`LLM_API_KEY`, `LLM_MODEL_NAME`)
- **Factory Pattern**: Lazy imports with optional dependencies for clean provider switching
- **Type-Safe Interface**: Abstract base class ensuring consistent LLM behavior across providers
- **Easy Extension**: Add new providers by implementing the `LLMProvider` interface

### Enhanced AI-Powered NLP Features

- **Multi-Model Sentiment Analysis**:
  - FinBERT for sentiment classification (positive/negative/neutral)
  - RoBERTa emotion detection (anger, joy, fear, sadness, surprise, disgust)
  - Gemini LLM for contextual interpretation explaining WHY the models produced their results
  - Clean UI with AI interpretation as the focal point, compact score visualization
- **Semantic Topic Extraction**:
  - Sentence-transformer embeddings for semantic similarity
  - DBSCAN clustering for intelligent topic grouping
  - LLM-generated cluster labels and comprehensive topic summaries
  - Interactive snippets showing topic occurrences in context
- **Centralized Configuration**: All NLP parameters (thresholds, model names, excluded verbs) configurable via environment variables

### Modular RAG Architecture (Code Refactoring)

- **Component Separation**: Extracted RAG functionality into dedicated, testable modules
  - `SearchEngine`: Hybrid search with semantic, BM25, and cross-encoder reranking
  - `ConfidenceCalculator`: Multi-factor confidence scoring
  - `EntityAnalyzer`: Entity extraction, sentiment, and co-occurrence analysis
  - `DocumentLoader`: Smart chunking with metadata tracking
- **Improved Testability**: 65%+ test coverage with component-level unit tests
- **Type Safety**: Pydantic models for all RAG data structures
- **Maintainability**: Clear separation of concerns, easier to extend and debug

### Production-Ready Logging

- **Dual-format logging**: JSON for production/cloud, colored for development
- **Automatic environment detection**: Zero configuration needed
- **Cloud-native**: Works with Azure Application Insights, CloudWatch, ELK stack
- **Smart filtering**: Suppresses noisy third-party library logs

### Professional Configuration Management

- **Type-safe settings**: Pydantic validation catches errors at startup
- **Environment-based**: Full `.env` file support for local and cloud deployment
- **Flexible**: Support for multiple LLM providers (Gemini, OpenAI, Anthropic)
- **Cloud-ready**: Seamless Azure/AWS deployment with environment variables

### Performance & Reliability

- **Smart deduplication**: Prevents re-indexing existing ChromaDB chunks
- **100x faster re-indexing**: Skip embedding computation for existing documents
- **Clean logs**: No more duplicate ID warnings or telemetry errors
- **Dependency injection**: Better testability and cleaner initialization

[See full changelog](docs/CHANGELOG.md)

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
   uv run uvicorn src.api:app --reload
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

3. **Or use Docker Compose**

   ```powershell
   docker-compose up -d
   ```

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

Built with testing in mind — comprehensive test suite with pytest, automated CI/CD, and modern Python tooling.

### Run Tests

```powershell
# Install dev dependencies
uv sync --group dev

# Run all tests with coverage
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run only integration tests
uv run pytest -m integration

# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
```

### Code Quality Checks

```powershell
# Format code
uv run ruff format src/

# Lint code
uv run flake8 src/

# Sort imports
uv run isort src/

# Type checking
uv run mypy src/

# Run all checks
uv run ruff format src/ && uv run ruff check src/ && uv run pytest
```

### CI/CD Pipeline

The project uses modular GitHub Actions workflows for continuous integration:

- ✅ **Automated testing** on Python 3.11, 3.12 ([`python-tests.yml`](.github/workflows/python-tests.yml))
- ✅ **Code quality** — Ruff linting and formatting ([`python-lint.yml`](.github/workflows/python-lint.yml))
- ✅ **Type checking** — Mypy static analysis ([`python-typecheck.yml`](.github/workflows/python-typecheck.yml))
- ✅ **Security scanning** — Bandit and pip-audit ([`security-audit.yml`](.github/workflows/security-audit.yml))
- ✅ **Documentation** — Auto-deploy to GitHub Pages ([`deploy-docs.yml`](.github/workflows/deploy-docs.yml))
- ✅ **Docker builds** — Automated image builds ([`build-push-docker.yml`](.github/workflows/build-push-docker.yml))

For detailed testing documentation, see [`docs/howto/testing.md`](docs/howto/testing.md).

## 📦 Core Dependencies

**RAG & LLM:**

- `chromadb` — Vector database for embeddings
- `google-generativeai` — Gemini LLM integration (default provider)
- `openai` — OpenAI GPT models (optional: `uv sync --group llm-openai`)
- `anthropic` — Anthropic Claude models (optional: `uv sync --group llm-anthropic`)
- `sentence-transformers` — MPNet embeddings (768d)
- `rank-bm25` — Keyword search for hybrid retrieval
- `langchain` — Text splitting utilities

**NLP & ML:**

- `transformers` + `torch` — FinBERT sentiment analysis, RoBERTa emotion detection
- `nltk` — Text preprocessing
- `scikit-learn` — DBSCAN clustering, cosine similarity

**API & Infrastructure:**

- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `pydantic` — Data validation

See `pyproject.toml` for complete dependency list.

## 💡 Project Structure

```text
Trump-Rally-Speeches-NLP-Chatbot/
│
├── src/                          # Production API code
│   ├── main.py                  # Application entry point
│   ├── api/                     # API routes and dependencies
│   │   ├── routes_chatbot.py    #    RAG endpoints
│   │   ├── routes_nlp.py        #    NLP analysis endpoints
│   │   ├── routes_health.py     #    Health checks
│   │   └── dependencies.py      #    Dependency injection
│   ├── services/
│   │   ├── rag_service.py       # ⭐ RAG orchestration
│   │   ├── llm/                 # ⭐ Pluggable LLM providers
│   │   │   ├── base.py          #    Abstract LLMProvider interface
│   │   │   ├── factory.py       #    Factory pattern with lazy imports
│   │   │   ├── gemini.py        #    Google Gemini implementation
│   │   │   ├── openai.py        #    OpenAI GPT (optional)
│   │   │   └── anthropic.py     #    Anthropic Claude (optional)
│   │   ├── rag/                 # Modular RAG components
│   │   │   ├── search_engine.py #    Hybrid search
│   │   │   ├── confidence.py    #    Confidence scoring
│   │   │   ├── entity_analyzer.py #  Entity extraction
│   │   │   ├── document_loader.py #  Document chunking
│   │   │   └── models.py        #    Pydantic data models
│   │   ├── sentiment_service.py # Multi-model sentiment
│   │   ├── topic_service.py     # Semantic topic extraction
│   │   └── nlp_service.py       # Word frequency, n-grams
│   ├── config/
│   │   └── settings.py          # Pydantic settings
│   ├── core/
│   │   ├── preprocessing.py     # Text preprocessing
│   │   └── logging_config.py    # Structured logging
│   ├── models/                  # ML models
│   │   └── sentiment.py         # FinBERT wrapper
│   └── utils/
│       └── data_loader.py       # Data loading utilities
│
├── data/
│   ├── Donald Trump Rally Speeches/  # 35 speech transcripts
│   └── chromadb/                     # Vector database persistence
│
├── src/
│   └── static/
│       └── index.html           # Web interface
│
├── notebooks/                   # Exploratory analysis
│   ├── 1. Word Frequency & Topics Analysis.ipynb
│   └── 2. Sentiment Analysis.ipynb
├── tests/                       # pytest test suite (65%+ coverage)
│   ├── conftest.py             # Test fixtures
│   ├── test_rag_integration.py # RAG system tests
│   ├── test_search_engine.py   # Hybrid search tests
│   ├── test_confidence.py      # Confidence scoring tests
│   └── ...                     # Additional test modules
├── docs/                        # Documentation (MkDocs site)
│   ├── index.md                # Docs homepage
│   ├── guides/                 # Getting started guides
│   ├── howto/                  # Task-oriented guides
│   ├── reference/              # Technical reference
│   ├── development/            # Development guides
│   └── copilot-artifacts/      # Deep-dive educational docs
├── .github/
│   └── workflows/              # CI/CD pipelines
│       ├── python-tests.yml    # Automated testing
│       ├── python-lint.yml     # Code quality checks
│       ├── security-audit.yml  # Security scanning
│       └── deploy-docs.yml     # Documentation deployment
├── configs/                     # Environment configurations
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # Container orchestration
├── mkdocs.yml                   # Documentation site config
├── pyproject.toml               # Dependencies & project metadata
├── .env.example                 # Environment variables template
└── LICENSE                      # MIT License
```

## 📚 Documentation

**📘 [Full Documentation Site](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)** — Complete guides, tutorials, and API reference

### View Documentation Locally (Optional)

```powershell
# Install docs dependencies
uv sync --group docs

# Serve docs with live reload (use port 8001 to avoid API conflict)
uv run mkdocs serve --dev-addr localhost:8001
```

Then open <http://localhost:8001> in your browser.

For more information on working with the documentation, see the [Documentation Guide](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/howto/documentation/).

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Copyright © 2025 Kristiyan Bonev and contributors

### Attribution

This repository is for educational and portfolio purposes. The speech transcripts are publicly available data used for demonstrative NLP analysis.

**Key Technologies:**

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — BERT-based models
- [FinBERT](https://huggingface.co/ProsusAI/finbert) — Sentiment analysis
- [sentence-transformers](https://www.sbert.net/) — MPNet embeddings
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [FastAPI](https://fastapi.tiangolo.com/) — Modern web framework
- [uv](https://docs.astral.sh/uv/) — Python package manager

---

## Get in Touch

### Kristiyan Bonev

- GitHub: [@JustaKris](https://github.com/JustaKris)
- LinkedIn: [Kristiyan Bonev](https://www.linkedin.com/in/kristiyan-bonev-profile/)
- Email: <k.s.bonev@gmail.com>

Built this from scratch to explore modern NLP techniques and production ML deployment. Dive into the code, try the API, or reach out if you want to chat about RAG systems, LLM integration, or anything else!
