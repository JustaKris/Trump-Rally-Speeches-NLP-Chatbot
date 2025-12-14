# Trump Speeches NLP Chatbot API — Technical Documentation

Production-ready AI/ML platform demonstrating enterprise-grade practices in natural language processing, retrieval-augmented generation, and modern backend development.

## Project Overview

This portfolio project demonstrates expertise in:

- **Advanced RAG Architecture** — Modular design with dedicated components for search, confidence scoring, entity analytics, and document loading
- **Hybrid Retrieval Systems** — Semantic search (MPNet 768d) combined with BM25 keyword matching and cross-encoder reranking
- **Production API Development** — FastAPI with 12+ RESTful endpoints, modular route organization, type-safe Pydantic models
- **Pluggable LLM Providers** — Support for Gemini, OpenAI, and Anthropic with unified interface and model-agnostic configuration
- **Entity Analytics** — Automated entity extraction with sentiment analysis and contextual associations
- **Professional DevOps** — Docker containerization, CI/CD pipelines, automated testing (65%+ coverage), code quality enforcement

## Documentation Structure

### Getting Started Guides

Quick setup and deployment:

- **[Quickstart Guide](guides/quickstart.md)** — Local setup and first API calls
- **[Deployment Guide](guides/deployment.md)** — Production deployment to Render, Azure, or Docker
- **[Documentation Guide](guides/documentation.md)** — Working with MkDocs and contributing to documentation
- **[CI/CD Local Testing](guides/ci-local-testing.md)** — Running all CI checks locally before pushing
- **[FAQ](faq.md)** — Frequently asked questions and troubleshooting

### Development Resources

Code quality and development standards:

- **[Code Style Guide](development/code-style.md)** — Python style guidelines and naming conventions
- **[Formatting Guide](development/formatting.md)** — Ruff formatting standards
- **[Linting Guide](development/linting.md)** — Code quality checks with Ruff and Mypy
- **[Testing Guide](development/testing.md)** — pytest practices, coverage requirements, and CI/CD
- **[Security Guide](development/security.md)** — Security scanning with Bandit and pip-audit
- **[Logging Setup](development/logging.md)** — Centralized logging configuration
- **[Markdown Linting](development/markdown-linting.md)** — Documentation formatting standards

### Technical Reference

In-depth technical documentation:

- **[System Architecture](reference/architecture.md)** — Component design, data flows, deployment patterns
- **[Configuration Reference](reference/configuration.md)** — Complete configuration options

**Core Features (The Big Three):**

- **[Q&A System](reference/qa-system.md)** — RAG-based question answering with semantic search, entity analytics, and confidence scoring
- **[Sentiment Analysis](reference/sentiment-analysis.md)** — Multi-model emotion detection with FinBERT, RoBERTa, and LLM interpretation
- **[Topic Analysis](reference/topic-analysis.md)** — AI-powered topic extraction with semantic clustering and contextual insights

- **[Changelog](CHANGELOG.md)** — Version history and recent improvements

## Quick Links

- **[GitHub Repository](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot)** — Source code and issue tracking
- **[API Documentation (Swagger)](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs)** — Interactive API documentation (local)
- **[API Documentation (ReDoc)](https://trump-speeches-nlp-chatbot.azurewebsites.net/redoc)** — Alternative API documentation

> NOTE: The Azure Web App links above point to a deployed instance that may take a minute or two to start (cold start) after a deployment or if idle. If the site doesn't load immediately, please wait 1–2 minutes and try again.

## Core Features

The system provides **three main AI-powered capabilities**, each demonstrating production-grade ML engineering:

### 1. Q&A System (RAG)

Intelligent question answering using Retrieval-Augmented Generation:

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What economic policies were discussed?", "top_k": 5}'
```

**Capabilities:**

- Semantic search using MPNet embeddings (768-dimensional)
- Hybrid search combining vector similarity and BM25 keyword matching
- Cross-encoder reranking for improved precision
- Multi-factor confidence scoring with human-readable explanations
- Entity extraction, sentiment analysis, and co-occurrence detection
- Pluggable LLM providers (Gemini/OpenAI/Anthropic) for answer generation
- Modular components: SearchEngine, ConfidenceCalculator, EntityAnalyzer, DocumentLoader

→ **[Full Q&A System Documentation](reference/qa-system.md)**

### 2. Sentiment Analysis

AI-powered emotional and sentimental analysis using multi-model ensemble:

```bash
curl -X POST http://localhost:8000/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

**Capabilities:**

- FinBERT for political/financial sentiment classification (positive/negative/neutral)
- RoBERTa for multi-emotion detection (anger, joy, fear, sadness, surprise, disgust)
- Gemini LLM for contextual interpretation explaining *why* the text has that tone
- Smart chunking for long texts with confidence aggregation
- Returns detailed emotion breakdown and AI-generated insights

→ **[Full Sentiment Analysis Documentation](reference/sentiment-analysis.md)**

### 3. Topic Analysis

AI-powered topic extraction with semantic clustering and contextual understanding:

```bash
curl -X POST http://localhost:8000/analyze/topics \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

**Capabilities:**

- Semantic clustering groups related keywords using embeddings
- AI-generated topic labels (e.g., "economy", "jobs" → "Economic Policy")
- Contextual snippets show keywords in actual use with highlighting
- LLM-generated summary provides interpretive analysis of main themes
- Smart filtering excludes common verbs and weak clusters

→ **[Full Topic Analysis Documentation](reference/topic-analysis.md)**

---

### Interactive Web Interface

Single-page application at the root (`/`) for testing all features without writing code.

## 🛠️ Technology Stack

**AI/ML:**

- ChromaDB (vector database)
- sentence-transformers (MPNet embeddings)
- Pluggable LLMs (Gemini, OpenAI, Claude)
- Hugging Face Transformers (FinBERT, RoBERTa)

**Backend:**

- FastAPI (REST API)
- Pydantic (validation)
- NLTK (preprocessing)

**DevOps:**

- Docker + Docker Compose
- GitHub Actions (CI/CD)
- pytest (testing, 65%+ coverage)
- Ruff (linting & formatting), mypy (type checking)

## 💡 Example Use Cases

1. **Political Speech Analysis** — Extract themes, sentiment, and talking points
2. **Q&A Over Documents** — Ask questions about large text corpora
3. **Entity Tracking** — Monitor mentions and sentiment toward specific entities
4. **Topic Discovery** — Identify main themes and discourse patterns
5. **Emotional Profiling** — Understand emotional tone and sentimental framing

## 🎓 Learning Resources

- **Core Features:**
  - [Q&A System (RAG)](reference/qa-system.md) - Hybrid search, entity analytics, confidence scoring
  - [Sentiment Analysis](reference/sentiment-analysis.md) - Multi-model emotion detection
  - [Topic Analysis](reference/topic-analysis.md) - Semantic clustering and theme extraction
- **Architecture & Design:**
  - [System Architecture](reference/architecture.md) - Component design and data flows
  - [Testing Strategy](development/testing.md) - pytest, coverage, and CI/CD integration
  - [Deployment Guide](guides/deployment.md) - Production deployment options

## 📞 Support & Contributing

- **Issues:** [GitHub Issues](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/issues)
- **Author:** Kristiyan Bonev
- **License:** MIT

---

**Ready to get started?** Head to the **[Quickstart Guide](guides/quickstart.md)** →
