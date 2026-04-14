# Trump Speeches NLP Chatbot — Documentation

A full-stack NLP platform combining retrieval-augmented generation, hybrid search, multi-model sentiment analysis, and AI-powered topic extraction. Built with FastAPI, ChromaDB, and pluggable LLM providers.

## Quick Start

```bash
git clone https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot.git
cd Trump-Rally-Speeches-NLP-Chatbot
uv sync
cp .env.example .env  # Add your LLM_API_KEY
uv run uvicorn speech_nlp.app:app --reload
```

→ **[Full Quickstart Guide](guides/quickstart.md)** | **[Developer Guide](dev-guide.md)** (daily reference)

## Live Demo

- **[Web App (Azure)](https://trump-speeches-nlp-chatbot.azurewebsites.net)** — Interactive UI
- **[Swagger Docs](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs)** — API playground

> **Note:** Azure Free Tier hosting — cold starts take 1-5 minutes after inactivity. Once warmed, responses are fast.

## The Three Features

| Feature | What It Does | Docs |
| --------- | ------------- | ------ |
| **Q&A System** | RAG pipeline with hybrid search, guardrails, query rewriting, entity analytics, and confidence scoring | [Reference](reference/qa-system.md) |
| **Sentiment Analysis** | FinBERT + RoBERTa + LLM interpretation for political/emotional tone analysis | [Reference](reference/sentiment-analysis.md) |
| **Topic Analysis** | Semantic clustering with AI-generated labels, contextual snippets, and theme summaries | [Reference](reference/topic-analysis.md) |

## Documentation Map

**Guides** — [Quickstart](guides/quickstart.md) · [Deployment](guides/deployment.md) · [CI/CD Local Testing](guides/ci-local-testing.md) · [Documentation](guides/documentation.md)

**Reference** — [Architecture](reference/architecture.md) · [Configuration](reference/configuration.md) · [FAQ](reference/faq.md)

**Development** — [Code Style](development/code-style.md) · [Testing](development/testing.md) · [Linting](development/linting.md) · [Security](development/security.md) · [Logging](development/logging.md)

**Project** — [Changelog](CHANGELOG.md) · [Roadmap](ROADMAP.md) · [GitHub](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot)
