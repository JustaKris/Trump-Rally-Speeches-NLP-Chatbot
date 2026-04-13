# Developer Guide

Quick reference for setting up, running, and shipping the project. See [deployment.md](guides/deployment.md) for the full details on CI/CD and cloud hosting.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (package manager)
- Docker (optional, for container workflows)
- A Gemini API key ([get one free](https://ai.google.dev/)) or another supported LLM provider key

---

## First-Time Setup

```powershell
# Clone and enter the repo
git clone https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot.git
cd Trump-Rally-Speeches-NLP-Chatbot

# Create the virtual environment and install all dependencies
uv sync --all-groups

# Activate the venv (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

Create a `.env` file in the project root with your API key:

```bash
LLM_API_KEY=your_gemini_api_key_here

# Optional overrides
# LLM_PROVIDER=openai
# LLM_MODEL_NAME=gpt-4o-mini
```

> **Gemini is the default provider.** For OpenAI or Anthropic, also set `LLM_PROVIDER` and `LLM_MODEL_NAME`.

---

## Running Locally

```powershell
# Start the dev server with hot reload
uv run uvicorn src.main:app --reload

# Or bind to all interfaces (useful for testing on a local network)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:

| URL | Purpose |
| ----- | --------- |
| <http://localhost:8000> | Web UI |
| <http://localhost:8000/docs> | Swagger API docs |
| <http://localhost:8000/redoc> | ReDoc API docs |
| <http://localhost:8000/health> | Health check |

**Note:** First startup takes ~30–60s while ML models load (~2GB: FinBERT, RoBERTa, MPNet).

---

## Running with Docker

```powershell
# Build the image (one-time, ~5-10 min — downloads and bakes in ML models)
docker build -t trump-speeches-nlp-chatbot .

# Run (models are already in the image, so startup is fast)
docker run --rm -it -p 8000:8000 --env-file .env --name nlp-chatbot trump-speeches-nlp-chatbot

# Run in the background
docker run -d -p 8000:8000 --env-file .env --name nlp-chatbot trump-speeches-nlp-chatbot

# Persist ChromaDB data across container runs
docker run --rm -it -p 8000:8000 `
  -v "${PWD}/data/chromadb:/app/data/chromadb" `
  --env-file .env --name nlp-chatbot trump-speeches-nlp-chatbot

# View logs
docker logs -f nlp-chatbot

# Stop and remove
docker stop nlp-chatbot
docker rm nlp-chatbot
```

### Docker Compose

```powershell
docker-compose up          # Foreground
docker-compose up -d       # Background
docker-compose down        # Stop and remove containers
```

---

## Pushing to Docker Hub

```powershell
docker login

# Tag
docker tag trump-speeches-nlp-chatbot yourusername/trump-speeches-nlp-chatbot:latest
docker tag trump-speeches-nlp-chatbot yourusername/trump-speeches-nlp-chatbot:v1.0.0

# Push
docker push yourusername/trump-speeches-nlp-chatbot:latest
```

Pushing to `main` also triggers the GitHub Actions workflows which build and push automatically — you only need to do this manually if you want to push a specific local build.

---

## Tests & Code Quality

```powershell
# Run tests
uv run pytest
uv run pytest -v --cov=src          # With coverage report

# Lint and format
uv run ruff check src/
uv run ruff check src/ --fix        # Auto-fix fixable issues
uv run ruff format src/

# Type checking
uv run mypy src/

# Security scan
uv run bandit -r src/ -c pyproject.toml
```

All of these run automatically in CI on every push. The pipeline requires tests and linting to pass; type checking and security scanning are informational (allowed to fail).

---

## Managing Dependencies

```powershell
uv add requests                    # Add a package
uv add --group dev pytest-xdist    # Add to a dependency group
uv remove requests                 # Remove a package
uv sync                            # Sync install to match lock file
uv sync --upgrade                  # Upgrade all to latest compatible versions
uv lock --upgrade-package fastapi  # Upgrade a single package
```

---

## Configuration

Config is loaded in this order (highest priority first):

1. Environment variables
2. `.env` file
3. `configs/<environment>.yaml` (default: `development.yaml`)
4. Code defaults

Set `ENVIRONMENT=staging` or `ENVIRONMENT=production` to switch config files.

---

## Project Structure (Key Paths)

```text
src/
  main.py                   # App entry point, startup hooks
  api/                      # FastAPI route handlers
  services/
    rag_service.py           # RAG pipeline orchestrator
    rag/                     # Search, chunking, confidence, entity analysis
    llm/                     # Pluggable LLM providers (Gemini, OpenAI, Anthropic)
    sentiment_service.py     # FinBERT + RoBERTa + LLM sentiment
    topic_service.py         # Semantic topic clustering
  config/settings.py         # Pydantic settings (all config fields live here)
  models/schemas.py          # Pydantic request/response models
  templates/index.html       # Single-page frontend
data/
  chromadb/                  # Persisted vector store
  Donald Trump Rally Speeches/  # Source speech transcripts
configs/                     # Environment YAML configs
scripts/                     # Utility scripts (model download, migrations)
tests/                       # pytest test suite
```

---

## Useful Links

- [Full Deployment Guide](guides/deployment.md) — CI/CD, Azure, Render setup
- [Quickstart](guides/quickstart.md) — Faster first-run walkthrough
- [Troubleshooting (Azure)](guides/troubleshooting_Azure.md)
- [ROADMAP](ROADMAP.md)
- [Live App](https://trump-speeches-nlp-chatbot.azurewebsites.net)
