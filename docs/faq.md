# Frequently Asked Questions (FAQ)

## General Questions

### What is this project?

A production-ready NLP application showcasing advanced AI/ML techniques including:

- Retrieval-Augmented Generation (RAG) with hybrid search
- Multi-model sentiment analysis with LLM interpretation
- AI-powered topic clustering and extraction
- Entity analytics with contextual associations
- Pluggable LLM provider architecture (Gemini, OpenAI, Claude)

Built with FastAPI, ChromaDB, PyTorch, and Transformers, deployed on Azure with full CI/CD pipelines.

### Is this production-ready?

Yes! The project includes:

- Comprehensive testing (65%+ coverage)
- Production logging (JSON format)
- Docker containerization
- GitHub Actions CI/CD
- Security scanning (Bandit, pip-audit)
- Automated deployment to Azure
- Health check endpoints
- Error handling and validation

### Can I use this for my own project?

Absolutely! This project is MIT licensed. You can:

- Use it as a template for your own NLP API
- Learn from the architecture and implementation
- Adapt components for your specific use case
- Deploy it with your own dataset

Just replace the Trump speech dataset with your own text corpus.

---

## Setup & Installation

### Why is my first request slow?

**Azure cold start:** The free tier Azure Web App goes to sleep after inactivity. The first request after idle time takes 1-2 minutes to wake up and load all ML models into memory.

**Local first run:** The first time you run locally, the application downloads ~2GB of ML models (FinBERT, RoBERTa, MPNet embeddings). Subsequent runs are fast.

### What are the system requirements?

**Minimum:**

- Python 3.11-3.14
- 2.5GB RAM (with RAG)
- 2GB disk space (models + data)
- 1-2 CPU cores

**Recommended:**

- Python 3.12
- 4-8GB RAM (for concurrent requests)
- 5GB disk space
- 4+ CPU cores

### Do I need a GPU?

No. The project uses CPU-only PyTorch builds. While GPU would speed up model inference, it's not required and the application performs well on CPU.

### Which Python version should I use?

**Recommended:** Python 3.12

**Supported:** Python 3.11, 3.12, 3.13, 3.14

The project uses modern Python features and type hints. Python 3.11+ is required.

---

## API Usage

### How do I ask questions with RAG?

```bash
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What economic policies were discussed?",
    "top_k": 5
  }'
```

The system will:

1. Extract entities from your question
2. Perform hybrid search (semantic + BM25)
3. Calculate confidence scores
4. Generate an AI answer using the LLM
5. Return sources and entity statistics

### What's the difference between semantic and hybrid search?

**Semantic Search (`/rag/search`):**

- Uses MPNet embeddings (768-dimensional vectors)
- Finds conceptually similar content
- Good for: Meaning-based queries, synonyms, paraphrasing

**Hybrid Search (used in `/rag/ask`):**

- Combines semantic search + BM25 keyword matching
- Weights: 70% semantic, 30% BM25 (configurable)
- Optional cross-encoder reranking
- Good for: Best of both worlds - meaning + keywords

### How accurate are the confidence scores?

Confidence is calculated using 4 weighted factors:

- **Retrieval Quality (40%):** Average semantic similarity of results
- **Consistency (25%):** Score variance (low variance = high confidence)
- **Coverage (20%):** Number of supporting chunks
- **Entity Coverage (15%):** Percentage of chunks mentioning query entities

**Levels:**

- **High:** combined_score ≥ 0.7 (reliable answer)
- **Medium:** 0.4 ≤ score < 0.7 (reasonable answer, verify sources)
- **Low:** score < 0.4 (limited information, answer may be speculative)

### Why does sentiment analysis return multiple scores?

The sentiment service uses an ensemble approach:

1. **FinBERT** → Overall sentiment (positive/negative/neutral)
2. **RoBERTa** → Six emotion categories (joy, anger, fear, sadness, surprise, disgust)
3. **LLM (Gemini)** → Contextual interpretation explaining WHY the models gave those scores

This multi-model approach provides more nuanced analysis than single-model systems.

---

## LLM Configuration

### How do I switch from Gemini to OpenAI?

1. Install OpenAI dependency:

   ```bash
   uv sync --group llm-openai
   ```

2. Update `.env`:

   ```bash
   LLM_PROVIDER=openai
   LLM_API_KEY=sk-your_openai_key
   LLM_MODEL_NAME=gpt-4o-mini
   ```

3. Restart the application

### How do I use Claude instead?

1. Install Anthropic dependency:

   ```bash
   uv sync --group llm-anthropic
   ```

2. Update `.env`:

   ```bash
   LLM_PROVIDER=anthropic
   LLM_API_KEY=sk-ant-your_anthropic_key
   LLM_MODEL_NAME=claude-3-5-sonnet-20241022
   ```

3. Restart the application

### Can I use multiple LLM providers simultaneously?

No, the application uses one provider at a time, configured via `LLM_PROVIDER`. However, switching is instant - just update the environment variable and restart.

The pluggable architecture makes it easy to A/B test different models.

### Which LLM provider is cheapest?

As of December 2025:

- **Gemini 2.5 Flash:** Free tier available, very cost-effective
- **GPT-4o-mini:** ~$0.15 per 1M input tokens
- **Claude 3.5 Sonnet:** ~$3 per 1M input tokens (higher quality)

Gemini is the default for a reason - excellent performance at minimal cost.

---

## Deployment

### Can I deploy this for free?

**Yes!** Options:

- **Render:** Free tier available (512MB RAM, enough for basic usage)
- **Azure:** Free tier (750 hours/month for Web Apps)
- **Docker:** Self-host anywhere

Note: Free tiers have cold starts and limited resources. For production traffic, upgrade to paid tiers.

### Why does my Docker build take so long?

The multi-stage build:

1. Downloads ~2GB of ML models
2. Installs 100+ Python dependencies
3. Compiles PyTorch extensions

**First build:** 10-15 minutes (downloads everything)
**Subsequent builds:** 2-5 minutes (uses layer caching)

Use `docker build --progress=plain` to see detailed progress.

### How do I enable HTTPS in production?

The application itself runs HTTP on port 8000. For HTTPS:

- **Azure/Render:** Automatic HTTPS (handled by platform)
- **Self-hosted:** Use Nginx reverse proxy with Let's Encrypt
- **Docker:** Add Nginx container with SSL certificates

See the [deployment guide](guides/deployment.md) for reverse proxy configuration.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'X'"

Run:

```bash
uv sync --all-groups
```

This installs all dependencies including optional LLM providers.

### "GEMINI_API_KEY not found"

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

Get a free API key at <https://ai.google.dev/>

### Tests are failing with "LLM provider not initialized"

Some tests are skipped when optional LLM providers aren't installed. This is expected.

To run all tests, install the specific provider group:

```bash
uv sync --group llm-openai
uv sync --group llm-anthropic
```

### RAG returns empty results

**Possible causes:**

1. **Empty collection:** Run `POST /rag/index` to index documents
2. **Query too specific:** Try broader questions
3. **Collection cleared:** Check `GET /rag/stats` to verify document count

### Sentiment analysis text is cut off

Check `LLM_MAX_OUTPUT_TOKENS` in your `.env`. Default is 2000 tokens (~1500 words).

If interpretations are still truncated, increase to 3000 or 4000.

### Port 8000 already in use

Another application is using port 8000. Run on a different port:

```bash
uv run uvicorn src.main:app --port 8001
```

---

## Development

### How do I run tests?

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html

# Specific test file
uv run pytest tests/test_rag_integration.py -v
```

### How do I format code?

```bash
# Format all Python files
uv run ruff format .

# Check for linting issues
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix
```

### How do I add a new LLM provider?

1. Create `src/services/llm/your_provider.py`
2. Implement the `LLMProvider` abstract base class
3. Add to `factory.py` provider mapping
4. Add dependency to `pyproject.toml` in a new optional group
5. Update documentation

See [anthropic.py](../src/services/llm/anthropic.py) as an example.

### How is logging configured?

The application uses structured logging:

- **Development:** Pretty colored output with loguru-style formatting
- **Production:** JSON format for log aggregation (ELK, CloudWatch, etc.)

Configured in [src/core/logging_config.py](../src/core/logging_config.py)

---

## Data & Models

### Can I use my own dataset?

Yes! Replace the speeches in `data/Donald Trump Rally Speeches/` with your own `.txt` files, then re-index:

```bash
POST /rag/index
```

The system will:

1. Load all `.txt` files from the directory
2. Chunk them (2048 chars, 150 overlap)
3. Generate embeddings (MPNet)
4. Store in ChromaDB

### Which ML models are used?

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| **FinBERT** | Sentiment classification | ~440MB | ProsusAI/finbert |
| **RoBERTa-Emotion** | Emotion detection | ~330MB | j-hartmann/emotion-english-distilroberta-base |
| **all-mpnet-base-v2** | Embeddings, clustering | ~420MB | sentence-transformers |
| **ms-marco-MiniLM** | Cross-encoder reranking | ~80MB | cross-encoder |
| **Gemini 2.5 Flash** | Answer generation | API | Google AI |

### How much data can the RAG system handle?

**Current dataset:** 35 documents, ~300,000 words, 1,082 chunks

**Tested limits:** Up to 10,000 chunks (several million words)

**Scalability:**

- Replace ChromaDB with pgvector or Pinecone for larger datasets
- Add Redis caching for frequent queries
- Use background jobs for re-indexing

### Where are embeddings stored?

ChromaDB stores embeddings in `data/chromadb/`:

- `chroma.sqlite3` - Metadata database
- `UUID folders` - Vector data

This directory persists across restarts. To reset:

```bash
POST /rag/index  # Re-indexes from source files
```

---

## Architecture

### Why use hybrid search instead of just semantic?

**Semantic search weaknesses:**

- Misses exact keyword matches
- Struggles with rare terms, names, acronyms
- Can return conceptually similar but irrelevant results

**BM25 search weaknesses:**

- Misses paraphrasing and synonyms
- No understanding of meaning
- Keyword-only matching

**Hybrid search:** Combines both strengths, weighted 70/30 by default.

### What's the difference between RAGService and SearchEngine?

**Architecture separation:**

- **`RAGService`** (orchestrator) - Manages ChromaDB, coordinates components, handles indexing
- **`SearchEngine`** (component) - Performs search operations (semantic, BM25, hybrid, reranking)
- **`ConfidenceCalculator`** (component) - Calculates multi-factor confidence scores
- **`EntityAnalyzer`** (component) - Extracts entities and generates statistics
- **`DocumentLoader`** (component) - Loads and chunks documents

This modular design enables:

- Independent testing of components
- Easy replacement/upgrades
- Clear separation of concerns

### Why not use LangChain for everything?

LangChain is great for rapid prototyping, but this project demonstrates:

- **Custom implementations** - Shows understanding of underlying concepts
- **Type safety** - Pydantic models throughout
- **Testability** - Modular components with 90%+ coverage
- **Control** - Fine-tuned search strategies and confidence scoring

LangChain is used selectively (text splitting utilities) where it adds value without obscuring the architecture.

---

## Contributing

### Can I contribute to this project?

This is primarily a portfolio project, but suggestions and feedback are welcome!

**To suggest improvements:**

1. Open an issue on GitHub
2. Describe the enhancement or bug
3. Provide context and use cases

**To report security issues:**

Email: <k.s.bonev@gmail.com> (do not open public issues for security vulnerabilities)

### How can I learn from this project?

**Recommended learning path:**

1. **Start with basics** - Review [quickstart.md](guides/quickstart.md) and run locally
2. **Explore architecture** - Read [architecture.md](reference/architecture.md) with diagrams
3. **Study components** - Review modular RAG components in `src/services/rag/`
4. **Run tests** - See how components are tested in isolation
5. **Try modifications** - Swap datasets, change parameters, add features
6. **Deploy** - Follow [deployment.md](guides/deployment.md) to deploy your own instance

---

## License

### What license is this under?

MIT License - you can use, modify, and distribute this code freely.

See [LICENSE](../LICENSE) for full text.

### Can I use this commercially?

Yes! The MIT license allows commercial use. Just include the copyright notice.

### What about the dataset?

The Trump rally speeches are public domain (government official speeches). You can use them freely.

If you replace with your own dataset, ensure you have rights to use that data.

---

## Still Have Questions?

- 📧 **Email:** <k.s.bonev@gmail.com>
- 🐙 **GitHub Issues:** [Open an issue](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot/issues)
- 📚 **Documentation:** [Full technical docs](https://justakris.github.io/Trump-Rally-Speeches-NLP-Chatbot/)
- 🔗 **Live Demo:** [Try the API](https://trump-speeches-nlp-chatbot.azurewebsites.net)
