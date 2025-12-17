# Topic Analysis System

This document provides complete reference for the AI-powered topic analysis system, which extracts and organizes key themes from text using semantic clustering and large language models.

**What It Does:**

- Extracts meaningful topics from text using AI embeddings
- Groups related keywords into semantic clusters (e.g., "economy", "jobs" → "Economic Policy")
- Provides contextual text snippets showing keywords in actual use
- Generates AI-powered summaries of main themes
- Filters out noise (common verbs, weak clusters)

**Perfect For:**

- Political speech thematic analysis
- Document summarization
- Content classification
- Research on discourse patterns

## Overview

The topic analysis system goes beyond simple word frequency by:

1. **Semantic Clustering** — Groups related keywords using embeddings
2. **AI-Generated Labels** — Creates meaningful topic names using LLM
3. **Contextual Snippets** — Shows keywords in actual use with highlighting
4. **AI Summaries** — Provides interpretive analysis of main themes
5. **Smart Filtering** — Excludes common verbs and weak clusters

**Note:** Uses the configured LLM provider (Gemini by default, with OpenAI and Claude support via optional dependencies) for label generation and summaries. Configure via `LLM_PROVIDER` environment variable.

## Basic vs Enhanced Topic Extraction

### Legacy Approach (Removed)

The old frequency-based extraction simply listed keywords by count.

### Current: AI-Powered Topic Analysis (`/analyze/topics`)

Returns semantically clustered topics with context:

```json
{
  "clustered_topics": [
    {
      "label": "National Pride",
      "keywords": [
        {"word": "great", "count": 40, "relevance": 1.0},
        {"word": "country", "count": 35, "relevance": 0.875}
      ],
      "avg_relevance": 0.9375,
      "total_mentions": 75
    }
  ],
  "snippets": [
    {
      "label": "National Pride",
      "snippets": [
        "We're going to make America **great** again. This **country** deserves better.",
        "Our **country** is the greatest nation on Earth..."
      ]
    }
  ],
  "summary": "The speech emphasizes themes of national pride and American exceptionalism, with recurring references to restoring greatness...",
  "metadata": {
    "total_keywords": 30,
    "num_clusters": 5,
    "has_ai_summary": true
  }
}
```

## Installation & Setup

### Prerequisites

**Python Version:** 3.11 or 3.12 (as specified in `pyproject.toml`)

**Package Manager:** This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot.git
cd Trump-Rally-Speeches-NLP-Chatbot

# Install dependencies (creates .venv automatically)
uv sync

# Configure environment
cp .env.example .env
# Edit .env: Set LLM_API_KEY and LLM_PROVIDER

# Run the server
uv run uvicorn src.main:app --reload
```

API available at `http://localhost:8000`.

### Dependencies

Core topic analysis dependencies (automatically installed with `uv sync`):

- `sentence-transformers>=3.3.0` — Embeddings with MPNet model
- `scikit-learn>=1.7.2` — KMeans clustering
- `numpy>=1.26.0,<2.0.0` — NumPy arrays (compatible with PyTorch 2.6)
- `google-generativeai>=0.8.0` — Gemini LLM (default)

**Optional LLM Providers:**

```bash
# Install OpenAI support
uv sync --group llm-openai

# Install Claude support
uv sync --group llm-anthropic
```

Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` in `.env` after installing.

## API Usage

### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze/topics" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "They're burning Minneapolis. You don't think of Minneapolis that way, right? You don't think of it. The city is burning down. You have this fake CNN reporter, what's his name? Nobody, the nice shaved head. Maybe I should try that! By that! No, I don't think... Donald Trump went down substantially in the polls, like about 40%. He showed up with a new haircut. It's called the shave head. Ah, hello. And remember he said, No, this is a friendly protest. It's a mostly genteel. And it's really quite nice. Now people are shooting bullets at him. He's being hit with tear gas. This is a friendly protest."
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze/topics",
    json={"text": "Your text here..."}
)

result = response.json()

# Access clustered topics
for cluster in result["clustered_topics"]:
    print(f"\n{cluster['label']} ({cluster['total_mentions']} mentions)")
    print(f"Keywords: {', '.join([kw['word'] for kw in cluster['keywords'][:3]])}")

# Access snippets
for snippet_group in result["snippets"]:
    print(f"\n{snippet_group['label']} examples:")
    for snippet in snippet_group["snippets"]:
        print(f"  - {snippet}")

# Access AI summary
if result["summary"]:
    print(f"\nSummary: {result['summary']}")
```

### JavaScript/Frontend Example

```javascript
async function analyzeTopics(text) {
  const response = await fetch('/analyze/topics', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  
  const data = await response.json();
  
  // Display AI summary
  if (data.summary) {
    console.log('Summary:', data.summary);
  }
  
  // Display clustered topics
  data.clustered_topics.forEach(cluster => {
    console.log(`${cluster.label}: ${cluster.total_mentions} mentions`);
  });
  
  return data;
}
```

## Parameters

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to analyze (in request body) |
| `top_n` | integer | 10 | Number of topic clusters to return (query param) |
| `num_clusters` | integer | auto | Number of clusters 3-6, auto-determined (query param) |
| `snippets_per_topic` | integer | 3 | Number of example snippets per cluster (query param) |

**Note:** `text` is passed in the JSON body, while `top_n`, `num_clusters`, and `snippets_per_topic` are query parameters.

### Parameter Examples

**Get more clusters:**

```bash
curl -X POST "http://localhost:8000/analyze/topics?top_n=8&num_clusters=6" \
  -H "Content-Type: application/json" \
  -d '{"text": "..."}'
```

**Get more snippets per topic:**

```bash
curl -X POST "http://localhost:8000/analyze/topics?snippets_per_topic=5" \
  -H "Content-Type: application/json" \
  -d '{"text": "..."}'
```

## Response Structure

### Clustered Topics

Each topic cluster contains:

- `label` — AI-generated semantic label (e.g., "Border Security")
- `keywords` — List of related keywords with counts and relevance scores
- `avg_relevance` — Average relevance score for the cluster (0-1)
- `total_mentions` — Total mentions of all keywords in cluster

### Snippets

Each snippet group contains:

- `label` — Matches the topic cluster label
- `snippets` — Array of text excerpts with keywords highlighted using `**bold**` markdown
- `keyword_count` — Number of distinct keywords in this cluster

### Summary

- AI-generated 2-3 sentence interpretation of main themes
- Identifies dominant topics and patterns
- Provides objective, analytical perspective
- Only present if Gemini LLM is configured

### Metadata

- `total_keywords` — Number of keywords extracted before clustering
- `num_clusters` — Number of clusters created
- `text_length` — Length of input text in characters
- `has_ai_summary` — Whether AI summary was generated

## Use Cases

### 1. Political Speech Analysis

Analyze political speeches to identify main themes:

```python
speech_text = load_speech("path/to/speech.txt")
result = analyze_topics_enhanced(speech_text, top_n=6)

print(f"Main themes: {', '.join([c['label'] for c in result['clustered_topics']])}")
print(f"Summary: {result['summary']}")
```

### 2. Document Summarization

Extract key topics from long documents:

```python
for cluster in result['clustered_topics'][:3]:
    print(f"\n{cluster['label']}:")
    for snippet in result['snippets'][idx]['snippets']:
        print(f"  '{snippet}'")
```

### 3. Content Classification

Categorize documents by topic clusters:

```python
def categorize_document(text):
    result = analyze_topics_enhanced(text, top_n=3)
    return [cluster['label'] for cluster in result['clustered_topics']]
```

### 4. Sentiment + Topic Analysis

Combine with sentiment analysis for deeper insights:

```python
# Get topics
topics = analyze_topics_enhanced(text)

# Get sentiment
sentiment = analyze_sentiment(text)

# Combine insights
print(f"Document sentiment: {sentiment['sentiment']}")
print(f"Main topics: {[c['label'] for c in topics['clustered_topics'][:3]]}")
```

## Technical Details

### Clustering Algorithm

The system uses **KMeans clustering** on **MPNet embeddings** (768-dimensional semantic vectors):

1. **Keyword Extraction** — Extract top keywords using frequency analysis with TF-IDF-style scoring
2. **Embedding Generation** — Generate 768-dimensional embeddings for each keyword using `all-mpnet-base-v2` from sentence-transformers
3. **Semantic Clustering** — Cluster embeddings into 3-6 groups using KMeans (number auto-determined based on keyword count)
4. **Ranking** — Sort clusters by total mentions to prioritize most important topics first

**Auto-Cluster Determination:**

- < 10 keywords: 3 clusters
- 10-20 keywords: 4 clusters  
- 20-30 keywords: 5 clusters
- 30+ keywords: 6 clusters

**Why KMeans?**

- Fast and deterministic
- Works well with fixed cluster counts
- Produces balanced clusters
- Efficient with high-dimensional embeddings

### Label Generation

Cluster labels are generated using **Gemini LLM** with a specialized prompt:

```text
Given these related keywords: economy, jobs, employment, market
Generate a concise 2-4 word label that captures the main theme.
```

If LLM is not available, falls back to using the top keyword as the label.

### Snippet Extraction

Snippets are extracted with context windows around keyword occurrences:

1. Find all positions of cluster keywords in text
2. Deduplicate nearby positions (min 200 chars apart)
3. Extract ±100 character context around each keyword
4. Clean up to sentence boundaries when possible
5. Highlight keywords with `**bold**` markdown

### AI Summary Generation

The summary is generated by providing Gemini with:

1. List of topic clusters with keywords and mention counts
2. Sample of the input text (first 2000 chars)
3. Prompt requesting 2-3 sentence analytical summary

## Performance Considerations

### Response Time

- **Without LLM:** ~1-2 seconds for typical documents (500-2000 words)
- **With LLM:** ~3-5 seconds (includes label generation + summary)

**Breakdown:**

- Keyword extraction: ~100-200ms
- Embedding generation: ~200-500ms (depends on keyword count)
- KMeans clustering: ~50-100ms
- Snippet extraction: ~200-400ms
- LLM calls (labels + summary): ~2-3 seconds total

**First Request:** May take 30-60 seconds for one-time model download (~500 MB for sentence-transformers).

### Optimal Text Length

- **Minimum:** 100+ words for meaningful clustering
- **Optimal:** 500-2000 words (typical political speech length)
- **Maximum:** No hard limit, but performance degrades linearly with length
  - 5000+ words: Consider text chunking or summarization first
  - Very long texts may produce too many clusters

### Memory Usage

- **sentence-transformers model:** ~500 MB (loaded once, cached)
- **LLM service:** Minimal (API-based, no local loading)
- **Runtime:** ~50-100 MB per request (temporary embeddings)
- **Recommendation:** Minimum 1.5 GB free RAM

### Caching

- Embedding model loads once at startup and persists
- LLM service initializes lazily on first use
- Keyword embeddings generated per request (not cached)
- **Optimization opportunity:** Cache embeddings for repeated analysis

## Configuration Best Practices

**Production Settings:**

```yaml
# configs/production.yaml
topic:
  max_keywords: 30
  min_cluster_size: 3
  topic_relevance_threshold: 0.3
  excluded_verbs: ["said", "going", "know", ...]  # Extensive list
```

**Development Settings:**

```yaml
# configs/development.yaml  
topic:
  max_keywords: 20  # Faster processing
  min_cluster_size: 2
  topic_relevance_threshold: 0.2  # More permissive
```

**Environment Variable Overrides:**

```bash
# .env
ENVIRONMENT=production  # Load production.yaml
LLM_API_KEY=your-key-here
LLM_PROVIDER=gemini
```

**Loading Order (Precedence):**

1. Environment variables (highest priority)
2. `.env` file
3. `configs/{ENVIRONMENT}.yaml`
4. Code defaults (lowest priority)

## Development Workflow

### Running Tests

```bash
# Run topic analysis tests
uv run pytest tests/test_topic_service.py -v

# Run with coverage
uv run pytest tests/test_topic_service.py --cov=src.services.topic_service

# Test topic extraction specifically
uv run pytest tests/test_topic_service.py::test_extract_topics_enhanced -v
```

### Code Quality

```bash
# Lint and format
uv run ruff check src/services/topic_service.py
uv run ruff format src/services/topic_service.py

# Type checking
uv run mypy src/services/topic_service.py
```

### Local Testing

```bash
# Start server with hot reload
uv run uvicorn src.main:app --reload --log-level debug

# Test endpoint
curl -X POST "http://localhost:8000/analyze/topics?top_n=5" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text for topic analysis..."}'
```

### Debugging Tips

**Enable verbose logging:**

```python
# In configs/development.yaml
logging:
  level: DEBUG
  format: pretty  # Colored console output
```

**Inspect cluster assignments:**

```python
from src.services.topic_service import TopicExtractionService

service = TopicExtractionService()
result = service.extract_topics_enhanced(text, top_n=5)

# Examine cluster details
for cluster in result['clustered_topics']:
    print(f"{cluster['label']}: {cluster['total_mentions']} mentions")
    print(f"Keywords: {[kw['word'] for kw in cluster['keywords']]}")
    print(f"Avg relevance: {cluster['avg_relevance']:.3f}\n")
```

## Troubleshooting

### No Clusters Generated

**Problem:** Empty `clustered_topics` array

**Solutions:**

- Ensure text has at least 50-100 words
- Check that text contains meaningful content (not just stopwords)
- Try increasing `top_n` parameter

### Missing AI Summary

**Problem:** `summary` field is `null`

**Solutions:**

- Ensure `LLM_API_KEY` is configured in `.env` (see `.env.example` for template)
- Set `LLM_PROVIDER` to your chosen provider (gemini/openai/anthropic)
- Check API logs for LLM errors
- Verify Gemini API quota/limits
- System still provides clustered topics even without LLM summary

### Service Unavailable (503)

**Problem:** 503 error with "Topic extraction not available. Service not initialized."

**Solutions:**

- **Verify startup logs:** Check for topic service initialization errors

  ```bash
  uv run uvicorn src.main:app --log-level debug
  # Look for "TopicExtractionService initialized" message
  ```
  
- **Check dependencies:** Ensure scikit-learn and sentence-transformers installed

  ```bash
  uv sync  # Reinstall all dependencies
  uv pip list | grep -E "scikit-learn|sentence-transformers"
  ```

- **Verify model download:** First request downloads ~500 MB model from HuggingFace
  - Check `~/.cache/huggingface/` for cached models
  - May take 1-2 minutes on first request
- **Memory check:** Ensure at least 1.5 GB free RAM
- **Restart service:** Clear any stuck initialization states

  ```bash
  # Kill process and restart
  uv run uvicorn src.main:app --reload
  ```

## See Also

- [Q&A System](qa-system.md) — RAG-based question answering with entity analytics
- [Sentiment Analysis](sentiment-analysis.md) — Multi-model emotion and sentiment detection
- [Architecture Documentation](architecture.md) - System architecture overview
- [API Reference](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs#/nlp/analyze_topics_enhanced_analyze_topics_enhanced_post) - Interactive API playground *(Azure Free Tier: allow 1-5min cold start)*
- [Development Guide](../development/testing.md) - Testing practices
- [GitHub Repository](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot) - Source code and pyproject.toml
