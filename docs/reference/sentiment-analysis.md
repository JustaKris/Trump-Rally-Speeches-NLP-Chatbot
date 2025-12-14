# Sentiment Analysis System

## Overview

The sentiment analysis system provides AI-powered emotional and sentimental analysis of text using a multi-model ensemble approach. It combines specialized transformer models with optional LLM interpretation to deliver comprehensive insights into the emotional tone and sentiment of text.

**What It Does:**

- Classifies overall sentiment (positive/negative/neutral)
- Detects specific emotions (anger, joy, fear, sadness, surprise, disgust, neutral)
- Provides AI-generated contextual interpretation explaining *why* the text has that emotional tone
- Handles long texts through smart chunking

**Perfect For:**

- Political speech analysis
- Customer feedback analysis  
- Social media sentiment tracking
- Document emotional profiling

---

## System Architecture

### Multi-Model Ensemble

The system uses **three specialized AI models** working together:

```text
┌─────────────────┐
│   Input Text    │
└────────┬────────┘
         │
         ├──────────┐
         │          │
    ┌────▼───┐  ┌──▼──────┐
    │FinBERT │  │ RoBERTa │
    │        │  │ Emotion │
    │Sentiment  │Detection│
    └────┬───┘  └──┬──────┘
         │         │
         └────┬────┘
              │
         ┌────▼────────┐
         │   Gemini    │
         │Contextual   │
         │Interpretation
         └─────────────┘
```

#### Model 1: FinBERT (Sentiment Classification)

**Purpose:** Primary sentiment classification

**Model:** `ProsusAI/finbert`  
**Specialization:** Financial and political text sentiment analysis  
**Output:** Probabilities for positive, negative, neutral

**Why FinBERT?**

- Trained on financial news and reports
- Excellent for political/economic discourse
- Better than generic sentiment models for formal speech
- Produces reliable confidence scores

**How It Works:**

1. Tokenizes input text
2. Chunks long texts into 510-token segments
3. Analyzes each chunk independently
4. Averages predictions across all chunks
5. Returns probability distribution

**Example Output:**

```python
{
    "positive": 0.15,
    "negative": 0.72,
    "neutral": 0.13,
    "dominant": "negative"
}
```

#### Model 2: RoBERTa Emotion Detection

**Purpose:** Multi-emotion classification

**Model:** `j-hartmann/emotion-english-distilroberta-base`  
**Output:** Probabilities for 7 emotions

**Emotions Detected:**

- 😠 Anger
- 😊 Joy  
- 😨 Fear
- 😢 Sadness
- 😮 Surprise
- 🤢 Disgust
- 😐 Neutral

**Why RoBERTa?**

- State-of-the-art emotion detection
- Trained on 58k emotion-labeled texts
- Distilled for faster inference
- Provides nuanced emotional profile beyond simple positive/negative

**Example Output:**

```python
{
    "anger": 0.62,
    "joy": 0.05,
    "fear": 0.12,
    "sadness": 0.08,
    "surprise": 0.03,
    "disgust": 0.07,
    "neutral": 0.03
}
```

#### Model 3: Gemini LLM (Contextual Interpretation)

**Purpose:** Human-readable explanation of sentiment

**Model:** Google Gemini 2.0 Flash (configurable via `LLM_MODEL_NAME`)  
**Output:** 2-3 sentence interpretation

**What It Explains:**

- **WHY** the text received its sentiment score
- **WHY** certain emotions dominate
- **WHAT** the speaker expresses emotion about
- Connections between sentiment and content

**Example Output:**
> "The text expresses strong negative sentiment about immigration policy, with anger emerging from perceived government failures. The speaker's frustration targets specific politicians and policies, explaining the high anger score and negative classification."

**Fallback:** If Gemini is unavailable, provides simple interpretation using model scores

---

## API Reference

### Endpoint

```http
POST /analyze/sentiment
```

### Request

**Body (JSON):**

```json
{
  "text": "Your text to analyze here..."
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Text to analyze (any length) |

### Response

**Success (200):**

```json
{
  "sentiment": "negative",
  "confidence": 0.72,
  "scores": {
    "positive": 0.15,
    "negative": 0.72,
    "neutral": 0.13
  },
  "emotions": {
    "anger": 0.62,
    "joy": 0.05,
    "fear": 0.12,
    "sadness": 0.08,
    "surprise": 0.03,
    "disgust": 0.07,
    "neutral": 0.03
  },
  "contextual_sentiment": "The text expresses strong negative sentiment about immigration policy, with anger emerging from perceived government failures. The speaker's frustration targets specific politicians and policies, explaining the high anger score and negative classification.",
  "num_chunks": 3
}
```

**Fields Explained:**

| Field | Type | Description |
|-------|------|-------------|
| `sentiment` | string | Dominant sentiment: "positive", "negative", or "neutral" |
| `confidence` | float | Confidence score (0-1) for dominant sentiment |
| `scores` | object | Probability distribution across all sentiments |
| `emotions` | object | Probability distribution across 7 emotions |
| `contextual_sentiment` | string | AI-generated interpretation (2-3 sentences) |
| `num_chunks` | int | Number of text chunks analyzed (for transparency) |

---

## Usage Examples

### cURL

```bash
curl -X POST "http://localhost:8000/analyze/sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "We are winning like never before. The economy is booming, jobs are coming back, and America is great again!"
  }'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze/sentiment",
    json={"text": "Your text here..."}
)

result = response.json()

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"\nTop Emotions:")
for emotion, score in sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {emotion}: {score:.0%}")
print(f"\nInterpretation:\n{result['contextual_sentiment']}")
```

### JavaScript

```javascript
async function analyzeSentiment(text) {
  const response = await fetch('/analyze/sentiment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  
  const data = await response.json();
  
  console.log(`Sentiment: ${data.sentiment} (${(data.confidence * 100).toFixed(0)}%)`);
  console.log(`Interpretation: ${data.contextual_sentiment}`);
  
  return data;
}
```

---

## Understanding the Results

### Sentiment Classification

**Positive (0.7+):**

- Optimistic language
- Achievement and success framing
- Praise and celebration
- Examples: economic wins, policy victories

**Negative (0.7+):**

- Critical language
- Problems and failures
- Opposition and conflict
- Examples: attacks on opponents, crisis framing

**Neutral (0.5+):**

- Factual statements
- Balanced perspectives
- Mixed emotions
- Examples: policy explanations, data presentation

**Mixed (No clear dominant):**

- Complex emotional landscape
- Multiple competing sentiments
- Nuanced arguments
- Check `contextual_sentiment` for interpretation

### Emotion Scores

**Primary Emotion:**
The highest-scoring emotion reveals the dominant emotional tone.

**Secondary Emotions:**

Look at the top 2-3 emotions for emotional complexity. For example:

- High anger + moderate fear = threat/danger framing
- High joy + moderate surprise = unexpected positive outcome
- Balanced emotions = complex or neutral tone

### Contextual Interpretation

The LLM-generated interpretation connects the **what** (numbers) to the **why** (meaning):

**Good Interpretation:**
> "The text conveys positive sentiment about economic achievements, with joy emerging from pride in policy success. However, underlying anger surfaces when discussing immigration, creating emotional complexity."

**What to Look For:**

- Explains *why* sentiment is classified as positive/negative/neutral
- Identifies *what* triggers dominant emotions
- Connects emotional tone to text content
- Helps understand model reasoning

---

## Technical Deep Dive

### Text Chunking Strategy

**Problem:** Transformer models have token limits (512 tokens for BERT-based models)

**Solution:** Smart chunking with averaging

```python
# Pseudocode
def analyze_long_text(text):
    chunks = split_into_510_token_chunks(text)  # Leave 2 tokens for [CLS], [SEP]
    predictions = []
    
    for chunk in chunks:
        prediction = model(chunk)
        predictions.append(prediction)
    
    # Average predictions across all chunks
    final_prediction = np.mean(predictions, axis=0)
    return final_prediction
```

**Benefits:**

- Handles texts of any length
- Maintains context within each chunk
- Stable predictions through averaging
- Transparent (returns `num_chunks` for validation)

### Model Initialization

**Lazy Loading:** Models download and load on first request, not at startup. This reduces startup time and memory usage when features aren't needed.

**Caching:** Once loaded, models stay in memory for subsequent requests, providing fast response times.

**First-Time Model Download:**

- Downloads models from HuggingFace Hub (~1.5 GB total)
- Caches to `~/.cache/huggingface/` for reuse
- Only happens once per environment

**Memory Usage (After Loading):**

- FinBERT: ~440 MB
- RoBERTa-Emotion: ~330 MB
- **Total:** ~770 MB (loaded once, reused)
- **Recommendation:** Minimum 2 GB free RAM

### LLM Integration

**Prompt Engineering:**

The system sends Gemini a structured prompt with:

1. Text excerpt (first 600 chars)
2. Sentiment scores from FinBERT
3. Emotion scores from RoBERTa
4. Task instructions

**Example Prompt:**

```text
You are analyzing the emotional and sentimental tone of a text excerpt...

TEXT ANALYZED:
"They're burning Minneapolis..."

SENTIMENT ANALYSIS RESULTS:
- Overall Sentiment: NEGATIVE (72% confidence)
- Positive: 15%
- Negative: 72%
- Neutral: 13%

EMOTION DETECTION RESULTS:
- Primary Emotion: Anger (62%)
- Top 3 Emotions: anger (62%), fear (12%), sadness (8%)

TASK:
Write a 2-3 sentence interpretation that:
1. Explains WHY the text received a negative sentiment score
2. Explains WHY anger is the dominant emotion
3. Connects both findings to specific aspects of the text content
```

**Safety Handling:**

- Checks for blocked responses
- Validates response completeness
- Logs finish reasons and safety ratings
- Provides detailed fallback if LLM fails

---

## Performance

### Latency

**First Request:** ~30-60 seconds

- One-time model downloads (~1.5 GB)
- Model initialization

**Subsequent Requests:**

- **Short text (<512 tokens):** ~500-1000ms
- **Medium text (1-3 chunks):** ~1-2 seconds
- **Long text (5+ chunks):** ~3-5 seconds

**Breakdown:**

- FinBERT: ~200-500ms per chunk
- RoBERTa: ~200-400ms
- Gemini LLM: ~1-2 seconds
- **Total:** 2-4 seconds typical

### Throughput

- Single request: 2-4 seconds
- Concurrent requests: Handled by FastAPI async
- Bottleneck: Model inference (CPU-bound)

### Optimization Opportunities

1. **GPU acceleration:** This project uses PyTorch CPU-only builds for portability. For GPU acceleration:

   ```bash
   # Install PyTorch GPU dependencies (not in default dependencies)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   Expected speedup: 5-10x on compatible GPUs

2. **Batch processing:** Process multiple texts together (currently processes one at a time)
3. **Async LLM:** LLM interpretation doesn't block model inference but adds 1-2s latency
4. **Caching:** Cache results for identical inputs (not currently implemented)

---

## Installation & Setup

### Prerequisites

**Python Version:** 3.11 or 3.12 (as specified in `pyproject.toml`)

**Package Manager:** This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot.git
cd Trump-Rally-Speeches-NLP-Chatbot

# Create virtual environment and install dependencies
uv sync

# Copy environment template and configure
cp .env.example .env
# Edit .env and add your LLM_API_KEY

# Run the API server
uv run uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Configuration

### Environment Variables

```bash
# LLM Configuration (required for contextual interpretation)
LLM_API_KEY=your_api_key_here
LLM_PROVIDER=gemini  # Options: gemini, openai, anthropic
LLM_MODEL_NAME=gemini-2.0-flash-exp  # Or gemini-2.5-flash

# Alternative: Use OpenAI
# LLM_API_KEY=sk-your_openai_key
# LLM_PROVIDER=openai
# LLM_MODEL_NAME=gpt-4o-mini

# Alternative: Use Claude
# LLM_API_KEY=sk-ant-your_key
# LLM_PROVIDER=anthropic
# LLM_MODEL_NAME=claude-3-5-sonnet-20241022
```

### Custom Models

You can override default models in code:

```python
from src.services.sentiment_service import EnhancedSentimentAnalyzer

analyzer = EnhancedSentimentAnalyzer(
    sentiment_model="distilbert-base-uncased-finetuned-sst-2-english",  # Generic sentiment
    emotion_model="j-hartmann/emotion-english-distilroberta-base",
    llm_service=my_llm_service
)
```

### LLM Provider Options

**Default:** Gemini (included in base dependencies)

**Optional Providers:** Install via dependency groups:

```bash
# OpenAI GPT support
uv sync --group llm-openai
# Set in .env: LLM_PROVIDER=openai, LLM_MODEL_NAME=gpt-4o-mini

# Anthropic Claude support  
uv sync --group llm-anthropic
# Set in .env: LLM_PROVIDER=anthropic, LLM_MODEL_NAME=claude-3-5-sonnet-20241022
```

**Benefits of Optional Dependencies:**

- Smaller install size (only install what you need)
- Faster dependency resolution
- Reduced security surface area
- Follows modern Python best practices (PEP 735)

---

## Common Use Cases

### 1. Political Speech Analysis

Analyze emotional tone and sentiment of political speeches:

```python
speech_text = load_speech("rally_speech.txt")
result = analyze_sentiment(speech_text)

print(f"Overall sentiment: {result['sentiment']}")
print(f"Top 3 emotions: {get_top_emotions(result['emotions'], 3)}")
print(f"Interpretation: {result['contextual_sentiment']}")
```

### 2. Sentiment Tracking Over Time

Track sentiment changes across multiple speeches:

```python
speeches = load_all_speeches()
sentiment_timeline = []

for speech in speeches:
    result = analyze_sentiment(speech['text'])
    sentiment_timeline.append({
        'date': speech['date'],
        'sentiment': result['sentiment'],
        'confidence': result['confidence'],
        'primary_emotion': max(result['emotions'].items(), key=lambda x: x[1])[0]
    })

# Plot sentiment over time
plot_sentiment_timeline(sentiment_timeline)
```

### 3. Emotion-Based Classification

Categorize documents by dominant emotion:

```python
def categorize_by_emotion(texts):
    categories = {emotion: [] for emotion in ["anger", "joy", "fear", "sadness", "surprise", "disgust", "neutral"]}
    
    for text in texts:
        result = analyze_sentiment(text)
        primary_emotion = max(result['emotions'].items(), key=lambda x: x[1])[0]
        categories[primary_emotion].append(text)
    
    return categories
```

### 4. Combined Sentiment + Topic Analysis

Understand what topics drive different sentiments:

```python
def analyze_sentiment_by_topic(text):
    # Get sentiment
    sentiment_result = analyze_sentiment(text)
    
    # Get topics
    topics_result = analyze_topics(text)
    
    return {
        'overall_sentiment': sentiment_result['sentiment'],
        'primary_emotion': max(sentiment_result['emotions'].items(), key=lambda x: x[1])[0],
        'main_topics': [t['label'] for t in topics_result['clustered_topics'][:3]],
        'interpretation': sentiment_result['contextual_sentiment']
    }
```

---

## Development Workflow

### Running Tests

```bash
# Run sentiment analysis tests
uv run pytest tests/test_confidence.py -v

# Run with coverage
uv run pytest tests/test_confidence.py --cov=src.services.sentiment_service

# Run all NLP tests
uv run pytest tests/test_*.py -k sentiment
```

### Code Quality

```bash
# Lint and format code
uv run ruff check src/services/sentiment_service.py
uv run ruff format src/services/sentiment_service.py

# Type checking
uv run mypy src/services/sentiment_service.py
```

### Local Development

```bash
# Run with hot reload
uv run uvicorn src.main:app --reload --log-level debug

# Test endpoint manually
curl -X POST http://localhost:8000/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Test sentiment analysis"}'
```

## Troubleshooting

### Service Unavailable (503)

**Error:**

```json
{
  "detail": "Sentiment models not loaded. Please try again later."
}
```

**Causes:**

- Models still downloading (first request, ~1.5 GB total)
- Model initialization failed
- Out of memory (need ~2 GB free)
- Dependency issues

**Solutions:**

1. **Wait for model download:** Check logs for download progress

   ```bash
   # View logs in development
   uv run uvicorn src.main:app --log-level debug
   ```

2. **Check available memory:** Models require ~770 MB + overhead

   ```bash
   # Linux/Mac
   free -h
   # Windows (PowerShell)
   Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory
   ```

3. **Verify dependencies:** Ensure all required packages installed

   ```bash
   uv sync  # Reinstall dependencies
   ```

4. **Check initialization logs:** Look for errors in startup logs
5. **Restart application:** Clear any stuck states

   ```bash
   # Kill existing process and restart
   uv run uvicorn src.main:app --reload
   ```

### Empty or Missing Contextual Interpretation

**Symptom:** `contextual_sentiment` is generic or missing

**Causes:**

- Gemini API key not configured
- LLM response blocked by safety filters
- Rate limiting

**Solutions:**

1. Set `LLM_API_KEY` in `.env` (see `.env.example` for template)
2. Verify `LLM_PROVIDER` is correctly set (gemini/openai/anthropic)
3. Check logs for LLM errors
4. Verify API quota/limits
5. System still works with fallback interpretation

### Unexpected Neutral Sentiment

**Symptom:** Everything classified as neutral

**Causes:**

- Text is genuinely balanced
- Complex/nuanced emotional tone
- FinBERT uncertain

**Analysis:**

- Check `scores` for probability distribution
- If all scores are ~0.33, truly neutral/mixed
- Check `contextual_sentiment` for LLM interpretation
- Review top 3 emotions for emotional complexity

### Performance Issues

**Symptom:** Requests taking >10 seconds

**Solutions:**

1. Check `num_chunks` in response (more chunks = slower)
2. Consider shorter input texts
3. Deploy on GPU for faster inference
4. Disable LLM interpretation if not needed

---

## API Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid input format"
}
```

**Cause:** Missing or malformed `text` field

### 500 Internal Server Error

```json
{
  "detail": "Analysis failed: [error details]"
}
```

**Causes:**

- Model inference error
- Out of memory
- Unexpected input format

### 503 Service Unavailable

```json
{
  "detail": "Sentiment models not loaded. Please try again later."
}
```

**Cause:** Models not initialized

---

## Limitations

### Current Limitations

1. **Political Bias:** FinBERT trained on financial news may miss political nuances
2. **Context Window:** 512 tokens per chunk may lose long-range context
3. **Emotion Granularity:** 7 emotions may not capture all emotional nuances
4. **Latency:** 2-4 seconds per request on CPU

### Future Enhancements

- [ ] Fine-tune models on political speech corpus
- [ ] Add support for multilingual sentiment
- [ ] Implement sentiment aspect extraction (sentiment per topic)
- [ ] Add temporal sentiment tracking
- [ ] Support for custom emotion taxonomies
- [ ] GPU acceleration support

---

## See Also

- [Q&A System](qa-system.md) - RAG-based question answering
- [Topic Analysis](topic-analysis.md) - AI-powered topic extraction
- [Architecture](architecture.md) - System architecture overview
- [API Documentation](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs) - Interactive API docs
- [Development Guide](../development/testing.md) - Testing and development practices
- [GitHub Repository](https://github.com/JustaKris/Trump-Rally-Speeches-NLP-Chatbot) - Source code and pyproject.toml
