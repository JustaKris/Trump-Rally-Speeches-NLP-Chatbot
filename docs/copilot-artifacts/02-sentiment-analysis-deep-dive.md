# Sentiment Analysis Deep Dive: Multi-Model Ensemble Explained

**Understanding Emotion Detection and Sentiment Classification**

---

## What Is Sentiment Analysis?

**Definition:**
Sentiment analysis is the computational task of identifying and categorizing opinions, emotions, and attitudes expressed in text.

**The Goal:**
Determine whether text is:
- **Positive** (e.g., praise, optimism, success)
- **Negative** (e.g., criticism, anger, failure)
- **Neutral** (e.g., factual, balanced, informational)

**Beyond Simple Classification:**
Your system doesn't just say "positive" or "negative"—it provides:
- Fine-grained emotion detection (anger, joy, fear, sadness, etc.)
- Confidence scores for each classification
- AI-generated contextual interpretation explaining *why*

---

## Why Multi-Model Ensemble?

### The Limitation of Single Models

**Problem:**
No single model is perfect for all aspects of sentiment.

**Example:**
```
Text: "The radical left is destroying our country, but we're fighting back!"

Single Model Approach:
- Sentiment: Negative (focuses on "destroying")
- Misses: The defiant tone, the "fighting back" rallying cry
```

**Your Solution: Use Multiple Specialized Models**

1. **FinBERT** — Political/financial sentiment (positive/negative/neutral)
2. **RoBERTa-Emotion** — 7 specific emotions (anger, joy, fear, sadness, surprise, disgust, neutral)
3. **Gemini LLM** — Contextual interpretation explaining emotional tone

**Why This Works:**
- Each model focuses on different aspects
- Combined output provides richer understanding
- Cross-validation between models increases confidence

---

## The Three Models Explained

### Model 1: FinBERT (Sentiment Classifier)

**Full Name:** Financial BERT (ProsusAI/finbert)

**What It Does:**
Classifies text into three sentiment categories with probability scores.

**Training Data:**
- Financial news articles and reports
- Political and economic discourse
- Formal, policy-oriented language

**Why FinBERT for Political Speeches?**
- Political speeches use similar language to financial discourse
- Trained on formal, persuasive text
- Better than generic sentiment models (which train on movie reviews)
- Produces reliable confidence scores

**Output Example:**
```python
Input: "The economy is booming, we've added millions of jobs!"

Output: {
    "positive": 0.89,
    "negative": 0.03,
    "neutral": 0.08,
    "dominant": "positive"
}
```

**How It Works (Simplified):**
1. **Tokenization** — Split text into subword tokens
2. **Encoding** — Pass through BERT transformer (12 layers)
3. **Classification Head** — Final layer predicts probabilities
4. **Softmax** — Normalize to sum to 1.0

**Technical Details:**
- **Architecture:** BERT-base (110M parameters)
- **Max Length:** 512 tokens
- **Processing:** Chunks long texts, averages predictions
- **Speed:** ~200-500ms per chunk on CPU

### Model 2: RoBERTa-Emotion (Emotion Detector)

**Full Name:** Emotion English DistilRoBERTa Base (j-hartmann/emotion-english-distilroberta-base)

**What It Does:**
Detects 7 specific emotions with probability scores.

**Emotions:**
1. 😠 **Anger** — Frustration, hostility, outrage
2. 😊 **Joy** — Happiness, celebration, optimism
3. 😨 **Fear** — Anxiety, threat, danger
4. 😢 **Sadness** — Disappointment, loss, despair
5. 😮 **Surprise** — Shock, unexpectedness
6. 🤢 **Disgust** — Revulsion, contempt, disdain
7. 😐 **Neutral** — Factual, informational, balanced

**Training Data:**
- 58,000 emotion-labeled texts
- Social media posts, news, dialogue
- Diverse sources for generalization

**Why This Model?**
- State-of-the-art for emotion detection
- Balanced across emotion categories
- Distilled for faster inference
- Provides nuanced emotional profile

**Output Example:**
```python
Input: "They're burning Minneapolis! This is outrageous!"

Output: {
    "anger": 0.72,
    "fear": 0.15,
    "disgust": 0.08,
    "surprise": 0.03,
    "joy": 0.01,
    "sadness": 0.01,
    "neutral": 0.00
}
```

**How It Works:**
1. **Tokenization** — RoBERTa-style byte-pair encoding
2. **Encoding** — DistilRoBERTa transformer (6 layers—faster than full)
3. **Emotion Head** — 7-class classification layer
4. **Softmax** — Normalize probabilities

**Technical Details:**
- **Architecture:** DistilRoBERTa (82M parameters, distilled from 125M)
- **Max Length:** 512 tokens
- **Processing:** Handles text chunking automatically
- **Speed:** ~200-400ms per chunk on CPU

### Model 3: Gemini LLM (Contextual Interpreter)

**Full Name:** Google Gemini 2.0 Flash (gemini-2.0-flash-exp)

**What It Does:**
Generates a 2-3 sentence human-readable explanation of:
- **WHY** the text has its sentiment classification
- **WHY** certain emotions dominate
- **WHAT** specifically triggers those emotions

**Input to Gemini:**
```
You are analyzing the emotional and sentimental tone of a text excerpt...

TEXT ANALYZED:
"They're burning Minneapolis! The city is burning down!"

SENTIMENT ANALYSIS RESULTS:
- Overall Sentiment: NEGATIVE (72% confidence)
- Positive: 15%
- Negative: 72%
- Neutral: 13%

EMOTION DETECTION RESULTS:
- Primary Emotion: Anger (72%)
- Top 3 Emotions: anger (72%), fear (15%), disgust (8%)

TASK:
Write a 2-3 sentence interpretation that:
1. Explains WHY the text received a negative sentiment score
2. Explains WHY anger is the dominant emotion
3. Connects both findings to specific aspects of the text content
```

**Output from Gemini:**
```
"The text expresses strong negative sentiment due to the depiction of 
Minneapolis burning, conveying a sense of crisis and destruction. Anger 
emerges as the dominant emotion, reflecting outrage at perceived urban 
unrest and lawlessness. The inflammatory language amplifies the negative 
tone and emotional intensity."
```

**Why Add LLM Interpretation?**
- **Numbers aren't enough** — Users want to understand *why*
- **Contextual understanding** — LLM connects sentiment to content
- **Educational** — Helps users learn to read sentiment signals
- **Transparency** — Makes model decisions interpretable

**Fallback Behavior:**
If Gemini is unavailable (no API key, rate limit, error):
```python
fallback = f"The text shows {sentiment} sentiment with {emotion} as the dominant emotion."
```

**Technical Details:**
- **Model:** Gemini 2.0 Flash (fast, cost-effective)
- **Temperature:** 0.3 (focused, less creative)
- **Max Tokens:** 1024 (plenty for 2-3 sentences)
- **Cost:** ~$0.000075 per request (very cheap)
- **Speed:** ~1-2 seconds per call

---

## How the Ensemble Works Together

### Step-by-Step Processing

#### Step 1: Text Chunking (if needed)

**Problem:** Transformer models have token limits (512 tokens)

**Your Solution:**
```python
def chunk_text_for_bert(text: str, max_length: int = 510) -> List[str]:
    """Split long texts into 510-token chunks."""
    # Leave 2 tokens for [CLS] and [SEP]
    chunks = []
    tokens = tokenizer.tokenize(text)
    
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks
```

**Example:**
```
Long speech (1500 words):
├── Chunk 1: Tokens 0-510
├── Chunk 2: Tokens 510-1020
└── Chunk 3: Tokens 1020-1250

Each chunk processed independently, results averaged.
```

#### Step 2: FinBERT Sentiment Analysis

**Process Each Chunk:**
```python
def analyze_sentiment_scores(text: str) -> Dict:
    chunks = chunk_text_for_bert(text)
    predictions = []
    
    for chunk in chunks:
        # Tokenize
        inputs = sentiment_tokenizer(chunk, return_tensors="pt")
        
        # Forward pass
        outputs = sentiment_model(**inputs)
        
        # Get probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        predictions.append({
            "positive": probs[0][0].item(),
            "negative": probs[0][1].item(),
            "neutral": probs[0][2].item()
        })
    
    # Average across chunks
    avg_positive = mean([p["positive"] for p in predictions])
    avg_negative = mean([p["negative"] for p in predictions])
    avg_neutral = mean([p["neutral"] for p in predictions])
    
    # Determine dominant
    dominant = max(
        [("positive", avg_positive), 
         ("negative", avg_negative), 
         ("neutral", avg_neutral)],
        key=lambda x: x[1]
    )[0]
    
    return {
        "positive": avg_positive,
        "negative": avg_negative,
        "neutral": avg_neutral,
        "dominant": dominant,
        "num_chunks": len(chunks)
    }
```

#### Step 3: RoBERTa Emotion Detection

**Process Each Chunk:**
```python
def analyze_emotions(text: str) -> Dict[str, float]:
    chunks = chunk_text_for_bert(text)
    predictions = []
    
    for chunk in chunks:
        # Tokenize
        inputs = emotion_tokenizer(chunk, return_tensors="pt")
        
        # Forward pass
        outputs = emotion_model(**inputs)
        
        # Get probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        predictions.append({
            "anger": probs[0][0].item(),
            "disgust": probs[0][1].item(),
            "fear": probs[0][2].item(),
            "joy": probs[0][3].item(),
            "neutral": probs[0][4].item(),
            "sadness": probs[0][5].item(),
            "surprise": probs[0][6].item()
        })
    
    # Average across chunks
    return {
        emotion: mean([p[emotion] for p in predictions])
        for emotion in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    }
```

#### Step 4: Gemini Contextual Interpretation

**Generate Explanation:**
```python
def generate_contextual_interpretation(
    text: str,
    sentiment_scores: Dict,
    emotions: Dict
) -> str:
    # Truncate text for API (first 600 chars)
    text_excerpt = text[:600]
    
    # Get top 3 emotions
    top_emotions = sorted(
        emotions.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    # Build prompt
    prompt = f"""You are analyzing the emotional and sentimental tone of a text excerpt...

TEXT ANALYZED:
"{text_excerpt}"

SENTIMENT ANALYSIS RESULTS:
- Overall Sentiment: {sentiment_scores['dominant'].upper()} ({sentiment_scores[sentiment_scores['dominant']] * 100:.0f}% confidence)
- Positive: {sentiment_scores['positive'] * 100:.0f}%
- Negative: {sentiment_scores['negative'] * 100:.0f}%
- Neutral: {sentiment_scores['neutral'] * 100:.0f}%

EMOTION DETECTION RESULTS:
- Primary Emotion: {top_emotions[0][0].title()} ({top_emotions[0][1] * 100:.0f}%)
- Top 3 Emotions: {', '.join([f"{e[0]} ({e[1]*100:.0f}%)" for e in top_emotions])}

TASK:
Write a 2-3 sentence interpretation that:
1. Explains WHY the text received a {sentiment_scores['dominant']} sentiment score
2. Explains WHY {top_emotions[0][0]} is the dominant emotion
3. Connects both findings to specific aspects of the text content

Keep it concise, analytical, and objective.
"""
    
    # Call Gemini
    response = gemini_client.generate_content(prompt)
    
    return response.text
```

#### Step 5: Combine All Results

**Final Response:**
```python
def analyze_sentiment(text: str) -> Dict:
    # Step 2: FinBERT
    sentiment_result = analyze_sentiment_scores(text)
    
    # Step 3: RoBERTa
    emotions = analyze_emotions(text)
    
    # Step 4: Gemini
    contextual_sentiment = generate_contextual_interpretation(
        text,
        sentiment_result,
        emotions
    )
    
    # Step 5: Combine
    return {
        "sentiment": sentiment_result["dominant"],
        "confidence": sentiment_result[sentiment_result["dominant"]],
        "scores": {
            "positive": sentiment_result["positive"],
            "negative": sentiment_result["negative"],
            "neutral": sentiment_result["neutral"]
        },
        "emotions": emotions,
        "contextual_sentiment": contextual_sentiment,
        "num_chunks": sentiment_result["num_chunks"]
    }
```

---

## Understanding the Output

### Example Analysis

**Input Text:**
```
"The economy is booming like never before! We've added 11 million jobs, 
manufacturing is returning to America, and unemployment is at historic 
lows. This is what winning looks like!"
```

**Output:**
```json
{
  "sentiment": "positive",
  "confidence": 0.89,
  "scores": {
    "positive": 0.89,
    "negative": 0.03,
    "neutral": 0.08
  },
  "emotions": {
    "joy": 0.75,
    "surprise": 0.12,
    "neutral": 0.08,
    "anger": 0.03,
    "sadness": 0.01,
    "fear": 0.01,
    "disgust": 0.00
  },
  "contextual_sentiment": "The text conveys overwhelmingly positive sentiment through its focus on economic achievements and success metrics like job creation and low unemployment. Joy emerges as the dominant emotion, reflecting pride and celebration of policy outcomes. The triumphant language ('winning', 'booming') reinforces the positive, optimistic tone.",
  "num_chunks": 1
}
```

### Interpreting the Results

**Sentiment Classification:**
- **Dominant:** Positive (89% confidence)
- **Interpretation:** Very confident classification, clear positive framing

**Emotion Breakdown:**
- **Primary:** Joy (75%)—celebration of achievements
- **Secondary:** Surprise (12%)—"booming like never before" suggests unexpected success
- **Low emotions:** Negative emotions nearly absent (anger 3%, sadness 1%)

**Contextual Interpretation:**
- Connects sentiment to specific content (economic achievements)
- Explains *why* joy dominates (pride in policy outcomes)
- Links language choices to emotional tone ("winning", "booming")

**Number of Chunks:**
- 1 chunk—short text, processed in single pass
- More chunks = longer text, averaged predictions

---

## Why This Approach Is Powerful

### 1. **Emotional Nuance**

**Traditional Approach:**
```
Input: "They're destroying our country, but we won't let them!"
Output: Negative sentiment
```

**Your Approach:**
```
Input: "They're destroying our country, but we won't let them!"
Output:
- Sentiment: Negative (72%)
- Primary Emotion: Anger (65%)
- Secondary: Fear (20%)
- Interpretation: "Negative sentiment stems from threat framing ('destroying'), 
  while anger reflects defiance ('we won't let them'). The language combines 
  perceived victimization with combative resistance."
```

### 2. **Cross-Validation**

**If Models Disagree:**
```
FinBERT: Negative (60%)
RoBERTa: Joy (55%) + Anger (30%)
Interpretation: Mixed signals detected
```

This indicates **complex emotional landscape** (not simple positive/negative).

### 3. **Explainability**

**Users Get:**
- Not just "negative"
- But "negative because X, with anger due to Y, reflecting Z"

**Value:**
- Builds trust in the system
- Educational for users
- Helps identify model weaknesses

---

## Real-World Applications

### 1. Political Speech Analysis

**Use Case:** Track emotional tone across multiple speeches

**Approach:**
```python
speeches = load_all_speeches()
emotional_profile = []

for speech in speeches:
    result = analyze_sentiment(speech["text"])
    emotional_profile.append({
        "date": speech["date"],
        "sentiment": result["sentiment"],
        "primary_emotion": max(result["emotions"].items(), key=lambda x: x[1])[0],
        "confidence": result["confidence"]
    })

# Plot sentiment over time
plot_sentiment_timeline(emotional_profile)
```

### 2. Social Media Monitoring

**Use Case:** Analyze public reaction to events

**Approach:**
- Collect tweets/posts about an event
- Run sentiment analysis on each
- Aggregate to understand overall sentiment
- Track dominant emotions (anger spike = controversy)

### 3. Customer Feedback Analysis

**Use Case:** Understand customer satisfaction

**Approach:**
- Analyze product reviews
- Categorize by sentiment and emotion
- Identify pain points (high sadness/disgust = bad UX)
- Track improvement over time

### 4. Content Moderation

**Use Case:** Flag potentially harmful content

**Approach:**
- High anger + high disgust = potential toxicity
- High fear + negative sentiment = alarming content
- Use for prioritization in moderation queues

---

## Performance Optimization

### Model Loading Strategy

**Lazy Loading:**
```python
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self._sentiment_model = None  # Load on first use
        self._emotion_model = None
        
    @property
    def sentiment_model(self):
        if self._sentiment_model is None:
            self._sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            )
        return self._sentiment_model
```

**Benefits:**
- Faster startup (don't load models until needed)
- Lower memory (only load what's used)
- Models cached after first use

### Memory Management

**Models in Memory:**
- FinBERT: ~440 MB
- RoBERTa-Emotion: ~330 MB
- **Total:** ~770 MB

**Optimization:**
- Load once, reuse for all requests
- Use CPU inference (GPU for production scale)
- Consider model quantization for smaller footprint

### Latency Breakdown

**Typical Request (500 words):**
```
Text chunking: 10ms
FinBERT inference: 500ms (1 chunk)
RoBERTa inference: 400ms (1 chunk)
Gemini API call: 1500ms
Total: ~2.4 seconds
```

**Long Text (2000 words):**
```
Text chunking: 20ms
FinBERT inference: 2000ms (4 chunks)
RoBERTa inference: 1600ms (4 chunks)
Gemini API call: 1500ms
Total: ~5.1 seconds
```

### Optimization Ideas

**Batch Processing:**
```python
# Instead of processing chunks sequentially
for chunk in chunks:
    result = model(chunk)  # Slow

# Batch process all chunks at once
results = model(chunks)  # Faster (uses GPU parallelism)
```

**Async LLM:**
```python
# Don't block on Gemini
async def analyze_sentiment(text):
    sentiment_future = async_analyze_sentiment(text)
    emotion_future = async_analyze_emotions(text)
    
    sentiment, emotions = await asyncio.gather(sentiment_future, emotion_future)
    
    # Gemini call happens in parallel
    interpretation = await async_generate_interpretation(text, sentiment, emotions)
```

**Caching:**
```python
# Cache LLM interpretations (expensive)
@lru_cache(maxsize=100)
def get_interpretation(text_hash):
    return gemini.generate(text_hash)
```

---

## Testing Strategy

### Unit Tests

**Model Initialization:**
```python
def test_models_load_successfully():
    analyzer = EnhancedSentimentAnalyzer()
    assert analyzer.sentiment_model is not None
    assert analyzer.emotion_model is not None
```

**Sentiment Classification:**
```python
def test_positive_sentiment():
    text = "The economy is booming, jobs are growing!"
    result = analyzer.analyze_sentiment(text)
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.7
```

**Emotion Detection:**
```python
def test_anger_detection():
    text = "They're destroying our country! This is outrageous!"
    result = analyzer.analyze_sentiment(text)
    assert result["emotions"]["anger"] > 0.5
```

### Integration Tests

**Full Pipeline:**
```python
def test_complete_sentiment_analysis():
    text = "Sample political speech text..."
    result = analyzer.analyze_sentiment(text)
    
    # Check all required fields
    assert "sentiment" in result
    assert "confidence" in result
    assert "scores" in result
    assert "emotions" in result
    assert "contextual_sentiment" in result
    assert "num_chunks" in result
    
    # Validate types
    assert isinstance(result["sentiment"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["emotions"], dict)
```

---

## Common Issues & Solutions

### Issue: All Sentiments Classified as Neutral

**Symptoms:** Everything gets neutral classification (33/33/33 split)

**Diagnosis:**
- Model not loaded properly
- Input text too short or empty
- Model weights corrupted

**Solutions:**
```python
# Check model outputs
logits = model(**inputs).logits
print(f"Logits: {logits}")  # Should show clear differences

# Verify input
print(f"Input length: {len(text)}")  # Should be > 10 characters
```

### Issue: Gemini Interpretation Missing

**Symptoms:** `contextual_sentiment` is generic fallback

**Diagnosis:**
- API key not set (`LLM_API_KEY`)
- Rate limit reached
- Gemini returned empty response

**Solutions:**
```python
# Check API key
import os
print(f"API Key set: {bool(os.getenv('LLM_API_KEY'))}")

# Check Gemini logs
logger.debug(f"Gemini response: {response.text}")
logger.debug(f"Finish reason: {response.finish_reason}")
```

### Issue: Slow Performance

**Symptoms:** Requests taking >10 seconds

**Diagnosis:**
- Long text (many chunks)
- CPU-only inference
- No batching

**Solutions:**
- Limit input length
- Use GPU (see deployment docs)
- Implement batching
- Cache frequent interpretations

---

## Next Steps

**Continue Learning:**
- **`03-topic-analysis-deep-dive.md`** — Semantic clustering and topic extraction
- **`05-llm-integration.md`** — Deep dive on LLM integration patterns
- **`06-concepts-glossary.md`** — Quick reference for all concepts

**Practice Explaining:**
- Why use 3 models instead of 1?
- How does chunking work and why?
- What makes FinBERT good for political text?
- How does the LLM add value?

**Interview Questions:**
- What's the difference between sentiment and emotion?
- Why average predictions across chunks?
- How would you handle non-English text?
- What are the limitations of your approach?

---

You now understand multi-model sentiment analysis at a deep level! 🎭
