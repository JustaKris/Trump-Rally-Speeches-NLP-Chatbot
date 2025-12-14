# LLM Integration Deep Dive: Pluggable Provider Architecture

## Understanding How Large Language Models Power Your AI Features

---

## What Are LLMs and Why Do You Use Them?

### Large Language Model Basics

**Definition:**
An LLM is a neural network trained on massive text datasets to understand and generate human-like text.

**What They Do:**

- **Understand** natural language (questions, statements, context)
- **Generate** coherent text responses
- **Reason** about complex topics
- **Interpret** data and provide insights

**Your LLMs:**

- **Gemini 1.5 Flash/Pro** (Google)
- **GPT-4/GPT-3.5** (OpenAI) - optional
- **Claude 3** (Anthropic) - optional

### Why Your Project Uses LLMs

#### 1. Q&A System (RAG)

```text
User: "What did Trump say about jobs in Michigan?"

Without LLM: Return raw document chunks
→ User sees: "...jobs...Michigan...great...economy..."

With LLM: Generate coherent answer from context
→ User sees: "Trump emphasized job creation in Michigan, highlighting 
   the return of manufacturing positions and economic growth in the state."
```

**Problem LLMs Solve:** Transform retrieval results into natural answers

#### 2. Sentiment Analysis

```text
Raw Output:
- FinBERT: {"positive": 0.85, "negative": 0.10, "neutral": 0.05}
- RoBERTa: {"joy": 0.60, "surprise": 0.25, "neutral": 0.15}

Without LLM: Show numbers
→ User sees: "Sentiment: 0.85 positive, Emotions: joy 0.60"

With LLM: Generate interpretation
→ User sees: "The speech expresses strong optimism about economic 
   achievements, with an enthusiastic and celebratory tone focused 
   on job creation and market performance."
```

**Problem LLMs Solve:** Make technical outputs human-readable

#### 3. Topic Analysis

```text
Cluster: ["economy", "jobs", "market", "growth", "prosperity"]

Without LLM: Use most frequent word as label
→ Label: "Economy"

With LLM: Generate meaningful label
→ Label: "Economic Policy & Job Growth"
```

**Problem LLMs Solve:** Create meaningful topic labels from keyword clusters

---

## Pluggable Provider Architecture

### The Problem: Vendor Lock-In

**Naive Approach (Hardcoded):**

```python
import google.generativeai as genai

def answer_question(question, context):
    # Problem: Tightly coupled to Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    response = model.generate_content(prompt)
    
    return response.text
```

**Problems:**

1. Can't swap providers without rewriting code
2. Hard to test (always hits real Gemini API)
3. No flexibility for different use cases
4. Vendor lock-in risk

### Your Solution: Strategy Pattern

#### Step 1: Define Interface

```python
# src/services/llm/base.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate text completion from prompt"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return provider name for logging"""
        pass
    
    @abstractmethod
    def get_model(self) -> str:
        """Return model name"""
        pass
```

**Benefits:**

- All providers must implement same interface
- Business logic doesn't care which provider
- Easy to swap providers
- Easy to mock for testing

#### Step 2: Implement Concrete Providers

**Gemini:**

```python
# src/services/llm/gemini_provider.py
import google.generativeai as genai

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.model_name = model
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        response = await self.client.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text
    
    def get_name(self) -> str:
        return "gemini"
    
    def get_model(self) -> str:
        return self.model_name
```

**OpenAI:**

```python
# src/services/llm/openai_provider.py
from openai import AsyncOpenAI

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def get_name(self) -> str:
        return "openai"
    
    def get_model(self) -> str:
        return self.model
```

**Claude:**

```python
# src/services/llm/claude_provider.py
import anthropic

class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def get_name(self) -> str:
        return "claude"
    
    def get_model(self) -> str:
        return self.model
```

#### Step 3: Factory Pattern

```python
# src/services/llm/factory.py
from typing import Dict, Type
from .base import LLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .claude_provider import ClaudeProvider

class LLMFactory:
    """Factory for creating LLM providers"""
    
    # Registry of available providers
    _providers: Dict[str, Type[LLMProvider]] = {
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
        "claude": ClaudeProvider
    }
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: str,
        model: str = None
    ) -> LLMProvider:
        """Create and return LLM provider instance"""
        
        # Validate provider exists
        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {available}"
            )
        
        # Get provider class
        provider_class = cls._providers[provider_name]
        
        # Create instance
        if model:
            return provider_class(api_key=api_key, model=model)
        else:
            return provider_class(api_key=api_key)
    
    @classmethod
    def register_provider(
        cls,
        name: str,
        provider_class: Type[LLMProvider]
    ):
        """Register a custom provider (for extensibility)"""
        cls._providers[name] = provider_class
```

#### Step 4: LLM Service (High-Level API)

```python
# src/services/llm_service.py
from .llm.base import LLMProvider

class LLMService:
    """High-level LLM service for business logic"""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate text with current provider"""
        try:
            response = await self.provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Log
            logger.info(
                f"LLM generation completed",
                extra={
                    "provider": self.provider.get_name(),
                    "model": self.provider.get_model(),
                    "prompt_length": len(prompt),
                    "response_length": len(response)
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"LLM generation failed: {e}",
                extra={"provider": self.provider.get_name()}
            )
            raise
    
    async def generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """Generate answer for Q&A system"""
        prompt = self._build_qa_prompt(question, context)
        return await self.generate(
            prompt=prompt,
            temperature=0.5,  # Lower temp for factual answers
            max_tokens=300
        )
    
    async def generate_interpretation(
        self,
        sentiment_data: Dict
    ) -> str:
        """Generate sentiment interpretation"""
        prompt = self._build_sentiment_prompt(sentiment_data)
        return await self.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=200
        )
    
    async def generate_topic_label(
        self,
        keywords: List[str]
    ) -> str:
        """Generate topic cluster label"""
        prompt = self._build_topic_label_prompt(keywords)
        return await self.generate(
            prompt=prompt,
            temperature=0.3,  # Low temp for concise labels
            max_tokens=20
        )
    
    # Prompt builders
    def _build_qa_prompt(self, question: str, context: str) -> str:
        return f"""You are a helpful AI assistant answering questions about 
political speeches. Use only the information provided in the context.

CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, factual answer based solely on the context. If the context 
doesn't contain relevant information, say "I don't have enough information 
to answer that question."

ANSWER:"""
    
    def _build_sentiment_prompt(self, data: Dict) -> str:
        sentiment = data["sentiment"]
        emotions = data["emotions"]
        
        return f"""Analyze the following sentiment analysis results and provide 
a 2-3 sentence interpretation in natural language.

SENTIMENT SCORES:
- Positive: {sentiment.get('positive', 0):.2f}
- Negative: {sentiment.get('negative', 0):.2f}
- Neutral: {sentiment.get('neutral', 0):.2f}

EMOTION SCORES:
{json.dumps(emotions, indent=2)}

Write a concise interpretation that explains the overall tone and emotional 
content of the text. Be objective and analytical.

INTERPRETATION:"""
    
    def _build_topic_label_prompt(self, keywords: List[str]) -> str:
        keyword_str = ", ".join(keywords)
        return f"""You are a topic labeling expert. Given a cluster of related 
keywords from a political speech, generate a concise, descriptive label 
(2-4 words) that captures the main theme.

KEYWORDS: {keyword_str}

Generate a label that represents this topic cluster. Use title case. Be 
specific and meaningful. Examples of good labels: "Economic Policy", 
"Border Security", "Healthcare Reform".

LABEL:"""
```

---

## Configuration & Environment Setup

### Environment Variables

**.env File:**

```bash
# Choose provider: gemini, openai, or claude
LLM_PROVIDER=gemini

# API key for chosen provider
LLM_API_KEY=your_api_key_here

# Optional: Specify model
LLM_MODEL=gemini-1.5-flash

# Optional: Override temperature
LLM_TEMPERATURE=0.7
```

### Configuration Files

**Development:**

```yaml
# configs/development.yaml
llm:
  provider: "gemini"
  model: "gemini-1.5-flash"  # Fast, cheap for dev
  temperature: 0.7
  max_tokens: 500
  fallback_enabled: true
```

**Production:**

```yaml
# configs/production.yaml
llm:
  provider: "gemini"
  model: "gemini-1.5-pro"  # More capable for prod
  temperature: 0.5  # More deterministic
  max_tokens: 500
  fallback_enabled: true
  timeout: 30
```

### Loading Configuration

```python
# src/config/llm_config.py
def load_llm_config() -> Dict:
    """Load LLM configuration with environment overrides"""
    
    # Load from YAML
    env = os.getenv("APP_ENV", "development")
    config = load_yaml(f"configs/{env}.yaml")["llm"]
    
    # Override with environment variables
    if "LLM_PROVIDER" in os.environ:
        config["provider"] = os.getenv("LLM_PROVIDER")
    
    if "LLM_API_KEY" in os.environ:
        config["api_key"] = os.getenv("LLM_API_KEY")
    
    if "LLM_MODEL" in os.environ:
        config["model"] = os.getenv("LLM_MODEL")
    
    if "LLM_TEMPERATURE" in os.environ:
        config["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
    
    # Validate
    if "api_key" not in config:
        raise ValueError("LLM_API_KEY must be set")
    
    return config
```

### Initialization

```python
# src/main.py
def initialize_llm_service() -> LLMService:
    """Initialize LLM service with configured provider"""
    
    # Load config
    config = load_llm_config()
    
    # Create provider
    provider = LLMFactory.create_provider(
        provider_name=config["provider"],
        api_key=config["api_key"],
        model=config.get("model")
    )
    
    # Create service
    service = LLMService(provider=provider)
    
    logger.info(
        f"Initialized LLM service",
        extra={
            "provider": provider.get_name(),
            "model": provider.get_model()
        }
    )
    
    return service
```

---

## Prompt Engineering Best Practices

### 1. Clear Instructions

**Bad:**

```python
prompt = f"Question: {question}\nContext: {context}"
```

**Good:**

```python
prompt = f"""You are a helpful AI assistant. Answer the question using ONLY 
the provided context. Be factual and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
```

**Why:** Clear role, constraints, and structure improve output quality

### 2. Few-Shot Examples

**Example:**

```python
prompt = f"""Generate a topic label for the following keywords.

EXAMPLES:
Keywords: economy, jobs, market, growth
Label: Economic Policy

Keywords: immigration, border, wall, security
Label: Border Security

YOUR TASK:
Keywords: {keywords}
Label:"""
```

**Why:** Examples demonstrate desired output format

### 3. Temperature Tuning

**Factual Tasks (Low Temperature):**

```python
# Q&A answers - want factual accuracy
answer = await llm.generate(prompt, temperature=0.3)

# Topic labels - want consistency
label = await llm.generate(prompt, temperature=0.2)
```

**Creative Tasks (High Temperature):**

```python
# Sentiment interpretation - want nuance
interpretation = await llm.generate(prompt, temperature=0.7)
```

**Temperature Guide:**

- **0.0-0.3**: Deterministic, factual (Q&A, labels)
- **0.4-0.7**: Balanced (interpretations, summaries)
- **0.8-1.0**: Creative (brainstorming, variations)

### 4. Output Constraints

**Specify Length:**

```python
prompt = f"""Generate a 2-3 sentence interpretation.
...
INTERPRETATION (2-3 sentences):"""

# Also limit tokens
await llm.generate(prompt, max_tokens=150)
```

**Specify Format:**

```python
prompt = f"""Generate a label in title case (2-4 words).
Examples: "Economic Policy", "Border Security"
...
LABEL (2-4 words, title case):"""
```

---

## Error Handling & Fallbacks

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMService:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate with automatic retries"""
        return await self.provider.generate(prompt, **kwargs)
```

### Timeout Handling

```python
import asyncio

async def generate_with_timeout(
    self,
    prompt: str,
    timeout: int = 30,
    **kwargs
) -> str:
    """Generate with timeout"""
    try:
        result = await asyncio.wait_for(
            self.provider.generate(prompt, **kwargs),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"LLM timeout after {timeout}s")
        raise LLMTimeoutError(f"Request exceeded {timeout}s")
```

### Fallback Strategies

#### Strategy 1: Fallback Provider

```python
class LLMService:
    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_provider: LLMProvider = None
    ):
        self.primary = primary_provider
        self.fallback = fallback_provider
    
    async def generate(self, prompt: str, **kwargs) -> str:
        try:
            return await self.primary.generate(prompt, **kwargs)
        except Exception as e:
            if self.fallback:
                logger.warning(
                    f"Primary provider failed, using fallback",
                    extra={"error": str(e)}
                )
                return await self.fallback.generate(prompt, **kwargs)
            raise
```

#### Strategy 2: Cached Fallback

```python
async def generate(self, prompt: str, **kwargs) -> str:
    # Try cache first
    cached = await self.cache.get(prompt)
    if cached:
        return cached
    
    # Generate new
    try:
        result = await self.provider.generate(prompt, **kwargs)
        await self.cache.set(prompt, result)
        return result
    except Exception:
        # If generation fails, return generic fallback
        logger.error("LLM generation failed, using fallback")
        return self._get_fallback_response()
```

#### Strategy 3: Degraded Mode

```python
async def answer_question(self, question: str, context: str) -> str:
    try:
        # Try LLM
        return await self.llm.generate_answer(question, context)
    except LLMError:
        # Fallback to simple extraction
        logger.warning("LLM unavailable, using extraction fallback")
        return self._extract_answer_simple(question, context)

def _extract_answer_simple(self, question: str, context: str) -> str:
    """Simple keyword-based extraction as fallback"""
    # Extract sentence with most question keywords
    sentences = context.split('.')
    # ... simple heuristic ...
    return best_sentence
```

---

## Cost Management

### Token Usage Tracking

```python
class LLMService:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.usage_tracker = UsageTracker()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        # Track input tokens
        input_tokens = self._count_tokens(prompt)
        
        # Generate
        result = await self.provider.generate(prompt, **kwargs)
        
        # Track output tokens
        output_tokens = self._count_tokens(result)
        
        # Log usage
        self.usage_tracker.log_request(
            provider=self.provider.get_name(),
            model=self.provider.get_model(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=self._calculate_cost(input_tokens, output_tokens)
        )
        
        return result
    
    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost based on provider pricing"""
        pricing = {
            "gemini-1.5-flash": {
                "input": 0.00035 / 1000,  # per token
                "output": 0.00105 / 1000
            },
            "gpt-4": {
                "input": 0.03 / 1000,
                "output": 0.06 / 1000
            }
        }
        
        model = self.provider.get_model()
        rates = pricing.get(model, {"input": 0, "output": 0})
        
        cost = (
            input_tokens * rates["input"] +
            output_tokens * rates["output"]
        )
        
        return cost
```

### Budget Limits

```python
class UsageTracker:
    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.daily_spend = 0.0
        self.last_reset = datetime.now().date()
    
    def log_request(self, cost: float, **kwargs):
        # Reset if new day
        if datetime.now().date() > self.last_reset:
            self.daily_spend = 0.0
            self.last_reset = datetime.now().date()
        
        # Check budget
        if self.daily_spend + cost > self.daily_budget:
            raise BudgetExceededError(
                f"Daily budget ${self.daily_budget} exceeded"
            )
        
        # Log spend
        self.daily_spend += cost
        
        # Save to DB/file for analytics
        self._save_usage(cost, **kwargs)
```

### Cost Optimization Strategies

**1. Cache Responses:**

```python
@lru_cache(maxsize=1000)
def generate_cached(prompt: str) -> str:
    return llm.generate(prompt)
```

**2. Use Cheaper Models for Simple Tasks:**

```python
# Topic labels - use Flash
label = await flash_llm.generate(label_prompt)

# Complex analysis - use Pro
analysis = await pro_llm.generate(analysis_prompt)
```

**3. Batch Requests:**

```python
# Instead of
labels = [await llm.generate(p) for p in prompts]

# Batch into one request
combined_prompt = "\n\n".join(
    f"Keywords {i}: {p}\nLabel {i}:" 
    for i, p in enumerate(prompts)
)
all_labels = await llm.generate(combined_prompt)
```

---

## Testing LLM Integration

### Unit Tests with Mocks

```python
# tests/test_llm_service.py
from unittest.mock import AsyncMock, Mock
import pytest

@pytest.fixture
def mock_provider():
    provider = Mock(spec=LLMProvider)
    provider.generate = AsyncMock(return_value="Mocked response")
    provider.get_name.return_value = "mock"
    provider.get_model.return_value = "mock-model"
    return provider

@pytest.mark.asyncio
async def test_generate_answer(mock_provider):
    service = LLMService(provider=mock_provider)
    
    result = await service.generate_answer(
        question="Test question?",
        context="Test context"
    )
    
    # Verify mock was called
    mock_provider.generate.assert_called_once()
    
    # Verify prompt format
    call_args = mock_provider.generate.call_args
    prompt = call_args[1]["prompt"]
    assert "Test question?" in prompt
    assert "Test context" in prompt
    
    # Verify result
    assert result == "Mocked response"
```

### Integration Tests (Live API)

```python
# tests/test_llm_integration.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_gemini_generation():
    provider = GeminiProvider(
        api_key=os.getenv("LLM_API_KEY"),
        model="gemini-1.5-flash"
    )
    service = LLMService(provider=provider)
    
    result = await service.generate(
        prompt="What is 2+2? Answer with just the number.",
        temperature=0.0,
        max_tokens=10
    )
    
    assert "4" in result
```

### Testing Prompt Quality

```python
# tests/test_prompt_quality.py
@pytest.mark.asyncio
async def test_qa_prompt_includes_context():
    service = LLMService(mock_provider)
    
    prompt = service._build_qa_prompt(
        question="What is X?",
        context="X is Y."
    )
    
    assert "What is X?" in prompt
    assert "X is Y." in prompt
    assert "CONTEXT" in prompt
    assert "QUESTION" in prompt

@pytest.mark.asyncio
async def test_sentiment_prompt_format():
    service = LLMService(mock_provider)
    
    prompt = service._build_sentiment_prompt({
        "sentiment": {"positive": 0.8},
        "emotions": {"joy": 0.6}
    })
    
    assert "0.8" in prompt
    assert "0.6" in prompt
```

---

## Advanced Topics

### Multi-Provider Ensemble

**Use Case:** Combine multiple LLMs for higher quality

```python
class EnsembleLLMService:
    def __init__(self, providers: List[LLMProvider]):
        self.providers = providers
    
    async def generate_with_voting(self, prompt: str) -> str:
        # Get responses from all providers
        tasks = [p.generate(prompt) for p in self.providers]
        responses = await asyncio.gather(*tasks)
        
        # Vote on best response (e.g., longest, most common)
        return max(responses, key=len)
    
    async def generate_with_consensus(self, prompt: str) -> str:
        # Generate from all, combine insights
        tasks = [p.generate(prompt) for p in self.providers]
        responses = await asyncio.gather(*tasks)
        
        # Use one LLM to synthesize consensus
        synthesis_prompt = f"""Analyze these responses and create a consensus:
        
{chr(10).join(f"Response {i+1}: {r}" for i, r in enumerate(responses))}

Synthesized answer:"""
        
        return await self.providers[0].generate(synthesis_prompt)
```

### Streaming Responses

**For Real-Time UX:**

```python
class StreamingLLMService:
    async def generate_stream(
        self,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """Stream tokens as they're generated"""
        
        async for chunk in self.provider.generate_stream(prompt):
            yield chunk
        
# Usage in API
@router.post("/ask/stream")
async def ask_streaming(request: QuestionRequest):
    async def generate():
        async for chunk in llm_service.generate_stream(prompt):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Fine-Tuning Integration

**Custom Model Support:**

```python
class FineTunedGeminiProvider(GeminiProvider):
    def __init__(self, api_key: str, tuned_model_name: str):
        # Use fine-tuned model
        super().__init__(api_key, model=tuned_model_name)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        # Add any custom preprocessing for fine-tuned model
        preprocessed_prompt = self._preprocess(prompt)
        return await super().generate(preprocessed_prompt, **kwargs)
```

---

## Next Steps

**Continue Learning:**

- **`06-concepts-glossary.md`** — Quick reference for all technical terms

**Practice Explaining:**

- Why use pluggable providers instead of hardcoding Gemini?
- What's the difference between temperature 0.3 and 0.7?
- How does prompt engineering affect output quality?
- What's your strategy for handling LLM failures?

**Interview Questions:**

- Explain the strategy pattern for LLM providers
- How do you manage API costs?
- What happens if Gemini API goes down?
- How would you add a new LLM provider?
- What testing strategies do you use for LLMs?

---

You now understand LLM integration at a deep level! 🤖
