# Technical Architecture Deep Dive: System Design & Patterns

Understanding the Software Engineering Behind Your AI Project

---

## Architecture Overview

### High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Streamlit   │  │   FastAPI    │  │  Static HTML │     │
│  │     UI       │  │  REST API    │  │   Frontend   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────┬────────────────┬───────────────┬──────────────┘
             │                │               │
             v                v               v
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Service Layer (Business Logic)           │  │
│  │                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │  │
│  │  │ RAG Service  │  │  Sentiment   │  │   Topic    │ │  │
│  │  │              │  │   Service    │  │  Service   │ │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │  │
│  │                                                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │  │
│  │  │     LLM      │  │  Confidence  │  │  Entity    │ │  │
│  │  │   Service    │  │  Calculator  │  │  Analyzer  │ │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────┬─────────────────────────────────┬─────────────┘
             │                                 │
             v                                 v
┌─────────────────────────────────────────────────────────────┐
│                        DATA LAYER                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Document     │  │    Search    │  │  Embedding   │     │
│  │   Loader     │  │    Engine    │  │   Service    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   ChromaDB   │  │  Model Cache │  │   Config     │     │
│  │ Vector Store │  │   Manager    │  │   Manager    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

#### 1. Separation of Concerns

- Each layer has distinct responsibilities
- Changes in one layer don't cascade
- Easy to test in isolation

#### 2. Dependency Injection

- Components receive dependencies rather than creating them
- Easier to mock and test
- More flexible configuration

#### 3. Single Responsibility

- Each service does one thing well
- DocumentLoader loads documents
- SearchEngine searches documents
- ConfidenceCalculator calculates confidence

#### 4. Interface Abstraction

- LLM providers are pluggable
- Can swap Gemini for OpenAI without changing business logic
- Easy to add new providers

---

## Layered Architecture

### 1. Presentation Layer

**Responsibilities:**

- Handle HTTP requests/responses
- Validate input
- Format output
- Manage user sessions

**Components:**

**FastAPI Routes:**

```python
# src/api/routes/qa_routes.py
@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    # 1. Validate input
    if not request.question:
        raise HTTPException(400, "Question required")
    
    # 2. Call service layer
    result = await rag_service.answer_question(
        question=request.question,
        top_k=request.top_k
    )
    
    # 3. Format response
    return QAResponse(**result)
```

**Streamlit UI:**

```python
# src/main.py
def main():
    st.title("Trump Rally Speeches Q&A")
    
    # User input
    question = st.text_input("Ask a question:")
    
    if st.button("Submit"):
        # Call service
        response = rag_service.answer_question(question)
        
        # Display results
        st.write(response["answer"])
        st.metric("Confidence", f"{response['confidence']:.1%}")
```

**Key Pattern:**

- No business logic in presentation layer
- Just input/output handling
- Delegates to service layer

### 2. Application Layer (Service Layer)

**Responsibilities:**

- Implement business logic
- Orchestrate data layer components
- Handle complex workflows
- Apply business rules

#### Example: RAG Service

```python
# src/services/rag_service.py
class RAGService:
    def __init__(
        self,
        search_engine: SearchEngine,
        llm_service: LLMService,
        confidence_calculator: ConfidenceCalculator,
        entity_analyzer: EntityAnalyzer
    ):
        # Dependency injection
        self.search_engine = search_engine
        self.llm_service = llm_service
        self.confidence_calculator = confidence_calculator
        self.entity_analyzer = entity_analyzer
    
    async def answer_question(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        # Step 1: Search for relevant documents
        search_results = await self.search_engine.hybrid_search(
            query=question,
            top_k=top_k
        )
        
        # Step 2: Build context from results
        context = self._build_context(search_results)
        
        # Step 3: Generate answer with LLM
        answer = await self.llm_service.generate(
            prompt=self._build_prompt(question, context)
        )
        
        # Step 4: Calculate confidence
        confidence = self.confidence_calculator.calculate(
            question=question,
            search_results=search_results,
            answer=answer
        )
        
        # Step 5: Extract entities
        entities = self.entity_analyzer.analyze(
            question=question,
            search_results=search_results
        )
        
        # Step 6: Return structured result
        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "entities": entities,
            "sources": [r["metadata"] for r in search_results]
        }
```

**Key Pattern:**

- Services coordinate multiple data layer components
- Each service method is a complete business operation
- Services have no knowledge of presentation layer

#### Example: Sentiment Service

```python
# src/services/sentiment_service.py
class SentimentService:
    def __init__(
        self,
        llm_service: LLMService,
        config: Dict[str, Any]
    ):
        self.llm_service = llm_service
        self.chunk_size = config.get("chunk_size", 510)
        
        # Lazy loading
        self._finbert = None
        self._roberta = None
    
    @property
    def finbert(self):
        if self._finbert is None:
            self._finbert = load_finbert_model()
        return self._finbert
    
    @property
    def roberta(self):
        if self._roberta is None:
            self._roberta = load_roberta_model()
        return self._roberta
    
    def analyze(self, text: str) -> Dict[str, Any]:
        # Step 1: Chunk text
        chunks = self._chunk_text(text, self.chunk_size)
        
        # Step 2: Analyze with FinBERT
        finbert_results = [
            self.finbert(chunk) for chunk in chunks
        ]
        
        # Step 3: Analyze with RoBERTa
        roberta_results = [
            self.roberta(chunk) for chunk in chunks
        ]
        
        # Step 4: Aggregate results
        sentiment = self._aggregate_sentiment(finbert_results)
        emotions = self._aggregate_emotions(roberta_results)
        
        # Step 5: Generate interpretation with LLM
        interpretation = self.llm_service.generate(
            prompt=self._build_interpretation_prompt(
                sentiment, emotions
            )
        )
        
        return {
            "sentiment": sentiment,
            "emotions": emotions,
            "interpretation": interpretation
        }
```

### 3. Data Layer

**Responsibilities:**

- Data access and persistence
- External API calls
- File I/O
- Model management

**Components:**

**DocumentLoader:**

```python
# src/core/document_loader.py
class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load_documents(self) -> List[Document]:
        """Load all text documents from data directory"""
        documents = []
        
        for file_path in self.data_dir.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8")
            
            metadata = {
                "source": file_path.name,
                "date": self._extract_date(file_path.name),
                "location": self._extract_location(file_path.name)
            }
            
            documents.append(Document(
                content=content,
                metadata=metadata
            ))
        
        return documents
```

**SearchEngine:**

```python
# src/core/search_engine.py
class SearchEngine:
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_service: EmbeddingService
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        # Step 1: Generate query embedding
        query_embedding = self.embedding_service.embed(query)
        
        # Step 2: Semantic search
        semantic_results = self.vector_db.similarity_search(
            embedding=query_embedding,
            top_k=top_k * 2  # Over-fetch for re-ranking
        )
        
        # Step 3: BM25 search
        bm25_results = self.vector_db.keyword_search(
            query=query,
            top_k=top_k * 2
        )
        
        # Step 4: Combine and re-rank
        combined = self._combine_results(
            semantic_results,
            bm25_results,
            semantic_weight=0.7,
            bm25_weight=0.3
        )
        
        return combined[:top_k]
```

**ChromaDB Wrapper:**

```python
# src/core/vector_database.py
class ChromaVectorDatabase:
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        self.collection = None
    
    def create_collection(
        self,
        name: str,
        embedding_function
    ):
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ):
        self.collection.add(
            embeddings=embeddings,
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            ids=[doc.id for doc in documents]
        )
    
    def similarity_search(
        self,
        embedding: List[float],
        top_k: int
    ) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_results(results)
```

---

## Design Patterns in Use

### 1. Strategy Pattern (LLM Providers)

**Problem:** Need to support multiple LLM providers (Gemini, OpenAI, Claude)

**Solution:** Define interface, implement concrete strategies

**Interface:**

```python
# src/services/llm/base.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate text completion"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return provider name"""
        pass
```

**Concrete Implementations:**

```python
# src/services/llm/gemini_provider.py
class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.client = genai.GenerativeModel(model)
        genai.configure(api_key=api_key)
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        return response.text
    
    def get_name(self) -> str:
        return "gemini"

# src/services/llm/openai_provider.py
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
```

**Factory:**

```python
# src/services/llm/factory.py
class LLMFactory:
    @staticmethod
    def create_provider(provider_name: str, config: Dict) -> LLMProvider:
        providers = {
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
            "claude": ClaudeProvider
        }
        
        provider_class = providers.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        return provider_class(
            api_key=config["api_key"],
            model=config.get("model")
        )
```

**Usage:**

```python
# Swap providers without changing business logic
provider = LLMFactory.create_provider(
    provider_name=os.getenv("LLM_PROVIDER", "gemini"),
    config={"api_key": os.getenv("LLM_API_KEY")}
)

llm_service = LLMService(provider=provider)
```

### 2. Singleton Pattern (Model Cache)

**Problem:** Loading ML models is expensive, should only happen once

**Solution:** Singleton to cache loaded models

**Implementation:**

```python
# src/utils/model_cache.py
class ModelCache:
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str, loader_fn):
        if model_name not in self._models:
            print(f"Loading {model_name}...")
            self._models[model_name] = loader_fn()
        return self._models[model_name]

# Usage
cache = ModelCache()
finbert = cache.get_model("finbert", load_finbert_model)
roberta = cache.get_model("roberta", load_roberta_model)
```

### 3. Builder Pattern (Confidence Calculation)

**Problem:** Confidence score has many factors with weights

**Solution:** Builder to construct confidence calculation

**Implementation:**

```python
# src/core/confidence_calculator.py
class ConfidenceCalculator:
    def __init__(self):
        self.factors = []
        self.weights = []
    
    def add_factor(
        self,
        name: str,
        weight: float,
        calculator: Callable[[Dict], float]
    ):
        self.factors.append({
            "name": name,
            "weight": weight,
            "calculator": calculator
        })
        return self
    
    def calculate(self, data: Dict) -> float:
        total_score = 0.0
        total_weight = sum(f["weight"] for f in self.factors)
        
        for factor in self.factors:
            score = factor["calculator"](data)
            weighted_score = score * (factor["weight"] / total_weight)
            total_score += weighted_score
        
        return min(max(total_score, 0.0), 1.0)

# Build confidence calculator
confidence_calc = (
    ConfidenceCalculator()
    .add_factor("similarity", 0.40, lambda d: d["avg_similarity"])
    .add_factor("keyword_match", 0.25, lambda d: d["keyword_overlap"])
    .add_factor("answer_length", 0.15, lambda d: d["answer_completeness"])
    .add_factor("entity_match", 0.20, lambda d: d["entity_overlap"])
)
```

### 4. Repository Pattern (Data Access)

**Problem:** Isolate data access logic from business logic

**Solution:** Repository for each data source

**Implementation:**

```python
# src/repositories/document_repository.py
class DocumentRepository:
    def __init__(self, vector_db: ChromaVectorDatabase):
        self.vector_db = vector_db
    
    def get_all(self) -> List[Document]:
        """Get all documents"""
        return self.vector_db.get_all_documents()
    
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.vector_db.get_document(doc_id)
    
    def search(self, query: str, top_k: int) -> List[Document]:
        """Search documents"""
        return self.vector_db.search(query, top_k)
    
    def add(self, document: Document):
        """Add document"""
        self.vector_db.add_document(document)
```

### 5. Observer Pattern (Logging)

**Problem:** Need to log events across the system

**Solution:** Observer pattern for event logging

**Implementation:**

```python
# src/utils/event_logger.py
class EventLogger:
    def __init__(self):
        self.observers = []
    
    def subscribe(self, observer: Callable):
        self.observers.append(observer)
    
    def log_event(self, event_type: str, data: Dict):
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        for observer in self.observers:
            observer(event)

# Create logger
logger = EventLogger()

# Subscribe console observer
def console_observer(event):
    print(f"[{event['timestamp']}] {event['type']}: {event['data']}")

logger.subscribe(console_observer)

# Subscribe file observer
def file_observer(event):
    with open("events.log", "a") as f:
        f.write(json.dumps(event) + "\n")

logger.subscribe(file_observer)

# Log events
logger.log_event("QUESTION_ASKED", {"question": "What about jobs?"})
logger.log_event("ANSWER_GENERATED", {"confidence": 0.85})
```

---

## Configuration Management

### Environment-Based Configuration

**Development:**

```yaml
# configs/development.yaml
llm:
  provider: "gemini"
  model: "gemini-1.5-flash"
  temperature: 0.7

database:
  persist_directory: "./data/chromadb"
  collection_name: "speeches_dev"

search:
  top_k: 5
  semantic_weight: 0.7
  bm25_weight: 0.3

logging:
  level: "DEBUG"
  format: "detailed"
```

**Production:**

```yaml
# configs/production.yaml
llm:
  provider: "gemini"
  model: "gemini-1.5-pro"  # More capable model
  temperature: 0.5  # More deterministic

database:
  persist_directory: "/var/data/chromadb"
  collection_name: "speeches_prod"

search:
  top_k: 10  # More results
  semantic_weight: 0.7
  bm25_weight: 0.3

logging:
  level: "INFO"
  format: "json"
```

**Loading:**

```python
# src/config/loader.py
def load_config(env: str = "development") -> Dict:
    config_path = Path(f"configs/{env}.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables
    config = override_with_env(config)
    
    return config

def override_with_env(config: Dict) -> Dict:
    # LLM_API_KEY overrides config file
    if "LLM_API_KEY" in os.environ:
        config["llm"]["api_key"] = os.getenv("LLM_API_KEY")
    
    # LLM_PROVIDER overrides config file
    if "LLM_PROVIDER" in os.environ:
        config["llm"]["provider"] = os.getenv("LLM_PROVIDER")
    
    return config
```

---

## Dependency Injection Container

**Problem:** Creating and wiring dependencies is complex

**Solution:** DI container to manage object graph

**Implementation:**

```python
# src/container.py
from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide

class Container(containers.DeclarativeContainer):
    # Configuration
    config = providers.Configuration()
    
    # Data Layer
    vector_db = providers.Singleton(
        ChromaVectorDatabase,
        persist_directory=config.database.persist_directory
    )
    
    embedding_service = providers.Singleton(
        EmbeddingService,
        model_name=config.embeddings.model
    )
    
    search_engine = providers.Singleton(
        SearchEngine,
        vector_db=vector_db,
        embedding_service=embedding_service
    )
    
    # LLM
    llm_provider = providers.Singleton(
        LLMFactory.create_provider,
        provider_name=config.llm.provider,
        config=config.llm
    )
    
    llm_service = providers.Singleton(
        LLMService,
        provider=llm_provider
    )
    
    # Services
    rag_service = providers.Singleton(
        RAGService,
        search_engine=search_engine,
        llm_service=llm_service,
        confidence_calculator=confidence_calculator,
        entity_analyzer=entity_analyzer
    )
    
    sentiment_service = providers.Singleton(
        SentimentService,
        llm_service=llm_service,
        config=config.sentiment
    )
    
    topic_service = providers.Singleton(
        TopicService,
        llm_service=llm_service,
        config=config.topic
    )
```

**Usage:**

```python
# Wire container to modules
container = Container()
container.config.from_yaml("configs/development.yaml")
container.wire(modules=[__name__])

# Inject dependencies
@inject
def main(
    rag_service: RAGService = Provide[Container.rag_service]
):
    result = rag_service.answer_question("What about jobs?")
    print(result)
```

---

## Testing Architecture

### Unit Tests (Isolated Components)

```python
# tests/test_confidence_calculator.py
def test_confidence_calculation():
    # Arrange
    calc = ConfidenceCalculator()
    calc.add_factor("similarity", 1.0, lambda d: d["score"])
    
    # Act
    confidence = calc.calculate({"score": 0.85})
    
    # Assert
    assert confidence == 0.85
```

### Integration Tests (Multiple Components)

```python
# tests/test_rag_integration.py
@pytest.fixture
def rag_service():
    # Use test configuration
    config = load_config("test")
    container = Container()
    container.config.from_dict(config)
    return container.rag_service()

def test_end_to_end_qa(rag_service):
    # Act
    result = rag_service.answer_question(
        question="What did Trump say about jobs?"
    )
    
    # Assert
    assert "answer" in result
    assert result["confidence"] > 0.0
    assert len(result["sources"]) > 0
```

### Mocking External Dependencies

```python
# tests/test_sentiment_service.py
from unittest.mock import Mock

def test_sentiment_with_mocked_llm():
    # Arrange
    mock_llm = Mock()
    mock_llm.generate.return_value = "Positive interpretation"
    
    service = SentimentService(
        llm_service=mock_llm,
        config={"chunk_size": 510}
    )
    
    # Act
    result = service.analyze("Great jobs numbers!")
    
    # Assert
    mock_llm.generate.assert_called_once()
    assert "interpretation" in result
```

---

## Performance Optimization

### 1. Lazy Loading

**Problem:** Loading all models at startup is slow

**Solution:** Load models only when needed

```python
class SentimentService:
    def __init__(self, llm_service, config):
        self.llm_service = llm_service
        self._finbert = None  # Not loaded yet
    
    @property
    def finbert(self):
        if self._finbert is None:
            self._finbert = load_finbert_model()  # Load on first use
        return self._finbert
```

### 2. Caching

**Problem:** Repeated queries waste computation

**Solution:** LRU cache for frequent queries

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def embed_text(text: str):
    return model.encode(text)
```

### 3. Async/Await

**Problem:** Blocking I/O wastes time

**Solution:** Async operations for LLM calls

```python
async def generate_labels(clusters):
    # Run LLM calls in parallel
    tasks = [
        llm_service.generate(build_prompt(cluster))
        for cluster in clusters
    ]
    labels = await asyncio.gather(*tasks)
    return labels
```

### 4. Batch Processing

**Problem:** Processing one item at a time is slow

**Solution:** Batch embeddings and predictions

```python
# Instead of
embeddings = [model.encode(text) for text in texts]

# Do
embeddings = model.encode(texts)  # Batch encode
```

---

## Error Handling Strategy

### Layered Error Handling

**Data Layer:**

```python
class SearchEngine:
    def search(self, query: str):
        try:
            results = self.vector_db.query(query)
            return results
        except DatabaseConnectionError as e:
            logger.error(f"DB error: {e}")
            raise DataLayerError("Search failed") from e
```

**Service Layer:**

```python
class RAGService:
    async def answer_question(self, question: str):
        try:
            results = await self.search_engine.search(question)
            return self._process_results(results)
        except DataLayerError as e:
            logger.warning(f"Search error: {e}")
            return {"answer": "Unable to search at this time"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ServiceError("Answer generation failed") from e
```

**Presentation Layer:**

```python
@router.post("/ask")
async def ask(request: QuestionRequest):
    try:
        result = await rag_service.answer_question(request.question)
        return result
    except ServiceError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.critical(f"Unhandled error: {e}")
        raise HTTPException(500, "Internal server error")
```

---

## Next Steps

**Continue Learning:**

- **`05-llm-integration.md`** — Deep dive on LLM provider integration
- **`06-concepts-glossary.md`** — Quick reference for all terms

**Practice Explaining:**

- What design patterns does your project use?
- Why use layered architecture?
- How does dependency injection help?
- What's the benefit of strategy pattern for LLM providers?

**Interview Questions:**

- Explain the separation of concerns in your architecture
- How would you add a new LLM provider?
- What testing strategies do you use?
- How do you handle errors across layers?

---

You now understand the software engineering architecture! 🏗️
