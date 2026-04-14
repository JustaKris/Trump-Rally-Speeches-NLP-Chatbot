"""RAG (Retrieval-Augmented Generation) components.

Provides modular components for the RAG pipeline:
- service: Main RAG orchestrator
- search: Semantic and hybrid search engine
- chunking: Document loading and chunking
- entities: Entity extraction and analysis
- rewriter: LLM-powered query rewriting
- confidence: Answer confidence scoring
- guardrails: Query validation and grounding checks
- models: Internal domain models
"""

from .chunking import DocumentLoader
from .confidence import ConfidenceCalculator
from .entities import EntityAnalyzer
from .guardrails import RAGGuardrails
from .models import (
    ConfidenceFactors,
    ConfidenceResult,
    ContextChunk,
    EntitySentiment,
    EntityStatistics,
    SearchResult,
)
from .rewriter import QueryRewriter
from .search import SearchEngine

__all__ = [
    "DocumentLoader",
    "ConfidenceCalculator",
    "EntityAnalyzer",
    "RAGGuardrails",
    "QueryRewriter",
    "SearchEngine",
    "ConfidenceFactors",
    "ConfidenceResult",
    "ContextChunk",
    "EntitySentiment",
    "EntityStatistics",
    "SearchResult",
]
