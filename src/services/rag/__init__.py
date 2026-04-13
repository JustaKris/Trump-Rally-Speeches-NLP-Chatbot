"""RAG (Retrieval-Augmented Generation) module.

Provides specialized components for document retrieval and answer generation:
- DocumentLoader: Document loading and chunking
- SearchEngine: Hybrid semantic and keyword search
- EntityAnalyzer: Entity extraction and analysis
- ConfidenceCalculator: Confidence scoring for answers
- Domain models: Type-safe data structures
"""

from .confidence import ConfidenceCalculator
from .document_loader import DocumentLoader
from .entity_analyzer import EntityAnalyzer
from .guardrails import RAGGuardrails
from .models import (
    ConfidenceFactors,
    ConfidenceResult,
    ContextChunk,
    EntitySentiment,
    EntityStatistics,
    SearchResult,
)
from .query_rewriter import QueryRewriter
from .search_engine import SearchEngine

__all__ = [
    # Components
    "DocumentLoader",
    "SearchEngine",
    "EntityAnalyzer",
    "ConfidenceCalculator",
    "RAGGuardrails",
    "QueryRewriter",
    # Models
    "SearchResult",
    "ContextChunk",
    "ConfidenceFactors",
    "ConfidenceResult",
    "EntitySentiment",
    "EntityStatistics",
]
