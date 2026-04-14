"""Pydantic schemas for API request and response models.

Organized into:
- requests: Input validation models (TextInput, RAGQueryRequest, etc.)
- responses: Output serialization models (SentimentResponse, RAGAnswerResponse, etc.)
"""

from .requests import (
    NGramRequest,
    RAGQueryRequest,
    RAGSearchRequest,
    TextInput,
)
from .responses import (
    EnhancedTopicResponse,
    RAGAnswerResponse,
    RAGStatsResponse,
    SentimentResponse,
    StatsResponse,
    TopicResponse,
)

__all__ = [
    # Request models
    "TextInput",
    "NGramRequest",
    "RAGQueryRequest",
    "RAGSearchRequest",
    # Response models
    "SentimentResponse",
    "TopicResponse",
    "EnhancedTopicResponse",
    "StatsResponse",
    "RAGAnswerResponse",
    "RAGStatsResponse",
]
