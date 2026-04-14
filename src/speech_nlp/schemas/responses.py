"""Pydantic schemas for API response models.

Defines output serialization and documentation for API endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SentimentResponse(BaseModel):
    """Response model for AI-powered sentiment analysis with emotion detection."""

    sentiment: str = Field(..., description="Dominant sentiment (positive/negative/neutral)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    scores: Dict[str, float] = Field(..., description="All sentiment scores")
    emotions: Dict[str, float] = Field(..., description="Emotion classification scores")
    contextual_sentiment: str = Field(
        ..., description="AI-generated contextual interpretation of emotional tone"
    )
    num_chunks: int = Field(..., description="Number of text chunks analyzed")
    llm_powered: bool = Field(..., description="Whether LLM was used for contextual interpretation")


class TopicResponse(BaseModel):
    """Response model for topic extraction."""

    topics: List[Dict[str, Any]]


class EnhancedTopicResponse(BaseModel):
    """Response model for enhanced topic extraction with clustering and snippets."""

    clustered_topics: List[Dict[str, Any]] = Field(
        ..., description="Semantically clustered topic groups with labels"
    )
    snippets: List[Dict[str, Any]] = Field(
        ..., description="Contextual text snippets for each topic cluster"
    )
    summary: Optional[str] = Field(None, description="AI-generated interpretation of main themes")
    metadata: Dict[str, Any] = Field(..., description="Analysis metadata")
    llm_powered: bool = Field(..., description="Whether LLM was used for summary generation")


class StatsResponse(BaseModel):
    """Response model for dataset statistics."""

    total_speeches: int
    total_words: int
    avg_words_per_speech: float
    date_range: Dict[str, str]
    years: List[str]
    locations: List[str]


class RAGAnswerResponse(BaseModel):
    """Response model for RAG answers."""

    answer: str = Field(..., description="Generated answer")
    context: List[Dict[str, Any]] = Field(..., description="Context chunks used")
    confidence: str = Field(..., description="Confidence level (high/medium/low)")
    confidence_score: float = Field(..., description="Numeric confidence score (0-1)")
    confidence_explanation: str = Field(..., description="Human-readable explanation of confidence")
    confidence_factors: Dict[str, Any] = Field(..., description="Breakdown of confidence factors")
    sources: List[str] = Field(..., description="Source documents")
    entity_statistics: Optional[Dict[str, Any]] = Field(
        None, description="Enhanced statistics about entities: mentions, sentiment, associations"
    )
    guardrails: Optional[Dict[str, Any]] = Field(
        None,
        description="Guardrails metadata: relevance filtering, grounding score, triggered status",
    )
    llm_powered: bool = Field(..., description="Whether LLM was used to generate the answer")


class RAGStatsResponse(BaseModel):
    """Response model for RAG statistics."""

    collection_name: str
    total_chunks: int
    unique_sources: int
    sources: List[str]
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
