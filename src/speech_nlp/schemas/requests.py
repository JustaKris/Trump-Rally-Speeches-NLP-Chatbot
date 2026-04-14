"""Pydantic schemas for API request models.

Defines input validation and serialization for API endpoints.
"""

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    """Input model for text analysis."""

    text: str = Field(..., min_length=1, description="Text to analyze")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {"text": "We're going to make America great again. Our economy is booming!"}
        }


class NGramRequest(BaseModel):
    """Request model for n-gram extraction."""

    text: str = Field(..., min_length=1)
    n: int = Field(2, ge=2, le=5, description="N-gram size (2-5)")
    top_n: int = Field(20, ge=1, le=100, description="Number of top n-grams to return")


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(..., min_length=1, description="Question to ask about the documents")
    top_k: int = Field(5, ge=1, le=15, description="Number of context chunks to retrieve")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "question": "What are the main economic policies discussed?",
                "top_k": 5,
            }
        }


class RAGSearchRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
