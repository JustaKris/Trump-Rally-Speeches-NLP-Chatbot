"""API module for the NLP Chatbot API.

Contains route handlers organized by functionality:
- chatbot: RAG-powered question answering
- analysis: Text analysis endpoints (sentiment, frequency, topics)
- health: Health check and metadata endpoints
"""

from .analysis import router as nlp_router
from .chatbot import router as chatbot_router
from .dependencies import (
    get_llm_service,
    get_nlp_service,
    get_rag_service,
    get_sentiment_analyzer_dep,
    get_settings_dep,
)
from .health import router as health_router

__all__ = [
    # Routers
    "chatbot_router",
    "nlp_router",
    "health_router",
    # Dependencies
    "get_settings_dep",
    "get_sentiment_analyzer_dep",
    "get_rag_service",
    "get_llm_service",
    "get_nlp_service",
]
