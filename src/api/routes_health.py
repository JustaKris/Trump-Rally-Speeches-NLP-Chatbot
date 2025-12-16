"""Health check and metadata endpoints.

Provides system health status, version information, and configuration details.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ..services import EnhancedSentimentAnalyzer, RAGService
from .dependencies import get_rag_service, get_sentiment_analyzer_dep, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Set up templates
templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main frontend page."""
    html_file = Path(__file__).parent.parent / "templates" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)

    # Fallback HTML if template not found
    return """
    <html>
        <head><title>Trump Speeches NLP Chatbot API</title></head>
        <body>
            <h1>🧠 Trump Speeches NLP Chatbot API</h1>
            <p>Production-ready NLP and RAG platform with AI Q&A, sentiment analysis, and semantic search</p>
            <ul>
                <li><a href="/docs">📚 Interactive API Documentation</a></li>
                <li><a href="/redoc">📖 ReDoc Documentation</a></li>
                <li><a href="/health">🏥 Health Check</a></li>
            </ul>
        </body>
    </html>
    """


@router.get("/health")
async def health_check(
    settings=Depends(get_settings_dep),
    sentiment_analyzer: Optional[EnhancedSentimentAnalyzer] = Depends(get_sentiment_analyzer_dep),
    rag_service: Optional[RAGService] = Depends(get_rag_service),
):
    """Health check endpoint.

    Returns system status and service availability.
    """
    status_code = "healthy" if rag_service is not None else "degraded"

    services = {
        "sentiment_analyzer": sentiment_analyzer is not None,
        "rag_service": rag_service is not None,
        "llm_configured": settings.is_llm_configured(),
    }

    # Add RAG details if available
    if rag_service:
        services["rag_chunks"] = rag_service.collection.count()

    return {
        "status": status_code,
        "version": settings.app_version,
        "environment": settings.environment,
        "services": services,
    }


@router.get("/config")
async def get_config(settings=Depends(get_settings_dep)):
    """Get public configuration information.

    Returns non-sensitive configuration details.
    """
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "environment": settings.environment,
        "llm_provider": settings.llm.provider,
        "llm_model": settings.get_llm_model_name() if settings.is_llm_configured() else None,
        "sentiment_model": settings.models.sentiment_model_name,
        "embedding_model": settings.models.embedding_model_name,
        "features": {
            "llm_enabled": settings.llm.enabled and settings.is_llm_configured(),
            "hybrid_search": settings.rag.use_hybrid_search,
            "reranking": settings.rag.use_reranking,
        },
    }


@router.get("/diagnostics")
async def diagnostics(
    settings=Depends(get_settings_dep),
    sentiment_analyzer: Optional[EnhancedSentimentAnalyzer] = Depends(get_sentiment_analyzer_dep),
    rag_service: Optional[RAGService] = Depends(get_rag_service),
):
    """Detailed diagnostics for troubleshooting production issues.

    Shows paths, API key status, model availability, etc.
    """
    import os
    from pathlib import Path

    diagnostics_info = {
        "environment": {
            "name": settings.environment,
            "log_level": settings.log_level,
        },
        "api_keys": {
            "llm_api_key_set": settings.llm.api_key is not None,
            "llm_api_key_length": len(settings.llm.api_key) if settings.llm.api_key else 0,
        },
        "paths": {
            "chromadb_dir": settings.rag.chromadb_persist_directory,
            "chromadb_exists": Path(settings.rag.chromadb_persist_directory).exists(),
            "speeches_dir": settings.paths.speeches_directory,
            "speeches_exists": Path(settings.paths.speeches_directory).exists(),
            "speeches_count": len(list(Path(settings.paths.speeches_directory).glob("*.txt")))
            if Path(settings.paths.speeches_directory).exists()
            else 0,
        },
        "services": {
            "sentiment_analyzer_loaded": sentiment_analyzer is not None,
            "rag_service_loaded": rag_service is not None,
            "llm_configured": settings.is_llm_configured(),
        },
        "environment_variables": {
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "not set"),
            "LLM_API_KEY": "***set***" if os.getenv("LLM_API_KEY") else "not set",
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "not set"),
        },
    }

    # Add RAG-specific diagnostics if loaded
    if rag_service:
        diagnostics_info["rag"] = {
            "chunk_count": rag_service.collection.count(),
            "collection_name": settings.rag.chromadb_collection_name,
            "hybrid_search": settings.rag.use_hybrid_search,
            "reranking": settings.rag.use_reranking,
        }

    return diagnostics_info
