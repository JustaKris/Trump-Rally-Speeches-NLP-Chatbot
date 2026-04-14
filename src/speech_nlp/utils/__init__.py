"""Utility functions for text processing, data loading, embeddings, and formatting."""

from .embeddings import batch_embeddings, cosine_similarity, normalize_scores
from .formatting import format_confidence_badge, format_sentiment_badge, format_sources
from .io import (
    get_dataset_statistics,
    get_project_root,
    load_speeches_from_directory,
)
from .text import (
    chunk_text_for_bert,
    clean_text,
    extract_ngrams,
    get_stopwords,
    tokenize_text,
)

__all__ = [
    # Text preprocessing
    "clean_text",
    "tokenize_text",
    "extract_ngrams",
    "get_stopwords",
    "chunk_text_for_bert",
    # IO helpers
    "get_project_root",
    "load_speeches_from_directory",
    "get_dataset_statistics",
    # Embeddings
    "cosine_similarity",
    "normalize_scores",
    "batch_embeddings",
    # Formatting
    "format_sentiment_badge",
    "format_confidence_badge",
    "format_sources",
]
