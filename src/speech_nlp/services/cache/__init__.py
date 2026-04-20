"""Response caching service.

Provides Redis-backed caching for RAG responses to reduce LLM costs and latency.
Falls back to in-memory caching when Redis is unavailable.
"""

from .base import CacheBackend, CacheService
from .redis import MemoryCache, RedisCache

__all__ = [
    "CacheBackend",
    "CacheService",
    "RedisCache",
    "MemoryCache",
]
