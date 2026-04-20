"""Abstract base class for cache backends.

Defines the interface that all cache implementations must follow.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract cache backend interface.

    All cache implementations (Redis, memory, etc.) must implement these methods.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value as dict, or None if not found/expired
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
        """Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl_seconds: Time-to-live in seconds (None for no expiration)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all cached values.

        Returns:
            Number of keys deleted
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the cache backend is available.

        Returns:
            True if the backend is connected and working
        """
        pass


class CacheService:
    """High-level caching service for RAG responses.

    Provides cache key generation, serialization, and metrics tracking.
    Uses a CacheBackend for actual storage.
    """

    def __init__(
        self,
        backend: CacheBackend,
        key_prefix: str = "rag",
        default_ttl_seconds: int = 3600,  # 1 hour default
    ):
        """Initialize the cache service.

        Args:
            backend: Cache backend implementation
            key_prefix: Prefix for all cache keys
            default_ttl_seconds: Default TTL for cached items
        """
        self.backend = backend
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl_seconds

        # Metrics
        self._hits = 0
        self._misses = 0

        logger.info(
            "Cache service initialized: backend=%s, prefix=%s, ttl=%ds",
            type(backend).__name__,
            key_prefix,
            default_ttl_seconds,
        )

    def generate_cache_key(self, question: str, top_k: int) -> str:
        """Generate a deterministic cache key for a RAG query.

        Uses SHA-256 hash of normalized query parameters to ensure
        consistent keys regardless of whitespace/case variations.

        Args:
            question: User's question
            top_k: Number of results requested

        Returns:
            Cache key string
        """
        # Normalize: lowercase, strip whitespace, remove extra spaces
        normalized_question = " ".join(question.lower().strip().split())

        # Create deterministic hash
        key_data = f"{normalized_question}|{top_k}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]

        return f"{self.key_prefix}:ask:{key_hash}"

    def get_response(self, question: str, top_k: int) -> Optional[Dict[str, Any]]:
        """Retrieve a cached RAG response.

        Args:
            question: User's question
            top_k: Number of results requested

        Returns:
            Cached response dict with 'cached': True added, or None if not cached
        """
        key = self.generate_cache_key(question, top_k)

        cached = self.backend.get(key)
        if cached is not None:
            self._hits += 1
            # Mark response as coming from cache
            cached["cached"] = True
            cached["cache_key"] = key
            logger.debug("Cache HIT: %s", key)
            return cached

        self._misses += 1
        logger.debug("Cache MISS: %s", key)
        return None

    def cache_response(
        self,
        question: str,
        top_k: int,
        response: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Cache a RAG response.

        Args:
            question: User's question
            top_k: Number of results requested
            response: RAG response dict to cache
            ttl_seconds: Override default TTL

        Returns:
            True if cached successfully
        """
        key = self.generate_cache_key(question, top_k)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        # Don't cache the 'cached' flag itself
        cache_data = {k: v for k, v in response.items() if k not in ("cached", "cache_key")}

        success = self.backend.set(key, cache_data, ttl)
        if success:
            logger.debug("Cached response: %s (ttl=%ds)", key, ttl)
        else:
            logger.warning("Failed to cache response: %s", key)

        return success

    def invalidate(self, question: str, top_k: int) -> bool:
        """Invalidate a specific cached response.

        Args:
            question: User's question
            top_k: Number of results

        Returns:
            True if invalidated
        """
        key = self.generate_cache_key(question, top_k)
        return self.backend.delete(key)

    def clear_all(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of keys cleared
        """
        count = self.backend.clear()
        logger.info("Cleared %d cached responses", count)
        return count

    @property
    def is_available(self) -> bool:
        """Check if caching is available."""
        return self.backend.is_available()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, and backend info
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_pct": f"{hit_rate * 100:.1f}%",
            "backend": type(self.backend).__name__,
            "available": self.backend.is_available(),
            "key_prefix": self.key_prefix,
            "default_ttl_seconds": self.default_ttl,
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = 0
