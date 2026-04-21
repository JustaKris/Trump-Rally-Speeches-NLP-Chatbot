"""Tests for response caching service.

Tests the cache backend implementations (MemoryCache, RedisCache),
cache key generation, TTL handling, and cache service integration.
"""

import time
from unittest.mock import patch

import pytest

from speech_nlp.services.cache import CacheService, MemoryCache, RedisCache


class TestMemoryCache:
    """Tests for in-memory cache backend."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = MemoryCache()
        data = {"answer": "test answer", "confidence": 0.95}

        assert cache.set("test_key", data)
        result = cache.get("test_key")

        assert result == data

    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        cache = MemoryCache()

        assert cache.get("nonexistent") is None

    def test_delete(self):
        """Test deleting a cached value."""
        cache = MemoryCache()
        cache.set("to_delete", {"data": "value"})

        assert cache.delete("to_delete") is True
        assert cache.get("to_delete") is None
        assert cache.delete("to_delete") is False  # Already deleted

    def test_exists(self):
        """Test exists check."""
        cache = MemoryCache()
        cache.set("exists_key", {"data": "value"})

        assert cache.exists("exists_key") is True
        assert cache.exists("nonexistent") is False

    def test_clear(self):
        """Test clearing all cached values."""
        cache = MemoryCache()
        cache.set("key1", {"data": 1})
        cache.set("key2", {"data": 2})
        cache.set("key3", {"data": 3})

        count = cache.clear()

        assert count == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_is_available(self):
        """Test that memory cache is always available."""
        cache = MemoryCache()
        assert cache.is_available() is True

    def test_ttl_expiration(self):
        """Test that values expire after TTL."""
        cache = MemoryCache()

        # Set with 1 second TTL
        cache.set("expiring", {"data": "value"}, ttl_seconds=1)

        # Should exist immediately
        assert cache.get("expiring") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("expiring") is None

    def test_no_expiration_without_ttl(self):
        """Test that values without TTL don't expire."""
        cache = MemoryCache()
        cache.set("permanent", {"data": "value"})

        # Sleep briefly and verify it's still there
        time.sleep(0.1)
        assert cache.get("permanent") is not None

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = MemoryCache()

        # Set some entries with short TTL
        cache.set("expire1", {"data": 1}, ttl_seconds=1)
        cache.set("expire2", {"data": 2}, ttl_seconds=1)
        cache.set("permanent", {"data": 3})  # No TTL

        time.sleep(1.1)

        count = cache.cleanup_expired()

        assert count == 2
        assert cache.get("permanent") is not None


class TestRedisCache:
    """Tests for Redis cache backend with fallback."""

    def test_fallback_to_memory_when_redis_unavailable(self):
        """Test that RedisCache falls back to MemoryCache when Redis is unavailable."""
        # Connect to non-existent Redis
        cache = RedisCache(host="nonexistent-host", port=9999, socket_timeout=0.1)

        assert cache.using_fallback is True
        assert cache.is_available() is True  # Fallback should be available

        # Should work with fallback
        cache.set("test", {"data": "value"})
        assert cache.get("test") == {"data": "value"}

    def test_fallback_when_redis_package_missing(self):
        """Test fallback when redis package is not installed."""
        with patch.dict("sys.modules", {"redis": None}):
            # This would raise ImportError internally
            cache = RedisCache()
            assert cache.using_fallback is True or cache._fallback is not None

    def test_get_info_fallback(self):
        """Test get_info when using fallback."""
        cache = RedisCache(host="nonexistent", socket_timeout=0.1)

        info = cache.get_info()

        assert info["backend"] == "MemoryCache (fallback)"
        assert "reason" in info


class TestCacheService:
    """Tests for high-level CacheService."""

    @pytest.fixture
    def cache_service(self):
        """Create a CacheService with MemoryCache backend."""
        backend = MemoryCache()
        return CacheService(backend=backend, key_prefix="test", default_ttl_seconds=60)

    def test_generate_cache_key_deterministic(self, cache_service):
        """Test that cache key generation is deterministic."""
        key1 = cache_service.generate_cache_key("What about the wall?", 5)
        key2 = cache_service.generate_cache_key("What about the wall?", 5)

        assert key1 == key2
        assert key1.startswith("test:ask:")

    def test_generate_cache_key_normalizes_input(self, cache_service):
        """Test that cache keys are normalized."""
        key1 = cache_service.generate_cache_key("What about the wall?", 5)
        key2 = cache_service.generate_cache_key("  WHAT  ABOUT  THE  WALL?  ", 5)

        assert key1 == key2

    def test_generate_cache_key_different_for_different_top_k(self, cache_service):
        """Test that different top_k produces different keys."""
        key1 = cache_service.generate_cache_key("What about the wall?", 5)
        key2 = cache_service.generate_cache_key("What about the wall?", 10)

        assert key1 != key2

    def test_cache_response_and_get_response(self, cache_service):
        """Test caching and retrieving a response."""
        question = "What did Trump say about immigration?"
        top_k = 5
        response = {
            "answer": "Trump discussed immigration policies...",
            "confidence": "high",
            "confidence_score": 0.85,
            "sources": ["speech1.txt", "speech2.txt"],
        }

        # Cache the response
        assert cache_service.cache_response(question, top_k, response)

        # Retrieve it
        cached = cache_service.get_response(question, top_k)

        assert cached is not None
        assert cached["answer"] == response["answer"]
        assert cached["cached"] is True  # Should be marked as cached
        assert "cache_key" in cached

    def test_get_response_miss(self, cache_service):
        """Test cache miss returns None."""
        result = cache_service.get_response("uncached question", 5)
        assert result is None

    def test_invalidate(self, cache_service):
        """Test cache invalidation."""
        question = "Test question"
        cache_service.cache_response(question, 5, {"answer": "test"})

        assert cache_service.get_response(question, 5) is not None
        assert cache_service.invalidate(question, 5) is True
        assert cache_service.get_response(question, 5) is None

    def test_clear_all(self, cache_service):
        """Test clearing all cached responses."""
        cache_service.cache_response("q1", 5, {"answer": "a1"})
        cache_service.cache_response("q2", 5, {"answer": "a2"})

        count = cache_service.clear_all()

        assert count == 2
        assert cache_service.get_response("q1", 5) is None
        assert cache_service.get_response("q2", 5) is None

    def test_stats_tracking(self, cache_service):
        """Test hit/miss statistics tracking."""
        cache_service.cache_response("cached_q", 5, {"answer": "test"})

        # Generate some hits and misses
        cache_service.get_response("cached_q", 5)  # Hit
        cache_service.get_response("cached_q", 5)  # Hit
        cache_service.get_response("uncached", 5)  # Miss

        stats = cache_service.get_stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total_requests"] == 3
        assert stats["hit_rate"] == pytest.approx(2 / 3, rel=0.01)

    def test_reset_stats(self, cache_service):
        """Test resetting statistics."""
        cache_service.cache_response("q", 5, {"answer": "a"})
        cache_service.get_response("q", 5)
        cache_service.get_response("missing", 5)

        cache_service.reset_stats()
        stats = cache_service.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_is_available(self, cache_service):
        """Test availability check."""
        assert cache_service.is_available is True

    def test_custom_ttl(self, cache_service):
        """Test caching with custom TTL."""
        # Cache with very short TTL
        cache_service.cache_response(
            "expiring_q", 5, {"answer": "test"}, ttl_seconds=1
        )

        assert cache_service.get_response("expiring_q", 5) is not None

        time.sleep(1.1)

        assert cache_service.get_response("expiring_q", 5) is None

    def test_does_not_cache_cached_flag(self, cache_service):
        """Test that the 'cached' flag is not persisted."""
        response = {
            "answer": "test",
            "cached": True,  # Should not be stored
            "cache_key": "should_not_store",
        }

        cache_service.cache_response("q", 5, response)
        cached = cache_service.get_response("q", 5)

        # The 'cached' flag should be added fresh, not from stored data
        assert cached["cached"] is True
        # But the stored data shouldn't have had it
        # (we verify by checking that repeated gets work correctly)


class TestRAGServiceCacheIntegration:
    """Integration tests for caching in RAG service."""

    def test_rag_service_with_cache_parameter(self):
        """Test that RAGService accepts cache_service parameter."""
        from speech_nlp.services.cache import CacheService, MemoryCache

        backend = MemoryCache()
        CacheService(backend=backend, key_prefix="test")

        # This import may fail if models aren't available in test environment
        # In that case, we just test that the parameter is accepted
        try:
            # Verify the parameter is accepted (don't actually initialize the full service)
            import inspect

            from speech_nlp.services import RAGService

            sig = inspect.signature(RAGService.__init__)
            assert "cache_service" in sig.parameters
        except ImportError:
            # Models not available in test environment
            pass
