"""Redis and in-memory cache backend implementations.

Redis is the primary backend for production use.
MemoryCache provides a fallback for development/testing or when Redis is unavailable.
"""

import json
import logging
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from speech_nlp.services.cache.base import CacheBackend

logger = logging.getLogger(__name__)


class MemoryCache(CacheBackend):
    """In-memory cache backend with TTL support.

    Thread-safe implementation using a simple dict.
    Suitable for development, testing, or single-instance deployments.

    Note: Data is lost on restart and not shared across processes.
    """

    def __init__(self):
        """Initialize the in-memory cache."""
        self._cache: Dict[str, Tuple[Dict[str, Any], Optional[float]]] = {}
        self._lock = Lock()
        logger.info("MemoryCache initialized")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached value, checking expiration."""
        with self._lock:
            if key not in self._cache:
                return None

            value, expires_at = self._cache[key]

            # Check expiration
            if expires_at is not None and time.time() > expires_at:
                del self._cache[key]
                return None

            return value

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
        """Store a value with optional TTL."""
        with self._lock:
            expires_at = None
            if ttl_seconds is not None and ttl_seconds > 0:
                expires_at = time.time() + ttl_seconds

            self._cache[key] = (value, expires_at)
            return True

    def delete(self, key: str) -> bool:
        """Delete a cached value."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists (and is not expired)."""
        return self.get(key) is not None

    def clear(self) -> int:
        """Clear all cached values."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def is_available(self) -> bool:
        """In-memory cache is always available."""
        return True

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        with self._lock:
            now = time.time()
            expired_keys = [
                key
                for key, (_, expires_at) in self._cache.items()
                if expires_at is not None and now > expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


class RedisCache(CacheBackend):
    """Redis-backed cache with automatic fallback to in-memory cache.

    Attempts to connect to Redis on initialization. If Redis is unavailable,
    automatically falls back to MemoryCache and logs a warning.

    This ensures the application works even without Redis configured.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 5.0,
        key_prefix: str = "speech_nlp",
    ):
        """Initialize Redis cache with fallback.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (optional)
            socket_timeout: Connection timeout in seconds
            key_prefix: Prefix for all keys (namespace isolation)
        """
        self._redis: Optional[Any] = None
        self._fallback: Optional[MemoryCache] = None
        self._key_prefix = key_prefix
        self._redis_available = False

        # Try to connect to Redis
        try:
            import redis  # type: ignore[import-not-found]

            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
                decode_responses=True,
            )

            # Test connection
            self._client.ping()
            self._redis_available = True
            logger.info("RedisCache connected: %s:%d (db=%d)", host, port, db)

        except ImportError:
            logger.warning(
                "redis package not installed. Install with: pip install redis. "
                "Falling back to in-memory cache."
            )
            self._fallback = MemoryCache()

        except Exception as e:
            logger.warning(
                "Could not connect to Redis at %s:%d: %s. Falling back to in-memory cache.",
                host,
                port,
                e,
            )
            self._fallback = MemoryCache()

    @property
    def _client(self) -> Any:
        """Return the Redis client. Only call after confirming fallback is not active."""
        assert self._redis is not None
        return self._redis

    def _prefixed_key(self, key: str) -> str:
        """Add namespace prefix to key."""
        return f"{self._key_prefix}:{key}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached value."""
        if self._fallback:
            return self._fallback.get(key)

        try:
            data = self._client.get(self._prefixed_key(key))
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.error("Redis GET error for %s: %s", key, e)
            return None

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
        """Store a value with optional TTL."""
        if self._fallback:
            return self._fallback.set(key, value, ttl_seconds)

        try:
            serialized = json.dumps(value)
            prefixed = self._prefixed_key(key)

            if ttl_seconds is not None and ttl_seconds > 0:
                self._client.setex(prefixed, ttl_seconds, serialized)
            else:
                self._client.set(prefixed, serialized)

            return True
        except Exception as e:
            logger.error("Redis SET error for %s: %s", key, e)
            return False

    def delete(self, key: str) -> bool:
        """Delete a cached value."""
        if self._fallback:
            return self._fallback.delete(key)

        try:
            result = self._client.delete(self._prefixed_key(key))
            return result > 0
        except Exception as e:
            logger.error("Redis DELETE error for %s: %s", key, e)
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if self._fallback:
            return self._fallback.exists(key)

        try:
            return bool(self._client.exists(self._prefixed_key(key)))
        except Exception as e:
            logger.error("Redis EXISTS error for %s: %s", key, e)
            return False

    def clear(self) -> int:
        """Clear all cached values with our prefix."""
        if self._fallback:
            return self._fallback.clear()

        try:
            # Find all keys with our prefix
            pattern = f"{self._key_prefix}:*"
            keys = list(self._client.scan_iter(pattern))

            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error("Redis CLEAR error: %s", e)
            return 0

    def is_available(self) -> bool:
        """Check if Redis is connected (or fallback is available)."""
        if self._fallback:
            return True  # Fallback is always available

        try:
            self._client.ping()
            return True
        except Exception:
            return False

    @property
    def using_fallback(self) -> bool:
        """Check if we're using the in-memory fallback."""
        return self._fallback is not None

    def get_info(self) -> Dict[str, Any]:
        """Get Redis server info (or fallback status)."""
        if self._fallback:
            return {
                "backend": "MemoryCache (fallback)",
                "reason": "Redis unavailable",
            }

        try:
            info = self._client.info()
            return {
                "backend": "Redis",
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            return {
                "backend": "Redis",
                "error": str(e),
            }
