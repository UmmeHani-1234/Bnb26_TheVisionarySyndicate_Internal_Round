"""In-memory and TTL caching layer for external product data requests."""

import time
from typing import Any, Dict, Optional, Tuple


class ProductDataCache:
    """Thread-safe TTL cache for product searches and queries."""

    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value if present and not expired."""
        if key in self._cache:
            expires_at, val = self._cache[key]
            if time.time() < expires_at:
                self.hits += 1
                return val
            # Expired
            del self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store value with expiration."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self._cache[key] = (time.time() + ttl, value)

    def clear(self) -> None:
        """Flush cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
        }
