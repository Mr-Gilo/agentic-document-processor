"""
Simple in-memory tool result cache.
Caches results per (document_hash, tool_name) pair.
Avoids redundant LLM calls when the same document is processed multiple times.
"""

import hashlib
import time
from typing import Optional, Dict, Any

class ToolCache:
    """
    In-memory LRU-style cache for tool results.
    Key: (document_hash, tool_name)
    TTL: 1 hour by default
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 500):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    def _make_key(self, text: str, tool_name: str) -> str:
        doc_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{doc_hash}:{tool_name}"

    def get(self, text: str, tool_name: str) -> Optional[Dict]:
        key = self._make_key(text, tool_name)
        entry = self.cache.get(key)

        if entry is None:
            self.misses += 1
            return None

        if time.time() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return entry["result"]

    def set(self, text: str, tool_name: str, result: Dict):
        if len(self.cache) >= self.max_entries:
            # Evict oldest entry
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]

        key = self._make_key(text, tool_name)
        self.cache[key] = {
            "result": result,
            "timestamp": time.time(),
            "tool": tool_name
        }

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(100 * self.hits / total, 1) if total > 0 else 0.0
        }

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# Global cache instance shared across requests
tool_cache = ToolCache()