"""Composite and Unified Product Provider Factory."""

from __future__ import annotations

import os
from typing import List, Optional

from .base import ProductDataProvider
from .cache import ProductDataCache
from .external_provider import ExternalProductProvider
from .local_provider import LocalCatalogueProvider
from .schema import NormalizedProduct, SearchFilter


class CompositeProductProvider(ProductDataProvider):
    """Unified provider that coordinates local deterministic data and external live APIs."""

    def __init__(
        self,
        local_provider: Optional[LocalCatalogueProvider] = None,
        external_provider: Optional[ExternalProductProvider] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.local = local_provider or LocalCatalogueProvider()
        self.external = external_provider or ExternalProductProvider()
        # Modes: "local" (default for benchmark isolation and deterministic evaluation), "hybrid", "external"
        self.mode = (mode or os.getenv("DATA_SOURCE_MODE") or "local").strip().lower()

    @property
    def name(self) -> str:
        return f"composite_{self.mode}"

    def list_categories(self) -> List[str]:
        local_cats = set(self.local.list_categories())
        ext_cats = set(self.external.list_categories())
        return sorted(local_cats | ext_cats)

    def search(self, filters: SearchFilter, limit: int = 10) -> List[NormalizedProduct]:
        """Search products respecting benchmark isolation and data source mode."""
        local_matches = self.local.search(filters=filters, limit=limit)

        if self.mode == "local" or len(local_matches) >= limit:
            return local_matches

        if self.mode in ("hybrid", "external"):
            needed = limit - len(local_matches)
            try:
                ext_matches = self.external.search(filters=filters, limit=needed)
                existing_names = {p.name.lower() for p in local_matches}
                combined = list(local_matches)
                for ext_p in ext_matches:
                    if ext_p.name.lower() not in existing_names:
                        combined.append(ext_p)
                return combined[:limit]
            except Exception:
                return local_matches

        return local_matches

    def get_product_by_name(self, name: str) -> Optional[NormalizedProduct]:
        # Local first for deterministic benchmark fidelity
        found = self.local.get_product_by_name(name)
        if found:
            return found
        if self.mode in ("hybrid", "external"):
            return self.external.get_product_by_name(name)
        return None

    def get_product_by_id(self, product_id: str) -> Optional[NormalizedProduct]:
        found = self.local.get_product_by_id(product_id)
        if found:
            return found
        if self.mode in ("hybrid", "external"):
            return self.external.get_product_by_id(product_id)
        return None


# Global singleton instance for the agent and tools
_global_provider: Optional[CompositeProductProvider] = None


def get_product_provider() -> CompositeProductProvider:
    """Retrieve or initialize the global product data provider singleton."""
    global _global_provider
    if _global_provider is None:
        _global_provider = CompositeProductProvider()
    return _global_provider


def reset_product_provider() -> None:
    """Reset provider singleton (used in test fixtures)."""
    global _global_provider
    _global_provider = None
