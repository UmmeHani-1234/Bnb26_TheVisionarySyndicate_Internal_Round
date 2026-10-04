"""Abstract Base Class for Product Data Providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .schema import NormalizedProduct, SearchFilter


class ProductDataProvider(ABC):
    """Abstract interface for all product data sources (Local, Best Buy, Open APIs, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name/identifier."""
        pass

    @abstractmethod
    def search(self, filters: SearchFilter, limit: int = 10) -> List[NormalizedProduct]:
        """Search products matching category, brand, budget, and specification filters."""
        pass

    @abstractmethod
    def get_product_by_name(self, name: str) -> Optional[NormalizedProduct]:
        """Fetch exact or closest product by name."""
        pass

    @abstractmethod
    def get_product_by_id(self, product_id: str) -> Optional[NormalizedProduct]:
        """Fetch product by unique ID."""
        pass

    @abstractmethod
    def list_categories(self) -> List[str]:
        """List all supported/available product categories."""
        pass
