"""Normalized product schema and data types for the Black Box Electronics Consultant."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class NormalizedProduct:
    """Universal normalized product representation across all electronics categories."""

    product_id: str
    name: str
    brand: str
    category: str
    subcategory: Optional[str] = None
    model: Optional[str] = None
    price: float = 0.0
    currency: str = "INR"
    availability: bool = True
    image_url: Optional[str] = None
    description: Optional[str] = None
    source: str = "local"
    source_url: Optional[str] = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    identifiers: Dict[str, str] = field(default_factory=dict)
    specifications: Dict[str, Any] = field(default_factory=dict)

    # Legacy/Computing convenience properties for backwards compatibility
    @property
    def ram_gb(self) -> Optional[int]:
        val = self.specifications.get("ram_gb", self.specifications.get("ram"))
        return int(val) if val is not None else None

    @property
    def storage_gb(self) -> Optional[int]:
        val = self.specifications.get("storage_gb", self.specifications.get("storage"))
        return int(val) if val is not None else None

    @property
    def processor(self) -> Optional[str]:
        return self.specifications.get("processor") or self.specifications.get("cpu")

    @property
    def gpu(self) -> Optional[str]:
        return self.specifications.get("gpu") or self.specifications.get("graphics")

    @property
    def dedicated_gpu(self) -> bool:
        return bool(self.specifications.get("dedicated_gpu", False))

    @property
    def programming_suitability(self) -> int:
        return int(self.specifications.get("programming_suitability", 3))

    @property
    def gaming_suitability(self) -> int:
        return int(self.specifications.get("gaming_suitability", 3))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary preserving both normalized and flat specification keys."""
        data: Dict[str, Any] = {
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "model": self.model or self.name,
            "category": self.category.lower(),
            "subcategory": self.subcategory,
            "price": self.price,
            "currency": self.currency,
            "availability": self.availability,
            "image_url": self.image_url,
            "description": self.description,
            "source": self.source,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "identifiers": self.identifiers,
            "specifications": self.specifications,
        }
        # Flatten category specifications into the root for direct tool/view access
        for key, value in self.specifications.items():
            if key not in data:
                data[key] = value
        return data


@dataclass
class SearchFilter:
    """Universal category-aware search query filters."""

    category: Optional[str] = None
    subcategory: Optional[str] = None
    query: Optional[str] = None
    brand: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    use_case: Optional[str] = None
    min_ram_gb: Optional[int] = None
    min_storage_gb: Optional[int] = None
    needs_gaming: bool = False
    needs_programming: bool = False
    required_features: List[str] = field(default_factory=list)
    spec_constraints: Dict[str, Any] = field(default_factory=dict)
