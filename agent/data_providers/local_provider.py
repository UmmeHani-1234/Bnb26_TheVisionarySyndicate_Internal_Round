"""Local deterministic product data provider for the Black Box benchmark and agent."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ProductDataProvider
from .schema import NormalizedProduct, SearchFilter

logger = logging.getLogger(__name__)

DEFAULT_LAPTOPS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "products.json"
DEFAULT_ELECTRONICS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "electronics_catalogue.json"


class LocalCatalogueProvider(ProductDataProvider):
    """Deterministic local catalogue provider for reproducible traces, benchmarks and testing."""

    def __init__(
        self,
        laptops_path: Optional[Path | str] = None,
        electronics_path: Optional[Path | str] = None,
    ) -> None:
        self.laptops_path = Path(laptops_path or os.getenv("PRODUCTS_PATH") or DEFAULT_LAPTOPS_PATH)
        self.electronics_path = Path(electronics_path or DEFAULT_ELECTRONICS_PATH)

    @property
    def name(self) -> str:
        return "local_catalogue"

    def _load_raw_data(self) -> List[NormalizedProduct]:
        """Loads and normalizes products from local JSON files."""
        normalized: List[NormalizedProduct] = []
        seen_names = set()

        # 1. Load benchmark laptops
        if self.laptops_path.exists():
            try:
                with open(self.laptops_path, "r", encoding="utf-8") as f:
                    laptop_items = json.load(f)
                    for idx, item in enumerate(laptop_items):
                        name = item.get("name", f"Laptop-{idx}")
                        seen_names.add(name.lower())
                        specs = {
                            "ram_gb": item.get("ram_gb"),
                            "storage_gb": item.get("storage_gb"),
                            "processor": item.get("processor"),
                            "gpu": item.get("gpu"),
                            "dedicated_gpu": bool(item.get("dedicated_gpu", False)),
                            "programming_suitability": item.get("programming_suitability", 3),
                            "gaming_suitability": item.get("gaming_suitability", 3),
                        }
                        # Preserve exact category name from benchmark (e.g. 'gaming', 'ultrabook', 'budget')
                        raw_cat = item.get("category", "laptop")
                        prod = NormalizedProduct(
                            product_id=f"lap-{idx+1:03d}",
                            name=name,
                            brand=name.split()[0],
                            category=raw_cat.lower(),
                            subcategory="laptop",
                            model=name,
                            price=float(item.get("price", 0)),
                            currency="INR",
                            availability=True,
                            image_url=item.get("image_url"),
                            description=f"{raw_cat.title()} laptop with {item.get('processor')} and {item.get('ram_gb')}GB RAM.",
                            source="local",
                            specifications=specs,
                        )
                        normalized.append(prod)
            except Exception as exc:
                logger.warning("Failed to load benchmark laptops from %s: %s", self.laptops_path, exc)

        # 2. Load multi-category electronics catalogue
        if self.electronics_path.exists():
            try:
                with open(self.electronics_path, "r", encoding="utf-8") as f:
                    multi_items = json.load(f)
                    for item in multi_items:
                        name = item.get("name", "")
                        if name.lower() in seen_names:
                            continue
                        seen_names.add(name.lower())
                        prod = NormalizedProduct(
                            product_id=item.get("product_id", f"prod-{len(normalized)+1:03d}"),
                            name=name,
                            brand=item.get("brand", name.split()[0] if name else "Generic"),
                            category=item.get("category", "electronics").lower(),
                            subcategory=item.get("subcategory"),
                            model=item.get("model", name),
                            price=float(item.get("price", 0)),
                            currency=item.get("currency", "INR"),
                            availability=bool(item.get("availability", True)),
                            image_url=item.get("image_url"),
                            description=item.get("description"),
                            source="local",
                            identifiers=item.get("identifiers", {}),
                            specifications=item.get("specifications", {}),
                        )
                        normalized.append(prod)
            except Exception as exc:
                logger.warning("Failed to load electronics catalogue from %s: %s", self.electronics_path, exc)

        return normalized

    def get_all_products(self) -> List[NormalizedProduct]:
        """Return all local catalogue products, reloading if needed."""
        return self._load_raw_data()

    def search(self, filters: SearchFilter, limit: int = 20) -> List[NormalizedProduct]:
        """Perform category-aware filtering across local products."""
        products = self.get_all_products()
        matches: List[NormalizedProduct] = []

        target_category = None
        if filters.category:
            cat_raw = str(filters.category).strip().lower()
            cat_map = {
                "laptops": "laptop",
                "notebook": "laptop",
                "phones": "smartphone",
                "phone": "smartphone",
                "mobile": "smartphone",
                "smartphones": "smartphone",
                "monitors": "monitor",
                "display": "monitor",
                "displays": "monitor",
                "screen": "monitor",
                "tvs": "tv",
                "television": "tv",
                "smart_tv": "tv",
                "headphone": "headphones",
                "audio": "headphones",
                "earphone": "headphones",
                "earbuds": "earbuds",
                "cameras": "camera",
                "watch": "smartwatch",
                "watches": "smartwatch",
                "tablets": "tablet",
                "ipad": "tablet",
                "speakers": "speaker",
                "soundbar": "speaker",
                "routers": "router",
                "networking": "router",
                "wifi": "router",
            }
            target_category = cat_map.get(cat_raw, cat_raw)

        for p in products:
            p_cat = p.category.lower()
            p_sub = (p.subcategory or "").lower()

            # Category match
            if target_category:
                if target_category == "laptop":
                    if p_cat not in ("laptop", "gaming", "ultrabook", "budget") and p_sub != "laptop":
                        continue
                elif p_cat != target_category and p_sub != target_category and target_category not in p_cat:
                    continue

            # Brand match
            if filters.brand:
                brand_q = filters.brand.strip().lower()
                if brand_q not in p.brand.lower() and brand_q not in p.name.lower():
                    continue

            # Query text match
            if filters.query:
                query_words = [w for w in filters.query.lower().split() if len(w) > 2]
                text_corpus = f"{p.name} {p.brand} {p.model} {p.category} {p.subcategory or ''} {p.description or ''} {json.dumps(p.specifications)}".lower()
                if query_words and not any(w in text_corpus for w in query_words):
                    continue

            # Price / Budget
            if filters.max_price is not None and p.price > filters.max_price:
                continue
            if filters.min_price is not None and p.price < filters.min_price:
                continue

            # Computing / Laptop specific constraints
            specs = p.specifications
            if filters.min_ram_gb is not None:
                ram = specs.get("ram_gb", specs.get("ram"))
                if ram is None or ram < filters.min_ram_gb:
                    continue
            if filters.min_storage_gb is not None:
                storage = specs.get("storage_gb", specs.get("storage"))
                if storage is None or storage < filters.min_storage_gb:
                    continue
            if filters.needs_gaming:
                if p_cat not in ("laptop", "gaming", "ultrabook", "budget") and p_sub != "laptop":
                    continue
                game_score = specs.get("gaming_suitability")
                if game_score is None or game_score < 3:
                    continue
            if filters.needs_programming:
                if p_cat not in ("laptop", "gaming", "ultrabook", "budget") and p_sub != "laptop":
                    continue
                prog_score = specs.get("programming_suitability")
                if prog_score is None or prog_score < 3:
                    continue

            matches.append(p)

        # Sort cheapest first deterministically
        matches.sort(key=lambda p: (p.price, p.name))
        return matches[:limit]

    def get_product_by_name(self, name: str) -> Optional[NormalizedProduct]:
        if not name or not isinstance(name, str) or not name.strip():
            return None
        wanted = name.strip().lower()
        products = self.get_all_products()

        for p in products:
            if p.name.lower() == wanted:
                return p

        partials = [p for p in products if wanted in p.name.lower()]
        if len(partials) == 1:
            return partials[0]
        return None

    def get_product_by_id(self, product_id: str) -> Optional[NormalizedProduct]:
        products = self.get_all_products()
        for p in products:
            if p.product_id == product_id:
                return p
        return None

    def list_categories(self) -> List[str]:
        products = self.get_all_products()
        cats = {p.category for p in products}
        for p in products:
            if p.subcategory:
                cats.add(p.subcategory)
        return sorted(cats)
