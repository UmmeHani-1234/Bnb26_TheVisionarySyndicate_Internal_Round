"""External Product Data Provider implementation with Best Buy and Open Electronics endpoints."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .base import ProductDataProvider
from .cache import ProductDataCache
from .schema import NormalizedProduct, SearchFilter

logger = logging.getLogger(__name__)


class ExternalProductProvider(ProductDataProvider):
    """External product data provider supporting Best Buy Products API and Open Electronics APIs.
    
    Security: API keys are strictly read from backend environment variables (BESTBUY_API_KEY).
    Caching: All external search queries and product lookups are cached with configurable TTL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[ProductDataCache] = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.api_key = api_key or os.getenv("BESTBUY_API_KEY")
        self.cache = cache or ProductDataCache(default_ttl_seconds=1800)
        self.timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "external_bestbuy_and_open_api"

    def list_categories(self) -> List[str]:
        return [
            "laptop",
            "smartphone",
            "tablet",
            "monitor",
            "tv",
            "headphones",
            "earbuds",
            "speaker",
            "camera",
            "smartwatch",
            "router",
            "gaming_console",
            "desktop_pc",
            "accessories",
        ]

    def _fetch_from_dummyjson(self, category: Optional[str], limit: int = 10) -> List[NormalizedProduct]:
        """Fetch real consumer electronics from DummyJSON Open Electronics API with INR conversion."""
        category_map = {
            "laptop": "laptops",
            "smartphone": "smartphones",
            "tablet": "tablets",
            "mobile": "smartphones",
        }
        dj_cat = category_map.get((category or "").lower(), "laptops")
        url = f"https://dummyjson.com/products/category/{dj_cat}?limit={limit}"
        
        cache_key = f"dj_{dj_cat}_{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BlackBox-AI-Consultant/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                products_raw = data.get("products", [])
                normalized = []
                for p in products_raw:
                    # Convert USD price to approximate INR (x85)
                    usd_price = float(p.get("price", 100))
                    inr_price = round(usd_price * 85, -1)
                    
                    prod = NormalizedProduct(
                        product_id=f"ext-dj-{p.get('id')}",
                        name=p.get("title", "Electronics Device"),
                        brand=p.get("brand", "Global"),
                        category=category or "electronics",
                        subcategory=p.get("category"),
                        model=p.get("title"),
                        price=inr_price,
                        currency="INR",
                        availability=p.get("availabilityStatus", "In Stock") == "In Stock",
                        image_url=p.get("thumbnail") or (p.get("images", [None])[0]),
                        description=p.get("description"),
                        source="external_open_api",
                        source_url=f"https://dummyjson.com/products/{p.get('id')}",
                        identifiers={"sku": str(p.get("sku", p.get("id")))},
                        specifications={
                            "rating": p.get("rating"),
                            "stock": p.get("stock"),
                            "weight_g": p.get("weight"),
                            "dimensions": p.get("dimensions"),
                            "warranty": p.get("warrantyInformation"),
                        },
                    )
                    normalized.append(prod)

                self.cache.set(cache_key, normalized, ttl_seconds=3600)
                return normalized
        except Exception as exc:
            logger.info("Open electronics API query failed or timed out: %s", exc)
            return []

    def _fetch_from_bestbuy(self, query: str, limit: int = 10) -> List[NormalizedProduct]:
        """Fetch structured products from Best Buy official API when API key is provided."""
        if not self.api_key:
            return []

        cache_key = f"bb_{query}_{limit}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        encoded_q = urllib.parse.quote(query)
        url = (
            f"https://api.bestbuy.com/v1/products((search={encoded_q}))"
            f"?apiKey={self.api_key}&format=json&show=sku,name,manufacturer,modelNumber,regularPrice,salePrice,shortDescription,image,categoryPath,details"
            f"&pageSize={limit}"
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BlackBox-AI-Consultant/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                products_raw = data.get("products", [])
                normalized = []
                for p in products_raw:
                    usd_price = float(p.get("salePrice") or p.get("regularPrice") or 0)
                    inr_price = round(usd_price * 85, -1)

                    specs: Dict[str, Any] = {}
                    for detail in p.get("details", []):
                        name = detail.get("name")
                        val = detail.get("value")
                        if name and val:
                            specs[name.lower().replace(" ", "_")] = val

                    prod = NormalizedProduct(
                        product_id=f"ext-bb-{p.get('sku')}",
                        name=p.get("name", "Product"),
                        brand=p.get("manufacturer", "Unknown"),
                        category="electronics",
                        model=p.get("modelNumber"),
                        price=inr_price,
                        currency="INR",
                        availability=True,
                        image_url=p.get("image"),
                        description=p.get("shortDescription"),
                        source="external_bestbuy",
                        source_url=f"https://www.bestbuy.com/site/{p.get('sku')}.p",
                        identifiers={"sku": str(p.get("sku"))},
                        specifications=specs,
                    )
                    normalized.append(prod)

                self.cache.set(cache_key, normalized, ttl_seconds=3600)
                return normalized
        except Exception as exc:
            logger.warning("Best Buy API query error: %s", exc)
            return []

    def search(self, filters: SearchFilter, limit: int = 10) -> List[NormalizedProduct]:
        """Search external sources with caching and budget filtering."""
        query = filters.query or filters.category or "electronics"
        
        # 1. Best Buy if key available
        results = self._fetch_from_bestbuy(query=query, limit=limit)
        
        # 2. Fall back to Open Electronics API
        if not results:
            results = self._fetch_from_dummyjson(category=filters.category, limit=limit)

        # Filter by budget
        if filters.max_price is not None:
            results = [p for p in results if p.price <= filters.max_price]

        return results[:limit]

    def get_product_by_name(self, name: str) -> Optional[NormalizedProduct]:
        if not name or not isinstance(name, str):
            return None
        wanted = name.strip().lower()
        results = self.search(SearchFilter(query=name), limit=5)
        for p in results:
            if p.name.lower() == wanted or wanted in p.name.lower():
                return p
        return None

    def get_product_by_id(self, product_id: str) -> Optional[NormalizedProduct]:
        # Look up in cache
        for _, val in self.cache._cache.values():
            if isinstance(val, list):
                for p in val:
                    if isinstance(p, NormalizedProduct) and p.product_id == product_id:
                        return p
        return None
