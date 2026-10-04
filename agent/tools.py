"""The observable tools used by the Black Box Electronics Consultant Agent.

Every tool returns a plain dict. Success results have `"status": "success"`.
Failures return a structured error:
    {"status": "error", "error_type": "...", "message": "..."}

Tools are category-aware and support laptops, smartphones, monitors, TVs, headphones,
cameras, smartwatches, tablets, speakers, routers, and accessories via the ProductDataProvider.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .data_providers import (
    CompositeProductProvider,
    NormalizedProduct,
    SearchFilter,
    get_product_provider,
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "products.json"

# "Suitable" in search means a suitability score of at least this (scale 1-5).
SEARCH_MIN_SUITABILITY = 3


# --------------------------------------------------------------------------- helpers


class ToolInputError(Exception):
    """Raised inside a tool for invalid input; converted into a structured error."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


def error_result(error_type: str, message: str, **extra: Any) -> dict:
    return {"status": "error", "error_type": error_type, "message": message, **extra}


def _guard(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Turn any exception inside a tool into a structured error result.

    Supports reproducible failure injection via FAILURE_MODE environment variable.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        mode = os.getenv("FAILURE_MODE", "").strip().lower()
        if mode == "timeout":
            return error_result("timeout", "Tool execution timed out after 30000ms")
        if mode == "unexpected_output":
            return {
                "status": "success",
                "malformed": True,
                "product": "Faulty Laptop",
                "price": None,
                "count": 1,
                "products": [{"name": "Faulty Laptop", "price": None}],
            }

        try:
            return fn(*args, **kwargs)
        except ToolInputError as exc:
            return error_result(exc.error_type, str(exc))
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all, reported as data
            return error_result("internal_error", f"{type(exc).__name__}: {exc}")

    return wrapper


def load_products(path: Optional[str] = None) -> list[dict]:
    """Load the benchmark product catalogue. Preserved for deterministic benchmark isolation."""
    data_path = Path(path or os.getenv("PRODUCTS_PATH") or DEFAULT_DATA_PATH)
    with data_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _find_product(products: list[dict], name: str) -> dict:
    if not isinstance(name, str) or not name.strip():
        raise ToolInputError("invalid_product_name", "product_name must be a non-empty string.")
    wanted = name.strip().lower()
    for product in products:
        if product["name"].lower() == wanted:
            return product
    partial = [p for p in products if wanted in p["name"].lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ToolInputError(
            "ambiguous_product",
            f"'{name}' matches several products: {[p['name'] for p in partial]}. Use the exact name.",
        )
    raise ToolInputError(
        "product_not_found",
        f"No product named '{name}'. Available products: {[p['name'] for p in products]}.",
    )


def _positive(label: str, value: Optional[float]) -> None:
    if value is not None and value <= 0:
        raise ToolInputError("invalid_input", f"{label} must be greater than 0 (got {value}).")


# --------------------------------------------------------------------------- schemas


class SearchProductsInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Product category, e.g. 'laptop', 'gaming', 'ultrabook', 'smartphone', 'monitor', 'tv', 'headphones', 'camera', 'smartwatch', 'tablet', 'speaker', 'router'.",
    )
    max_price: Optional[float] = Field(default=None, description="Maximum price in Indian rupees (₹).")
    min_ram_gb: Optional[int] = Field(default=None, description="Minimum RAM in GB.")
    min_storage_gb: Optional[int] = Field(default=None, description="Minimum storage in GB.")
    needs_gaming: bool = Field(default=False, description="True if the user wants a device suitable for gaming.")
    needs_programming: bool = Field(
        default=False, description="True if the user wants a device suitable for programming."
    )
    brand: Optional[str] = Field(default=None, description="Optional brand name filter (e.g. Sony, Apple, Samsung, Dell, Asus).")
    query: Optional[str] = Field(default=None, description="General search keyword or specification term.")


class CheckSpecificationsInput(BaseModel):
    product_name: str = Field(description="Exact product name as returned by search_products.")
    min_ram_gb: Optional[int] = Field(default=None, description="Required minimum RAM in GB.")
    min_storage_gb: Optional[int] = Field(default=None, description="Required minimum storage in GB.")
    requires_dedicated_gpu: Optional[bool] = Field(
        default=None, description="Set to true to require a dedicated (discrete) GPU."
    )
    min_gaming_score: Optional[int] = Field(
        default=None, description="Required minimum gaming suitability score (1-5)."
    )
    min_programming_score: Optional[int] = Field(
        default=None, description="Required minimum programming suitability score (1-5)."
    )
    processor_contains: Optional[str] = Field(
        default=None, description="Text the processor name must contain, e.g. 'Ryzen', 'i7', 'M2', 'Snapdragon'."
    )
    screen_size_min: Optional[float] = Field(default=None, description="Minimum screen size in inches (for TVs, monitors, tablets).")
    refresh_rate_min_hz: Optional[int] = Field(default=None, description="Minimum display refresh rate in Hz.")
    requires_noise_cancellation: Optional[bool] = Field(default=None, description="Set to true to require Active Noise Cancellation.")


class CalculateBudgetInput(BaseModel):
    product_name: str = Field(description="Exact product name as returned by search_products.")
    budget: float = Field(description="The user's maximum budget in Indian rupees (₹).")


# --------------------------------------------------------------------------- tools


@tool("search_products", args_schema=SearchProductsInput)
@_guard
def search_products(
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_ram_gb: Optional[int] = None,
    min_storage_gb: Optional[int] = None,
    needs_gaming: bool = False,
    needs_programming: bool = False,
    brand: Optional[str] = None,
    query: Optional[str] = None,
) -> dict:
    """Search the electronics catalogue across all categories (laptops, monitors, TVs, phones, audio, cameras, etc.) and return matching products, cheapest first.

    All filters are optional and combined with AND.
    Returns {"status", "filters", "count", "products"}.
    """
    _positive("max_price", max_price)
    _positive("min_ram_gb", min_ram_gb)
    _positive("min_storage_gb", min_storage_gb)

    # 1. Check if PRODUCTS_PATH is overridden or if standard local catalogue is loaded
    provider = get_product_provider()
    known_categories = provider.list_categories()
    # Add common aliases for known validation
    all_known_cats = {
        "gaming", "ultrabook", "budget", "laptop", "laptops", "notebook",
        "smartphone", "smartphones", "phone", "phones", "mobile",
        "monitor", "monitors", "display", "displays", "screen",
        "tv", "tvs", "television", "smart_tv", "oled_tv", "budget_4k_tv",
        "headphones", "headphone", "earbuds", "audio", "wireless_anc_headphones", "true_wireless_anc",
        "camera", "cameras", "mirrorless_camera",
        "smartwatch", "smartwatches", "watch", "fitness_smartwatch",
        "tablet", "tablets", "ipad", "productivity_tablet",
        "speaker", "speakers", "smart_home_speaker",
        "router", "routers", "networking", "gaming_router",
        "desktop_pc", "pc", "accessories", "electronics",
    } | set(known_categories)

    cat = None
    if category is not None and str(category).strip():
        cat = str(category).strip().lower()
        if cat not in all_known_cats:
            raise ToolInputError("invalid_category", f"Unknown category '{category}'. Known categories: {sorted(all_known_cats)}.")

    # 2. If PRODUCTS_PATH is explicitly set (e.g. unit test fixture with custom file), read directly
    if os.getenv("PRODUCTS_PATH"):
        raw_products = load_products()
        matches = []
        for p in raw_products:
            if cat and p.get("category", "").lower() != cat and cat not in p.get("category", "").lower():
                continue
            if max_price is not None and p.get("price", 0) > max_price:
                continue
            if min_ram_gb is not None and p.get("ram_gb", 0) < min_ram_gb:
                continue
            if min_storage_gb is not None and p.get("storage_gb", 0) < min_storage_gb:
                continue
            if needs_gaming and p.get("gaming_suitability", 0) < SEARCH_MIN_SUITABILITY:
                continue
            if needs_programming and p.get("programming_suitability", 0) < SEARCH_MIN_SUITABILITY:
                continue
            matches.append(p)
        matches.sort(key=lambda p: (p.get("price", 0), p.get("name", "")))
        count = len(matches)
        dict_products = matches
    else:
        # Standard multi-category search via Provider
        search_filter = SearchFilter(
            category=cat,
            brand=brand,
            query=query,
            max_price=max_price,
            min_ram_gb=min_ram_gb,
            min_storage_gb=min_storage_gb,
            needs_gaming=needs_gaming,
            needs_programming=needs_programming,
        )
        norm_matches = provider.search(search_filter, limit=20)
        dict_products = [p.to_dict() for p in norm_matches]
        count = len(dict_products)

    filters = {
        "category": cat,
        "max_price": max_price,
        "min_ram_gb": min_ram_gb,
        "min_storage_gb": min_storage_gb,
        "needs_gaming": needs_gaming,
        "needs_programming": needs_programming,
        "brand": brand,
        "query": query,
    }
    return {
        "status": "success",
        "filters": {k: v for k, v in filters.items() if v not in (None, False)},
        "count": count,
        "products": dict_products,
    }


@tool("check_specifications", args_schema=CheckSpecificationsInput)
@_guard
def check_specifications(
    product_name: str,
    min_ram_gb: Optional[int] = None,
    min_storage_gb: Optional[int] = None,
    requires_dedicated_gpu: Optional[bool] = None,
    min_gaming_score: Optional[int] = None,
    min_programming_score: Optional[int] = None,
    processor_contains: Optional[str] = None,
    screen_size_min: Optional[float] = None,
    refresh_rate_min_hz: Optional[int] = None,
    requires_noise_cancellation: Optional[bool] = None,
) -> dict:
    """Check one product against specific hardware/category requirements and report which passed or failed.

    Provide at least one requirement besides product_name. Returns
    {"status", "product", "price", "checks", "passed_count", "failed_count", "all_passed"}
    where each check is {"requirement", "required", "actual", "passed"}.
    """
    # Fetch product either from PRODUCTS_PATH file or unified provider
    if os.getenv("PRODUCTS_PATH"):
        product_dict = _find_product(load_products(), product_name)
    else:
        provider = get_product_provider()
        norm_p = provider.get_product_by_name(product_name)
        if norm_p is not None:
            product_dict = norm_p.to_dict()
        else:
            # Fall back to raw search list
            product_dict = _find_product(load_products(), product_name)

    checks: list[dict] = []

    def add(requirement: str, required: Any, actual: Any, passed: bool) -> None:
        checks.append({"requirement": requirement, "required": required, "actual": actual, "passed": bool(passed)})

    # Computing / Laptop checks
    if min_ram_gb is not None:
        actual_ram = product_dict.get("ram_gb", product_dict.get("specifications", {}).get("ram_gb"))
        add("min_ram_gb", min_ram_gb, actual_ram, actual_ram is not None and actual_ram >= min_ram_gb)

    if min_storage_gb is not None:
        actual_storage = product_dict.get("storage_gb", product_dict.get("specifications", {}).get("storage_gb"))
        add("min_storage_gb", min_storage_gb, actual_storage, actual_storage is not None and actual_storage >= min_storage_gb)

    if requires_dedicated_gpu is not None:
        actual_gpu = product_dict.get("dedicated_gpu", product_dict.get("specifications", {}).get("dedicated_gpu", False))
        add("dedicated_gpu", requires_dedicated_gpu, actual_gpu, actual_gpu == requires_dedicated_gpu)

    if min_gaming_score is not None:
        actual_game = product_dict.get("gaming_suitability", product_dict.get("specifications", {}).get("gaming_suitability", 1))
        add("min_gaming_score", min_gaming_score, actual_game, actual_game >= min_gaming_score)

    if min_programming_score is not None:
        actual_prog = product_dict.get("programming_suitability", product_dict.get("specifications", {}).get("programming_suitability", 1))
        add("min_programming_score", min_programming_score, actual_prog, actual_prog >= min_programming_score)

    if processor_contains:
        actual_proc = product_dict.get("processor", product_dict.get("specifications", {}).get("processor", ""))
        add("processor_contains", processor_contains, actual_proc, processor_contains.lower() in str(actual_proc).lower())

    # Multi-category specific checks
    specs = product_dict.get("specifications", {})
    if screen_size_min is not None:
        actual_screen = specs.get("screen_size")
        add("screen_size_min", screen_size_min, actual_screen, actual_screen is not None and actual_screen >= screen_size_min)

    if refresh_rate_min_hz is not None:
        actual_hz = specs.get("refresh_rate_hz")
        add("refresh_rate_min_hz", refresh_rate_min_hz, actual_hz, actual_hz is not None and actual_hz >= refresh_rate_min_hz)

    if requires_noise_cancellation is not None:
        actual_anc = specs.get("noise_cancellation", False)
        add("noise_cancellation", requires_noise_cancellation, actual_anc, actual_anc == requires_noise_cancellation)

    if not checks:
        raise ToolInputError(
            "no_requirements",
            "Provide at least one requirement (min_ram_gb, min_storage_gb, requires_dedicated_gpu, "
            "min_gaming_score, min_programming_score, processor_contains, screen_size_min, etc.).",
        )

    passed = sum(1 for c in checks if c["passed"])
    return {
        "status": "success",
        "product": product_dict.get("name", product_name),
        "price": product_dict.get("price"),
        "checks": checks,
        "passed_count": passed,
        "failed_count": len(checks) - passed,
        "all_passed": passed == len(checks),
    }


@tool("calculate_budget", args_schema=CalculateBudgetInput)
@_guard
def calculate_budget(product_name: str, budget: float) -> dict:
    """Compare any electronics product's catalogue price with the user's maximum budget.

    The price is looked up from the catalogue by product name (never supplied by the caller).
    Returns {"status", "product", "price", "budget", "difference", "within_budget", "note"}
    where difference = budget - price (negative means over budget).
    """
    _positive("budget", budget)

    if os.getenv("PRODUCTS_PATH"):
        product = _find_product(load_products(), product_name)
    else:
        provider = get_product_provider()
        norm_p = provider.get_product_by_name(product_name)
        if norm_p is not None:
            product = norm_p.to_dict()
        else:
            product = _find_product(load_products(), product_name)

    price = product["price"]
    difference = budget - price
    return {
        "status": "success",
        "product": product["name"],
        "price": price,
        "budget": budget,
        "difference": difference,
        "within_budget": difference >= 0,
        "note": "difference = budget - price; a negative value means the product is over budget",
    }


TOOLS = [search_products, check_specifications, calculate_budget]
