"""The three tools the agent can use. All of them read only the local dataset.

Every tool returns a plain dict. Success results have `"status": "success"`.
Failures never raise out of a tool; they come back as a structured error:

    {"status": "error", "error_type": "...", "message": "..."}

so the failure is visible as an observable event and the agent can decide how to
proceed (retry with corrected input, or explain the problem to the user).

Note: arguments with the wrong *type* (e.g. a non-numeric budget) are rejected by
LangChain's schema validation before the tool body runs. That surfaces to the
recorder as `on_tool_error`, which the callback handler also turns into a tool_error event.
"""

import functools
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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
    """Turn any exception inside a tool into a structured error result."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except ToolInputError as exc:
            return error_result(exc.error_type, str(exc))
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all, reported as data
            return error_result("internal_error", f"{type(exc).__name__}: {exc}")

    return wrapper


def load_products(path: Optional[str] = None) -> list[dict]:
    """Load the product catalogue. Re-read on every call so the file can be swapped later."""
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
        default=None, description="Product category to filter on, e.g. 'gaming', 'ultrabook', 'budget'."
    )
    max_price: Optional[float] = Field(default=None, description="Maximum price in Indian rupees (₹).")
    min_ram_gb: Optional[int] = Field(default=None, description="Minimum RAM in GB.")
    min_storage_gb: Optional[int] = Field(default=None, description="Minimum storage in GB.")
    needs_gaming: bool = Field(default=False, description="True if the user wants a laptop suitable for gaming.")
    needs_programming: bool = Field(
        default=False, description="True if the user wants a laptop suitable for programming."
    )


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
        default=None, description="Text the processor name must contain, e.g. 'Ryzen' or 'i7'."
    )


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
) -> dict:
    """Search the local laptop catalogue and return matching products, cheapest first.

    All filters are optional and combined with AND. needs_gaming / needs_programming
    keep only laptops with a suitability score of 3 or more (scale 1-5).
    Returns {"status", "filters", "count", "products"}.
    """
    _positive("max_price", max_price)
    _positive("min_ram_gb", min_ram_gb)
    _positive("min_storage_gb", min_storage_gb)

    products = load_products()

    cat = None
    if category is not None and str(category).strip():
        cat = str(category).strip().lower()
        known = sorted({p["category"] for p in products})
        if cat not in known:
            raise ToolInputError("invalid_category", f"Unknown category '{category}'. Known categories: {known}.")

    matches = []
    for p in products:
        if cat and p["category"] != cat:
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        if min_ram_gb is not None and p["ram_gb"] < min_ram_gb:
            continue
        if min_storage_gb is not None and p["storage_gb"] < min_storage_gb:
            continue
        if needs_gaming and p["gaming_suitability"] < SEARCH_MIN_SUITABILITY:
            continue
        if needs_programming and p["programming_suitability"] < SEARCH_MIN_SUITABILITY:
            continue
        matches.append(p)

    matches.sort(key=lambda p: (p["price"], p["name"]))
    filters = {
        "category": cat,
        "max_price": max_price,
        "min_ram_gb": min_ram_gb,
        "min_storage_gb": min_storage_gb,
        "needs_gaming": needs_gaming,
        "needs_programming": needs_programming,
    }
    return {
        "status": "success",
        "filters": {k: v for k, v in filters.items() if v not in (None, False)},
        "count": len(matches),
        "products": matches,
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
) -> dict:
    """Check one product against specific requirements and report which passed or failed.

    Provide at least one requirement besides product_name. Returns
    {"status", "product", "price", "checks", "passed_count", "failed_count", "all_passed"}
    where each check is {"requirement", "required", "actual", "passed"}.
    """
    product = _find_product(load_products(), product_name)

    checks: list[dict] = []

    def add(requirement: str, required: Any, actual: Any, passed: bool) -> None:
        checks.append({"requirement": requirement, "required": required, "actual": actual, "passed": bool(passed)})

    if min_ram_gb is not None:
        add("min_ram_gb", min_ram_gb, product["ram_gb"], product["ram_gb"] >= min_ram_gb)
    if min_storage_gb is not None:
        add("min_storage_gb", min_storage_gb, product["storage_gb"], product["storage_gb"] >= min_storage_gb)
    if requires_dedicated_gpu:
        add("dedicated_gpu", True, product["dedicated_gpu"], product["dedicated_gpu"])
    if min_gaming_score is not None:
        add(
            "min_gaming_score",
            min_gaming_score,
            product["gaming_suitability"],
            product["gaming_suitability"] >= min_gaming_score,
        )
    if min_programming_score is not None:
        add(
            "min_programming_score",
            min_programming_score,
            product["programming_suitability"],
            product["programming_suitability"] >= min_programming_score,
        )
    if processor_contains:
        add(
            "processor_contains",
            processor_contains,
            product["processor"],
            processor_contains.lower() in product["processor"].lower(),
        )

    if not checks:
        raise ToolInputError(
            "no_requirements",
            "Provide at least one requirement (min_ram_gb, min_storage_gb, requires_dedicated_gpu, "
            "min_gaming_score, min_programming_score or processor_contains).",
        )

    passed = sum(1 for c in checks if c["passed"])
    return {
        "status": "success",
        "product": product["name"],
        "price": product["price"],
        "checks": checks,
        "passed_count": passed,
        "failed_count": len(checks) - passed,
        "all_passed": passed == len(checks),
    }


@tool("calculate_budget", args_schema=CalculateBudgetInput)
@_guard
def calculate_budget(product_name: str, budget: float) -> dict:
    """Compare a product's catalogue price with the user's maximum budget.

    The price is looked up from the catalogue by product name (never supplied by the caller).
    Returns {"status", "product", "price", "budget", "difference", "within_budget", "note"}
    where difference = budget - price (negative means over budget).
    """
    _positive("budget", budget)
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
