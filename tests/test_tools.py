"""Unit tests for the three tools. They need no LLM and no network."""

import json

import pytest

from agent.tools import (
    TOOLS,
    calculate_budget,
    check_specifications,
    load_products,
    search_products,
)


def test_tool_names_are_exact():
    assert [t.name for t in TOOLS] == ["search_products", "check_specifications", "calculate_budget"]


def test_dataset_shape():
    products = load_products()
    assert len(products) >= 10
    required = {
        "name", "price", "ram_gb", "storage_gb", "processor", "gpu",
        "dedicated_gpu", "category", "programming_suitability", "gaming_suitability",
    }
    for product in products:
        assert required <= set(product)
    assert len({p["name"] for p in products}) == len(products)


# ------------------------------------------------------------------ search_products


def test_search_programming_and_gaming_under_70k():
    result = search_products.invoke({"max_price": 70000, "needs_gaming": True, "needs_programming": True})
    assert result["status"] == "success"
    assert result["count"] == 4
    assert all(p["price"] <= 70000 for p in result["products"])
    prices = [p["price"] for p in result["products"]]
    assert prices == sorted(prices)  # deterministic order: cheapest first


def test_search_min_ram_filter():
    result = search_products.invoke({"max_price": 66000, "min_ram_gb": 16})
    assert [p["name"] for p in result["products"]] == ["Lenovo IdeaPad Slim 5"]


def test_search_by_category():
    result = search_products.invoke({"category": "Ultrabook"})
    assert {p["category"] for p in result["products"]} == {"ultrabook"}


def test_search_no_matches_is_success_with_zero_count():
    result = search_products.invoke({"max_price": 10000})
    assert result["status"] == "success"
    assert result["count"] == 0
    assert result["products"] == []


@pytest.mark.parametrize("args", [{"max_price": -5}, {"min_ram_gb": 0}, {"category": "toaster"}])
def test_search_invalid_input_returns_structured_error(args):
    result = search_products.invoke(args)
    assert result["status"] == "error"
    assert result["error_type"] in {"invalid_input", "invalid_category"}
    assert result["message"]


def test_search_missing_data_file_is_structured_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PRODUCTS_PATH", str(tmp_path / "missing.json"))
    result = search_products.invoke({})
    assert result["status"] == "error"
    assert result["error_type"] == "internal_error"


def test_products_path_override(monkeypatch, tmp_path):
    custom = [dict(load_products()[0], name="Only Laptop", price=1000)]
    path = tmp_path / "p.json"
    path.write_text(json.dumps(custom), encoding="utf-8")
    monkeypatch.setenv("PRODUCTS_PATH", str(path))
    assert search_products.invoke({})["count"] == 1


# ------------------------------------------------------------------ check_specifications


def test_check_specs_all_pass():
    result = check_specifications.invoke(
        {"product_name": "ASUS TUF Gaming F15", "min_ram_gb": 16, "requires_dedicated_gpu": True}
    )
    assert result["status"] == "success"
    assert result["all_passed"] is True
    assert result["passed_count"] == 2 and result["failed_count"] == 0


def test_check_specs_reports_failures():
    result = check_specifications.invoke(
        {"product_name": "HP Victus 15", "min_ram_gb": 16, "min_gaming_score": 3, "processor_contains": "ryzen"}
    )
    assert result["all_passed"] is False
    by_name = {c["requirement"]: c for c in result["checks"]}
    assert by_name["min_ram_gb"]["passed"] is False and by_name["min_ram_gb"]["actual"] == 8
    assert by_name["min_gaming_score"]["passed"] is True
    assert by_name["processor_contains"]["passed"] is True


def test_check_specs_unknown_product():
    result = check_specifications.invoke({"product_name": "Nokia 3310", "min_ram_gb": 8})
    assert result["status"] == "error"
    assert result["error_type"] == "product_not_found"


def test_check_specs_requires_at_least_one_requirement():
    result = check_specifications.invoke({"product_name": "Dell G15"})
    assert result["status"] == "error"
    assert result["error_type"] == "no_requirements"


# ------------------------------------------------------------------ calculate_budget


def test_budget_within():
    result = calculate_budget.invoke({"product_name": "ASUS TUF Gaming F15", "budget": 70000})
    assert result["status"] == "success"
    assert result["price"] == 68990
    assert result["difference"] == 1010
    assert result["within_budget"] is True
    assert {"product", "price", "budget", "difference", "within_budget"} <= set(result)


def test_budget_over():
    result = calculate_budget.invoke({"product_name": "Dell G15", "budget": 70000})
    assert result["difference"] == -4990
    assert result["within_budget"] is False


def test_budget_exactly_equal_is_within():
    result = calculate_budget.invoke({"product_name": "Dell G15", "budget": 74990})
    assert result["difference"] == 0 and result["within_budget"] is True


def test_budget_invalid_budget():
    result = calculate_budget.invoke({"product_name": "Dell G15", "budget": -1})
    assert result["status"] == "error"
    assert result["error_type"] == "invalid_input"


def test_budget_unknown_product():
    result = calculate_budget.invoke({"product_name": "Imaginary Book", "budget": 70000})
    assert result["status"] == "error"
    assert result["error_type"] == "product_not_found"
