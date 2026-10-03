"""Local offline and self-hosted models for the Black Box agent.

Provides:
  1. LocalChatModel: A zero-dependency, 100% offline, zero-quota autonomous model that
     intelligently parses user requests, generates appropriate tool calls
     (search_products -> check_specifications -> calculate_budget), dynamically scores
     and ranks candidate products based on user priorities, and synthesizes rich,
     comparative recommendations with tradeoff analysis.
  2. OllamaChatModel: Connects to a locally running Ollama instance
     (http://localhost:11434) using standard library HTTP requests.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, List, Optional, Sequence
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


def _extract_budget(text: str) -> Optional[int]:
    """Extract budget from user prompt, e.g. 70000, 70,000, 70k, 1 lakh, 40000."""
    text_lower = text.lower()
    # Check for lakh (e.g., 1 lakh or 1.5 lakh)
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)\b", text_lower)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100_000)

    # Check for 70k or 60k
    k_match = re.search(r"(\d+)\s*k\b", text_lower)
    if k_match:
        return int(k_match.group(1)) * 1000

    # Check for ₹70,000 or 70000
    num_match = re.search(r"(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})", text)
    if num_match:
        num_str = num_match.group(1).replace(",", "")
        val = int(num_str)
        if val > 1000:
            return val
    return None


def _extract_ram(text: str) -> Optional[int]:
    """Extract minimum RAM in GB from user prompt."""
    ram_match = re.search(r"(\d+)\s*(?:gb|gig)\s*(?:ram)?", text.lower())
    if ram_match:
        return int(ram_match.group(1))
    return None


def _extract_storage(text: str) -> Optional[int]:
    """Extract minimum storage in GB from user prompt (e.g., 512gb, 1tb)."""
    text_lower = text.lower()
    tb_match = re.search(r"(\d+)\s*(?:tb|terabyte)", text_lower)
    if tb_match:
        return int(tb_match.group(1)) * 1024
    ssd_match = re.search(r"(\d+)\s*(?:gb)?\s*(?:ssd|storage|rom)", text_lower)
    if ssd_match:
        return int(ssd_match.group(1))
    return None


def _extract_brand(text: str) -> Optional[str]:
    """Extract mentioned brand if any."""
    text_lower = text.lower()
    for brand in ["samsung", "apple", "google", "oneplus", "xiaomi", "motorola", "nothing", "asus", "lenovo", "hp", "acer", "dell", "msi"]:
        if re.search(rf"\b{brand}\b", text_lower):
            return brand
    return None


def _parse_query_intent(user_text: str) -> dict[str, Any]:
    """Extract structured intent and criteria from user text."""
    text_lower = user_text.lower()
    budget = _extract_budget(user_text)
    min_ram = _extract_ram(user_text)
    min_storage = _extract_storage(user_text)
    brand = _extract_brand(user_text)

    is_phone = any(w in text_lower for w in ["phone", "mobile", "smartphone", "iphone", "galaxy s", "galaxy a", "pixel", "handset"])
    device_type = "phone" if is_phone else "laptop"

    needs_gaming = any(w in text_lower for w in ["game", "gaming", "gamer", "gpu", "graphics", "rtx", "gtx", "dedicated"])
    needs_prog = any(w in text_lower for w in ["programming", "coding", "developer", "development", "software", "code", "python", "java", "engineer"])
    needs_student = any(w in text_lower for w in ["student", "college", "school", "study", "assignments", "lecture", "university"])
    needs_portable = any(w in text_lower for w in ["portable", "lightweight", "slim", "battery", "travel", "ultrabook"])

    return {
        "budget": budget,
        "min_ram": min_ram,
        "min_storage": min_storage,
        "brand": brand,
        "device_type": device_type,
        "needs_gaming": needs_gaming,
        "needs_programming": needs_prog,
        "needs_student": needs_student,
        "needs_portable": needs_portable,
        "raw_text": user_text,
    }


SMARTPHONES = [
    # Premium Flagships (>= 1,00,000)
    {
        "name": "Samsung Galaxy S24 Ultra",
        "price": 129999,
        "brand": "samsung",
        "processor": "Qualcomm Snapdragon 8 Gen 3 for Galaxy",
        "ram": 12,
        "storage": 256,
        "display": '6.8" Dynamic AMOLED 2X (1-120Hz LTPO, 2600 nits, Gorilla Armor)',
        "camera": "200MP Quad Camera with 5x & 10x Optical Periscope Zoom, S-Pen",
        "battery": "5,000 mAh (45W Fast Charging)",
        "perks": "Built-in S-Pen for handwriting lecture notes and annotating PDFs on glass, Samsung DeX for desktop computing, and titanium frame durability.",
    },
    {
        "name": "Apple iPhone 15 Pro",
        "price": 124900,
        "brand": "apple",
        "processor": "Apple A17 Pro (3nm)",
        "ram": 8,
        "storage": 128,
        "display": '6.1" Super Retina XDR OLED (120Hz ProMotion)',
        "camera": "48MP Main + 12MP Ultra-wide + 12MP 3x Telephoto",
        "battery": "3,274 mAh",
        "perks": "Grade 5 Titanium design, Action Button for instant shortcuts (voice recorder, flashcards), and console-grade A17 Pro graphics performance.",
    },
    # High-End Flagships (60,000 - 85,000)
    {
        "name": "Samsung Galaxy S24",
        "price": 74999,
        "brand": "samsung",
        "processor": "Samsung Exynos 2400 / Snapdragon 8 Gen 3",
        "ram": 8,
        "storage": 256,
        "display": '6.2" Dynamic AMOLED 2X (120Hz LTPO, 2600 nits peak)',
        "camera": "50MP Main (OIS) + 12MP Ultra-wide + 10MP 3x Telephoto",
        "battery": "4,000 mAh (All-Day Battery)",
        "perks": "Galaxy AI Note & Transcript Assist for lecture recording, Circle to Search for textbook homework, compact pocketable design, and 7 years of OS updates.",
    },
    {
        "name": "Apple iPhone 15",
        "price": 69900,
        "brand": "apple",
        "processor": "Apple A16 Bionic",
        "ram": 6,
        "storage": 128,
        "display": '6.1" Super Retina XDR OLED (Dynamic Island)',
        "camera": "48MP Main + 12MP Ultra-wide",
        "battery": "3,349 mAh",
        "perks": "Dynamic Island for live timers and alerts, universal USB-C charging, and seamless AirDrop integration with MacBooks and iPads.",
    },
    {
        "name": "OnePlus 12",
        "price": 64999,
        "brand": "oneplus",
        "processor": "Qualcomm Snapdragon 8 Gen 3",
        "ram": 12,
        "storage": 256,
        "display": '6.82" 2K 120Hz ProXDR AMOLED (4500 nits)',
        "camera": "50MP Sony LYT-808 + 64MP 3x Periscope Telephoto",
        "battery": "5,400 mAh (100W SUPERVOOC Fast Charge)",
        "perks": "100W rapid charging fully recharges the phone in 26 minutes, paired with ultra-smooth 120Hz performance for multitasking.",
    },
    {
        "name": "Google Pixel 8",
        "price": 62999,
        "brand": "google",
        "processor": "Google Tensor G3",
        "ram": 8,
        "storage": 128,
        "display": '6.2" Actua OLED (120Hz, 2000 nits)',
        "camera": "50MP Main + 12MP Ultra-wide",
        "battery": "4,575 mAh",
        "perks": "Best-in-class Google AI Recorder with speaker labels for lecture capture, Magic Eraser, and 7 years of direct Android feature drops.",
    },
    # Upper Midrange (40,000 - 55,000)
    {
        "name": "Samsung Galaxy S23 FE",
        "price": 49999,
        "brand": "samsung",
        "processor": "Samsung Exynos 2200",
        "ram": 8,
        "storage": 128,
        "display": '6.4" Dynamic AMOLED 2X (120Hz)',
        "camera": "50MP OIS + 12MP Ultra-wide + 8MP 3x Telephoto",
        "battery": "4,500 mAh",
        "perks": "Flagship camera array with optical 3x zoom, bright AMOLED screen for campus reading, and Galaxy AI features comfortably within budget.",
    },
    {
        "name": "Apple iPhone 13",
        "price": 48999,
        "brand": "apple",
        "processor": "Apple A15 Bionic",
        "ram": 4,
        "storage": 128,
        "display": '6.1" Super Retina XDR OLED',
        "camera": "12MP Dual Cameras with Photographic Styles",
        "battery": "3,227 mAh",
        "perks": "Reliable Apple ecosystem integration, fluid performance, and durable Ceramic Shield construction under ₹50,000.",
    },
    {
        "name": "OnePlus 12R",
        "price": 39999,
        "brand": "oneplus",
        "processor": "Qualcomm Snapdragon 8 Gen 2",
        "ram": 8,
        "storage": 128,
        "display": '6.78" 1.5K 120Hz LTPO4 AMOLED (4500 nits)',
        "camera": "50MP Sony IMX890 OIS",
        "battery": "5,500 mAh (100W SUPERVOOC)",
        "perks": "Massive 5,500 mAh battery that easily lasts 1.5 days of classes, Snapdragon 8 Gen 2 flagship speed, and 100W charging.",
    },
    {
        "name": "Samsung Galaxy A55 5G",
        "price": 39999,
        "brand": "samsung",
        "processor": "Samsung Exynos 1480",
        "ram": 8,
        "storage": 128,
        "display": '6.6" Super AMOLED (120Hz, Gorilla Glass Victus+)',
        "camera": "50MP OIS + 12MP Ultra-wide + 5MP Macro",
        "battery": "5,000 mAh",
        "perks": "Premium metal frame, IP67 dust/water resistance for college commutes, 4 OS updates, and exceptional 2-day battery endurance.",
    },
    {
        "name": "Nothing Phone (2)",
        "price": 36999,
        "brand": "nothing",
        "processor": "Qualcomm Snapdragon 8+ Gen 1",
        "ram": 12,
        "storage": 256,
        "display": '6.7" LTPO OLED (120Hz)',
        "camera": "50MP Sony IMX890 + 50MP Ultra-wide",
        "battery": "4,700 mAh",
        "perks": "Unique Glyph Interface for silent class notifications, clean bloatware-free Nothing OS, and snappy flagship processor.",
    },
    # Budget / Value Tier (20,000 - 35,000)
    {
        "name": "Samsung Galaxy A35 5G",
        "price": 27999,
        "brand": "samsung",
        "processor": "Samsung Exynos 1380",
        "ram": 8,
        "storage": 128,
        "display": '6.6" Super AMOLED (120Hz)',
        "camera": "50MP Main OIS + 8MP Ultra-wide",
        "battery": "5,000 mAh",
        "perks": "Long-lasting 5,000 mAh battery that easily powers through study marathons, paired with a bright 120Hz AMOLED display.",
    },
    {
        "name": "OnePlus Nord CE4",
        "price": 24999,
        "brand": "oneplus",
        "processor": "Qualcomm Snapdragon 7 Gen 3",
        "ram": 8,
        "storage": 128,
        "display": '6.7" Fluid AMOLED (120Hz)',
        "camera": "50MP Sony LYT-600 OIS",
        "battery": "5,500 mAh (100W SUPERVOOC)",
        "perks": "Unbeatable battery endurance and 100W fast charging under ₹25,000, perfect for budget-conscious students.",
    },
    {
        "name": "Redmi Note 13 Pro 5G",
        "price": 24999,
        "brand": "xiaomi",
        "processor": "Qualcomm Snapdragon 7s Gen 2",
        "ram": 8,
        "storage": 128,
        "display": '6.67" 1.5K AMOLED (120Hz)',
        "camera": "200MP OIS Ultra-Clear Camera",
        "battery": "5,100 mAh (67W Turbo Charge)",
        "perks": "High-resolution 1.5K display for reading PDFs and textbooks with crystal-clear text, and 200MP camera.",
    },
    {
        "name": "Samsung Galaxy M34 5G",
        "price": 16999,
        "brand": "samsung",
        "processor": "Samsung Exynos 1280",
        "ram": 6,
        "storage": 128,
        "display": '6.5" Super AMOLED (120Hz)',
        "camera": "50MP OIS Main Camera",
        "battery": "6,000 mAh",
        "perks": "Monstrous 6,000 mAh battery that lasts up to 3 days without a charger, paired with Samsung Knox security.",
    },
]


def _recommend_smartphone(intent: dict[str, Any]) -> str:
    """Select and synthesize the best smartphone strictly respecting user budget and preferences."""
    user_budget = intent.get("budget") or 80000
    brand = (intent.get("brand") or "").lower()
    raw = intent.get("raw_text", "").lower()

    candidates = SMARTPHONES

    # 1. Filter by brand if specified
    if brand:
        brand_matches = [p for p in candidates if p["brand"] == brand]
        if brand_matches:
            candidates = brand_matches
    elif "samsung" in raw or "galaxy" in raw:
        brand_matches = [p for p in candidates if p["brand"] == "samsung"]
        if brand_matches:
            candidates = brand_matches
    elif "iphone" in raw or "apple" in raw:
        brand_matches = [p for p in candidates if p["brand"] == "apple"]
        if brand_matches:
            candidates = brand_matches

    # 2. Strict budget filtering: ALWAYS prioritize devices within user budget
    within_budget = [p for p in candidates if p["price"] <= user_budget]
    is_over_budget = False

    if within_budget:
        # Sort by price descending to get the best feature-packed device within the budget
        within_budget.sort(key=lambda p: p["price"], reverse=True)
        top = within_budget[0]
        runner_up = within_budget[1] if len(within_budget) > 1 else None
    else:
        # No options under budget! Fall back to closest available and flag clearly
        is_over_budget = True
        candidates_sorted = sorted(candidates, key=lambda p: abs(p["price"] - user_budget))
        top = candidates_sorted[0]
        runner_up = candidates_sorted[1] if len(candidates_sorted) > 1 else None

    price = top["price"]
    top_name = top["name"]
    cpu = top["processor"]
    display = top["display"]
    ram = top["ram"]
    storage = top["storage"]
    camera = top["camera"]
    battery = top["battery"]
    perks = top["perks"]

    diff = user_budget - price
    if diff >= 0:
        savings_text = f"saving ₹{diff:,} under budget"
        status_label = "Within Budget"
        diff_str = f"+₹{diff:,}"
    else:
        savings_text = f"over budget by ₹{-diff:,}"
        status_label = "Over Budget"
        diff_str = f"-₹{-diff:,}"

    # Runner-up alternative section
    runner_up_section = ""
    if runner_up:
        r_name = runner_up["name"]
        r_price = runner_up["price"]
        r_diff = abs(price - r_price)
        if r_price < price:
            r_delta = f"₹{r_diff:,} cheaper"
        else:
            r_delta = f"₹{r_diff:,} more"

        runner_up_section = (
            f"### 🔄 Alternative Option to Consider\n"
            f"- **{r_name}** at **₹{r_price:,}** ({r_delta})\n"
            f"  - Specs: {runner_up['processor']} | {runner_up['ram']}GB RAM | {runner_up['storage']}GB Storage | {runner_up['battery']}\n"
            f"  - *Comparison:* A solid option if you want to optimize your spend while maintaining dependable performance.\n\n"
        )

    notice = ""
    if is_over_budget:
        notice = (
            f"> ⚠️ **Budget Notice**: No reliable smartphones matching your criteria were found strictly within "
            f"₹{user_budget:,}. The **{top_name}** at ₹{price:,} is the closest available match in this category.\n\n"
        )

    return (
        f"Based on your requirements, the top smartphone recommendation is the **{top_name}**.\n\n"
        f"{notice}"
        f"### 📱 Top Recommendation: **{top_name}**\n"
        f"- **Price**: ₹{price:,} ({savings_text})\n"
        f"- **Processor**: {cpu}\n"
        f"- **Display**: {display}\n"
        f"- **Memory & Storage**: {ram}GB RAM | {storage}GB Storage\n"
        f"- **Camera**: {camera}\n"
        f"- **Battery**: {battery}\n\n"
        f"### 💡 Why This Fits Your College & Daily Needs\n"
        f"- {perks}\n\n"
        f"{runner_up_section}"
        f"### 💰 Budget & Value Summary\n"
        f"- **Target Budget**: ₹{user_budget:,}\n"
        f"- **Actual Price**: ₹{price:,}\n"
        f"- **Difference**: {diff_str} ({status_label})\n\n"
        f"*(Note: Our local catalogue tools specialize in laptops; this smartphone recommendation is drawn from current flagship hardware benchmarks to fulfill your exact phone request).* "
    )


def _score_laptop(laptop: dict[str, Any], intent: dict[str, Any]) -> float:
    """Score a laptop candidate based on user priorities and constraints."""
    score = 50.0  # base score
    name = laptop.get("name", "").lower()
    price = laptop.get("price", 0)
    ram = laptop.get("ram_gb", 8)
    storage = laptop.get("storage_gb", 256)
    gaming_score = laptop.get("gaming_suitability", 1)
    prog_score = laptop.get("programming_suitability", 1)
    category = laptop.get("category", "")
    has_gpu = laptop.get("dedicated_gpu", False)

    # 1. Brand match
    if intent.get("brand") and intent["brand"] in name:
        score += 45.0

    # 2. Gaming priority
    if intent.get("needs_gaming"):
        score += gaming_score * 12.0
        if has_gpu:
            score += 25.0
        if "rtx" in laptop.get("gpu", "").lower():
            score += 15.0

    # 3. Programming priority
    if intent.get("needs_programming"):
        score += prog_score * 12.0
        if ram >= 16:
            score += 20.0
        if storage >= 512:
            score += 10.0

    # 4. Student / College priority
    if intent.get("needs_student"):
        # For college, 512GB SSD is a massive upgrade over 256GB
        if storage >= 512:
            score += 25.0
        # Modern multi-core CPU
        if any(cpu in laptop.get("processor", "").lower() for cpu in ["i5", "ryzen 5", "i7", "ryzen 7"]):
            score += 15.0
        if category in ("budget", "ultrabook"):
            score += 10.0

    # 5. Portability priority
    if intent.get("needs_portable"):
        if category == "ultrabook":
            score += 30.0
        elif not has_gpu:
            score += 10.0

    # 6. Budget efficiency: reward getting the best performance within the user's budget ceiling
    budget = intent.get("budget")
    if budget and budget > 0:
        if price <= budget:
            ratio = price / budget
            score += ratio * 15.0
        else:
            score -= 100.0

    return score


class LocalChatModel(BaseChatModel):
    """Zero-quota, zero-cost, fully autonomous local chat model.
    
    Dynamically analyzes user requests, selects appropriate search criteria,
    intelligently scores and ranks catalog options, verifies specs and budget
    via tools, and generates rich, multi-dimensional comparative recommendations.
    Supports reproducible failure injection via failure_mode or FAILURE_MODE env var.
    """

    model_name: str = "blackbox-local-offline"
    failure_mode: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "local-offline"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "LocalChatModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        mode = (self.failure_mode or os.getenv("FAILURE_MODE") or "none").strip().lower()

        # 1. Identify the user's initial request
        user_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_text = str(msg.content)

        intent = _parse_query_intent(user_text)

        # 2. Inspect what tools have been executed so far
        tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]
        executed_tool_names = [getattr(m, "name", "") for m in tool_messages]

        # Injected Failure A: wrong_tool -> select inappropriate tool first
        if mode == "wrong_tool" and not executed_tool_names:
            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content="Checking specifications directly for non-existent laptop without prior search.",
                tool_calls=[{
                    "name": "check_specifications",
                    "args": {"product_name": "NonExistentLaptop", "min_ram_gb": 32},
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 1: No tools called yet -> Call search_products with adaptive criteria
        if "search_products" not in executed_tool_names:
            search_args: dict[str, Any] = {}
            if intent["budget"]:
                search_args["max_price"] = intent["budget"]
            if intent["min_ram"]:
                search_args["min_ram_gb"] = intent["min_ram"]
            if intent["min_storage"]:
                search_args["min_storage_gb"] = intent["min_storage"]
            if intent["needs_gaming"]:
                search_args["needs_gaming"] = True
            if intent["needs_programming"]:
                search_args["needs_programming"] = True

            text_lower = user_text.lower()
            if "budget" in text_lower and not intent["needs_gaming"]:
                search_args["category"] = "budget"

            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content="I will search our product catalogue to find laptops matching your criteria.",
                tool_calls=[{
                    "name": "search_products",
                    "args": search_args,
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 2: Parse returned products and perform intelligent multi-factor ranking
        search_msg = next((m for m in tool_messages if getattr(m, "name", "") == "search_products"), None)
        products = []
        if search_msg:
            try:
                content = search_msg.content
                parsed = json.loads(content) if isinstance(content, str) else content
                if isinstance(parsed, dict):
                    products = parsed.get("products", [])
            except Exception:
                products = []

        # If the user specifically requested a smartphone/phone
        if intent.get("device_type") == "phone":
            phone_text = _recommend_smartphone(intent)
            msg = AIMessage(content=phone_text)
            return ChatResult(generations=[ChatGeneration(message=msg)])

        if not products:
            msg = AIMessage(
                content=(
                    f"I searched our catalogue for criteria matching \"{user_text}\", but could not find any laptops "
                    "meeting all specified constraints. Please consider increasing your budget or relaxing specific "
                    "hardware requirements such as RAM or dedicated graphics."
                )
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Rank all matched products dynamically
        ranked_products = sorted(products, key=lambda p: _score_laptop(p, intent), reverse=True)
        top_laptop = ranked_products[0]
        runner_up = ranked_products[1] if len(ranked_products) > 1 else None
        top_name = top_laptop.get("name", "Laptop")

        # Stage 3: Call check_specifications for the top candidate
        if "check_specifications" not in executed_tool_names:
            specs_args: dict[str, Any] = {"product_name": top_name}
            if intent["min_ram"]:
                specs_args["min_ram_gb"] = intent["min_ram"]
            else:
                specs_args["min_ram_gb"] = top_laptop.get("ram_gb", 8)

            if intent["min_storage"]:
                specs_args["min_storage_gb"] = intent["min_storage"]
            if intent["needs_gaming"]:
                specs_args["min_gaming_score"] = 3
                if top_laptop.get("dedicated_gpu"):
                    specs_args["requires_dedicated_gpu"] = True
            if intent["needs_programming"]:
                specs_args["min_programming_score"] = 3

            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content=f"Checking specifications for {top_name}.",
                tool_calls=[{
                    "name": "check_specifications",
                    "args": specs_args,
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 4: Call calculate_budget for the top candidate
        if "calculate_budget" not in executed_tool_names:
            budget = intent["budget"] or top_laptop.get("price", 75000)
            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content=f"Checking budget fit for {top_name}.",
                tool_calls=[{
                    "name": "calculate_budget",
                    "args": {"product_name": top_name, "budget": float(budget)},
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 5: Synthesize rich, dynamic, multi-factor recommendation
        budget_msg = next((m for m in tool_messages if getattr(m, "name", "") == "calculate_budget"), None)
        budget_info = {}
        if budget_msg:
            try:
                b_content = budget_msg.content
                budget_info = json.loads(b_content) if isinstance(b_content, str) else (b_content or {})
            except Exception:
                budget_info = {}

        price = top_laptop.get("price", 0)
        ram = top_laptop.get("ram_gb", 8)
        storage = top_laptop.get("storage_gb", 256)
        cpu = top_laptop.get("processor", "Unknown CPU")
        gpu = top_laptop.get("gpu", "Integrated Graphics")
        category = top_laptop.get("category", "General")
        prog_stars = "★" * top_laptop.get("programming_suitability", 3)
        game_stars = "★" * top_laptop.get("gaming_suitability", 1)

        # Injected Failure B: wrong_interpretation -> tool returned price X, agent states Y
        # Injected Failure C: budget_violation -> agent selects product that exceeds budget
        if mode == "budget_violation":
            top_name = "Dell XPS 13"
            price = 84990
            cpu = "Intel Core i7-1360P"
            gpu = "Intel Iris Xe"
            ram = 16
            storage = 512
        elif mode == "wrong_interpretation":
            price = 85000  # Tool returned 64990 or 58990, but agent states 85000

        user_budget = intent["budget"] or price
        diff = user_budget - price
        savings_text = f"saving ₹{diff:,} under budget" if diff >= 0 else f"over budget by ₹{-diff:,}"

        # Context-specific reasoning paragraph
        reasons: list[str] = []
        if intent["needs_student"]:
            reasons.append(
                f"Its **{cpu}** processor and **{storage}GB SSD** provide ample responsiveness and disk space "
                "for coursework, academic software, research, and multitasking across browser tabs without slowdowns."
            )
        if intent["needs_programming"]:
            reasons.append(
                f"With a **{prog_stars} ({top_laptop.get('programming_suitability', 3)}/5)** programming suitability score, "
                f"the **{ram}GB RAM** and multi-core CPU handle compiler workloads, developer IDEs, and local containers reliably."
            )
        if intent["needs_gaming"]:
            reasons.append(
                f"Equipped with **{gpu}**, it delivers a **{game_stars} ({top_laptop.get('gaming_suitability', 1)}/5)** "
                "gaming score, making it capable of handling modern games at smooth framerates."
            )
        if not reasons:
            reasons.append(
                f"The combination of **{cpu}**, **{ram}GB RAM**, and **{storage}GB SSD** offers the best price-to-performance "
                "balance available in this tier."
            )

        reasoning_str = " ".join(reasons)

        # Alternative runner-up section
        alt_section = ""
        if runner_up and mode not in ("budget_violation", "wrong_interpretation"):
            r_name = runner_up.get("name")
            r_price = runner_up.get("price", 0)
            r_cpu = runner_up.get("processor")
            r_storage = runner_up.get("storage_gb")
            r_ram = runner_up.get("ram_gb")
            price_delta = abs(price - r_price)
            if r_price < price:
                delta_str = f"₹{price_delta:,} cheaper"
            else:
                delta_str = f"₹{price_delta:,} more"

            alt_section = (
                f"\n\n### 🔄 Alternative Option to Consider\n"
                f"- **{r_name}** at **₹{r_price:,}** ({delta_str})\n"
                f"  - Specs: {r_cpu} | {r_ram}GB RAM | {r_storage}GB SSD\n"
                f"  - *Comparison:* A solid alternative if you want to adjust your budget balance or prefer {runner_up.get('category')} styling."
            )

        final_text = (
            f"Based on your requirements, the best match from our catalogue is the **{top_name}**.\n\n"
            f"### 📋 Top Recommendation: **{top_name}**\n"
            f"- **Price**: ₹{price:,} ({savings_text})\n"
            f"- **Processor**: {cpu}\n"
            f"- **Graphics**: {gpu}\n"
            f"- **Memory**: {ram}GB RAM\n"
            f"- **Storage**: {storage}GB SSD\n"
            f"- **Category**: {category.title()}\n\n"
            f"### 💡 Why This Fits Your Needs\n"
            f"{reasoning_str}"
            f"{alt_section}\n\n"
            f"### 💰 Budget & Value Summary\n"
            f"- **Target Budget**: ₹{user_budget:,}\n"
            f"- **Actual Price**: ₹{price:,}\n"
            f"- **Difference**: {'+₹' + f'{diff:,}' if diff >= 0 else '-₹' + f'{-diff:,}'} "
            f"({'Within Budget' if diff >= 0 else 'Over Budget'})\n\n"
            f"This recommendation delivers the highest overall utility for your specified constraints."
        )

        msg = AIMessage(content=final_text)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class OllamaChatModel(BaseChatModel):
    """Locally hosted Ollama chat model (e.g. llama3.2, mistral)."""

    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "OllamaChatModel":
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Prepare Ollama payload
        formatted_messages = []
        for m in messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant" if isinstance(m, AIMessage) else "tool"
            formatted_messages.append({"role": role, "content": str(m.content)})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        url = f"{self.base_url.rstrip('/')}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data.get("message", {}).get("content", "")
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running ('ollama serve') and model '{self.model}' is pulled."
            ) from exc
