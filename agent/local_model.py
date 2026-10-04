"""Local offline and self-hosted models for the Black Box Electronics Consultant Agent.

Provides:
  1. LocalChatModel: A zero-dependency, 100% offline, autonomous model that
     intelligently parses user requests across all consumer electronics categories,
     generates observable tool calls (search_products -> check_specifications -> calculate_budget),
     handles general technical questions conversationally, and synthesizes structured recommendations.
  2. OllamaChatModel: Connects to a locally running Ollama instance using HTTP requests.
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
    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|l)\b", text_lower)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100_000)

    k_match = re.search(r"(\d+)\s*k\b", text_lower)
    if k_match:
        return int(k_match.group(1)) * 1000

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
    """Extract minimum storage in GB from user prompt."""
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
    brands = [
        "samsung", "apple", "google", "oneplus", "xiaomi", "motorola", "nothing",
        "asus", "lenovo", "hp", "acer", "dell", "msi", "sony", "lg", "bose",
        "fujifilm", "canon", "nikon", "sonos", "jbl", "tp-link", "logitech",
    ]
    for brand in brands:
        if re.search(rf"\b{brand}\b", text_lower):
            return brand
    return None


def _detect_category(text_lower: str) -> str:
    """Detect product category from query text."""
    if any(w in text_lower for w in ["monitor", "display", "screen", "refresh rate", "240hz", "144hz"]):
        if not any(w in text_lower for w in ["laptop", "phone", "tv"]):
            return "monitor"
    if any(w in text_lower for w in ["tv", "television", "oled tv", "smart tv", "bravia"]):
        return "tv"
    if any(w in text_lower for w in ["headphone", "headphones", "earphone", "anc", "noise cancel"]):
        return "headphones"
    if any(w in text_lower for w in ["earbud", "earbuds", "airpods", "tws"]):
        return "earbuds"
    if any(w in text_lower for w in ["camera", "mirrorless", "dslr", "lens", "megapixels"]):
        return "camera"
    if any(w in text_lower for w in ["smartwatch", "watch", "apple watch", "galaxy watch", "fitness band"]):
        return "smartwatch"
    if any(w in text_lower for w in ["tablet", "ipad", "galaxy tab"]):
        return "tablet"
    if any(w in text_lower for w in ["speaker", "speakers", "soundbar", "sonos", "bluetooth speaker"]):
        return "speaker"
    if any(w in text_lower for w in ["router", "wifi", "wi-fi", "mesh", "networking"]):
        return "router"
    if any(w in text_lower for w in ["phone", "mobile", "smartphone", "iphone", "galaxy s", "galaxy a", "pixel", "handset"]):
        return "smartphone"
    return "laptop"


def _is_general_chat(text: str) -> bool:
    """Detect general conversational / informational query rather than explicit product recommendation."""
    text_lower = text.strip().lower()
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "how are you", "who are you", "what can you do", "help"]
    if text_lower in greetings or any(text_lower.startswith(g + " ") or text_lower.startswith(g + "!") for g in greetings):
        if not _extract_budget(text) and not any(w in text_lower for w in ["need", "buy", "find", "suggest", "recommend", "under", "within"]):
            return True

    general_questions = [
        "what is refresh rate", "what is oled", "oled vs ips", "how much ram",
        "difference between intel and amd", "what is anc", "how does noise cancellation work",
        "what is an ssd", "why is 16gb ram better", "explain 4k vs 2k"
    ]
    if any(q in text_lower for q in general_questions):
        return True

    if text_lower.startswith("what is") or text_lower.startswith("how does") or text_lower.startswith("explain "):
        if not any(w in text_lower for w in ["find", "recommend", "suggest", "buy", "under", "best price"]):
            return True

    return False


def _answer_general_chat(text: str) -> str:
    """Provide a knowledgeable, friendly, concise technical answer as an electronics consultant."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["hi", "hello", "hey", "who are you", "what can you do"]):
        return (
            "Hello! I am your AI Electronics Product Consultant. I can help you find, compare, and verify "
            "specifications and budgets for laptops, smartphones, monitors, TVs, audio equipment, cameras, "
            "smartwatches, tablets, and more from our catalogue. What kind of electronics are you looking for today?"
        )
    if "refresh rate" in text_lower or "hz" in text_lower:
        return (
            "**Display Refresh Rate (Hz)** refers to how many times per second the screen updates its image. "
            "A standard display operates at **60Hz**, while gaming and high-fluidity monitors/phones run at "
            "**120Hz, 144Hz, or 240Hz**. Higher refresh rates deliver significantly smoother motion, reduced eye fatigue, "
            "and faster reaction times in games and UI navigation."
        )
    if "oled vs ips" in text_lower or "oled" in text_lower:
        return (
            "**OLED vs. IPS Panels:**\n\n"
            "- **OLED (Organic LED)**: Each individual pixel emits its own light, delivering true absolute blacks, infinite contrast, and instant response times (<0.1ms). Perfect for cinematic media and dark-room gaming.\n"
            "- **IPS (In-Plane Switching)**: Uses an LED backlight. It provides exceptional color accuracy and wide viewing angles at a more accessible price point, making it ideal for office productivity and graphic design."
        )
    if "ram" in text_lower:
        return (
            "**RAM Guidelines:**\n\n"
            "- **8GB**: Suitable for everyday web browsing, office documents, and light multitasking.\n"
            "- **16GB**: The sweet spot for modern software development, heavy browser tabs, and gaming.\n"
            "- **32GB+**: Recommended for heavy video rendering, local machine learning models, and complex virtualization."
        )
    return (
        f"As an electronics consultant, I can certainly assist with that! For {text.strip()}, our catalogue "
        "contains options tailored to productivity, gaming, and creative workflows. Would you like me to recommend "
        "specific products within a target budget?"
    )


def _parse_query_intent(user_text: str, history_texts: Optional[List[str]] = None) -> dict[str, Any]:
    """Extract structured intent and criteria from user text and multi-turn conversation context."""
    full_context = " ".join((history_texts or []) + [user_text])
    text_lower = user_text.lower()
    context_lower = full_context.lower()

    budget = _extract_budget(user_text) or _extract_budget(full_context)
    min_ram = _extract_ram(user_text) or _extract_ram(full_context)
    min_storage = _extract_storage(user_text) or _extract_storage(full_context)
    brand = _extract_brand(user_text) or _extract_brand(full_context)

    category = _detect_category(text_lower)
    if category == "laptop" and history_texts:
        category = _detect_category(context_lower)

    needs_gaming = any(w in context_lower for w in ["game", "gaming", "gamer", "gpu", "graphics", "rtx", "gtx", "dedicated", "240hz", "144hz"])
    needs_prog = any(w in context_lower for w in ["programming", "coding", "developer", "development", "software", "code", "python", "java", "engineer"])
    needs_student = any(w in context_lower for w in ["student", "college", "school", "study", "assignments", "lecture", "university"])
    needs_portable = any(w in context_lower for w in ["portable", "lightweight", "slim", "battery", "travel", "ultrabook", "anc", "noise cancel"])

    is_general = _is_general_chat(user_text)

    return {
        "budget": budget,
        "min_ram": min_ram,
        "min_storage": min_storage,
        "brand": brand,
        "category": category,
        "device_type": category,
        "needs_gaming": needs_gaming,
        "needs_programming": needs_prog,
        "needs_student": needs_student,
        "needs_portable": needs_portable,
        "is_general_chat": is_general,
        "raw_text": user_text,
    }


class LocalChatModel(BaseChatModel):
    """Zero-quota, zero-cost, fully autonomous local chat model for all electronics categories."""

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

        # Extract all user texts and latest message
        user_texts = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        user_text = user_texts[-1] if user_texts else ""
        history_texts = user_texts[:-1] if len(user_texts) > 1 else []

        intent = _parse_query_intent(user_text, history_texts)

        # Handle general non-product chat directly if no tools have been called
        tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]
        executed_tool_names = [getattr(m, "name", "") for m in tool_messages]

        if intent["is_general_chat"] and not executed_tool_names and mode == "none":
            answer = _answer_general_chat(user_text)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

        # Injected Failure A: wrong_tool -> select inappropriate tool first
        if mode == "wrong_tool" and not executed_tool_names:
            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content="Checking specifications directly for non-existent product without prior search.",
                tool_calls=[{
                    "name": "check_specifications",
                    "args": {"product_name": "NonExistentDevice", "min_ram_gb": 32},
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 1: Call search_products
        if "search_products" not in executed_tool_names:
            search_args: dict[str, Any] = {}
            cat = intent["category"]
            if cat and cat != "laptop":
                search_args["category"] = cat

            if intent["budget"]:
                search_args["max_price"] = intent["budget"]
            if intent["brand"]:
                search_args["brand"] = intent["brand"]

            if cat == "laptop":
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
                content=f"Searching our electronics catalogue for {cat} matching your criteria.",
                tool_calls=[{
                    "name": "search_products",
                    "args": search_args,
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 2: Parse search_products output
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

        if not products:
            cat_name = intent["category"] or "products"
            msg = AIMessage(
                content=(
                    f"I couldn't find an option that satisfies all of your requirements for {cat_name} within the available catalogue. "
                    "You might want to consider adjusting your budget ceiling or relaxing specific specification filters."
                )
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        top_prod = products[0]
        top_name = top_prod.get("name", "Product")
        runner_up = products[1] if len(products) > 1 else None

        # Stage 3: Call check_specifications for top candidate
        if "check_specifications" not in executed_tool_names:
            specs_args: dict[str, Any] = {"product_name": top_name}
            if intent["category"] == "laptop":
                if intent["min_ram"]:
                    specs_args["min_ram_gb"] = intent["min_ram"]
                else:
                    specs_args["min_ram_gb"] = top_prod.get("ram_gb", 8)

                if intent["min_storage"]:
                    specs_args["min_storage_gb"] = intent["min_storage"]
                if intent["needs_gaming"]:
                    specs_args["min_gaming_score"] = 3
                    if top_prod.get("dedicated_gpu"):
                        specs_args["requires_dedicated_gpu"] = True
                if intent["needs_programming"]:
                    specs_args["min_programming_score"] = 3
            elif intent["category"] in ("monitor", "tv"):
                specs_args["screen_size_min"] = 24.0
            elif intent["category"] in ("headphones", "earbuds"):
                specs_args["requires_noise_cancellation"] = True
            else:
                specs_args["min_ram_gb"] = top_prod.get("ram_gb", 8)

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

        # Stage 4: Call calculate_budget for top candidate
        if "calculate_budget" not in executed_tool_names:
            raw_price = top_prod.get("price", 50000)
            budget = intent["budget"] or raw_price
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

        # Stage 5: Synthesize final response
        raw_price = top_prod.get("price", 0)
        price = int(raw_price) if raw_price is not None else 0

        # Injected failures handling
        if mode == "budget_violation":
            top_name = "Dell XPS 13"
            price = 84990
        elif mode == "wrong_interpretation":
            price = 85000

        user_budget = intent["budget"] or price or 0
        diff = user_budget - price
        savings_text = f"saving ₹{diff:,} under budget" if diff >= 0 else f"over budget by ₹{-diff:,}"
        budget_status = "Within budget" if diff >= 0 else f"Above budget by ₹{-diff:,}"

        # Category-aware spec summary and rationale
        category = (top_prod.get("category") or intent["category"] or "Electronics").lower()
        specs = top_prod.get("specifications", {})

        spec_lines = []
        if "processor" in top_prod or "processor" in specs:
            spec_lines.append(f"- Processor: {top_prod.get('processor') or specs.get('processor')}")
        if "ram_gb" in top_prod or "ram_gb" in specs or "ram" in specs:
            spec_lines.append(f"- RAM: {top_prod.get('ram_gb') or specs.get('ram_gb') or specs.get('ram')}GB")
        if "storage_gb" in top_prod or "storage_gb" in specs or "storage" in specs:
            spec_lines.append(f"- Storage: {top_prod.get('storage_gb') or specs.get('storage_gb') or specs.get('storage')}GB")
        if "screen_size" in specs:
            spec_lines.append(f"- Display: {specs.get('screen_size')}\" {specs.get('resolution', '')} ({specs.get('panel_type', '')})")
        elif "display" in specs:
            spec_lines.append(f"- Display: {specs.get('display')}")
        if "gpu" in top_prod or "gpu" in specs:
            spec_lines.append(f"- Graphics: {top_prod.get('gpu') or specs.get('gpu')}")
        if "noise_cancellation" in specs:
            spec_lines.append(f"- Active Noise Cancellation: {'Yes (Industry Leading)' if specs.get('noise_cancellation') else 'No'}")
        if "camera" in specs:
            spec_lines.append(f"- Camera: {specs.get('camera')}")
        if "battery_mah" in specs:
            spec_lines.append(f"- Battery: {specs.get('battery_mah')} mAh")

        spec_block = "\n".join(spec_lines) if spec_lines else f"- Category: {category.title()}\n- Model: {top_name}"

        # Rationale
        if intent["needs_programming"] and category in ("laptop", "ultrabook", "gaming"):
            why_fits = (
                f"With high suitability for developer workflows, the system provides reliable multi-core processing "
                f"and disk throughput for compiling code, running local containers, and developer IDEs."
            )
        elif intent["needs_gaming"]:
            why_fits = "Delivers high framerates and fluid motion for modern gaming and visual workloads."
        elif intent["needs_student"]:
            why_fits = "Offers great day-to-day responsiveness, ample disk space for coursework, and reliable battery life."
        else:
            why_fits = f"Provides the best performance-to-price balance available in our {category} catalogue."

        alt_section = ""
        if runner_up and mode not in ("budget_violation", "wrong_interpretation"):
            r_name = runner_up.get("name")
            r_price = int(runner_up.get("price", 0))
            alt_section = (
                f"\n\n### 🔄 Alternative Option to Consider\n"
                f"- **{r_name}** at **₹{r_price:,}**\n"
                f"  - *Comparison:* A solid alternative if you wish to balance features or adjust your spend."
            )

        final_text = (
            f"I found {len(products)} options that match your requirements.\n\n"
            f"### {top_name}\n"
            f"**₹{price:,}**\n"
            f"{spec_block}\n\n"
            f"**Why it fits**\n"
            f"{why_fits}\n\n"
            f"**Budget**\n"
            f"{budget_status}"
            f"{alt_section}\n\n"
            f"Would you like me to compare these options?"
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
