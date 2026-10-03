"""Local offline and self-hosted models for the Black Box agent.

Provides:
  1. LocalChatModel: A zero-dependency, 100% offline, zero-quota model that
     intelligently parses user requests, generates appropriate tool calls
     (search_products -> check_specifications -> calculate_budget), and
     synthesizes structured recommendations.
  2. OllamaChatModel: Connects to a locally running Ollama instance
     (http://localhost:11434) using standard library HTTP requests.
"""

import json
import logging
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
    """Extract budget from user prompt, e.g. 70000, 70,000, 70k, 1 lakh."""
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


class LocalChatModel(BaseChatModel):
    """Zero-quota, zero-cost, fully local chat model.
    
    Acts as an autonomous recommendation engine that drives the LangChain
    tool loop (search -> specs -> budget -> final response) deterministically.
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

        # Stage 1: No tools called yet -> Call search_products
        if "search_products" not in executed_tool_names:
            budget = _extract_budget(user_text)
            min_ram = _extract_ram(user_text)
            text_lower = user_text.lower()
            needs_gaming = any(w in text_lower for w in ["game", "gaming", "gpu", "graphics", "rtx", "gtx"])
            needs_prog = any(w in text_lower for w in ["programming", "coding", "developer", "development", "software", "code", "python"])
            needs_student = any(w in text_lower for w in ["student", "college", "school", "study"])

            search_args: dict[str, Any] = {}
            if budget:
                search_args["max_price"] = budget
            if min_ram:
                search_args["min_ram_gb"] = min_ram
            if needs_gaming:
                search_args["needs_gaming"] = True
            if needs_prog:
                search_args["needs_programming"] = True
            if needs_student:
                search_args["category"] = "budget"

            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content="I will search our product database to find matching laptops based on your criteria.",
                tool_calls=[{
                    "name": "search_products",
                    "args": search_args,
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 2: search_products completed -> parse candidate laptops
        search_msg = next((m for m in tool_messages if getattr(m, "name", "") == "search_products"), None)
        products = []
        if search_msg:
            try:
                content = search_msg.content
                if isinstance(content, str):
                    parsed = json.loads(content)
                else:
                    parsed = content
                if isinstance(parsed, dict):
                    products = parsed.get("products", [])
            except Exception:
                products = []

        top_laptop = products[0] if products else None
        top_name = top_laptop.get("name", "Generic Laptop") if top_laptop else None

        if not top_name:
            # No products matched
            msg = AIMessage(
                content="I searched our catalog but could not find any laptops matching all your criteria. "
                "Please consider increasing your budget or relaxing specific hardware requirements."
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 3: Call check_specifications for the top candidate
        if "check_specifications" not in executed_tool_names:
            min_ram = _extract_ram(user_text) or 8
            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content=f"Checking specifications for {top_name}.",
                tool_calls=[{
                    "name": "check_specifications",
                    "args": {"product_name": top_name, "min_ram_gb": min_ram},
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 4: Call calculate_budget for the top candidate
        if "calculate_budget" not in executed_tool_names:
            budget = _extract_budget(user_text) or top_laptop.get("price", 75000)
            call_id = f"call_{uuid4().hex[:8]}"
            msg = AIMessage(
                content=f"Checking budget fit for {top_name}.",
                tool_calls=[{
                    "name": "calculate_budget",
                    "args": {"product_name": top_name, "budget": budget},
                    "id": call_id,
                    "type": "tool_call",
                }],
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Stage 5: All tools executed -> Generate final recommendation
        price = top_laptop.get("price", "N/A")
        ram = top_laptop.get("ram_gb", "N/A")
        storage = top_laptop.get("storage_gb", "N/A")
        cpu = top_laptop.get("processor", "N/A")
        gpu = top_laptop.get("gpu", "N/A")

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
            price = 85000  # Tool returned 64990 or 58990, but agent hallucinates 85000

        price_str = f"₹{price:,}" if isinstance(price, (int, float)) else f"₹{price}"
        final_text = (
            f"Based on your requirements, I recommend the **{top_name}**.\n\n"
            f"- **Price**: {price_str}\n"
            f"- **Processor**: {cpu}\n"
            f"- **Graphics**: {gpu}\n"
            f"- **Memory**: {ram}GB RAM\n"
            f"- **Storage**: {storage}GB SSD\n\n"
            f"It satisfies all required specifications and stays comfortably within your budget."
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
