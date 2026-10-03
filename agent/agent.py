"""Creates and runs the LangChain agent, and turns its execution into observable events.

Layout of this file:
  1. build_llm()            - the ONLY place that knows about LLM providers.
  2. EventCallbackHandler   - adapts LangChain callbacks into events.Event objects.
  3. LaptopAgent            - wires llm + tools + prompt, runs a request, returns events.
"""

import ast
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence
from uuid import UUID

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .events import (
    Event,
    EventLog,
    EventSink,
    EventStatus,
    EventType,
    summarize_tool_output,
)
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- 1. LLM provider


class ConfigError(RuntimeError):
    """Missing or invalid LLM configuration (reported cleanly by main.py)."""


# Friendly LLM_PROVIDER values -> LangChain `model_provider` names.
_PROVIDER_ALIASES = {
    "google": "google_genai",
    "gemini": "google_genai",
    "google_genai": "google_genai",
    "openai": "openai",
    "anthropic": "anthropic",
}
_DEFAULT_MODELS = {"google_genai": "gemini-2.5-flash"}
_API_KEY_ENV = {"google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
_PACKAGE_HINT = {"google_genai": "langchain-google-genai", "openai": "langchain-openai", "anthropic": "langchain-anthropic"}


def build_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Build the chat model from arguments or environment variables.

    LLM_PROVIDER     default "google" (Gemini)
    LLM_MODEL        default "gemini-2.5-flash" for google; required for other providers
    LLM_TEMPERATURE  default 0 (reproducible runs)
    GOOGLE_API_KEY   Gemini API key (GEMINI_API_KEY is also accepted)

    Nothing outside this function needs to change to switch provider or model.
    """
    load_dotenv()  # does not override variables already set in the environment

    raw_provider = (provider or os.getenv("LLM_PROVIDER") or "google").strip().lower()
    lc_provider = _PROVIDER_ALIASES.get(raw_provider, raw_provider)

    model_name = model or os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(lc_provider)
    if not model_name:
        raise ConfigError(f"LLM_MODEL must be set when LLM_PROVIDER='{raw_provider}'.")

    if temperature is None:
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        except ValueError as exc:
            raise ConfigError("LLM_TEMPERATURE must be a number.") from exc

    key_vars = _API_KEY_ENV.get(lc_provider)
    if key_vars:
        found = next((os.environ[v] for v in key_vars if os.getenv(v)), None)
        if not found:
            raise ConfigError(
                f"No API key found. Set {key_vars[0]} in your environment or .env file (see .env.example)."
            )
        os.environ.setdefault(key_vars[0], found)  # the integration reads the first name

    try:
        return init_chat_model(model_name, model_provider=lc_provider, temperature=temperature)
    except ImportError as exc:
        package = _PACKAGE_HINT.get(lc_provider, f"the LangChain package for '{raw_provider}'")
        raise ConfigError(f"Provider package missing. Install it with: pip install {package}") from exc
    except ValueError as exc:
        raise ConfigError(f"Could not create LLM (provider='{raw_provider}', model='{model_name}'): {exc}") from exc


# --------------------------------------------------------------------------- 2. callbacks -> events


def _parse_input_str(input_str: Any) -> dict:
    """LangChain passes tool input as str(dict); recover the dict when `inputs` is absent."""
    if isinstance(input_str, dict):
        return input_str
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(input_str)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, TypeError):
            continue
    return {"raw": str(input_str)}


def _normalize_output(output: Any) -> Any:
    """Tool output may arrive as a ToolMessage with a JSON string; recover the structure."""
    content = getattr(output, "content", output)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


class EventCallbackHandler(BaseCallbackHandler):
    """Turns LangChain model/tool callbacks into observable events on an EventLog."""

    raise_error = False  # observability must never break the agent

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self._active_tools: dict[UUID, dict] = {}

    # Chat-model start is irrelevant here, but defining it avoids LangChain's fallback warning.
    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        return None

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """The model finished; each tool call it requested is a 'tool selected' event."""
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                for call in getattr(message, "tool_calls", None) or []:
                    self.log.emit(
                        EventType.TOOL_SELECTED,
                        tool_name=call["name"],
                        input=call.get("args") or {},
                        status=EventStatus.SUCCESS,
                        summary=f"Model selected tool '{call['name']}'",
                        metadata={"tool_call_id": call.get("id")},
                    )

    def on_tool_start(self, serialized: Any, input_str: Any, *, run_id: UUID, inputs: Optional[dict] = None, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown_tool"
        args = inputs if isinstance(inputs, dict) else _parse_input_str(input_str)
        self._active_tools[run_id] = {"name": name, "input": args, "started": time.perf_counter()}
        self.log.emit(
            EventType.TOOL_CALLED,
            tool_name=name,
            input=args,
            status=EventStatus.STARTED,
            summary=f"Tool '{name}' called",
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        ctx = self._active_tools.pop(run_id, {"name": kwargs.get("name"), "input": None, "started": None})
        payload = _normalize_output(output)
        duration = _elapsed_ms(ctx["started"])
        failed = isinstance(payload, dict) and payload.get("status") == "error"
        self.log.emit(
            EventType.TOOL_ERROR if failed else EventType.TOOL_COMPLETED,
            tool_name=ctx["name"],
            input=ctx["input"],
            output=payload,
            status=EventStatus.ERROR if failed else EventStatus.SUCCESS,
            summary=summarize_tool_output(ctx["name"], payload),
            metadata={"duration_ms": duration},
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Raised errors (e.g. schema validation failures) before/inside a tool."""
        ctx = self._active_tools.pop(run_id, {"name": kwargs.get("name"), "input": None, "started": None})
        message = f"{type(error).__name__}: {error}"
        self.log.emit(
            EventType.TOOL_ERROR,
            tool_name=ctx["name"],
            input=ctx["input"],
            output={"status": "error", "error_type": type(error).__name__, "message": str(error)},
            status=EventStatus.ERROR,
            summary=message[:300],
            metadata={"duration_ms": _elapsed_ms(ctx["started"])},
        )


def _elapsed_ms(started: Optional[float]) -> Optional[float]:
    return None if started is None else round((time.perf_counter() - started) * 1000, 2)


# --------------------------------------------------------------------------- 3. the agent


@dataclass
class AgentResult:
    request: str
    final_response: str
    status: str  # "success" or "error"
    events: list[Event] = field(default_factory=list)


def _message_text(content: Any) -> str:
    """Gemini may return a list of content blocks; flatten to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return ""


def _final_text(result: dict) -> str:
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _message_text(message.content)
    return ""


class LaptopAgent:
    """Controlled research/recommendation agent: one LLM, three local tools, observable events."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Any]] = None,
        sinks: Optional[Iterable[EventSink]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        recursion_limit: int = 20,
    ) -> None:
        self.llm = llm if llm is not None else build_llm()
        self.tools = list(tools) if tools is not None else list(TOOLS)
        self.sinks = list(sinks or [])  # e.g. a future ExecutionRecorder.record
        self.recursion_limit = recursion_limit
        self._graph = create_agent(model=self.llm, tools=self.tools, system_prompt=system_prompt)

    def run(self, request: str) -> AgentResult:
        log = EventLog(sinks=self.sinks)
        log.emit(EventType.AGENT_STARTED, summary="Agent started")
        log.emit(
            EventType.USER_REQUEST_RECEIVED,
            input={"request": request},
            summary="User request received",
        )

        try:
            if not request or not request.strip():
                raise ValueError("The request is empty.")
            result = self._graph.invoke(
                {"messages": [{"role": "user", "content": request}]},
                config={"callbacks": [EventCallbackHandler(log)], "recursion_limit": self.recursion_limit},
            )
            final_text = _final_text(result)
            if not final_text:
                raise RuntimeError("The model returned an empty final response.")
        except Exception as exc:  # noqa: BLE001 - the run must end with a recorded outcome
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("Agent run failed")
            log.emit(
                EventType.AGENT_ERROR,
                output={"error_type": type(exc).__name__, "message": str(exc)},
                status=EventStatus.ERROR,
                summary=message[:300],
            )
            return AgentResult(
                request=request,
                final_response=f"The agent could not complete this request. ({message})",
                status="error",
                events=log.events,
            )

        log.emit(
            EventType.FINAL_RESPONSE_GENERATED,
            output={"response": final_text},
            summary=f"Final response generated ({len(final_text)} characters)",
        )
        return AgentResult(request=request, final_response=final_text, status="success", events=log.events)
