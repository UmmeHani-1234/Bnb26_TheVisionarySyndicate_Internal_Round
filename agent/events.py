"""Lightweight observable-event model for the Black Box target agent.

This module is deliberately free of LangChain imports. The agent adapts LangChain
callbacks into `Event` objects (see `EventCallbackHandler` in agent.py) and pushes
them into an `EventLog`. A future `ExecutionRecorder` only needs to be registered
as a sink; the agent itself does not change:

    log = EventLog(sinks=[my_recorder.record])

Only observable facts are recorded: what was requested, which tool was selected,
what went into and came out of each tool, statuses and short summaries.
No chain-of-thought is captured.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


class EventType:
    AGENT_STARTED = "agent_started"
    USER_REQUEST_RECEIVED = "user_request_received"
    TOOL_SELECTED = "tool_selected"  # the model chose a tool (carries the arguments)
    TOOL_CALLED = "tool_called"  # the tool started executing (carries the input)
    TOOL_COMPLETED = "tool_completed"  # the tool returned normally (input + output)
    TOOL_ERROR = "tool_error"  # the tool returned a structured error or raised
    FINAL_RESPONSE_GENERATED = "final_response_generated"
    AGENT_ERROR = "agent_error"


class EventStatus:
    STARTED = "started"
    SUCCESS = "success"
    ERROR = "error"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Return a JSON-safe copy of `value` (unknown objects become strings)."""
    if value is None:
        return None
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


@dataclass
class Event:
    event_type: str
    tool_name: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    status: str = EventStatus.SUCCESS
    summary: str = ""
    metadata: dict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=utc_now_iso)
    sequence: int = 0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "input": to_jsonable(self.input),
            "output": to_jsonable(self.output),
            "status": self.status,
            "summary": self.summary,
            "sequence": self.sequence,
            "metadata": to_jsonable(self.metadata),
        }


EventSink = Callable[[Event], None]


class EventLog:
    """Ordered, in-memory list of events plus optional sinks notified on every emit."""

    def __init__(
        self,
        sinks: Optional[Iterable[EventSink]] = None,
        id_factory: Optional[Callable[[], str]] = None,
        clock: Optional[Callable[[], str]] = None,
    ) -> None:
        self._events: list[Event] = []
        self._sinks: list[EventSink] = list(sinks or [])
        self._id_factory = id_factory
        self._clock = clock

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def add_sink(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def emit(
        self,
        event_type: str,
        *,
        tool_name: Optional[str] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        status: str = EventStatus.SUCCESS,
        summary: str = "",
        metadata: Optional[dict] = None,
    ) -> Event:
        event = Event(
            event_type=event_type,
            tool_name=tool_name,
            input=input,
            output=output,
            status=status,
            summary=summary,
            metadata=metadata or {},
            sequence=len(self._events) + 1,
        )
        if self._id_factory:
            event.event_id = self._id_factory()
        if self._clock:
            event.timestamp = self._clock()
        self._events.append(event)
        for sink in self._sinks:
            try:
                sink(event)
            except Exception:  # a broken sink must never break the agent
                logger.exception("Event sink failed for %s", event.event_type)
        return event

    def to_dicts(self) -> list[dict]:
        return [e.to_dict() for e in self._events]


# --------------------------------------------------------------------------- summaries


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def summarize_tool_output(tool_name: Optional[str], output: Any) -> str:
    """One-line, human-readable summary of a tool result (no reasoning, just facts)."""
    if not isinstance(output, dict):
        return f"Returned {type(output).__name__}"
    if output.get("status") == "error":
        return f"{output.get('error_type', 'error')}: {output.get('message', '')}".strip()
    if tool_name == "search_products":
        count = output.get("count", len(output.get("products", [])))
        return f"{_plural(int(count), 'product')} matched"
    if tool_name == "check_specifications":
        passed = output.get("passed_count", 0)
        total = passed + output.get("failed_count", 0)
        return f"{output.get('product')}: {passed}/{total} requirements passed"
    if tool_name == "calculate_budget":
        diff = output.get("difference") or 0
        name = output.get("product") or "Product"
        price = output.get("price") or 0
        try:
            p_val = float(price)
            d_val = float(diff)
        except (ValueError, TypeError):
            p_val, d_val = 0.0, 0.0
        if output.get("within_budget"):
            return f"{name}: ₹{p_val:,.0f} is ₹{d_val:,.0f} under budget"
        return f"{name}: ₹{p_val:,.0f} is ₹{abs(d_val):,.0f} over budget"
    return "Completed"


# --------------------------------------------------------------------------- timeline

_RULE = "─" * 60


def _compact(value: Any, limit: int = 110) -> str:
    text = json.dumps(to_jsonable(value), ensure_ascii=False, separators=(", ", ": "))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe(event: Event) -> tuple[str, list[str]]:
    t = event.event_type
    tool = f"Tool: {event.tool_name}"
    if t == EventType.AGENT_STARTED:
        return "Agent started", []
    if t == EventType.USER_REQUEST_RECEIVED:
        return "User request received", []
    if t == EventType.TOOL_SELECTED:
        return "Tool selected", [tool, f"Input: {_compact(event.input)}"]
    if t == EventType.TOOL_CALLED:
        return "Tool called", [tool]
    if t == EventType.TOOL_COMPLETED:
        return "Tool completed", [tool, f"Status: {event.status.upper()}", f"Result: {event.summary}"]
    if t == EventType.TOOL_ERROR:
        return "Tool failed", [tool, f"Status: {event.status.upper()}", f"Error: {event.summary}"]
    if t == EventType.FINAL_RESPONSE_GENERATED:
        return "Final response generated", []
    if t == EventType.AGENT_ERROR:
        return "Agent error", [f"Error: {event.summary}"]
    return t, [event.summary] if event.summary else []


def format_timeline(events: Iterable[Event], request: str, final_response: str) -> str:
    """Render the terminal view: request, numbered observable steps, final response."""
    lines = ["USER REQUEST", request, "", _RULE, "", "AGENT EXECUTION", ""]
    for i, event in enumerate(events, start=1):
        label, details = _describe(event)
        lines.append(f"[{i}] {label}")
        lines.extend(f"    {d}" for d in details)
        lines.append("")
    lines += [_RULE, "", "FINAL RESPONSE", "", final_response, ""]
    return "\n".join(lines)
