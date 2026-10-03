"""Agent tests using a scripted fake chat model: no API key, no network, fully deterministic."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.agent import ConfigError, LaptopAgent, build_llm
from agent.events import EventLog, EventType, format_timeline


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of AIMessages, one per model call."""

    responses: list[Any]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # tools are already wired by the test script
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.responses[min(self.cursor, len(self.responses) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def happy_path_script() -> list[AIMessage]:
    return [
        tool_call("search_products", {"max_price": 70000, "needs_gaming": True, "needs_programming": True}, "c1"),
        tool_call("check_specifications", {"product_name": "ASUS TUF Gaming F15", "min_ram_gb": 16}, "c2"),
        tool_call("calculate_budget", {"product_name": "ASUS TUF Gaming F15", "budget": 70000}, "c3"),
        AIMessage(content="I recommend the ASUS TUF Gaming F15 at ₹68,990, which is ₹1,010 under budget."),
    ]


def test_happy_path_event_sequence():
    agent = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()))
    result = agent.run("Find a laptop under ₹70,000 suitable for programming and gaming.")

    assert result.status == "success"
    assert "ASUS TUF Gaming F15" in result.final_response

    types = [e.event_type for e in result.events]
    assert types == [
        EventType.AGENT_STARTED,
        EventType.USER_REQUEST_RECEIVED,
        EventType.TOOL_SELECTED, EventType.TOOL_CALLED, EventType.TOOL_COMPLETED,
        EventType.TOOL_SELECTED, EventType.TOOL_CALLED, EventType.TOOL_COMPLETED,
        EventType.TOOL_SELECTED, EventType.TOOL_CALLED, EventType.TOOL_COMPLETED,
        EventType.FINAL_RESPONSE_GENERATED,
    ]
    tool_names = [e.tool_name for e in result.events if e.event_type == EventType.TOOL_COMPLETED]
    assert tool_names == ["search_products", "check_specifications", "calculate_budget"]

    completed = [e for e in result.events if e.event_type == EventType.TOOL_COMPLETED]
    assert completed[0].input["max_price"] == 70000
    assert completed[0].output["count"] == 4
    assert completed[0].summary == "4 products matched"
    assert completed[2].output["within_budget"] is True
    assert [e.sequence for e in result.events] == list(range(1, 13))


def test_events_have_required_fields():
    agent = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()))
    result = agent.run("Find a laptop under ₹70,000 suitable for programming and gaming.")
    ids = set()
    for event in result.events:
        d = event.to_dict()
        assert {"event_id", "event_type", "timestamp", "tool_name", "input", "output", "status"} <= set(d)
        ids.add(d["event_id"])
    assert len(ids) == len(result.events)  # unique ids
    assert result.events[0].tool_name is None


def test_tool_error_is_observable_and_agent_continues():
    script = [
        tool_call("calculate_budget", {"product_name": "Imaginary Book", "budget": 70000}, "c1"),
        AIMessage(content="I could not find that laptop in the catalogue."),
    ]
    result = LaptopAgent(llm=ScriptedChatModel(responses=script)).run("Is Imaginary Book within ₹70,000?")

    assert result.status == "success"  # the run itself finished
    errors = [e for e in result.events if e.event_type == EventType.TOOL_ERROR]
    assert len(errors) == 1
    assert errors[0].tool_name == "calculate_budget"
    assert errors[0].status == "error"
    assert errors[0].output["error_type"] == "product_not_found"
    assert result.events[-1].event_type == EventType.FINAL_RESPONSE_GENERATED


def test_no_matches_flow():
    script = [
        tool_call("search_products", {"max_price": 10000, "needs_gaming": True}, "c1"),
        AIMessage(content="No matching products were found in the catalogue."),
    ]
    result = LaptopAgent(llm=ScriptedChatModel(responses=script)).run("Gaming laptop under ₹10,000")
    completed = [e for e in result.events if e.event_type == EventType.TOOL_COMPLETED][0]
    assert completed.output["count"] == 0
    assert "No matching products" in result.final_response


def test_empty_request_is_recorded_as_agent_error():
    result = LaptopAgent(llm=ScriptedChatModel(responses=[AIMessage(content="x")])).run("   ")
    assert result.status == "error"
    assert result.events[-1].event_type == EventType.AGENT_ERROR


def test_sinks_receive_every_event():
    received = []
    agent = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()), sinks=[received.append])
    result = agent.run("Find a laptop under ₹70,000 suitable for programming and gaming.")
    assert [e.event_id for e in received] == [e.event_id for e in result.events]


def test_broken_sink_does_not_break_agent():
    def bad_sink(event):
        raise RuntimeError("recorder down")

    agent = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()), sinks=[bad_sink])
    assert agent.run("Find a laptop under ₹70,000 suitable for programming and gaming.").status == "success"


def test_timeline_rendering():
    agent = LaptopAgent(llm=ScriptedChatModel(responses=happy_path_script()))
    result = agent.run("Find a laptop under ₹70,000 suitable for programming and gaming.")
    text = format_timeline(result.events, result.request, result.final_response)
    for expected in ("USER REQUEST", "AGENT EXECUTION", "FINAL RESPONSE", "Tool selected",
                     "Tool: search_products", "Status: SUCCESS", "Tool: calculate_budget",
                     "Final response generated"):
        assert expected in text


def test_event_log_is_independent_of_langchain():
    log = EventLog(id_factory=iter(["a", "b"]).__next__, clock=lambda: "T0")
    e1 = log.emit(EventType.AGENT_STARTED)
    e2 = log.emit(EventType.TOOL_CALLED, tool_name="x", input={"k": 1})
    assert (e1.event_id, e2.event_id, e1.timestamp) == ("a", "b", "T0")
    assert log.to_dicts()[1]["input"] == {"k": 1}


# ------------------------------------------------------------------ provider config


def test_build_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr("agent.agent.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "google")
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        build_llm()


def test_build_llm_requires_model_for_unknown_provider(monkeypatch):
    monkeypatch.setattr("agent.agent.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        build_llm(provider="openai")