# Black Box: target agent (phase 1)

A small, controlled LangChain agent that Black Box will later monitor and debug.
It answers laptop-recommendation requests using **three tools over a local dataset**
(`data/products.json`, 10 laptops; prices are illustrative sample data) and records
every observable step as a structured event. No live web search, no external APIs
besides the LLM.

**Not in this phase:** debugger, dashboard, frontend, ML failure detection, replay, checkpoints.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
# edit .env and set GOOGLE_API_KEY (get one from Google AI Studio)
```

## Run

```bash
python main.py                                   # default request
python main.py "I need a laptop around ₹60,000 with at least 16GB RAM."
python main.py "Which laptop is best for gaming within ₹75,000?"
python main.py "Show me options that are within ₹65,000."
python main.py --json                            # also print raw events
python main.py --save-events events.json         # save raw events for later tooling
```

## Test

```bash
python -m pytest -q
```

Tests use a scripted fake chat model, so they need no API key and no network.

## Configuration (environment variables / `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `google` | `google` (alias `gemini`) uses Gemini. Others (`openai`, `anthropic`) work if their package is installed. |
| `GOOGLE_API_KEY` | none | Gemini API key (`GEMINI_API_KEY` also accepted). Never hard-coded. |
| `LLM_MODEL` | `gemini-2.5-flash` | Model name. Required for non-Google providers. |
| `LLM_TEMPERATURE` | `0` | Kept at 0 for reproducible runs. |
| `PRODUCTS_PATH` | `data/products.json` | Alternate dataset (handy for failure injection later). |

Provider-specific code lives only in `build_llm()` in `agent/agent.py`, plus the single
`langchain-google-genai` line in `requirements.txt`.

## Architecture

```
main.py ── LaptopAgent.run(request)
              │
              ├─ EventLog ◄── EventCallbackHandler ◄── LangChain callbacks
              │      │            (model chose a tool / tool start / tool end / tool error)
              │      └─ sinks  ← a future ExecutionRecorder plugs in here
              │
              └─ create_agent(llm, tools, system_prompt)
                     tools: search_products, check_specifications, calculate_budget
```

### Event schema

Every event serialises to:

```json
{
  "event_id": "uuid",
  "event_type": "tool_completed",
  "timestamp": "2026-10-03T13:16:35+00:00",
  "tool_name": "search_products",
  "input": {"max_price": 70000},
  "output": {"status": "success", "count": 4, "products": ["..."]},
  "status": "success",
  "summary": "4 products matched",
  "sequence": 5,
  "metadata": {"duration_ms": 0.4}
}
```

`tool_name` is `null` for non-tool events. Event types: `agent_started`,
`user_request_received`, `tool_selected`, `tool_called`, `tool_completed`, `tool_error`,
`final_response_generated`, `agent_error`. Statuses: `started`, `success`, `error`.
Only observable facts are stored; no chain-of-thought.

### Adding an ExecutionRecorder later

```python
class ExecutionRecorder:
    def record(self, event: Event) -> None: ...

agent = LaptopAgent(sinks=[ExecutionRecorder().record])
```

Nothing else in the agent changes. `LaptopAgent` also accepts `tools=[...]` and `llm=...`,
so wrapped or fault-injected tools and fake models can be swapped in.

### Error handling

Tools never raise on bad input; they return
`{"status": "error", "error_type": "...", "message": "..."}`, which is emitted as a `tool_error`
event and handed back to the model so it can retry or explain. Wrong-typed arguments are rejected
by LangChain's schema validation and are recorded through the `on_tool_error` callback.
A failed run ends with an `agent_error` event instead of crashing.

### Reproducibility

Temperature is 0 and the dataset and result ordering (cheapest first) are fixed.
LLM output can still vary slightly between runs; event IDs and timestamps are unique per run by design.
