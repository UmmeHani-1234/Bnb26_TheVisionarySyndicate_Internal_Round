# 🔬 Black Box — AI Agent Debugging & Observability System

> **A mission-control observability platform and automated debugging pipeline for production AI agents.**

Black Box diagnoses, replays, and evaluates failures in multi-step AI agents. When an agent hallucinates, breaches constraints, picks bad tools, or fails silently, Black Box isolates the root cause using a 6-signal anomaly detector and machine learning classifier, allows deterministic state-restored replay from checkpoints, and verifies fixes with an independent judge.

---

## 🏗️ Architecture

```text
User Request
     ↓
Chat Interface (Streamlit Mission Control UI)
     ↓
8-Step Observable AI Agent (Controlled ReAct Cycle)
     ↓
Execution Recorder (Structured AgentEvents & In-Memory Snapshots)
     ↓
Persistent Database (PostgreSQL / SQLite via SQLAlchemy)
     ↓
Failure Intelligence Pipeline (6-Signal Fusion: Heuristic + Random Forest)
     ↓
Evidence Extraction (Parameter Delta, Spec Check, Budget Math, Constraint Log)
     ↓
Checkpoint State Restoration (In-memory & DB Checkpoints)
     ↓
Deterministic Replay & Alternative Execution (Controlled Parameter Overrides)
     ↓
3-Way Trace Comparison & Independent Verification (Original vs Alternative vs Reference)
     ↓
System Benchmark & Evaluation (Leakage-Safe Held-Out Baseline & Method Comparisons)
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python 3.10+
- (Optional) Docker & Docker Compose
- (Optional) PostgreSQL

### 2. Installation
```bash
git clone https://github.com/UmmeHani-1234/Bnb26_TheVisionarySyndicate_Internal_Round.git black-box
cd black-box
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Setup
Copy the template configuration:
```bash
cp .env.example .env
```
Default configuration works **100% offline out-of-the-box** using `LLM_PROVIDER=local` with zero API keys or external costs. To use cloud models, configure `GROQ_API_KEY` or `GOOGLE_API_KEY`.

### 4. Running the Application
#### Unified Production Runner (Recommended):
```bash
python run_production.py
```
This starts both:
* **Streamlit Dashboard / Chat UI**: `http://localhost:8501`
* **FastAPI Observability API**: `http://localhost:8000` (`http://localhost:8000/docs`)

---

## 🐳 Docker Deployment

Run with persistent PostgreSQL database and automated health checks:
```bash
docker compose up -d
```
- Dashboard: `http://localhost:8501`
- REST API: `http://localhost:8000`
- PostgreSQL: `localhost:5432`

---

## ☁️ Cloud Deployment (Render / Railway / Heroku / Fly.io)

### Render (1-Click Blueprint)
Push repository to GitHub, connect to [Render](https://render.com), and click **New > Blueprint**. Render automatically detects `render.yaml`, provisions a PostgreSQL database, builds the container, and deploys.

### Heroku / Railway
The repository includes a production `Procfile`:
```text
web: python run_production.py
```
Set environment variables in your platform dashboard:
- `PORT` (assigned automatically by host)
- `DATABASE_URL` (PostgreSQL connection string)
- `LLM_PROVIDER` (`local` or `groq`)

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql://...`) | Fallback to `sqlite:///traces.db` |
| `LLM_PROVIDER` | Agent model provider: `local`, `groq`, `google`, `ollama` | `local` |
| `LLM_MODEL` | Model identifier | `blackbox-local-offline` |
| `GROQ_API_KEY` | Groq Cloud API key (if `LLM_PROVIDER=groq`) | - |
| `GOOGLE_API_KEY` | Gemini API key (if `LLM_PROVIDER=google`) | - |
| `PORT` | Public web UI port | `8501` |
| `API_PORT` | Backend REST API port | `8000` |
| `BACKEND_URL` | Public URL for backend API | `http://localhost:8000` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `*` |

---

## 🧪 Testing & Validation

Execute the complete test suite (101 unit, integration, and end-to-end tests):
```bash
pytest tests/ -v
```

---

## 🔁 Complete 10-Step Workflow

1. **User Chat**: Enter natural-language product or research queries on the main dashboard.
2. **Execution Steps**: Watch real-time 8-stage execution events (Thought, Tool Selection, Input Formulation, Tool Execution, Output Observation, State Update, Final Decision, User Response).
3. **Controlled Failures**: Inject real-world agent failure modes (Over Budget, Wrong Specs, Tool Hallucination, Bad Args, Empty Results).
4. **Execution Traces**: Inspect deep JSON payloads, input/output structures, tool latencies, and state snapshots.
5. **6-Signal Diagnosis**: Automated root-cause localization combining heuristic signals with a Random Forest classifier.
6. **Observable Evidence**: Mathematical proofs, parameter deltas, and constraint violations pinpointing exact bug causes.
7. **Deterministic Checkpoints**: Restore agent execution state from any historical intermediate step.
8. **Alternative Execution**: Replay with patched budgets or overrides without re-executing previous steps.
9. **3-Way Trace Comparison**: Side-by-side alignment of Original vs Alternative vs Reference runs with Independent Verifier ratings.
10. **Benchmark Evaluation**: Leakage-safe empirical evaluation comparing baseline methods across Top-1, Top-3, and MRR.
