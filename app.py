"""Streamlit Developer Observability & Debugging Dashboard for Black Box.

Stage 7: Final Product Integration, Unified Navigation, Polish & Demo Mode.
Implements the complete developer debugging lifecycle:
Run Agent / Demo -> Failed Run -> Execution Trace -> Diagnose -> Top-3 Suspicious Steps
-> Evidence -> Checkpoint -> Controlled Change -> Alternative Run -> 3-Way Trace Comparison
-> Independent Verifier -> Benchmark Evaluation.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from agent.agent import ConfigError, LaptopAgent
from agent.events import EventType
from failure_intelligence.service import FailureIntelligenceService
from recorder.recorder import ExecutionRecorder
from replay.checkpoint_manager import CheckpointManager
from replay.replay_engine import ReplayEngine
from storage.database import SessionLocal, init_db
from storage.repository import TraceRepository

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- Page Config
st.set_page_config(
    page_title="Black Box | AI Agent Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- Styling & Theme
st.markdown("""
<style>
/* Developer Observability Theme */
.brand-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 1.2rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 10px;
    border: 1px solid #334155;
    margin-bottom: 1.5rem;
}
.brand-title {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.brand-subtitle {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 0.2rem;
}
.badge-tag {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-success { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.badge-failed { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
.badge-suspicious { background: #78350f; color: #fbbf24; border: 1px solid #d97706; }
.badge-checkpoint { background: #1e1b4b; color: #a5b4fc; border: 1px solid #6366f1; }
.badge-replay { background: #312e81; color: #c7d2fe; border: 1px solid #4f46e5; }
.badge-recovered { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.badge-not-recovered { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }

.step-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.step-card-suspicious {
    border-left: 5px solid #f59e0b !important;
    background: #18181b;
}
.step-card-failed {
    border-left: 5px solid #ef4444 !important;
}
.code-block {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 0.75rem;
    font-family: monospace;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- Header
st.markdown("""
<div class="brand-header">
    <div>
        <div class="brand-title">🔬 BLACK BOX · AI Agent Debugging System</div>
        <div class="brand-subtitle">Observable Agent Traces · Failure Localization · Checkpoint Replay · Trace Comparison · Independent Verification</div>
    </div>
    <div style="text-align: right;">
        <span class="badge-tag badge-checkpoint">Developer Observability Dashboard</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- Database & Services
init_db()
repo = TraceRepository()
fi_service = FailureIntelligenceService(repository=repo)
DATA_PATH = Path("data/products.json")


@st.cache_data
def load_products() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def build_ml_recommender(products: List[Dict[str, Any]]):
    if not products:
        return None, None, None
    features = []
    for p in products:
        features.append([
            float(p.get("price", 0)),
            float(p.get("ram_gb", 8)),
            float(p.get("storage_gb", 512)),
            float(p.get("gaming_suitability", 1)),
            float(p.get("programming_suitability", 1)),
            1.0 if p.get("dedicated_gpu", False) else 0.0,
        ])
    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nn_model = NearestNeighbors(n_neighbors=min(4, len(products)), metric="cosine")
    nn_model.fit(X_scaled)
    return nn_model, scaler, X_scaled


def get_similar_laptops(product_name, products, model, scaler):
    names = [p["name"] for p in products]
    if product_name not in names or model is None:
        return []
    idx = names.index(product_name)
    target = np.array([[
        float(products[idx].get("price", 0)),
        float(products[idx].get("ram_gb", 8)),
        float(products[idx].get("storage_gb", 512)),
        float(products[idx].get("gaming_suitability", 1)),
        float(products[idx].get("programming_suitability", 1)),
        1.0 if products[idx].get("dedicated_gpu", False) else 0.0,
    ]])
    scaled_target = scaler.transform(target)
    distances, indices = model.kneighbors(scaled_target)
    similar = []
    for dist, i in zip(distances[0], indices[0]):
        if i != idx:
            similar.append({"product": products[i], "similarity_score": round((1.0 - float(dist)) * 100, 1)})
    return similar


NAV_PAGES = [
    "📊 Dashboard",
    "🚀 Demo & Agent Runner",
    "🗂️ Runs Browser",
    "🔬 Execution Trace",
    "🧠 Diagnosis & Evidence",
    "🔄 Checkpoint & Replay",
    "⚖️ Trace Comparison",
    "📈 Evaluation & Metrics",
    "📦 Catalogue & Tools",
]

def set_nav(target_page: str):
    st.session_state["main_nav_selection"] = target_page

def select_and_debug_run(run_id: str, target_page: str = "🔬 Execution Trace"):
    st.session_state["active_run_id"] = run_id
    st.session_state["main_nav_selection"] = target_page

if "main_nav_selection" not in st.session_state:
    st.session_state["main_nav_selection"] = NAV_PAGES[0]

# --------------------------------------------------------------------------- Sidebar Navigation
with st.sidebar:
    st.subheader("🧭 Black Box Navigation")
    nav_selection = st.radio(
        "Go to View:",
        NAV_PAGES,
        key="main_nav_selection",
    )

    st.divider()

    # Active Run Selection Sync
    all_runs_summary = repo.list_runs(limit=100)
    all_run_ids = [r["run_id"] for r in all_runs_summary]

    st.subheader("🎯 Active Debugging Run")
    if not all_run_ids:
        st.caption("No runs in storage yet. Launch a run in Demo Mode.")
        active_run_id = None
    else:
        # Default to session state run or latest run
        default_idx = 0
        if "active_run_id" in st.session_state and st.session_state["active_run_id"] in all_run_ids:
            default_idx = all_run_ids.index(st.session_state["active_run_id"])

        active_run_id = st.selectbox(
            "Selected Execution:",
            all_run_ids,
            index=default_idx,
            format_func=lambda rid: f"{'❌' if any(r['run_id'] == rid and r['status'] == 'failed' for r in all_runs_summary) else '✅'} {rid}",
            key="sidebar_active_run_selector",
        )
        st.session_state["active_run_id"] = active_run_id

    st.divider()
    st.caption("🤖 Active LLM Engine:")
    provider_val = os.getenv("LLM_PROVIDER", "local")
    model_val = os.getenv("LLM_MODEL", "blackbox-local-offline")
    st.code(f"provider: {provider_val}\nmodel: {model_val}", language="yaml")


# =====================================================================
# VIEW 1: DASHBOARD
# =====================================================================
if nav_selection == "📊 Dashboard":
    st.markdown("### 📊 System Overview & Live Telemetry")
    st.caption("Real database metrics for all recorded agent executions, replay lineages, and failure distributions.")

    all_runs = repo.list_runs(limit=200)
    total_runs = len(all_runs)
    successful_runs = sum(1 for r in all_runs if r.get("status") == "success")
    failed_runs = sum(1 for r in all_runs if r.get("status") == "failed")
    replay_branches = sum(1 for r in all_runs if r.get("parent_run_id") is not None)

    # Calculate real recovery rate across replays
    replays_recovered = 0
    total_replays_checked = 0
    for r in all_runs:
        if r.get("parent_run_id"):
            total_replays_checked += 1
            if r.get("status") == "success":
                replays_recovered += 1
    recovery_rate = (replays_recovered / total_replays_checked * 100) if total_replays_checked > 0 else 0.0

    # Collect failure categories
    failure_types = set()
    for r in all_runs:
        fm = r.get("failure_mode")
        if fm and fm != "none":
            failure_types.add(fm)

    # Metric Cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Runs", total_runs)
    with c2:
        st.metric("Successful", successful_runs)
    with c3:
        st.metric("Failed", failed_runs)
    with c4:
        st.metric("Failure Types", len(failure_types))
    with c5:
        st.metric("Replay Branches", replay_branches)
    with c6:
        st.metric("Recovery Rate", f"{recovery_rate:.1f}%")

    st.divider()

    # Quick Action Buttons
    st.markdown("#### ⚡ Quick Debugging Actions")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        st.button("🚀 Launch Controlled Demo Run", use_container_width=True, on_click=set_nav, args=("🚀 Demo & Agent Runner",))
    with qa2:
        st.button("🔬 Inspect Active Trace", use_container_width=True, disabled=(not active_run_id), on_click=set_nav, args=("🔬 Execution Trace",))
    with qa3:
        st.button("🧠 View Diagnosis & Evidence", use_container_width=True, disabled=(not active_run_id), on_click=set_nav, args=("🧠 Diagnosis & Evidence",))
    with qa4:
        st.button("📈 Open Benchmark Evaluation", use_container_width=True, on_click=set_nav, args=("📈 Evaluation & Metrics",))

    st.divider()

    # Recent Runs Table
    st.markdown("#### 📜 Recent Agent Executions")
    if not all_runs:
        st.info("No executions recorded yet. Launch a run in the **🚀 Demo & Agent Runner** tab.")
    else:
        table_data = []
        for r in all_runs[:15]:
            status_badge = "✅ SUCCESS" if r["status"] == "success" else "❌ FAILED"
            run_type = "🔁 Alternative Replay" if r.get("parent_run_id") else "Original Execution"
            table_data.append({
                "Run ID": r["run_id"],
                "Status": status_badge,
                "Run Type": run_type,
                "Failure Mode": r.get("failure_mode") or "None",
                "User Request": (r.get("user_request") or "")[:50] + ("..." if len(r.get("user_request") or "") > 50 else ""),
                "Parent Run": r.get("parent_run_id") or "—",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# =====================================================================
# VIEW 2: DEMO MODE & AGENT RUNNER
# =====================================================================
elif nav_selection == "🚀 Demo & Agent Runner":
    st.markdown("### 🚀 Demo Mode & Controlled Agent Runner")
    st.caption("Trigger real agent executions with controlled fault injections to observe end-to-end failure diagnosis and recovery.")

    col_demo, col_custom = st.columns([1, 1])

    with col_demo:
        st.markdown("#### ⚡ 1-Click Controlled Demo Scenarios")
        st.write("Select a pre-configured scenario to demonstrate Black Box debugging capabilities:")

        demo_scenario = st.selectbox(
            "Demo Scenario:",
            [
                "Wrong Tool Selection (Premature specification check)",
                "Budget Violation (Agent exceeds max budget constraint)",
                "Wrong Interpretation (Mismatched price hallucination)",
                "Unexpected Output (Corrupted tool response format)",
                "Tool Timeout (Simulated network latency failure)",
                "Clean Success Execution (Normal coursework query)",
            ],
            index=0,
        )

        demo_btn = st.button("🎯 Execute Demo Scenario", type="primary", use_container_width=True)

        if demo_btn:
            # Map selection to configuration
            scenario_map = {
                "Wrong Tool Selection (Premature specification check)": ("Find a laptop under 50000 for coursework", "wrong_tool"),
                "Budget Violation (Agent exceeds max budget constraint)": ("Find a gaming laptop under ₹40,000", "budget_violation"),
                "Wrong Interpretation (Mismatched price hallucination)": ("Recommend ultrabook within ₹60,000", "wrong_interpretation"),
                "Unexpected Output (Corrupted tool response format)": ("Find lightweight laptop within ₹55,000", "unexpected_output"),
                "Tool Timeout (Simulated network latency failure)": ("High-performance laptop under ₹75,000", "timeout"),
                "Clean Success Execution (Normal coursework query)": ("Find a laptop under 50000 for coursework", None),
            }
            query, f_mode = scenario_map[demo_scenario]

            with st.spinner(f"Running agent with failure_mode='{f_mode}'..."):
                recorder = ExecutionRecorder(repository=repo)
                agent = LaptopAgent(sinks=[recorder.record])
                agent_res = agent.run(query, failure_mode=f_mode)
                new_run_id = recorder._current_run_id

                # Auto-generate checkpoints
                trace = repo.get_run_trace(new_run_id)
                if trace:
                    cm = CheckpointManager(repository=repo)
                    cm.create_checkpoints_for_run(
                        run_id=new_run_id,
                        user_request=query,
                        steps=trace.get("steps", []),
                        failure_mode=f_mode,
                    )

                st.session_state["active_run_id"] = new_run_id
                st.success(f"✅ Demo execution complete! Active run set to **`{new_run_id}`** ({agent_res.status.upper()})")

    with col_custom:
        st.markdown("#### 🛠️ Custom Agent Query")
        custom_query = st.text_input(
            "Device or Computing Query:",
            value="Recommend a laptop under ₹70,000 for programming and gaming",
        )
        custom_failure = st.selectbox(
            "Controlled Failure Mode:",
            ["none", "wrong_tool", "budget_violation", "wrong_interpretation", "unexpected_output", "timeout"],
            index=0,
        )
        custom_run_btn = st.button("🚀 Run Custom Query", use_container_width=True)

        if custom_run_btn:
            if not custom_query.strip():
                st.warning("Please provide a non-empty request.")
            else:
                f_mode = None if custom_failure == "none" else custom_failure
                with st.spinner("Executing agent..."):
                    recorder = ExecutionRecorder(repository=repo)
                    agent = LaptopAgent(sinks=[recorder.record])
                    agent_res = agent.run(custom_query, failure_mode=f_mode)
                    new_run_id = recorder._current_run_id

                    trace = repo.get_run_trace(new_run_id)
                    if trace:
                        cm = CheckpointManager(repository=repo)
                        cm.create_checkpoints_for_run(
                            run_id=new_run_id,
                            user_request=custom_query,
                            steps=trace.get("steps", []),
                            failure_mode=f_mode,
                        )

                    st.session_state["active_run_id"] = new_run_id
                    st.success(f"✅ Run finished! Saved as **`{new_run_id}`** ({agent_res.status.upper()})")

    st.divider()

    # Active Run Summary if present
    if active_run_id:
        active_trace = repo.get_run_trace(active_run_id)
        if active_trace:
            st.markdown(f"#### 🔎 Current Active Run: `{active_run_id}`")
            status_col, steps_col, action_col = st.columns([1, 1, 2])
            with status_col:
                if active_trace.get("status") == "success":
                    st.success("Status: ✅ SUCCESS")
                else:
                    st.error(f"Status: ❌ FAILED ({active_trace.get('failure_mode') or 'error'})")
            with steps_col:
                st.info(f"Steps Recorded: **{len(active_trace.get('steps', []))}**")
            with action_col:
                ca1, ca2 = st.columns(2)
                with ca1:
                    st.button("🔬 View 8-Step Trace", use_container_width=True, on_click=set_nav, args=("🔬 Execution Trace",))
                with ca2:
                    st.button("🧠 Open Diagnosis", use_container_width=True, on_click=set_nav, args=("🧠 Diagnosis & Evidence",))


# =====================================================================
# VIEW 3: RUNS BROWSER
# =====================================================================
elif nav_selection == "🗂️ Runs Browser":
    st.markdown("### 🗂️ Execution Runs Browser")
    st.caption("Filter, inspect, and select any execution run across the database.")

    all_runs = repo.list_runs(limit=150)
    if not all_runs:
        st.info("No runs recorded in database yet.")
    else:
        # Filters
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            status_filter = st.selectbox("Filter Status:", ["All", "Success", "Failed"], index=0)
        with c_f2:
            type_filter = st.selectbox("Run Type:", ["All", "Original Executions", "Replay Branches"], index=0)
        with c_f3:
            all_f_modes = ["All"] + sorted(list({r.get("failure_mode") for r in all_runs if r.get("failure_mode")}))
            fmode_filter = st.selectbox("Failure Mode:", all_f_modes, index=0)

        # Apply filters
        filtered_runs = all_runs
        if status_filter != "All":
            filtered_runs = [r for r in filtered_runs if r.get("status") == status_filter.lower()]
        if type_filter == "Original Executions":
            filtered_runs = [r for r in filtered_runs if not r.get("parent_run_id")]
        elif type_filter == "Replay Branches":
            filtered_runs = [r for r in filtered_runs if r.get("parent_run_id")]
        if fmode_filter != "All":
            filtered_runs = [r for r in filtered_runs if r.get("failure_mode") == fmode_filter]

        st.write(f"Showing **{len(filtered_runs)}** runs:")

        for r in filtered_runs:
            with st.container(border=True):
                rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
                with rc1:
                    is_active = (r["run_id"] == active_run_id)
                    title_text = f"**`{r['run_id']}`**" + (" 🎯 *(Active)*" if is_active else "")
                    st.markdown(title_text)
                    st.caption(f"Query: {r.get('user_request') or 'N/A'}")
                with rc2:
                    if r["status"] == "success":
                        st.markdown('<span class="badge-tag badge-success">SUCCESS</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-tag badge-failed">FAILED</span>', unsafe_allow_html=True)
                    if r.get("failure_mode"):
                        st.caption(f"Mode: `{r['failure_mode']}`")
                with rc3:
                    if r.get("parent_run_id"):
                        st.markdown('<span class="badge-tag badge-replay">REPLAY</span>', unsafe_allow_html=True)
                        st.caption(f"Parent: `{r['parent_run_id']}`")
                    else:
                        st.caption("Original Run")
                with rc4:
                    st.button("🔍 Debug Run", key=f"sel_btn_{r['run_id']}", use_container_width=True, on_click=select_and_debug_run, args=(r["run_id"], "🔬 Execution Trace"))


# =====================================================================
# VIEW 4: EXECUTION TRACE
# =====================================================================
elif nav_selection == "🔬 Execution Trace":
    st.markdown("### 🔬 Observable Execution Trace")
    st.caption("Chronological timeline of all 8 standardized execution stages. Only observable execution data is shown.")

    if not active_run_id:
        st.warning("No active run selected. Please launch a run or select one in the Runs Browser.")
    else:
        trace = repo.get_run_trace(active_run_id)
        if not trace:
            st.error(f"Trace for run `{active_run_id}` could not be found.")
        else:
            steps = trace.get("steps", [])

            # Status Banner
            if trace.get("status") == "failed":
                st.error(f"🚨 **RUN FAILED** — Failure Mode: `{trace.get('failure_mode') or 'Observable Violation'}`")
            else:
                st.success("✅ **RUN PASSED** — All observable constraints and verifier checks passed.")

            st.write(f"**User Request:** *\"{trace.get('user_request')}\"*")
            st.write(f"**Run ID:** `{active_run_id}` | **Total Observable Steps:** `{len(steps)}`")

            st.divider()

            # Diagnostic peek to highlight suspicious step
            try:
                diag = fi_service.diagnose_run(active_run_id)
                top_cand = diag.get("likely_failure_causing_step")
                flagged_step_id = top_cand.get("step_id") if top_cand else None
            except Exception:
                flagged_step_id = None

            STAGE_NAMES = {
                1: "Request Understanding",
                2: "Planning",
                3: "Information Retrieval",
                4: "Tool Selection",
                5: "Tool Execution",
                6: "Result Processing / Interpretation",
                7: "Decision / State Update",
                8: "Final Response",
            }

            # Render 8 Steps
            for step in steps:
                sid = step.get("step_id")
                stype = step.get("step_type", "unknown")
                sname = STAGE_NAMES.get(sid, stype.replace("_", " ").title())
                status = step.get("status", "unknown")
                tname = step.get("tool_name")
                lat = step.get("latency")
                is_flagged = (sid == flagged_step_id)

                card_border = "border-left: 5px solid #ef4444;" if is_flagged else ("border-left: 5px solid #10b981;" if status == "success" else "border-left: 5px solid #f59e0b;")

                header_label = f"Step {sid}: {sname}"
                if tname:
                    header_label += f" — Tool: [{tname}]"
                if is_flagged:
                    header_label += " ⚠️ [Likely Failure-Causing Step]"

                with st.expander(header_label, expanded=is_flagged):
                    col_info1, col_info2, col_info3 = st.columns([1, 1, 1])
                    with col_info1:
                        st.write(f"**Step ID:** `{sid}`")
                        st.write(f"**Type:** `{stype}`")
                    with col_info2:
                        st.write(f"**Status:** `{status.upper()}`")
                        st.write(f"**Latency:** `{lat:.3f}s`" if lat is not None else "**Latency:** `N/A`")
                    with col_info3:
                        if is_flagged:
                            st.warning(f"**Suspicion Score:** `{top_cand.get('suspicion_score', 0.0):.4f}`")

                    # Step observable inputs & outputs
                    col_in, col_out = st.columns(2)
                    with col_in:
                        st.write("**Observable Input:**")
                        st.json(step.get("input") or {})
                    with col_out:
                        st.write("**Observable Output:**")
                        st.json(step.get("output") or {})

                    # State transitions
                    if step.get("state_before") or step.get("state_after"):
                        st.write("**Observable State Snapshots:**")
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.caption("State Before:")
                            st.code(json.dumps(step.get("state_before", {}), indent=2), language="json")
                        with sc2:
                            st.caption("State After:")
                            st.code(json.dumps(step.get("state_after", {}), indent=2), language="json")

            st.divider()
            c_bot1, c_bot2 = st.columns(2)
            with c_bot1:
                st.button("🧠 Proceed to Diagnosis & Evidence ➔", type="primary", use_container_width=True, on_click=set_nav, args=("🧠 Diagnosis & Evidence",))
            with c_bot2:
                st.button("🔄 Jump to Checkpoints & Replay ➔", use_container_width=True, on_click=set_nav, args=("🔄 Checkpoint & Replay",))


# =====================================================================
# VIEW 5: DIAGNOSIS & EVIDENCE
# =====================================================================
elif nav_selection == "🧠 Diagnosis & Evidence":
    st.markdown("### 🧠 Failure Diagnosis & Evidence Extraction")
    st.caption("Ranks candidate failure-causing steps and extracts concrete observable trace facts. Never uses LLM hallucinations as evidence.")

    if not active_run_id:
        st.warning("Please select an active run to diagnose.")
    else:
        try:
            with st.spinner("Analyzing trace and computing structured evidence signals..."):
                diag_packet = fi_service.diagnose_run(active_run_id)

            top1 = diag_packet.get("likely_failure_causing_step")
            top_candidates = diag_packet.get("top_candidates", [])

            if not top_candidates:
                st.info("No failure detected in this execution. All observable verification checks passed.")
            else:
                st.markdown("#### 🎯 Ranked Candidates for Investigation")
                st.caption("Top candidate steps ranked by normalized suspicion score:")

                for rank_idx, cand in enumerate(top_candidates, start=1):
                    badge_style = "badge-suspicious" if rank_idx == 1 else "badge-checkpoint"
                    st.markdown(
                        f"""
                        <div class="step-card {'step-card-suspicious' if rank_idx == 1 else ''}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="badge-tag {badge_style}">Candidate #{rank_idx}</span>
                                    <span style="font-size: 1.1rem; font-weight: 700; margin-left: 0.5rem;">
                                        Step {cand['step_id']} — {cand['step_type'].replace('_', ' ').title()}
                                    </span>
                                    {f'<span style="color: #94a3b8; margin-left: 0.5rem;">(Tool: {cand["tool_name"]})</span>' if cand.get("tool_name") else ''}
                                </div>
                                <div>
                                    <span style="font-size: 1.1rem; font-weight: 800; color: #f59e0b;">
                                        Suspicion Score: {cand['suspicion_score']:.4f}
                                    </span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.divider()

                # Detailed Evidence Panel for #1 Candidate
                st.markdown("#### 📋 Observable Evidence Packet (Why was this step flagged?)")
                if top1:
                    signals = top1.get("signals", {})
                    st.markdown(f"**Investigating Step {top1['step_id']}** ({top1['step_type']}) with Suspicion Score **`{top1['suspicion_score']:.4f}`**:")

                    # 6 Structured Signals breakdown
                    st.markdown("##### 🔬 6 Observable Signals Breakdown:")
                    sig_cols = st.columns(3)
                    with sig_cols[0]:
                        st.metric("Tool Mismatch", f"{signals.get('tool_mismatch', 0.0):.2f}")
                        st.caption("Discrepancy with reference tool")
                        st.metric("Output Divergence", f"{signals.get('output_divergence', 0.0):.2f}")
                        st.caption("Payload & schema deviation")
                    with sig_cols[1]:
                        st.metric("State Divergence", f"{signals.get('state_divergence', 0.0):.2f}")
                        st.caption("State mutation deviation")
                        st.metric("Downstream Impact", f"{signals.get('downstream_impact', 0.0):.2f}")
                        st.caption("Error cascade in next steps")
                    with sig_cols[2]:
                        st.metric("Latency Anomaly", f"{signals.get('latency_anomaly', 0.0):.2f}")
                        st.caption("Normalized latency deviation")
                        st.metric("Error / Status Signal", f"{signals.get('error_status', 0.0):.2f}")
                        st.caption("Explicit failure or timeout")

                    # Concrete Observable Facts
                    st.markdown("##### 📝 Concrete Observable Trace Facts:")
                    facts = diag_packet.get("trace_facts", [])
                    if facts:
                        for fact in facts:
                            st.markdown(f"• {fact}")
                    else:
                        st.write("No abnormal facts recorded for this candidate.")

            st.divider()
            c_diag1, c_diag2 = st.columns(2)
            with c_diag1:
                st.button("🔄 Restore Checkpoint & Replay ➔", type="primary", use_container_width=True, on_click=set_nav, args=("🔄 Checkpoint & Replay",))
            with c_diag2:
                st.button("⚖️ Open 3-Way Trace Comparison ➔", use_container_width=True, on_click=set_nav, args=("⚖️ Trace Comparison",))

        except Exception as exc:
            st.error(f"Error computing diagnosis: {exc}")


# =====================================================================
# VIEW 6: CHECKPOINT & REPLAY
# =====================================================================
elif nav_selection == "🔄 Checkpoint & Replay":
    st.markdown("### 🔄 Checkpointing & Controlled Replay")
    st.caption("Restore immutable agent state prior to the failure, apply controlled parameter/tool overrides, and generate an alternative execution branch.")

    if not active_run_id:
        st.warning("Please select an active run to replay.")
    else:
        checkpoints = repo.list_checkpoints_for_run(active_run_id)

        if not checkpoints:
            st.warning(f"No checkpoints found for run `{active_run_id}`. Checkpoints are automatically saved during agent execution.")
        else:
            st.markdown(f"#### 💾 Available Checkpoints for `{active_run_id}` ({len(checkpoints)})")

            cp_options = [f"Step {c['step_id']} ({c['step_type']}) — ID: {c['checkpoint_id']}" for c in checkpoints]
            selected_cp_idx = st.selectbox("Select Checkpoint to Restore State From:", range(len(checkpoints)), format_func=lambda i: cp_options[i])
            chosen_cp = checkpoints[selected_cp_idx]

            # Display Checkpoint State
            with st.expander("🔍 Inspect Saved Checkpoint State", expanded=False):
                st.write(f"**Checkpoint ID:** `{chosen_cp['checkpoint_id']}`")
                st.write(f"**Step ID:** `{chosen_cp['step_id']}` | **Step Type:** `{chosen_cp['step_type']}`")
                st.write(f"**Timestamp:** `{chosen_cp.get('timestamp') or 'N/A'}`")
                st.write("**Restorable Observable State:**")
                st.json(chosen_cp.get("state_snapshot") or {})

            st.divider()

            # Controlled Replay Interface
            st.markdown("#### 🎬 Controlled Replay Configuration")
            st.info("ℹ️ **Notice:** Controlled replay executes an alternative branch starting strictly from this checkpoint. **The original execution is never modified or overwritten.**")

            c_ov1, c_ov2 = st.columns(2)
            with c_ov1:
                override_type = st.radio(
                    "Override Type to Test:",
                    ["Tool Name", "Max Budget", "Clear Failure Mode", "Combined Tool + Clean Mode"],
                    index=0,
                )
            with c_ov2:
                override_dict = {}
                if override_type == "Tool Name":
                    new_tool = st.selectbox("Force Initial Tool Execution:", ["search_products", "calculate_budget", "check_specifications"])
                    override_dict["tool_name"] = new_tool
                elif override_type == "Max Budget":
                    new_budget = st.number_input("Override Max Budget (₹):", min_value=20000, max_value=200000, value=75000, step=5000)
                    override_dict["max_budget"] = int(new_budget)
                elif override_type == "Clear Failure Mode":
                    override_dict["failure_mode"] = "none"
                    st.write("Clears injected failure to test normal downstream execution.")
                elif override_type == "Combined Tool + Clean Mode":
                    new_tool = st.selectbox("Force Initial Tool Execution:", ["search_products", "check_specifications"])
                    override_dict["tool_name"] = new_tool
                    override_dict["failure_mode"] = "none"

            st.write(f"**Override Payload:** `{override_dict}`")
            replay_btn = st.button("🚀 Execute Controlled Alternative Replay", type="primary", use_container_width=True)

            if replay_btn:
                with st.spinner("Restoring state and executing alternative replay branch..."):
                    try:
                        cm_obj = CheckpointManager(repository=repo)
                        cp_model = cm_obj.get_checkpoint(chosen_cp["checkpoint_id"])
                        engine = ReplayEngine(repository=repo)
                        replay_res = engine.replay(
                            run_id=active_run_id,
                            checkpoint=cp_model,
                            override=override_dict if override_dict else None,
                        )

                        # Auto-create checkpoints for replay run
                        rep_trace = repo.get_run_trace(replay_res.replay_run_id)
                        if rep_trace:
                            cm_obj.create_checkpoints_for_run(
                                run_id=replay_res.replay_run_id,
                                user_request=rep_trace.get("user_request", ""),
                                steps=rep_trace.get("steps", []),
                            )

                        st.session_state["latest_replay_id"] = replay_res.replay_run_id
                        st.success(f"🎉 Replay execution finished! Alternative Run ID: **`{replay_res.replay_run_id}`** ({replay_res.status.upper()})")

                        # Replay Summary
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            st.write(f"**Original Run:** `{replay_res.original_run_id}`")
                            st.write(f"**Parent Checkpoint:** `{replay_res.checkpoint_id}`")
                            st.write(f"**Override Applied:** `{replay_res.override_applied}`")
                        with r_col2:
                            st.write(f"**Alternative Status:** `{replay_res.status.upper()}`")
                            st.write(f"**Final Response:** {replay_res.final_response}")

                    except Exception as exc:
                        st.error(f"Replay execution failed: {exc}")

            st.divider()

            # Existing Replays list
            replays = repo.list_replays_for_run(active_run_id)
            if replays:
                st.markdown(f"#### 📜 Alternative Replays of `{active_run_id}` ({len(replays)})")
                for rr in replays:
                    with st.container(border=True):
                        st.write(f"• **`{rr['run_id']}`** — Status: `{rr['status'].upper()}` — Checkpoint: `{rr.get('replay_metadata', {}).get('checkpoint_id', 'N/A')}`")


# =====================================================================
# VIEW 7: TRACE COMPARISON
# =====================================================================
elif nav_selection == "⚖️ Trace Comparison":
    st.markdown("### ⚖️ 3-Way Trace Comparison & Recovery Verification")
    st.caption("Aligns execution steps side-by-side across Original (Failed), Alternative (Replay), and Reference (Verified Successful) runs to verify outcome recovery.")

    all_db_runs = repo.list_runs(limit=100)
    failed_runs = [r["run_id"] for r in all_db_runs if r.get("status") == "failed"]
    all_rids = [r["run_id"] for r in all_db_runs]

    if not all_rids:
        st.info("No runs available in database.")
    else:
        c_sel1, c_sel2, c_sel3 = st.columns(3)
        with c_sel1:
            default_orig = active_run_id if active_run_id in all_rids else (failed_runs[0] if failed_runs else all_rids[0])
            orig_choice = st.selectbox("1. Original Run (Failed):", all_rids, index=all_rids.index(default_orig))
        with c_sel2:
            # Replays of original run
            replays_of_orig = [r["run_id"] for r in all_db_runs if r.get("parent_run_id") == orig_choice]
            alt_candidates = replays_of_orig if replays_of_orig else [r for r in all_rids if r != orig_choice]
            alt_choice = st.selectbox("2. Alternative Run (Replay):", alt_candidates if alt_candidates else all_rids)
        with c_sel3:
            succ_runs = [r["run_id"] for r in all_db_runs if r.get("status") == "success"]
            ref_opts = ["(Auto-Select Compatible Reference)"] + succ_runs
            ref_choice = st.selectbox("3. Reference Run (Success):", ref_opts)

        compare_btn = st.button("⚖️ Compare 3 Executions Side-by-Side", type="primary", use_container_width=True)

        if compare_btn and orig_choice and alt_choice:
            ref_id_param = None if ref_choice.startswith("(") else ref_choice
            try:
                with st.spinner("Aligning execution steps and running independent recovery verification..."):
                    comp_res = fi_service.compare_traces(
                        original_run_id=orig_choice,
                        alternative_run_id=alt_choice,
                        reference_run_id=ref_id_param,
                    )

                # Prominent Recovery Badge
                st.markdown("#### 🔬 Independent Verifier Recovery Result")
                cv1, cv2, cv3 = st.columns([2, 2, 2])
                with cv1:
                    v_orig = comp_res.get("original_verifier", {})
                    if v_orig.get("passed"):
                        st.success("Original Verifier: ✅ PASSED")
                    else:
                        st.error(f"Original Verifier: ❌ {v_orig.get('reason', 'Failed')}")
                with cv2:
                    v_alt = comp_res.get("alternative_verifier", {})
                    if v_alt.get("passed"):
                        st.success("Alternative Verifier: ✅ PASSED")
                    else:
                        st.error(f"Alternative Verifier: ❌ {v_alt.get('reason', 'Failed')}")
                with cv3:
                    if comp_res.get("recovered"):
                        st.balloons()
                        st.success("🎉 **VERIFIED RECOVERED**\n\nOutcome successfully recovered!")
                    else:
                        st.warning("⚠️ **RECOVERY NOT CONFIRMED**\n\nRequirements still violated.")

                # Downstream Propagation Effects
                if comp_res.get("downstream_effects"):
                    st.markdown("#### 🌊 Downstream Observable Effects")
                    for eff in comp_res["downstream_effects"]:
                        st.markdown(f"• {eff}")

                # 3-Way Aligned Steps Table
                st.markdown("#### 📐 Aligned 8-Step Execution Table")
                steps_rows = []
                for s in comp_res.get("steps", []):
                    ch_badge = "🔄 CHANGED" if s.get("is_changed") else "Identical"
                    steps_rows.append({
                        "Step ID": s.get("step_id"),
                        "Stage": s.get("step_type").replace("_", " ").title(),
                        "Original Execution": f"{s.get('original_tool') or s.get('step_type')} ({s.get('original_status')})",
                        "Alternative Execution": f"{s.get('alternative_tool') or s.get('step_type')} ({s.get('alternative_status')})",
                        "Reference Execution": f"{s.get('reference_tool') or s.get('step_type')} ({s.get('reference_status')})",
                        "Change Status": ch_badge,
                        "Observable Delta": s.get("change_description"),
                    })

                st.dataframe(pd.DataFrame(steps_rows), use_container_width=True, hide_index=True)

                st.divider()
                st.button("📈 Open Benchmark Evaluation ➔", type="primary", use_container_width=True, on_click=set_nav, args=("📈 Evaluation & Metrics",))

            except Exception as exc:
                st.error(f"Trace comparison failed: {exc}")


# =====================================================================
# VIEW 8: EVALUATION
# =====================================================================
elif nav_selection == "📈 Evaluation & Metrics":
    st.markdown("### 📈 Failure Localization Evaluation & Benchmarks")
    st.caption("Real measured evaluation performance across leakage-safe splits. Random Baseline vs Rule-Based vs Random Forest.")

    # Benchmark Controls
    with st.expander("⚙️ Benchmark Generation & Evaluation Controls", expanded=False):
        c_b1, c_b2, c_b3 = st.columns([2, 1, 1])
        with c_b1:
            st.write("**Controlled Evaluation Dataset Generation**")
            st.caption("Generates labeled prototype-scale runs across scenarios and strictly enforces train/val/test/held-out split isolation.")
        with c_b2:
            num_success = st.number_input("Target Success Runs:", min_value=10, max_value=60, value=20, step=5)
            num_failure = st.number_input("Target Failure Runs:", min_value=20, max_value=120, value=40, step=10)
        with c_b3:
            run_eval_btn = st.button("⚡ Run / Refresh Benchmark", type="primary", use_container_width=True)

    if run_eval_btn:
        with st.spinner("Generating controlled runs and computing leakage-safe benchmark evaluation..."):
            try:
                eval_res = fi_service.get_or_run_benchmark(
                    force_regenerate=True,
                    target_success_count=int(num_success),
                    target_failure_count=int(num_failure),
                )
                st.session_state["cached_eval"] = eval_res
                st.success("✅ Benchmark evaluation complete!")
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

    eval_data = st.session_state.get("cached_eval")
    if not eval_data:
        eval_data = fi_service.get_evaluation_summary()

    if eval_data.get("status") == "evaluation_not_ready":
        st.info(f"ℹ️ {eval_data.get('reason', 'Evaluation benchmark has not been run yet. Click \"Run / Refresh Benchmark\" above.')}")
    elif eval_data.get("summary"):
        summ = eval_data["summary"]
        rand_base = summ.get("random_baseline", {})
        rule_base = summ.get("rule_based", {})
        rf_base = summ.get("random_forest", {})

        st.markdown("#### 🏆 Overall Localization Performance Summary")
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.markdown("##### 🎲 Random Baseline (1/N)")
            st.metric("Top-1 Accuracy", f"{rand_base.get('top1_accuracy', 0.0) * 100:.1f}%")
            st.metric("Top-3 Accuracy", f"{rand_base.get('top3_accuracy', 0.0) * 100:.1f}%")
            st.metric("MRR", f"{rand_base.get('mrr', 0.0):.3f}")

        with m_c2:
            st.markdown("##### 📐 Improved Rule-Based Localizer")
            st.metric(
                "Top-1 Accuracy",
                f"{rule_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                delta=f"{(rule_base.get('top1_accuracy', 0.0) - rand_base.get('top1_accuracy', 0.0)) * 100:+.1f}% vs Random",
            )
            st.metric(
                "Top-3 Accuracy",
                f"{rule_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                delta=f"{(rule_base.get('top3_accuracy', 0.0) - rand_base.get('top3_accuracy', 0.0)) * 100:+.1f}% vs Random",
            )
            st.metric("MRR", f"{rule_base.get('mrr', 0.0):.3f}")

        with m_c3:
            st.markdown("##### 🌲 Random Forest Classifier")
            st.metric(
                "Top-1 Accuracy",
                f"{rf_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                delta=f"{(rf_base.get('top1_accuracy', 0.0) - rand_base.get('top1_accuracy', 0.0)) * 100:+.1f}% vs Random",
            )
            st.metric(
                "Top-3 Accuracy",
                f"{rf_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                delta=f"{(rf_base.get('top3_accuracy', 0.0) - rand_base.get('top3_accuracy', 0.0)) * 100:+.1f}% vs Random",
            )
            st.metric("MRR", f"{rf_base.get('mrr', 0.0):.3f}")

        st.divider()

        # Method Comparison Table
        st.markdown("#### 📊 Comparative Model Table")
        comp_df = pd.DataFrame([
            {
                "Method": "Random Baseline (Theoretical)",
                "Top-1 Accuracy": f"{rand_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rand_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rand_base.get('mrr', 0.0):.3f}",
                "Basis": "1 / N candidate steps",
            },
            {
                "Method": "Improved Rule-Based Localizer",
                "Top-1 Accuracy": f"{rule_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rule_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rule_base.get('mrr', 0.0):.3f}",
                "Basis": "6 Structured Signals + Validation-Tuned Weights",
            },
            {
                "Method": "Random Forest Classifier",
                "Top-1 Accuracy": f"{rf_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rf_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rf_base.get('mrr', 0.0):.3f}",
                "Basis": "Scikit-Learn Balanced Forest on 6 Signals",
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()

        # Known vs Held-Out Categories
        kvh = eval_data.get("known_vs_held_out")
        if kvh:
            st.markdown("#### 🎯 Generalization: Known vs Held-Out Failure Categories")
            k_col, h_col = st.columns(2)
            with k_col:
                k_data = kvh.get("known_categories", {})
                st.markdown(f"**Known Categories ({k_data.get('total_runs', 0)} test runs):**")
                k_rule = k_data.get("rule_based", {})
                k_rf = k_data.get("random_forest", {})
                st.write(f"• **Rule-Based:** Top-1: `{k_rule.get('top1', 0.0)*100:.1f}%` | Top-3: `{k_rule.get('top3', 0.0)*100:.1f}%` | MRR: `{k_rule.get('mrr', 0.0):.3f}`")
                st.write(f"• **Random Forest:** Top-1: `{k_rf.get('top1', 0.0)*100:.1f}%` | Top-3: `{k_rf.get('top3', 0.0)*100:.1f}%` | MRR: `{k_rf.get('mrr', 0.0):.3f}`")

            with h_col:
                h_data = kvh.get("held_out_category", {})
                h_cat = h_data.get("category", "unexpected_output")
                st.markdown(f"**Held-Out Category: `{h_cat}` ({h_data.get('total_runs', 0)} test runs):**")
                h_rule = h_data.get("rule_based", {})
                h_rf = h_data.get("random_forest", {})
                st.write(f"• **Rule-Based:** Top-1: `{h_rule.get('top1', 0.0)*100:.1f}%` | Top-3: `{h_rule.get('top3', 0.0)*100:.1f}%` | MRR: `{h_rule.get('mrr', 0.0):.3f}`")
                st.write(f"• **Random Forest:** Top-1: `{h_rf.get('top1', 0.0)*100:.1f}%` | Top-3: `{h_rf.get('top3', 0.0)*100:.1f}%` | MRR: `{h_rf.get('mrr', 0.0):.3f}`")

        # Category Breakdown
        cats = eval_data.get("by_category", [])
        if cats:
            st.divider()
            st.markdown("#### 📋 Category Breakdown Table")
            c_rows = []
            for c in cats:
                sample_chk = "⚠️ Small Sample (<5)" if c.get("small_sample_warning") else "✅ Normal"
                c_rows.append({
                    "Failure Category": c.get("category"),
                    "Type": "🔒 Held-Out" if c.get("is_held_out") else "Known",
                    "Test Runs": c.get("total_test_runs"),
                    "Sample Verification": sample_chk,
                    "Rule Top-1": f"{c.get('rule_based_top1', 0.0)*100:.1f}%",
                    "Rule Top-3": f"{c.get('rule_based_top3', 0.0)*100:.1f}%",
                    "RF Top-1": f"{c.get('random_forest_top1', 0.0)*100:.1f}%",
                    "RF Top-3": f"{c.get('random_forest_top3', 0.0)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)

        # Replay Recovery Stats
        rep_rec = eval_data.get("replay_recovery", {})
        st.divider()
        st.markdown("#### 🔁 Replay Recovery Verification Statistics")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Branches Attempted", rep_rec.get("branches_attempted", 0))
        with r2:
            st.metric("Recovered", f"✅ {rep_rec.get('recovered', 0)}")
        with r3:
            st.metric("Not Recovered", f"❌ {rep_rec.get('not_recovered', 0)}")
        with r4:
            st.metric("Recovery Rate", f"{rep_rec.get('recovery_rate', 0.0)*100:.1f}%")


# =====================================================================
# VIEW 9: CATALOGUE & TOOLS
# =====================================================================
elif nav_selection == "📦 Catalogue & Tools":
    st.markdown("### 📦 Product Catalogue & Machine Learning Recommender")
    st.caption("Inspect the underlying device catalogue and Scikit-Learn NearestNeighbors model used by the controlled AI agent.")

    tab_cat, tab_ml_tool = st.tabs(["📦 Catalogue Browser", "🧠 Nearest Neighbors Similarity"])

    products = load_products()
    nn_model, scaler, _ = build_ml_recommender(products)

    with tab_cat:
        st.markdown("#### Available Hardware Devices (`data/products.json`)")
        if products:
            st.dataframe(pd.DataFrame(products), use_container_width=True)
        else:
            st.info("Catalogue is empty.")

    with tab_ml_tool:
        st.markdown("#### Scikit-Learn Cosine Nearest Neighbors Model")
        if products:
            pnames = [p["name"] for p in products]
            sel_device = st.selectbox("Select Device to Find Similar Hardware:", pnames)
            sim_matches = get_similar_laptops(sel_device, products, nn_model, scaler)
            for m in sim_matches:
                p = m["product"]
                with st.container(border=True):
                    sc1, sc2, sc3 = st.columns([2, 1, 1])
                    with sc1:
                        st.markdown(f"**{p['name']}**")
                        st.caption(f"{p['processor']} | {p['ram_gb']}GB RAM | {p['storage_gb']}GB SSD")
                    with sc2:
                        st.markdown(f"**₹{p['price']:,}**")
                    with sc3:
                        st.metric("Similarity", f"{m['similarity_score']}%")
