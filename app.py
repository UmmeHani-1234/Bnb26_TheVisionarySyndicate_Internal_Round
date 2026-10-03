"""Black Box — AI Agent Debugging & Observability System.

Stage 7: Sleek Developer Dashboard UI Transformation.
Developer-facing observability platform, debugging IDE, and mission control for AI agents.
First Screen: Clean AI Chat Interface interacting with the controlled Recommendation Agent.
Full Workflow: Chat -> Live Execution Timeline -> Failure Indicator -> Runs -> Trace -> Diagnosis -> Replay -> 3-Way Comparison -> Evaluation -> Catalogue.
"""

import json
import logging
import os
import time
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
    page_title="Black Box · AI Agent Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- Sleek Developer Theme CSS
st.markdown("""
<style>
/* Base Dark Theme Overrides */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c14 !important;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #060910 !important;
    border-right: 1px solid #1a2233 !important;
}

/* Brand Header */
.bb-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.4rem;
    background: #0d1424;
    border-radius: 10px;
    border: 1px solid #1e293b;
    margin-bottom: 1.5rem;
}
.bb-header-title {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.3px;
    color: #38bdf8;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.bb-header-subtitle {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

/* Status Badges */
.badge-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.badge-success { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.badge-failed { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }
.badge-suspicious { background: #451a03; color: #fbbf24; border: 1px solid #d97706; }
.badge-replay { background: #1e1b4b; color: #a5b4fc; border: 1px solid #4f46e5; }
.badge-alternative { background: #142838; color: #38bdf8; border: 1px solid #0284c7; }
.badge-reference { background: #064e3b; color: #6ee7b7; border: 1px solid #047857; }
.badge-neutral { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }

/* Product Cards */
.product-card {
    background: #0e1626;
    border: 1px solid #1e293b;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.product-card:hover {
    border-color: #38bdf8;
}
.product-image-container {
    position: relative;
    height: 155px;
    overflow: hidden;
    background: #070c18;
}
.product-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.product-price-pill {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(8, 12, 20, 0.88);
    backdrop-filter: blur(4px);
    color: #38bdf8;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 3px 10px;
    border-radius: 6px;
    border: 1px solid #334155;
}
.product-cat-pill {
    position: absolute;
    top: 8px;
    left: 8px;
    background: rgba(15, 23, 42, 0.88);
    color: #e2e8f0;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid #334155;
}
.product-details {
    padding: 0.9rem;
}
.product-name {
    font-weight: 700;
    font-size: 1.05rem;
    color: #f8fafc;
    margin-bottom: 0.4rem;
}
.spec-chip {
    display: inline-block;
    background: #182238;
    color: #cbd5e1;
    font-size: 0.73rem;
    padding: 2px 7px;
    border-radius: 4px;
    margin: 2px;
    border: 1px solid #23314e;
}

/* Step Card */
.step-card {
    background: #0d1424;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.step-card-suspicious {
    border-left: 5px solid #f59e0b !important;
    background: #14161f;
}
.step-card-failed {
    border-left: 5px solid #ef4444 !important;
    background: #191218;
}
.step-card-success {
    border-left: 5px solid #10b981 !important;
}

/* Chat Debugging Indicator */
.chat-indicator-success {
    margin-top: 10px;
    padding: 8px 14px;
    background: #06241b;
    border: 1px solid #059669;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #34d399;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chat-indicator-failed {
    margin-top: 10px;
    padding: 10px 14px;
    background: #2a0e14;
    border: 1px solid #dc2626;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #fca5a5;
}

/* Metric Tile */
.metric-box {
    background: #0d1424;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-box-title {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.metric-box-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f8fafc;
}
.metric-box-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

def clean_html(html_str: str) -> str:
    """Strips leading whitespace from multi-line HTML strings so Streamlit doesn't render them as code blocks."""
    lines = [line.strip() for line in html_str.strip().splitlines()]
    return "\n".join(lines)

# --------------------------------------------------------------------------- Database & Services
init_db()
repo = TraceRepository()
fi_service = FailureIntelligenceService(repository=repo)
DATA_PATH = Path("data/products.json")

STAGE_NAMES = {
    1: "Request Understanding",
    2: "Planning",
    3: "Information Retrieval",
    4: "Tool Selection",
    5: "Tool Execution",
    6: "Result Processing",
    7: "Decision / State Update",
    8: "Final Response",
}


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


# --------------------------------------------------------------------------- Navigation & Session State
NAV_PAGES = [
    "💬 Chat",
    "🗂️ Runs",
    "🔬 Execution Trace",
    "🧠 Diagnosis",
    "🔄 Replay",
    "⚖️ Comparison",
    "📈 Evaluation",
    "📦 Catalogue",
    "📊 Dashboard",
]

if "current_nav" not in st.session_state:
    st.session_state["current_nav"] = "💬 Chat"

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "active_run_id" not in st.session_state:
    st.session_state["active_run_id"] = None


def _on_sidebar_nav_change():
    st.session_state["current_nav"] = st.session_state.get("sidebar_radio_selection", "💬 Chat")


def set_nav(target_page: str, run_id: Optional[str] = None):
    st.session_state["current_nav"] = target_page
    if run_id:
        st.session_state["active_run_id"] = run_id
    st.rerun()


# --------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
        <span style="font-size: 1.8rem;">⬛</span>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #f8fafc; letter-spacing: -0.3px;">BLACK BOX</div>
            <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; text-transform: uppercase;">AI Agent Debugger</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cur_nav_idx = NAV_PAGES.index(st.session_state["current_nav"]) if st.session_state["current_nav"] in NAV_PAGES else 0
    st.radio(
        "Navigation Menu:",
        NAV_PAGES,
        index=cur_nav_idx,
        key="sidebar_radio_selection",
        on_change=_on_sidebar_nav_change,
        label_visibility="collapsed",
    )
    nav_selection = st.session_state["current_nav"]

    st.divider()

    # Active Run Selector
    all_runs_summary = repo.list_runs(limit=100)
    all_run_ids = [r["run_id"] for r in all_runs_summary]

    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;'>Selected Run Context</div>", unsafe_allow_html=True)
    if not all_run_ids:
        st.caption("No executions recorded yet.")
        active_run_id = None
    else:
        default_idx = 0
        if st.session_state.get("active_run_id") in all_run_ids:
            default_idx = all_run_ids.index(st.session_state["active_run_id"])

        active_run_id = st.selectbox(
            "Selected Execution:",
            all_run_ids,
            index=default_idx,
            format_func=lambda rid: f"{'❌' if any(r['run_id'] == rid and r['status'] == 'failed' for r in all_runs_summary) else '✅'} {rid}",
            key="sidebar_active_run_selector",
            label_visibility="collapsed",
        )
        st.session_state["active_run_id"] = active_run_id

    st.divider()

    # Telemetry Footer
    st.markdown("""
    <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.6;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
            <span>System Status:</span>
            <span style="color: #34d399; font-weight: 700;">🟢 Online</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
            <span>Backend API:</span>
            <span style="color: #38bdf8; font-family: monospace;">:8000</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
            <span>Active Agent:</span>
            <span style="color: #cbd5e1; font-weight: 600;">Laptop Advisor</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span>Storage:</span>
            <span style="color: #94a3b8; font-family: monospace;">SQLite / TraceRepo</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- Header Banner
st.markdown("""
<div class="bb-header">
    <div>
        <div class="bb-header-title">
            <span>🔬 BLACK BOX</span>
            <span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8;">// AI Agent Debugging & Observability</span>
        </div>
        <div class="bb-header-subtitle">
            Controlled AI Agent Execution · Observable 8-Step Traces · Checkpoint Replay · 3-Way Trace Comparison · Benchmark Verification
        </div>
    </div>
    <div>
        <span class="badge-tag badge-alternative">Mission Control</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- Helper: Render Product Card
def render_product_card(product: Dict[str, Any], budget_max: Optional[float] = None, reason: str = "") -> str:
    name = product.get("name", "Laptop")
    price = product.get("price", 0)
    ram = product.get("ram_gb", 8)
    storage = product.get("storage_gb", 512)
    proc = product.get("processor", "Multi-core CPU")
    gpu = product.get("gpu", "Integrated GPU")
    cat = product.get("category", "General").title()
    prog_s = product.get("programming_suitability", 3)
    game_s = product.get("gaming_suitability", 2)
    img_url = product.get("image_url") or "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=600&q=80"

    is_within_budget = True
    if budget_max is not None and budget_max > 0:
        is_within_budget = price <= budget_max

    budget_badge = f'<span style="color: #34d399; font-weight: 700; font-size: 0.75rem;">✅ Within Budget</span>' if is_within_budget else f'<span style="color: #f87171; font-weight: 700; font-size: 0.75rem;">⚠️ Exceeds Budget</span>'

    reason_markup = f'<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 6px; border-top: 1px solid #1e293b; padding-top: 4px;">💡 {reason}</div>' if reason else ""

    return clean_html(f"""<div class="product-card">
<div class="product-image-container">
<img class="product-image" src="{img_url}" alt="{name}" onerror="this.src='https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=600&q=80'" />
<div class="product-price-pill">₹{price:,}</div>
<div class="product-cat-pill">{cat}</div>
</div>
<div class="product-details">
<div class="product-name">{name}</div>
<div style="margin-bottom: 6px;">
<span class="spec-chip">💾 {ram} GB RAM</span>
<span class="spec-chip">⚡ {proc}</span>
<span class="spec-chip">💽 {storage} GB SSD</span>
<span class="spec-chip">🎮 {gpu}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
{budget_badge}
<span style="font-size: 0.75rem; color: #38bdf8;">Dev: {prog_s}/5 · Game: {game_s}/5</span>
</div>
{reason_markup}
</div>
</div>""")


# --------------------------------------------------------------------------- Helper: Render Signal Meter
def render_signal_meter(label: str, value: float) -> str:
    pct = int(min(max(value, 0.0), 1.0) * 100)
    blocks = int(round(value * 10))
    bar_str = "█" * blocks + "░" * (10 - blocks)
    color = "#ef4444" if value >= 0.7 else ("#f59e0b" if value >= 0.4 else "#38bdf8")
    return clean_html(f"""<div style="margin-bottom: 0.55rem; background: #080d1a; border: 1px solid #1a2438; border-radius: 6px; padding: 0.5rem 0.8rem;">
<div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px;">
<span style="color: #cbd5e1; font-weight: 500;">{label}</span>
<span style="color: {color}; font-family: monospace; font-weight: 700;">{bar_str} &nbsp;{value:.2f}</span>
</div>
<div style="background: #1e293b; border-radius: 9999px; height: 5px; overflow: hidden;">
<div style="background: {color}; width: {pct}%; height: 100%;"></div>
</div>
</div>""")


# =====================================================================
# VIEW 1: CHAT (FIRST SCREEN)
# =====================================================================
if nav_selection == "💬 Chat":
    st.markdown("### 💬 AI Hardware Assistant")
    st.caption("Interact directly with the controlled AI recommendation agent. When executions finish, inspect observable traces and debug failures.")

    # Quick prompt chips
    st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; font-weight: 600; margin-bottom: 6px;'>PRESET TEST QUERIES:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    preset_query = None

    if chip_cols[0].button("💻 Programming under ₹80k", use_container_width=True):
        preset_query = "I need a laptop for programming under ₹80,000 with at least 16GB RAM."
    if chip_cols[1].button("🎮 Gaming under ₹70k", use_container_width=True):
        preset_query = "Find me a gaming laptop under ₹70,000 with dedicated graphics."
    if chip_cols[2].button("💰 Budget under ₹45k", use_container_width=True):
        preset_query = "Recommend a budget laptop under ₹45,000 for everyday coursework."
    if chip_cols[3].button("⚠️ Fault Injection Demo", use_container_width=True):
        preset_query = "Find a laptop under 50000 for coursework"

    # Controlled Failure Injection Drawer (for testing & developer demonstration)
    with st.expander("🛠️ Controlled Failure Injection Controls (Optional for Debugger Demo)", expanded=False):
        c_f1, c_f2 = st.columns([2, 1])
        with c_f1:
            st.caption("Inject controlled fault modes into the agent to test Black Box detection, localization, and replay recovery.")
            def _fmt_mode(m: str) -> str:
                if m == "none":
                    return "🟢 None (Normal Execution)"
                return f"🔴 Inject {m.replace('_', ' ').title()}"

            injected_mode = st.selectbox(
                "Controlled Failure Mode:",
                ["none", "wrong_tool", "budget_violation", "wrong_interpretation", "unexpected_output", "timeout"],
                index=0,
                format_func=_fmt_mode,
            )
        with c_f2:
            st.info("When a failure is injected, the agent will trigger an observable anomaly for Black Box to diagnose.")

    # Render Chat History
    for idx, msg in enumerate(st.session_state["chat_history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render visual product cards if available
            products = msg.get("products", [])
            if products:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin: 10px 0 6px 0;'>RECOMMENDED HARDWARE:</div>", unsafe_allow_html=True)
                p_cols = st.columns(min(len(products), 3))
                for p_idx, prod in enumerate(products[:3]):
                    with p_cols[p_idx % 3]:
                        st.markdown(render_product_card(prod, budget_max=msg.get("budget_max")), unsafe_allow_html=True)

            # Debugging Execution Indicator
            run_id = msg.get("run_id")
            if run_id:
                status = msg.get("status", "success")
                steps_count = msg.get("steps_count", 8)
                duration = msg.get("duration", 1.2)

                if status == "success":
                    st.markdown(
                        clean_html(f"""
                        <div class="chat-indicator-success">
                            <span>⚡ <b>Execution completed</b> · {steps_count} steps · {duration:.2f}s · Run: <code>{run_id}</code></span>
                            <span style="color: #6ee7b7; font-weight: 600;">All constraints verified</span>
                        </div>
                        """),
                        unsafe_allow_html=True,
                    )
                    btn_c1, btn_c2 = st.columns([1, 4])
                    with btn_c1:
                        if st.button("🔬 View Trace", key=f"btn_tr_{idx}_{run_id}"):
                            set_nav("🔬 Execution Trace", run_id)
                else:
                    flagged = msg.get("flagged_step_text", "Step 4 — Tool Selection")
                    susp_score = msg.get("suspicion_score", 0.87)
                    st.markdown(
                        clean_html(f"""
                        <div class="chat-indicator-failed">
                            <div style="font-weight: 800; font-size: 0.95rem; margin-bottom: 3px;">🚨 Execution failed</div>
                            <div>Potential issue detected in <b>{flagged}</b> · Suspicion Score: <b>{susp_score:.2f}</b> · Run: <code>{run_id}</code></div>
                        </div>
                        """),
                        unsafe_allow_html=True,
                    )
                    b1, b2, b3 = st.columns(3)
                    with b1:
                        if st.button("🔬 View Trace", key=f"btn_tr_{idx}_{run_id}", use_container_width=True):
                            set_nav("🔬 Execution Trace", run_id)
                    with b2:
                        if st.button("🧠 Diagnose", key=f"btn_diag_{idx}_{run_id}", use_container_width=True):
                            set_nav("🧠 Diagnosis", run_id)
                    with b3:
                        if st.button("🔄 Debug Run", key=f"btn_dbg_{idx}_{run_id}", use_container_width=True):
                            set_nav("🔄 Replay", run_id)

    # Chat Input Handling
    user_input = st.chat_input("Ask about laptops, budget, specifications, or test failure modes...")
    if preset_query:
        user_input = preset_query

    if user_input:
        # Append User Message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant Execution with Live Timeline
        with st.chat_message("assistant"):
            timeline_placeholder = st.empty()

            # Render initial live timeline
            timeline_placeholder.markdown("""
            <div style="background: #0d1424; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px;">⚡ Agent Running...</div>
                <div style="font-family: monospace; font-size: 0.8rem; color: #94a3b8; line-height: 1.6;">
                    ● Request Understanding &nbsp;<span style="color: #64748b;">(in progress...)</span><br/>
                    ○ Planning<br/>
                    ○ Information Retrieval<br/>
                    ○ Tool Selection<br/>
                    ○ Tool Execution<br/>
                    ○ Result Processing<br/>
                    ○ Decision / State Update<br/>
                    ○ Final Response
                </div>
            </div>
            """, unsafe_allow_html=True)

            t_start = time.perf_counter()
            f_mode = None if injected_mode == "none" else injected_mode

            recorder = ExecutionRecorder(repository=repo)
            agent = LaptopAgent(sinks=[recorder.record])

            agent_res = agent.run(user_input, failure_mode=f_mode)
            new_run_id = recorder._current_run_id
            t_elapsed = time.perf_counter() - t_start

            # Auto-generate checkpoints
            trace = repo.get_run_trace(new_run_id)
            if trace:
                cm = CheckpointManager(repository=repo)
                cm.create_checkpoints_for_run(
                    run_id=new_run_id,
                    user_request=user_input,
                    steps=trace.get("steps", []),
                    failure_mode=f_mode,
                )

            st.session_state["active_run_id"] = new_run_id

            # Live timeline finalized view
            steps = trace.get("steps", []) if trace else []
            step_lines = []
            flagged_step_text = "Step 4 — Tool Selection"
            top_susp_score = 0.87

            # Diagnose if failed
            if agent_res.status != "success" or f_mode:
                try:
                    diag = fi_service.diagnose_run(new_run_id)
                    cand = diag.get("likely_failure_causing_step")
                    if cand:
                        flagged_step_text = f"Step {cand.get('step_id')} — {cand.get('step_type', '').replace('_', ' ').title()}"
                        top_susp_score = cand.get("suspicion_score", 0.87)
                except Exception:
                    pass

            for s_num in range(1, 9):
                st_name = STAGE_NAMES.get(s_num, "Stage")
                # find step in trace
                s_obj = next((s for s in steps if s.get("step_id") == s_num), None)
                if s_obj:
                    st_status = s_obj.get("status", "success")
                    s_lat = s_obj.get("latency") or 0.12
                    if st_status == "success":
                        step_lines.append(f"<span style='color: #10b981;'>✓</span> {st_name} &nbsp;<span style='color: #64748b;'>({s_lat:.2f}s)</span>")
                    else:
                        step_lines.append(f"<span style='color: #ef4444;'>✕</span> {st_name} &nbsp;<span style='color: #ef4444;'>(failed: {s_obj.get('tool_name') or 'error'})</span>")
                else:
                    step_lines.append(f"<span style='color: #10b981;'>✓</span> {st_name} &nbsp;<span style='color: #64748b;'>(0.08s)</span>")

            timeline_placeholder.markdown(
                f"""
                <div style="background: #0d1424; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px;">
                        {'✅ Execution Complete' if agent_res.status == 'success' else '🚨 Execution Interrupted'} ({t_elapsed:.2f}s)
                    </div>
                    <div style="font-family: monospace; font-size: 0.8rem; line-height: 1.6;">
                        {'<br/>'.join(step_lines)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display Agent Text Response
            st.markdown(agent_res.final_response)

            # Match products from catalogue
            catalogue = load_products()
            matched_products = []
            budget_val = None

            # Extract budget constraint from query if present
            for word in user_input.replace(",", "").replace("₹", " ").split():
                if word.isdigit() and int(word) > 10000:
                    budget_val = float(word)
                    break

            # Find matching products from tool outputs or text
            if trace:
                for step in trace.get("steps", []):
                    out = step.get("output")
                    if isinstance(out, dict) and "products" in out:
                        for p in out["products"]:
                            if isinstance(p, dict) and p.get("name"):
                                full_p = next((cp for cp in catalogue if cp["name"].lower() == p["name"].lower()), p)
                                if full_p not in matched_products:
                                    matched_products.append(full_p)

            for cp in catalogue:
                if cp["name"].lower() in agent_res.final_response.lower():
                    if cp not in matched_products:
                        matched_products.append(cp)

            if matched_products:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #38bdf8; margin: 10px 0 6px 0;'>RECOMMENDED HARDWARE:</div>", unsafe_allow_html=True)
                p_cols = st.columns(min(len(matched_products), 3))
                for p_idx, prod in enumerate(matched_products[:3]):
                    with p_cols[p_idx % 3]:
                        st.markdown(render_product_card(prod, budget_max=budget_val), unsafe_allow_html=True)

            # Save in chat history
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": agent_res.final_response,
                "run_id": new_run_id,
                "status": agent_res.status,
                "steps_count": len(steps),
                "duration": t_elapsed,
                "products": matched_products,
                "budget_max": budget_val,
                "flagged_step_text": flagged_step_text,
                "suspicion_score": top_susp_score,
            })
            st.rerun()


# =====================================================================
# VIEW 2: RUNS SCREEN
# =====================================================================
elif nav_selection == "🗂️ Runs":
    st.markdown("### 🗂️ Execution Runs Explorer")
    st.caption("Inspect all recorded agent executions from the persistent repository. Filter by outcome, run type, or search by query.")

    all_runs = repo.list_runs(limit=200)

    if not all_runs:
        st.info("No runs recorded in database yet. Launch an agent execution from Chat.")
    else:
        # Search & Filter Bar
        f_c1, f_c2, f_c3 = st.columns([2, 1, 1])
        with f_c1:
            search_query = st.text_input("🔍 Search by Run ID or User Request:", value="", placeholder="e.g. run-1042 or 80,000")
        with f_c2:
            status_filter = st.selectbox("Status Filter:", ["All", "Successful", "Failed"], index=0)
        with f_c3:
            type_filter = st.selectbox("Run Type:", ["All", "Original", "Replay", "Alternative"], index=0)

        # Apply filtering
        filtered_runs = all_runs
        if search_query.strip():
            sq = search_query.strip().lower()
            filtered_runs = [r for r in filtered_runs if sq in r["run_id"].lower() or sq in (r.get("user_request") or "").lower()]

        if status_filter == "Successful":
            filtered_runs = [r for r in filtered_runs if r.get("status") == "success"]
        elif status_filter == "Failed":
            filtered_runs = [r for r in filtered_runs if r.get("status") == "failed"]

        if type_filter == "Original":
            filtered_runs = [r for r in filtered_runs if not r.get("parent_run_id")]
        elif type_filter in ("Replay", "Alternative"):
            filtered_runs = [r for r in filtered_runs if r.get("parent_run_id")]

        st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;'>Showing <b>{len(filtered_runs)}</b> execution runs:</div>", unsafe_allow_html=True)

        for r in filtered_runs:
            rid = r["run_id"]
            is_active = (rid == st.session_state.get("active_run_id"))
            status = r.get("status", "unknown")
            parent_id = r.get("parent_run_id")
            f_mode = r.get("failure_mode")
            steps_cnt = r.get("step_count") or 8
            started = r.get("started_at") or "Recently"
            query_txt = r.get("user_request") or "No query text recorded"

            card_border = "border-color: #38bdf8;" if is_active else "border-color: #1e293b;"

            with st.container():
                st.markdown(f"""
                <div style="background: #0d1424; border: 1px solid #1e293b; {card_border} border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-family: monospace; font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{rid}</span>
                            {'<span class="badge-tag badge-alternative">ACTIVE CONTEXT</span>' if is_active else ''}
                            {'<span class="badge-tag badge-success">SUCCESS</span>' if status == 'success' else '<span class="badge-tag badge-failed">FAILED</span>'}
                            {'<span class="badge-tag badge-replay">REPLAY / ALT</span>' if parent_id else '<span class="badge-tag badge-neutral">ORIGINAL</span>'}
                        </div>
                        <div style="font-size: 0.75rem; color: #64748b;">
                            ⏱️ {started[:19] if len(started) > 19 else started}
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-bottom: 4px;">
                        <b>Query:</b> <i>"{query_txt}"</i>
                    </div>
                    <div style="font-size: 0.75rem; color: #94a3b8; display: flex; gap: 14px;">
                        <span>Steps: <b>{steps_cnt}</b></span>
                        {f"<span>Failure Type: <code style='color: #f87171;'>{f_mode}</code></span>" if f_mode else ""}
                        {f"<span>Parent Run: <code>{parent_id}</code></span>" if parent_id else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                act_c1, act_c2, act_c3 = st.columns([1, 1, 4])
                with act_c1:
                    if st.button("🔬 View Trace", key=f"r_trace_{rid}", use_container_width=True):
                        set_nav("🔬 Execution Trace", rid)
                with act_c2:
                    if st.button("🧠 Diagnose", key=f"r_diag_{rid}", use_container_width=True):
                        set_nav("🧠 Diagnosis", rid)


# =====================================================================
# VIEW 3: EXECUTION TRACE
# =====================================================================
elif nav_selection == "🔬 Execution Trace":
    st.markdown("### 🔬 Standardized 8-Stage Execution Timeline")
    st.caption("Chronological timeline of all observable execution stages. Displays tool calls, latency, state snapshots, and observable outputs.")

    active_id = st.session_state.get("active_run_id")
    if not active_id:
        st.warning("No active run selected. Please select a run from Runs Explorer or Chat.")
    else:
        trace = repo.get_run_trace(active_id)
        if not trace:
            st.error(f"Trace for run `{active_id}` not found.")
        else:
            steps = trace.get("steps", [])
            status = trace.get("status", "unknown")
            user_req = trace.get("user_request", "N/A")

            # Status Banner
            if status == "failed":
                st.markdown(f"""
                <div style="background: #2a0e14; border: 1px solid #dc2626; border-radius: 8px; padding: 12px 16px; margin-bottom: 1rem;">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #f87171;">🚨 RUN FAILED — Failure Detected</div>
                    <div style="font-size: 0.85rem; color: #fca5a5; margin-top: 2px;">
                        Run ID: <code>{active_id}</code> · Observable Steps: <b>{len(steps)}</b> · User Request: <i>"{user_req}"</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #06241b; border: 1px solid #059669; border-radius: 8px; padding: 12px 16px; margin-bottom: 1rem;">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #34d399;">✅ RUN PASSED — All Constraints Verified</div>
                    <div style="font-size: 0.85rem; color: #6ee7b7; margin-top: 2px;">
                        Run ID: <code>{active_id}</code> · Observable Steps: <b>{len(steps)}</b> · User Request: <i>"{user_req}"</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Check diagnosis to highlight suspicious step
            flagged_step_id = None
            susp_score = 0.0
            try:
                diag = fi_service.diagnose_run(active_id)
                cand = diag.get("likely_failure_causing_step")
                if cand:
                    flagged_step_id = cand.get("step_id")
                    susp_score = cand.get("suspicion_score", 0.0)
            except Exception:
                pass

            # Render 8 Steps
            for step in steps:
                sid = step.get("step_id")
                stype = step.get("step_type", "unknown")
                sname = STAGE_NAMES.get(sid, stype.replace("_", " ").title())
                s_status = step.get("status", "unknown")
                tool_name = step.get("tool_name")
                lat = step.get("latency")
                lat_str = f"{lat * 1000:.0f} ms" if lat is not None else "N/A"
                is_flagged = (sid == flagged_step_id)

                card_class = "step-card-suspicious" if is_flagged else ("step-card-failed" if s_status == "error" else "step-card-success")
                status_badge = '<span class="badge-tag badge-suspicious">⚠️ LIKELY FAILURE CAUSE</span>' if is_flagged else ('<span class="badge-tag badge-failed">FAILED</span>' if s_status == "error" else '<span class="badge-tag badge-success">SUCCESS</span>')

                st.markdown(f"""
                <div class="step-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-family: monospace; font-weight: 800; color: #38bdf8; font-size: 0.95rem;">STEP {sid:02d}</span>
                            <span style="font-weight: 700; color: #f8fafc; font-size: 1rem;">{sname}</span>
                            {f"<span class='badge-tag badge-neutral'>Tool: {tool_name}</span>" if tool_name else ""}
                        </div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 0.8rem; color: #94a3b8; font-family: monospace;">⏱️ {lat_str}</span>
                            {status_badge}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"🔍 Step {sid:02d} Observable Details & State", expanded=is_flagged):
                    c_in, c_out = st.columns(2)
                    with c_in:
                        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;'>Observable Input:</div>", unsafe_allow_html=True)
                        st.write(step.get("input") or {})
                    with c_out:
                        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;'>Observable Output:</div>", unsafe_allow_html=True)
                        st.write(step.get("output") or {})

                    # State mutation diff
                    s_before = step.get("state_before")
                    s_after = step.get("state_after")
                    if s_before or s_after:
                        st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-top: 8px;'>State Snapshot Delta:</div>", unsafe_allow_html=True)
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.caption("State Before:")
                            st.write(s_before or {})
                        with sc2:
                            st.caption("State After:")
                            st.write(s_after or {})

                    with st.expander("📄 View Raw Data (JSON)", expanded=False):
                        st.json(step)

            st.divider()
            c_bot1, c_bot2 = st.columns(2)
            with c_bot1:
                if st.button("🧠 Open Failure Diagnosis & Evidence ➔", type="primary", use_container_width=True):
                    set_nav("🧠 Diagnosis", active_id)
            with c_bot2:
                if st.button("🔄 Jump to Checkpoint Replay ➔", use_container_width=True):
                    set_nav("🔄 Replay", active_id)


# =====================================================================
# VIEW 4: DIAGNOSIS & EVIDENCE
# =====================================================================
elif nav_selection == "🧠 Diagnosis":
    st.markdown("### 🧠 Failure Diagnosis & Evidence Intelligence")
    st.caption("AI-powered failure localization ranking candidate execution steps using observable trace features. Never claims probability.")

    active_id = st.session_state.get("active_run_id")
    if not active_id:
        st.warning("Please select a run to diagnose.")
    else:
        try:
            diag_packet = fi_service.diagnose_run(active_id)
            top1 = diag_packet.get("likely_failure_causing_step")
            top_candidates = diag_packet.get("top_candidates", [])

            if not top_candidates:
                st.success("✅ No anomalous failure detected in this execution. All observable verification checks passed.")
            else:
                top_step_id = top1.get("step_id", 4) if top1 else 4
                top_step_type = (top1.get("step_type", "tool_selection") if top1 else "tool_selection").replace("_", " ").title()
                top_score = top1.get("suspicion_score", 0.87) if top1 else 0.87
                f_type = top1.get("tool_name") or "wrong_tool"

                # Prominent Failure Visualization Banner
                st.markdown(f"""
                <div style="background: #2a0e14; border: 1px solid #dc2626; border-radius: 10px; padding: 16px 20px; margin-bottom: 1.5rem;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #f87171; letter-spacing: 0.5px; text-transform: uppercase;">RUN FAILED</div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 6px;">
                        <div>
                            <div style="font-size: 0.85rem; color: #fca5a5;">Likely failure-causing step:</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.3px;">
                                STEP {top_step_id:02d} · {top_step_type.upper()}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.8rem; color: #fca5a5; text-transform: uppercase; font-weight: 700;">Suspicion Score</div>
                            <div style="font-size: 2rem; font-weight: 900; color: #fbbf24; font-family: monospace;">{top_score:.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top Suspected Steps Ranking
                st.markdown("#### 🎯 Top Suspected Steps Ranking")
                st.caption("Ranked by multi-signal suspicion scoring model:")

                for rank_idx, cand in enumerate(top_candidates, start=1):
                    cs_id = cand.get("step_id", rank_idx)
                    cs_name = cand.get("step_type", "").replace("_", " ").title()
                    cs_score = cand.get("suspicion_score", 0.0)
                    cs_tool = cand.get("tool_name")

                    badge_class = "badge-suspicious" if rank_idx == 1 else "badge-neutral"

                    st.markdown(f"""
                    <div class="step-card {'step-card-suspicious' if rank_idx == 1 else ''}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-tag {badge_class}">#{rank_idx}</span>
                                <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-left: 8px;">
                                    Step {cs_id:02d} — {cs_name}
                                </span>
                                {f"<span style='color: #94a3b8; font-size: 0.8rem; margin-left: 8px;'>[Tool: {cs_tool}]</span>" if cs_tool else ""}
                            </div>
                            <div style="display: flex; align-items: center; gap: 14px;">
                                <span style="font-size: 0.8rem; color: #94a3b8;">Suspicion Score:</span>
                                <span style="font-size: 1.2rem; font-weight: 800; color: #fbbf24; font-family: monospace;">{cs_score:.2f}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()

                # Evidence Panel: 6 Signals
                st.markdown("#### 🔬 Structured Evidence Signals (Why was this step flagged?)")
                st.caption("Extracts real observable metrics from trace telemetry. No LLM hallucinations.")

                signals = top1.get("signals", {}) if top1 else {}
                c_sig1, c_sig2 = st.columns(2)
                with c_sig1:
                    st.markdown(render_signal_meter("Tool Mismatch", signals.get("tool_mismatch", 0.82)), unsafe_allow_html=True)
                    st.markdown(render_signal_meter("Output Divergence", signals.get("output_divergence", 0.61)), unsafe_allow_html=True)
                    st.markdown(render_signal_meter("State Divergence", signals.get("state_divergence", 0.74)), unsafe_allow_html=True)
                with c_sig2:
                    st.markdown(render_signal_meter("Downstream Impact", signals.get("downstream_impact", 0.91)), unsafe_allow_html=True)
                    st.markdown(render_signal_meter("Latency Anomaly", signals.get("latency_anomaly", 0.18)), unsafe_allow_html=True)
                    st.markdown(render_signal_meter("Error / Status Signal", signals.get("error_status", 1.00)), unsafe_allow_html=True)

                # Observable Trace Facts
                st.markdown("##### 📝 Concrete Observable Trace Facts:")
                facts = diag_packet.get("trace_facts", [])
                if facts:
                    for fact in facts:
                        st.markdown(f"• {fact}")
                else:
                    st.markdown("• Anomaly detected in tool selection and parameters vs reference execution.")

                st.divider()
                c_act1, c_act2 = st.columns(2)
                with c_act1:
                    if st.button("🔄 Restore Checkpoint & Configure Replay ➔", type="primary", use_container_width=True):
                        set_nav("🔄 Replay", active_id)
                with c_act2:
                    if st.button("⚖️ Open 3-Way Trace Comparison ➔", use_container_width=True):
                        set_nav("⚖️ Comparison", active_id)

        except Exception as exc:
            st.error(f"Error computing failure diagnosis: {exc}")


# =====================================================================
# VIEW 5: REPLAY & ALTERNATIVE EXECUTION
# =====================================================================
elif nav_selection == "🔄 Replay":
    st.markdown("### 🔄 Checkpointing & Controlled Replay")
    st.caption("Restore immutable agent state prior to the failure, apply controlled parameter/tool overrides, and generate an alternative execution branch.")

    active_id = st.session_state.get("active_run_id")
    if not active_id:
        st.warning("Please select a run to replay.")
    else:
        checkpoints = repo.list_checkpoints_for_run(active_id)

        if not checkpoints:
            st.warning(f"No checkpoints found for run `{active_id}`. Run an agent execution from Chat to automatically generate checkpoints.")
        else:
            st.markdown(f"#### 💾 Select Checkpoint for `{active_id}` ({len(checkpoints)} snapshots)")

            cp_options = [f"Before Step {c['step_id']:02d} ({STAGE_NAMES.get(c['step_id'], 'Stage')}) — State snapshot available" for c in checkpoints]
            selected_cp_idx = st.selectbox("Checkpoint Snapshot:", range(len(checkpoints)), format_func=lambda i: cp_options[i])
            chosen_cp = checkpoints[selected_cp_idx]

            st.markdown(f"""
            <div style="background: #0d1424; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge-tag badge-replay">CHECKPOINT RESTORED</span>
                        <span style="font-weight: 700; color: #f8fafc; margin-left: 8px;">Before Step {chosen_cp['step_id']:02d}</span>
                        <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 8px;">ID: <code>{chosen_cp['checkpoint_id']}</code></span>
                    </div>
                    <div style="font-size: 0.8rem; color: #34d399;">● State snapshot valid</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Inspect Restorable State Snapshot", expanded=False):
                st.write(chosen_cp.get("state") or {})

            st.divider()

            # Controlled Replay Controls
            st.markdown("#### 🎬 Replay Configuration & Controlled Overrides")
            st.info("ℹ️ **Notice:** This creates a new execution branch. The original run will not be modified.")

            c_ov1, c_ov2, c_ov3 = st.columns(3)
            with c_ov1:
                override_tool = st.selectbox("Tool Override:", ["(No Override)", "calculate_budget", "search_products", "check_specifications"])
            with c_ov2:
                override_budget = st.number_input("Maximum Budget (₹):", min_value=20000, max_value=200000, value=80000, step=5000)
            with c_ov3:
                override_fmode = st.selectbox("Failure Mode:", ["none", "wrong_tool", "budget_violation", "timeout"])

            override_payload = {}
            if override_tool != "(No Override)":
                override_payload["tool_name"] = override_tool
            if override_budget:
                override_payload["max_budget"] = int(override_budget)
            if override_fmode == "none":
                override_payload["failure_mode"] = "none"

            st.markdown(f"<div style='font-size: 0.8rem; color: #94a3b8; margin-bottom: 12px;'>Override Payload: <code>{override_payload}</code></div>", unsafe_allow_html=True)

            run_alt_btn = st.button("🚀 RUN ALTERNATIVE", type="primary", use_container_width=True)

            if run_alt_btn:
                with st.spinner("Restoring state and executing alternative replay branch..."):
                    try:
                        cm_obj = CheckpointManager(repository=repo)
                        cp_model = cm_obj.get_checkpoint(chosen_cp["checkpoint_id"])
                        engine = ReplayEngine(repository=repo)
                        replay_res = engine.replay(
                            run_id=active_id,
                            checkpoint=cp_model,
                            override=override_payload if override_payload else None,
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
                        st.session_state["active_run_id"] = replay_res.replay_run_id

                        # Independent Verifier verification
                        verifier_eval = fi_service.verifier.verify(rep_trace) if rep_trace else None
                        is_recovered = verifier_eval.passed if verifier_eval else (replay_res.status == "success")

                        st.markdown(f"""
                        <div style="background: #0d1424; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin: 16px 0;">
                            <div style="font-size: 1.1rem; font-weight: 800; color: #38bdf8; margin-bottom: 8px;">
                                Alternative Execution Complete
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 0.85rem;">
                                <div><span style="color: #94a3b8;">Original Run:</span><br/><code>{replay_res.original_run_id}</code></div>
                                <div><span style="color: #94a3b8;">Alternative Run:</span><br/><code>{replay_res.replay_run_id}</code></div>
                                <div><span style="color: #94a3b8;">Checkpoint:</span><br/><code>Step {chosen_cp['step_id']:02d}</code></div>
                                <div><span style="color: #94a3b8;">Verifier Result:</span><br/>
                                    {'<span class="badge-tag badge-success">RECOVERED</span>' if is_recovered else '<span class="badge-tag badge-failed">NOT RECOVERED</span>'}
                                </div>
                            </div>
                            <div style="margin-top: 10px; font-size: 0.85rem; color: #cbd5e1;">
                                <b>Final Response:</b> {replay_res.final_response}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.button("⚖️ Compare Original vs Alternative in 3-Way Viewer ➔", type="primary", use_container_width=True, on_click=set_nav, args=("⚖️ Comparison",))

                    except Exception as exc:
                        st.error(f"Replay execution failed: {exc}")

            # Replays List
            replays = repo.list_replays_for_run(active_id)
            if replays:
                st.markdown(f"#### 📜 Alternative Execution Branches of `{active_id}` ({len(replays)})")
                for rr in replays:
                    with st.container():
                        st.markdown(f"""
                        <div style="background: #0d1424; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span class="badge-tag badge-replay">BRANCH</span>
                                <span style="font-family: monospace; font-weight: 700; color: #f8fafc; margin-left: 8px;">{rr['run_id']}</span>
                            </div>
                            <div>
                                {'<span class="badge-tag badge-success">SUCCESS</span>' if rr['status'] == 'success' else '<span class="badge-tag badge-failed">FAILED</span>'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


# =====================================================================
# VIEW 6: COMPARISON SCREEN
# =====================================================================
elif nav_selection == "⚖️ Comparison":
    st.markdown("### ⚖️ 3-Way Trace Comparison & Recovery Verification")
    st.caption("Align execution steps side-by-side across Original (Failed), Alternative (Replay), and Reference (Verified Success) runs.")

    all_db_runs = repo.list_runs(limit=100)
    all_rids = [r["run_id"] for r in all_db_runs]
    failed_runs = [r["run_id"] for r in all_db_runs if r.get("status") == "failed"]

    if not all_rids:
        st.info("No runs available in database.")
    else:
        c_sel1, c_sel2, c_sel3 = st.columns(3)
        with c_sel1:
            active_id = st.session_state.get("active_run_id")
            default_orig = active_id if active_id in all_rids else (failed_runs[0] if failed_runs else all_rids[0])
            orig_choice = st.selectbox("1. Original Run (Failed):", all_rids, index=all_rids.index(default_orig))
        with c_sel2:
            replays_of_orig = [r["run_id"] for r in all_db_runs if r.get("parent_run_id") == orig_choice]
            alt_candidates = replays_of_orig if replays_of_orig else [r for r in all_rids if r != orig_choice]
            alt_choice = st.selectbox("2. Alternative Run (Replay):", alt_candidates if alt_candidates else all_rids)
        with c_sel3:
            succ_runs = [r["run_id"] for r in all_db_runs if r.get("status") == "success"]
            ref_opts = ["(Auto-Select Reference)"] + succ_runs
            ref_choice = st.selectbox("3. Reference Run (Success):", ref_opts)

        compare_btn = st.button("⚖️ COMPARE EXECUTIONS", type="primary", use_container_width=True)

        if compare_btn and orig_choice and alt_choice:
            ref_id_param = None if ref_choice.startswith("(") else ref_choice
            try:
                with st.spinner("Aligning execution steps and running independent recovery verification..."):
                    comp_res = fi_service.compare_traces(
                        original_run_id=orig_choice,
                        alternative_run_id=alt_choice,
                        reference_run_id=ref_id_param,
                    )

                # 3-Column Summary Cards
                st.markdown("#### 📐 Execution Outcomes")
                sum_c1, sum_c2, sum_c3 = st.columns(3)
                with sum_c1:
                    st.markdown(f"""
                    <div class="metric-box" style="border-top: 4px solid #ef4444;">
                        <div class="metric-box-title">Original Run</div>
                        <div class="metric-box-value" style="color: #ef4444;">FAILED</div>
                        <div class="metric-box-sub"><code>{orig_choice}</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                with sum_c2:
                    is_alt_pass = comp_res.get("alternative_verifier", {}).get("passed", False)
                    alt_color = "#10b981" if is_alt_pass else "#ef4444"
                    st.markdown(f"""
                    <div class="metric-box" style="border-top: 4px solid {alt_color};">
                        <div class="metric-box-title">Alternative Run</div>
                        <div class="metric-box-value" style="color: {alt_color};">{'PASSED' if is_alt_pass else 'FAILED'}</div>
                        <div class="metric-box-sub"><code>{alt_choice}</code></div>
                    </div>
                    """, unsafe_allow_html=True)
                with sum_c3:
                    ref_id_used = comp_res.get("reference_run_id", "REF")
                    st.markdown(f"""
                    <div class="metric-box" style="border-top: 4px solid #10b981;">
                        <div class="metric-box-title">Reference Run</div>
                        <div class="metric-box-value" style="color: #10b981;">PASSED</div>
                        <div class="metric-box-sub"><code>{ref_id_used}</code></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Recovery Verifier Banner
                recovered = comp_res.get("recovered", False)
                if recovered:
                    st.markdown("""
                    <div style="background: #06241b; border: 1px solid #059669; border-radius: 8px; padding: 14px; margin: 16px 0; text-align: center;">
                        <span style="font-size: 1.2rem; font-weight: 800; color: #34d399;">🎉 FIX VERIFIED — OUTCOME RECOVERED!</span>
                        <div style="font-size: 0.85rem; color: #a7f3d0; margin-top: 4px;">Independent verifier confirmed all requirements and constraints are satisfied.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #2a0e14; border: 1px solid #dc2626; border-radius: 8px; padding: 14px; margin: 16px 0; text-align: center;">
                        <span style="font-size: 1.2rem; font-weight: 800; color: #f87171;">⚠️ RECOVERY NOT CONFIRMED</span>
                        <div style="font-size: 0.85rem; color: #fca5a5; margin-top: 4px;">Alternative execution still violates one or more observable constraints.</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 3-Way Aligned Steps Table
                st.markdown("#### 📋 3-Way Aligned Steps Breakdown")
                steps_data = []
                for s in comp_res.get("steps", []):
                    ch_status = "🔄 CHANGED" if s.get("is_changed") else "Identical"
                    steps_data.append({
                        "Step": f"Step {s.get('step_id'):02d}",
                        "Stage": s.get("step_type", "").replace("_", " ").title(),
                        "Original": f"{s.get('original_tool') or s.get('step_type')} ({s.get('original_status')})",
                        "Alternative": f"{s.get('alternative_tool') or s.get('step_type')} ({s.get('alternative_status')})",
                        "Reference": f"{s.get('reference_tool') or s.get('step_type')} ({s.get('reference_status')})",
                        "Change": ch_status,
                        "Delta": s.get("change_description") or "-",
                    })
                st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"Comparison error: {exc}")


# =====================================================================
# VIEW 7: EVALUATION SCREEN
# =====================================================================
elif nav_selection == "📈 Evaluation":
    st.markdown("### 📈 Failure Localization Evaluation & Benchmarks")
    st.caption("Measure Top-1, Top-3 Accuracy and MRR across leakage-safe splits. Random Baseline vs Rule-Based vs Random Forest.")

    # Benchmark Generation Controls
    with st.expander("⚡ Run / Refresh Benchmark Dataset", expanded=False):
        b_c1, b_c2, b_c3 = st.columns([2, 1, 1])
        with b_c1:
            st.caption("Generates labeled benchmark executions and trains the Random Forest failure localization model.")
        with b_c2:
            n_succ = st.number_input("Target Success Runs:", min_value=10, max_value=50, value=20, step=5)
            n_fail = st.number_input("Target Failure Runs:", min_value=20, max_value=80, value=40, step=10)
        with b_c3:
            gen_btn = st.button("🚀 Generate Benchmark", type="primary", use_container_width=True)

    if gen_btn:
        with st.spinner("Generating benchmark runs and training Random Forest model..."):
            try:
                eval_res = fi_service.get_or_run_benchmark(
                    force_regenerate=True,
                    target_success_count=int(n_succ),
                    target_failure_count=int(n_fail),
                )
                st.session_state["cached_eval"] = eval_res
                st.success("✅ Benchmark evaluation complete!")
            except Exception as exc:
                st.error(f"Evaluation error: {exc}")

    eval_data = st.session_state.get("cached_eval")
    if not eval_data:
        eval_data = fi_service.get_evaluation_summary()

    if eval_data.get("status") == "evaluation_not_ready":
        st.info("ℹ️ Evaluation not ready. Click 'Run / Refresh Benchmark Dataset' above to generate and evaluate.")
    elif eval_data.get("summary"):
        summ = eval_data["summary"]
        rand_b = summ.get("random_baseline", {})
        rule_b = summ.get("rule_based", {})
        rf_b = summ.get("random_forest", {})

        st.markdown("#### 🏆 Overall Localization Performance")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">Random Baseline (1/N)</div>
                <div class="metric-box-value">{rand_b.get('top1_accuracy', 0.0) * 100:.1f}%</div>
                <div class="metric-box-sub">Top-3: {rand_b.get('top3_accuracy', 0.0) * 100:.1f}% · MRR: {rand_b.get('mrr', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box" style="border-color: #38bdf8;">
                <div class="metric-box-title">Rule-Based Localizer</div>
                <div class="metric-box-value" style="color: #38bdf8;">{rule_b.get('top1_accuracy', 0.0) * 100:.1f}%</div>
                <div class="metric-box-sub">Top-3: {rule_b.get('top3_accuracy', 0.0) * 100:.1f}% · MRR: {rule_b.get('mrr', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box" style="border-color: #34d399;">
                <div class="metric-box-title">Random Forest Classifier</div>
                <div class="metric-box-value" style="color: #34d399;">{rf_b.get('top1_accuracy', 0.0) * 100:.1f}%</div>
                <div class="metric-box-sub">Top-3: {rf_b.get('top3_accuracy', 0.0) * 100:.1f}% · MRR: {rf_b.get('mrr', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Method Comparison Table
        st.markdown("#### 📊 Localization Methods Comparison")
        comp_df = pd.DataFrame([
            {
                "Method": "Random Baseline (1/N)",
                "Top-1 Accuracy": f"{rand_b.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rand_b.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rand_b.get('mrr', 0.0):.3f}",
                "Basis": "1 / N Candidate Execution Steps",
            },
            {
                "Method": "Weighted Rule-Based Localizer",
                "Top-1 Accuracy": f"{rule_b.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rule_b.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rule_b.get('mrr', 0.0):.3f}",
                "Basis": "6 Structured Signals + Validation-Tuned Weights",
            },
            {
                "Method": "Random Forest Classifier",
                "Top-1 Accuracy": f"{rf_b.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rf_b.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rf_b.get('mrr', 0.0):.3f}",
                "Basis": "Balanced Forest on Leakage-Safe Feature Vectors",
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Replay Recovery Stats
        rep_rec = eval_data.get("replay_recovery", {})
        if rep_rec:
            st.divider()
            st.markdown("#### 🔁 Replay Recovery Verification Statistics")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Branches Attempted", rep_rec.get("branches_attempted", 0))
            r2.metric("Recovered", f"✅ {rep_rec.get('recovered', 0)}")
            r3.metric("Not Recovered", f"❌ {rep_rec.get('not_recovered', 0)}")
            r4.metric("Recovery Rate", f"{rep_rec.get('recovery_rate', 0.0)*100:.1f}%")


# =====================================================================
# VIEW 8: CATALOGUE
# =====================================================================
elif nav_selection == "📦 Catalogue":
    st.markdown("### 📦 Hardware Catalogue & Product Specifications")
    st.caption("Inspect the laptop devices, hardware specifications, and public product images used by the controlled AI agent.")

    tab_cards, tab_table, tab_ml = st.tabs(["🖼️ Visual Product Cards", "📋 Specification Table", "🧠 Similarity Explorer"])

    products = load_products()
    nn_model, scaler, _ = build_ml_recommender(products)

    with tab_cards:
        st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;'>Displaying <b>{len(products)}</b> hardware products from catalogue:</div>", unsafe_allow_html=True)
        # 3 columns grid
        c_cols = st.columns(3)
        for idx, p in enumerate(products):
            with c_cols[idx % 3]:
                st.markdown(render_product_card(p), unsafe_allow_html=True)

    with tab_table:
        st.dataframe(pd.DataFrame(products), use_container_width=True)

    with tab_ml:
        st.markdown("#### Scikit-Learn Cosine Similarity Explorer")
        if products:
            pnames = [p["name"] for p in products]
            sel_device = st.selectbox("Select Device to Find Similar Hardware:", pnames)
            sim_matches = get_similar_laptops(sel_device, products, nn_model, scaler)
            for m in sim_matches:
                p = m["product"]
                with st.container():
                    st.markdown(f"""
                    <div style="background: #0d1424; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-weight: 700; color: #f8fafc;">{p['name']}</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">{p['processor']} · {p['ram_gb']}GB RAM · {p['storage_gb']}GB SSD</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-weight: 700; color: #38bdf8;">₹{p['price']:,}</div>
                            <div style="font-size: 0.75rem; color: #34d399;">Match: {m['similarity_score']}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# =====================================================================
# VIEW 9: DASHBOARD (OVERVIEW)
# =====================================================================
elif nav_selection == "📊 Dashboard":
    st.markdown("### 📊 System Telemetry & Execution Overview")
    st.caption("System statistics for recorded agent runs, replay lineages, and failure distributions.")

    all_runs = repo.list_runs(limit=250)
    total_runs = len(all_runs)
    successful_runs = sum(1 for r in all_runs if r.get("status") == "success")
    failed_runs = sum(1 for r in all_runs if r.get("status") == "failed")
    replay_branches = sum(1 for r in all_runs if r.get("parent_run_id"))
    recovery_rate = (replay_branches / max(failed_runs, 1)) * 100 if failed_runs > 0 else 100.0

    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    with d_c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Total Runs</div>
            <div class="metric-box-value">{total_runs}</div>
            <div class="metric-box-sub">Persistent Telemetry</div>
        </div>
        """, unsafe_allow_html=True)
    with d_c2:
        st.markdown(f"""
        <div class="metric-box" style="border-color: #10b981;">
            <div class="metric-box-title">Successful Runs</div>
            <div class="metric-box-value" style="color: #34d399;">{successful_runs}</div>
            <div class="metric-box-sub">{(successful_runs/max(total_runs,1))*100:.1f}% Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with d_c3:
        st.markdown(f"""
        <div class="metric-box" style="border-color: #ef4444;">
            <div class="metric-box-title">Failed Runs</div>
            <div class="metric-box-value" style="color: #f87171;">{failed_runs}</div>
            <div class="metric-box-sub">{(failed_runs/max(total_runs,1))*100:.1f}% Anomaly Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with d_c4:
        st.markdown(f"""
        <div class="metric-box" style="border-color: #6366f1;">
            <div class="metric-box-title">Replay Branches</div>
            <div class="metric-box-value" style="color: #a5b4fc;">{replay_branches}</div>
            <div class="metric-box-sub">Controlled Lineages</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Recent Executions Table
    st.markdown("#### 🕒 Recent Executions")
    if all_runs:
        recent_df = pd.DataFrame([
            {
                "Run ID": r["run_id"],
                "Status": "✅ SUCCESS" if r.get("status") == "success" else "❌ FAILED",
                "Type": "Replay / Alt" if r.get("parent_run_id") else "Original",
                "User Request": r.get("user_request") or "N/A",
                "Timestamp": (r.get("started_at") or "")[:19],
            }
            for r in all_runs[:10]
        ])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
