"""Streamlit Web Interface for Black Box — Stages 1-5.

Tabs:
  1. Agent Execution  — run the LangChain agent, view timeline
  2. Trace Debugger   — Stage 5: open runs, view failure diagnosis, replay from checkpoints
  3. ML Recommender   — Scikit-Learn NearestNeighbors
  4. Laptop Catalogue — browse products.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from agent.agent import ConfigError, LaptopAgent
from agent.events import EventType
from recorder.recorder import ExecutionRecorder
from storage.repository import TraceRepository
from replay.checkpoint_manager import CheckpointManager
from replay.replay_engine import ReplayEngine

load_dotenv(override=True)

API_BASE = "http://localhost:8000"


# --------------------------------------------------------------------------- page config
st.set_page_config(
    page_title="Black Box | Agent Debugger",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# --------------------------------------------------------------------------- CSS
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #3B82F6, #10B981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { color: #6B7280; font-size: 1rem; margin-bottom: 1.5rem; }
.suspicious { border-left: 4px solid #EF4444; padding-left: 8px; }
.replay-panel { background: #1e293b; border-radius: 8px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 Black Box | AI Agent Debugger</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Stages 1–5: Observable Agent · Trace Storage · Failure Intelligence · Checkpoint Replay</div>', unsafe_allow_html=True)

products = load_products()
nn_model, scaler, _ = build_ml_recommender(products)

with st.sidebar:
    st.subheader("🤖 Active LLM Engine")
    provider_val = os.getenv("LLM_PROVIDER", "local")
    model_val = os.getenv("LLM_MODEL", "blackbox-local-offline")
    st.success(f"**Provider:** `{provider_val}`\n\n**Model:** `{model_val}`")
    st.divider()
    st.subheader("💡 Quick Queries")
    presets = [
        "Find a laptop under ₹70,000 for programming and gaming.",
        "Best laptop for college under ₹50,000.",
        "Gaming laptop within ₹90,000.",
        "Show options within ₹60,000 with 16GB RAM.",
    ]
    selected_preset = st.radio("Template:", presets, index=0)
    st.divider()
    st.caption("Black Box | Stage 5 — Replay & Debug")


# --------------------------------------------------------------------------- Tabs
tab_agent, tab_debug, tab_stage6, tab_ml, tab_catalog = st.tabs([
    "🤖 Agent Execution",
    "🔬 Trace Debugger (Stage 5)",
    "📊 Evaluation & Trace Comparison (Stage 6)",
    "🧠 ML Recommender",
    "📦 Catalogue",
])


# =====================================================================  TAB 1: Agent Run
with tab_agent:
    st.subheader("Run the AI Agent")
    user_query = st.text_input("Enter your device or computing requirement (e.g. laptop, phone):", value=selected_preset)
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        run_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)
    with col2:
        fail_options = ["none", "budget_violation", "wrong_tool", "wrong_interpretation", "timeout"]
        failure_mode = st.selectbox("Failure Mode", fail_options, index=0, label_visibility="collapsed")

    if run_btn:
        if not user_query.strip():
            st.warning("Please enter a valid request.")
        else:
            with st.spinner("Agent is reasoning and executing tools..."):
                try:
                    repo = TraceRepository()
                    recorder = ExecutionRecorder(repository=repo)
                    agent = LaptopAgent(sinks=[recorder.record])
                    fm = failure_mode if failure_mode != "none" else None
                    result = agent.run(user_query, failure_mode=fm)

                    # Auto-generate checkpoints
                    run_id = recorder._current_run_id
                    trace = repo.get_run_trace(run_id)
                    if trace:
                        cm = CheckpointManager(repository=repo)
                        cps = cm.create_checkpoints_for_run(
                            run_id=run_id,
                            user_request=user_query,
                            steps=trace.get("steps", []),
                            failure_mode=fm,
                        )
                        st.success(f"✅ Run saved as **`{run_id}`** with **{len(cps)} checkpoints**. Open in Trace Debugger tab to replay.")

                    st.subheader("📋 Final Recommendation")
                    if result.status == "success":
                        st.markdown(result.final_response)
                    else:
                        st.error(result.final_response)

                    st.subheader("⏱️ Observable Execution Timeline")
                    for idx, ev in enumerate(result.events, start=1):
                        ev_type = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                        with st.expander(
                            f"[{idx}] {ev_type.upper()} {f'— Tool: {ev.tool_name}' if ev.tool_name else ''}",
                            expanded=(idx <= 3),
                        ):
                            st.write(f"**Status:** `{ev.status.value if hasattr(ev.status, 'value') else ev.status}`")
                            if ev.summary:
                                st.info(f"**Summary:** {ev.summary}")
                            if ev.input:
                                st.json({"input": ev.input})
                            if ev.output:
                                st.json({"output": ev.output})

                    st.download_button(
                        label="📥 Download Trace Events (JSON)",
                        data=json.dumps([e.to_dict() for e in result.events], indent=2),
                        file_name="agent_events.json",
                        mime="application/json",
                    )

                except ConfigError as ce:
                    st.error(f"Configuration Error: {ce}")
                except Exception as e:
                    st.error(f"Execution Error: {e}")


# =====================================================================  TAB 2: Stage 5 Debugger
with tab_debug:
    st.subheader("🔬 Trace Debugger — Open a Run, Diagnose, and Replay")

    repo = TraceRepository()
    runs = repo.list_runs(limit=30)

    if not runs:
        st.info("No runs yet. Go to the Agent Execution tab and run a query first.")
    else:
        # Run selector
        run_labels = []
        for r in runs:
            label = f"{'⚠️ FAILED' if r['status'] == 'failed' else '✅'} {r['run_id']} — {r['user_request'][:50]}"
            if r.get("parent_run_id"):
                label += f" (replay of {r['parent_run_id']})"
            run_labels.append(label)

        selected_idx = st.selectbox("Select a run to inspect:", range(len(runs)), format_func=lambda i: run_labels[i])
        selected_run = runs[selected_idx]
        run_id = selected_run["run_id"]

        st.divider()
        trace = repo.get_run_trace(run_id)
        steps = trace.get("steps", []) if trace else []

        # Run summary
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Run ID", run_id)
        with col_b:
            status_emoji = "✅" if selected_run["status"] == "success" else "❌"
            st.metric("Status", f"{status_emoji} {selected_run['status'].upper()}")
        with col_c:
            st.metric("Steps", len(steps))

        if selected_run.get("parent_run_id"):
            st.info(f"🔁 This is a **replay** of `{selected_run['parent_run_id']}`")

        # Failure Intelligence
        st.subheader("🧠 Failure Intelligence")
        try:
            from intelligence.rules import RuleBasedLocalizer
            localizer = RuleBasedLocalizer()
            diagnosis = localizer.diagnose(trace)
            diag_dict = diagnosis.to_dict()

            if diag_dict["failure_detected"]:
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.error(f"**Failure Type:** {diag_dict['failure_type']}")
                    st.write(f"**Summary:** {diag_dict['summary']}")
                with col_f2:
                    resp_step = diag_dict.get("likely_responsible_step")
                    if resp_step:
                        st.warning(
                            f"**Likely Responsible Step:** `Step {resp_step['step_id']}` "
                            f"— {resp_step['step_type']} (Suspicion: **{resp_step['score']:.2f}**)"
                        )
                with st.expander("📋 Evidence", expanded=False):
                    for ev in diag_dict.get("evidence", []):
                        st.write(f"• {ev.get('description', ev)}")
            else:
                st.success("✅ No failure detected in this trace.")
        except Exception as exc:
            st.warning(f"Diagnosis error: {exc}")
            diag_dict = {}

        # Execution Trace with Checkpoint/Replay actions
        st.subheader("📊 Execution Trace")
        checkpoints = repo.list_checkpoints_for_run(run_id)
        cp_by_step = {cp["step_id"]: cp for cp in checkpoints}
        suspicious_step_id = None
        resp_step = diag_dict.get("likely_responsible_step") if diag_dict else None
        if resp_step:
            suspicious_step_id = resp_step.get("step_id")

        for step in steps:
            step_id = step["step_id"]
            step_type = step["step_type"]
            tool = step.get("tool_name", "")
            is_suspicious = (step_id == suspicious_step_id)
            cp = cp_by_step.get(step_id)

            label_prefix = "⚠️ " if is_suspicious else f"[{step_id}] "
            label = f"{label_prefix} Step {step_id} — {step_type}" + (f" ({tool})" if tool else "")

            with st.expander(label, expanded=is_suspicious):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"**Status:** `{step.get('status', 'unknown')}`")
                    if step.get("input"):
                        st.json({"input": step["input"]})
                    if step.get("output"):
                        st.json({"output": step["output"]})
                with c2:
                    if cp:
                        st.success(f"✅ Checkpoint available")
                        st.caption(f"`{cp['checkpoint_id']}`")
                        # Store checkpoint id for replay panel
                        if st.button(f"🔄 Replay from here", key=f"replay_btn_{step_id}"):
                            st.session_state["replay_checkpoint"] = cp
                            st.session_state["replay_run_id"] = run_id
                    else:
                        st.caption("No checkpoint")

        # ------------------------------------------------ Replay Panel
        st.divider()
        st.subheader("🔄 Replay Panel")

        if "replay_checkpoint" in st.session_state and st.session_state.get("replay_run_id") == run_id:
            cp = st.session_state["replay_checkpoint"]
            cp_state = cp.get("state", {})

            st.markdown(f"""
**Original Run:** `{run_id}`  
**Checkpoint:** `{cp['checkpoint_id']}`  
**Resume From:** Step {cp['step_id']}  
**Checkpoint Type:** {cp['checkpoint_type']}
""")
            with st.expander("📦 Current State at Checkpoint", expanded=True):
                st.json(cp_state)

            st.markdown("### Optional Override")
            col_ov1, col_ov2 = st.columns(2)
            with col_ov1:
                current_budget = cp_state.get("budget")
                budget_val = st.number_input(
                    "Budget (₹) — leave 0 to keep original",
                    min_value=0,
                    value=current_budget or 0,
                    step=1000,
                )
            with col_ov2:
                fm_opts = ["Keep original", "none (clear failure)", "budget_violation", "wrong_tool", "wrong_interpretation"]
                fm_override = st.selectbox("Failure Mode", fm_opts, index=0)

            if st.button("▶️ Run Replay", type="primary"):
                override: Dict[str, Any] = {}
                if budget_val and budget_val > 0 and budget_val != current_budget:
                    override["max_budget"] = int(budget_val)
                if fm_override != "Keep original":
                    fm_val = fm_override.split(" ")[0]
                    override["failure_mode"] = fm_val

                with st.spinner("Running replay..."):
                    try:
                        engine = ReplayEngine(repository=repo)
                        from replay.checkpoint_manager import CheckpointManager
                        cm_obj = CheckpointManager(repository=repo)
                        checkpoint_obj = cm_obj.get_checkpoint(cp["checkpoint_id"])
                        result = engine.replay(
                            run_id=run_id,
                            checkpoint=checkpoint_obj,
                            override=override or None,
                        )

                        # Auto-generate checkpoints for replay run
                        replay_trace = repo.get_run_trace(result.replay_run_id)
                        if replay_trace:
                            cm_obj.create_checkpoints_for_run(
                                run_id=result.replay_run_id,
                                user_request=replay_trace.get("user_request", ""),
                                steps=replay_trace.get("steps", []),
                            )

                        orig_status = selected_run["status"]
                        replay_status = result.status

                        st.markdown("---")
                        st.markdown("### 🎬 Replay Complete")
                        c_orig, c_replay = st.columns(2)
                        with c_orig:
                            if orig_status == "success":
                                st.success(f"**Original Run:** ✅ SUCCESS")
                            else:
                                st.error(f"**Original Run:** ❌ FAILED")
                        with c_replay:
                            if replay_status == "success":
                                st.success(f"**Replay Run:** ✅ SUCCESS")
                            else:
                                st.error(f"**Replay Run:** ❌ FAILED")

                        st.info(f"**Replay Run ID:** `{result.replay_run_id}`")
                        st.info(f"**Replay Type:** {result.replay_type}")
                        if result.override_applied:
                            st.info(f"**Override Applied:** {result.override_applied}")
                        st.write(f"**Replay Response:** {result.final_response}")

                        if orig_status == "failed" and replay_status == "failed":
                            st.warning("The modified execution did not resolve the failure.")
                        elif orig_status == "failed" and replay_status == "success":
                            st.balloons()
                            st.success("🎉 Replay successfully recovered from the failure!")

                    except Exception as exc:
                        st.error(f"Replay error: {exc}")
        else:
            st.info("Click **'🔄 Replay from here'** next to a checkpoint in the trace above to open this panel.")

        # Show existing replays
        replays = repo.list_replays_for_run(run_id)
        if replays:
            st.divider()
            st.subheader(f"📜 Previous Replays of `{run_id}` ({len(replays)})")
            for rr in replays:
                status_icon = "✅" if rr["status"] == "success" else "❌"
                rm = rr.get("replay_metadata") or {}
                st.write(f"{status_icon} **`{rr['run_id']}`** — {rr['status']} — Checkpoint: `{rm.get('checkpoint_id', 'N/A')}`")


# =====================================================================  TAB 3: Stage 6 Evaluation & Trace Comparison
with tab_stage6:
    st.subheader("📊 Failure Intelligence Upgrade, Trace Comparison & Evaluation (Stage 6)")
    st.caption("Benchmark evaluation across leakage-safe splits · Random Baseline · Rule-Based vs Random Forest · 3-Way Trace Comparison")

    from failure_intelligence.service import FailureIntelligenceService
    repo_eval = TraceRepository()
    fi_service = FailureIntelligenceService(repository=repo_eval)

    # ---------------- Benchmark Controls
    with st.expander("⚙️ Benchmark Generation & Evaluation Controls", expanded=True):
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 1, 1])
        with c_ctrl1:
            st.write("**Generate Controlled Evaluation Dataset & Evaluate Models**")
            st.caption("Generates controlled runs across scenarios and evaluates Top-1, Top-3, and MRR using strict train/val/test/held-out splits.")
        with c_ctrl2:
            num_success = st.number_input("Target Success Runs", min_value=10, max_value=60, value=20, step=5)
            num_failure = st.number_input("Target Failure Runs", min_value=20, max_value=120, value=40, step=10)
        with c_ctrl3:
            run_eval_btn = st.button("⚡ Run Benchmark Evaluation", type="primary", use_container_width=True)

    if run_eval_btn:
        with st.spinner("Generating controlled runs and computing leakage-safe benchmark evaluation..."):
            try:
                eval_data = fi_service.get_or_run_benchmark(
                    force_regenerate=True,
                    target_success_count=int(num_success),
                    target_failure_count=int(num_failure),
                )
                st.session_state["stage6_eval_data"] = eval_data
                st.success("✅ Benchmark evaluation completed successfully!")
            except Exception as exc:
                st.error(f"Benchmark error: {exc}")

    eval_data = st.session_state.get("stage6_eval_data")
    if not eval_data:
        eval_data = fi_service.get_evaluation_summary()

    if eval_data.get("status") == "evaluation_not_ready":
        st.info("ℹ️ " + eval_data.get("reason", "Evaluation benchmark has not been run yet. Click 'Run Benchmark Evaluation' above."))
    elif eval_data.get("summary"):
        summ = eval_data["summary"]
        rand_base = summ.get("random_baseline", {})
        rule_base = summ.get("rule_based", {})
        rf_base = summ.get("random_forest", {})

        st.markdown("### 🏆 Localization Performance Summary")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("#### 🎲 Random Baseline")
            st.metric("Top-1 Accuracy", f"{rand_base.get('top1_accuracy', 0.0) * 100:.1f}%")
            st.metric("Top-3 Accuracy", f"{rand_base.get('top3_accuracy', 0.0) * 100:.1f}%")
            st.metric("MRR", f"{rand_base.get('mrr', 0.0):.3f}")

        with col_m2:
            st.markdown("#### 📐 Rule-Based Model")
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

        with col_m3:
            st.markdown("#### 🌲 Random Forest")
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

        # Method Comparison Table
        st.markdown("### 📊 Method Comparison")
        comp_df = pd.DataFrame([
            {
                "Method": "Random Baseline (Theoretical)",
                "Top-1 Accuracy": f"{rand_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rand_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rand_base.get('mrr', 0.0):.3f}",
                "Evaluation Basis": "1 / N candidate steps",
            },
            {
                "Method": "Improved Rule-Based Localizer",
                "Top-1 Accuracy": f"{rule_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rule_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rule_base.get('mrr', 0.0):.3f}",
                "Evaluation Basis": "6 Structured Signals + Validation-Tuned Weights",
            },
            {
                "Method": "Random Forest Classifier",
                "Top-1 Accuracy": f"{rf_base.get('top1_accuracy', 0.0) * 100:.1f}%",
                "Top-3 Accuracy": f"{rf_base.get('top3_accuracy', 0.0) * 100:.1f}%",
                "MRR": f"{rf_base.get('mrr', 0.0):.3f}",
                "Evaluation Basis": "Scikit-Learn Balanced Forest on 6 Signals",
            },
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Known vs Held-Out Categories
        kvh = eval_data.get("known_vs_held_out")
        if kvh:
            st.markdown("### 🎯 Known Categories vs Held-Out Category")
            col_k, col_h = st.columns(2)
            with col_k:
                k_data = kvh.get("known_categories", {})
                st.markdown(f"#### 🏷️ Known Categories ({k_data.get('total_runs', 0)} test runs)")
                k_rule = k_data.get("rule_based", {})
                k_rf = k_data.get("random_forest", {})
                st.write(f"• **Rule-Based:** Top-1: `{k_rule.get('top1', 0.0)*100:.1f}%` | Top-3: `{k_rule.get('top3', 0.0)*100:.1f}%` | MRR: `{k_rule.get('mrr', 0.0):.3f}`")
                st.write(f"• **Random Forest:** Top-1: `{k_rf.get('top1', 0.0)*100:.1f}%` | Top-3: `{k_rf.get('top3', 0.0)*100:.1f}%` | MRR: `{k_rf.get('mrr', 0.0):.3f}`")

            with col_h:
                h_data = kvh.get("held_out_category", {})
                h_cat = h_data.get("category", "Held-Out")
                st.markdown(f"#### 🔒 Held-Out Category: `{h_cat}` ({h_data.get('total_runs', 0)} test runs)")
                h_rule = h_data.get("rule_based", {})
                h_rf = h_data.get("random_forest", {})
                st.write(f"• **Rule-Based:** Top-1: `{h_rule.get('top1', 0.0)*100:.1f}%` | Top-3: `{h_rule.get('top3', 0.0)*100:.1f}%` | MRR: `{h_rule.get('mrr', 0.0):.3f}`")
                st.write(f"• **Random Forest:** Top-1: `{h_rf.get('top1', 0.0)*100:.1f}%` | Top-3: `{h_rf.get('top3', 0.0)*100:.1f}%` | MRR: `{h_rf.get('mrr', 0.0):.3f}`")

        # Category Breakdown
        cats = eval_data.get("by_category", [])
        if cats:
            st.markdown("### 📋 Failure Category Breakdown")
            cat_rows = []
            for c in cats:
                sample_badge = "⚠️ Small Sample (<5)" if c.get("small_sample_warning") else "✅ Normal"
                cat_rows.append({
                    "Failure Category": c.get("category"),
                    "Type": "🔒 Held-Out" if c.get("is_held_out") else "Known",
                    "Test Runs": c.get("total_test_runs"),
                    "Sample Check": sample_badge,
                    "Rule Top-1": f"{c.get('rule_based_top1', 0.0)*100:.1f}%",
                    "Rule Top-3": f"{c.get('rule_based_top3', 0.0)*100:.1f}%",
                    "RF Top-1": f"{c.get('random_forest_top1', 0.0)*100:.1f}%",
                    "RF Top-3": f"{c.get('random_forest_top3', 0.0)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

        # Replay Recovery
        rep_rec = eval_data.get("replay_recovery", {})
        st.markdown("### 🔁 Replay Recovery Verification")
        c_r1, c_r2, c_r3, c_r4 = st.columns(4)
        with c_r1:
            st.metric("Branches Attempted", rep_rec.get("branches_attempted", 0))
        with c_r2:
            st.metric("Recovered", f"✅ {rep_rec.get('recovered', 0)}")
        with c_r3:
            st.metric("Not Recovered", f"❌ {rep_rec.get('not_recovered', 0)}")
        with c_r4:
            st.metric("Recovery Rate", f"{rep_rec.get('recovery_rate', 0.0)*100:.1f}%")

    st.divider()

    # ---------------- 3-Way Trace Comparator
    st.markdown("### 🔍 3-Way Trace Comparator (Original vs Alternative vs Reference)")
    st.caption("Aligns execution steps across Original, Alternative, and Verified Reference runs to observe intermediate divergence and recovery.")

    all_db_runs = repo_eval.list_runs(limit=60)
    failed_runs = [r for r in all_db_runs if r.get("status") == "failed"]
    all_run_ids = [r["run_id"] for r in all_db_runs]

    if not all_run_ids:
        st.info("No runs available in storage to compare.")
    else:
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        with col_sel1:
            orig_choice = st.selectbox(
                "Original Run (Failed)",
                [r["run_id"] for r in failed_runs] if failed_runs else all_run_ids,
                key="tc_orig_select",
            )
        with col_sel2:
            # Prefer replays of original run
            possible_alts = [r["run_id"] for r in all_db_runs if r.get("parent_run_id") == orig_choice]
            if not possible_alts:
                possible_alts = [r for r in all_run_ids if r != orig_choice]
            alt_choice = st.selectbox(
                "Alternative Run (Replay)",
                possible_alts if possible_alts else all_run_ids,
                key="tc_alt_select",
            )
        with col_sel3:
            succ_runs = [r["run_id"] for r in all_db_runs if r.get("status") == "success"]
            ref_opts = ["(Auto-Select Compatible Reference)"] + succ_runs
            ref_choice = st.selectbox("Reference Run", ref_opts, key="tc_ref_select")

        compare_btn = st.button("⚖️ Compare Executions Side-by-Side", type="secondary", use_container_width=True)

        if compare_btn and orig_choice and alt_choice:
            ref_id_arg = None if ref_choice.startswith("(") else ref_choice
            try:
                comp_result = fi_service.compare_traces(
                    original_run_id=orig_choice,
                    alternative_run_id=alt_choice,
                    reference_run_id=ref_id_arg,
                )

                st.markdown("#### 🔬 Recovery Verification")
                c_ver1, c_ver2, c_badge = st.columns([2, 2, 2])
                with c_ver1:
                    v_orig = comp_result.get("original_verifier", {})
                    if v_orig.get("passed"):
                        st.success("Original Verifier: ✅ PASSED")
                    else:
                        st.error(f"Original Verifier: ❌ {v_orig.get('reason', 'Failed')}")
                with c_ver2:
                    v_alt = comp_result.get("alternative_verifier", {})
                    if v_alt.get("passed"):
                        st.success("Alternative Verifier: ✅ PASSED")
                    else:
                        st.error(f"Alternative Verifier: ❌ {v_alt.get('reason', 'Failed')}")
                with c_badge:
                    if comp_result.get("recovered"):
                        st.balloons()
                        st.success("🎉 **VERIFIED RECOVERED**\n\nOutcome restored!")
                    else:
                        st.warning("⚠️ **NOT RECOVERED**\n\nConstraints violated.")

                if comp_result.get("downstream_effects"):
                    st.markdown("#### 🌊 Downstream Observable Effects")
                    for eff in comp_result["downstream_effects"]:
                        st.write(f"• {eff}")

                st.markdown("#### 📐 Aligned 3-Way Steps Table")
                steps_data = []
                for s in comp_result.get("steps", []):
                    ch_badge = "🔄 YES" if s.get("is_changed") else "Identical"
                    steps_data.append({
                        "Step ID": s.get("step_id"),
                        "Step Type": s.get("step_type"),
                        "Original Step": f"{s.get('original_tool') or s.get('step_type')} ({s.get('original_status')})",
                        "Alternative Step": f"{s.get('alternative_tool') or s.get('step_type')} ({s.get('alternative_status')})",
                        "Reference Step": f"{s.get('reference_tool') or s.get('step_type')} ({s.get('reference_status')})",
                        "Intervention / Changed?": ch_badge,
                        "Change Summary": s.get("change_description"),
                    })

                st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"Trace comparison error: {exc}")


# =====================================================================  TAB 4: ML Recommender
with tab_ml:
    st.subheader("🧠 Scikit-Learn Nearest Neighbors Laptop Similarity")
    st.write("Using Scikit-Learn's `StandardScaler` and `NearestNeighbors` (Cosine Distance).")
    if products:
        product_names = [p["name"] for p in products]
        selected_laptop = st.selectbox("Select a base laptop:", product_names)
        col_base, col_sim = st.columns([1, 2])
        base_item = next(p for p in products if p["name"] == selected_laptop)
        with col_base:
            st.markdown(f"### 🎯 {base_item['name']}")
            st.metric("Price", f"₹{base_item['price']:,}")
            st.metric("RAM / Storage", f"{base_item['ram_gb']}GB / {base_item['storage_gb']}GB")
            st.write(f"🎮 **Gaming:** {base_item.get('gaming_suitability', 'N/A')}/5")
            st.write(f"💻 **Programming:** {base_item.get('programming_suitability', 'N/A')}/5")
            st.write(f"⚡ **Dedicated GPU:** {'Yes' if base_item.get('dedicated_gpu') else 'No'}")
        with col_sim:
            st.markdown("### 🔍 Top Similar Alternatives")
            matches = get_similar_laptops(selected_laptop, products, nn_model, scaler)
            for m in matches:
                p = m["product"]
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.markdown(f"**{p['name']}**")
                        st.caption(f"{p['processor']} | {p['ram_gb']}GB RAM | {p['storage_gb']}GB SSD")
                    with c2:
                        st.markdown(f"**₹{p['price']:,}**")
                    with c3:
                        st.metric("Similarity", f"{m['similarity_score']}%")


# =====================================================================  TAB 4: Catalogue
with tab_catalog:
    st.subheader("📦 Available Catalogue (`data/products.json`)")
    if products:
        df = pd.DataFrame(products)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No products found.")
