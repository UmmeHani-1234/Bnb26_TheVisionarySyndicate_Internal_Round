"""Streamlit Web Interface for the Black Box Laptop Recommendation Agent.

Features:
- Live Agent Execution & Interactive Query Input
- Observable Execution Timeline (Agent events, tool calls, and outputs)
- Scikit-Learn Content-Based ML Laptop Recommender (Nearest Neighbors on specs)
- Dataset Explorer and Event Log Inspector
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from agent.agent import ConfigError, LaptopAgent
from agent.events import EventType

load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Black Box | Laptop Agent & ML Recommender",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/products.json")


@st.cache_data
def load_products() -> List[Dict[str, Any]]:
    """Loads products dataset."""
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def build_ml_recommender(products: List[Dict[str, Any]]):
    """Builds a Scikit-Learn NearestNeighbors model for laptop similarity."""
    if not products:
        return None, None, None

    features = []
    for p in products:
        # Vector: [price, ram_gb, storage_gb, gaming_suitability, programming_suitability, dedicated_gpu]
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

    # Scikit-Learn NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=min(4, len(products)), metric="cosine")
    nn_model.fit(X_scaled)
    return nn_model, scaler, X_scaled


def get_similar_laptops(product_name: str, products: List[Dict[str, Any]], model, scaler) -> List[Dict[str, Any]]:
    """Finds top similar laptops using Scikit-Learn NearestNeighbors."""
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
        if i != idx:  # exclude self
            similar.append({"product": products[i], "similarity_score": round((1.0 - float(dist)) * 100, 1)})
    return similar


# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-title">💻 Black Box Agent & ML Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">LangChain Observable Target Agent with Scikit-Learn Laptop Intelligence</div>', unsafe_allow_html=True)

# Load data and ML model
products = load_products()
nn_model, scaler, _ = build_ml_recommender(products)

# Sidebar
with st.sidebar:
    st.subheader("💡 Quick Example Queries")
    presets = [
        "Find a laptop under ₹70,000 suitable for programming and gaming.",
        "I need a lightweight laptop for college under ₹50,000 with good battery.",
        "Suggest the best laptop for heavy gaming within ₹90,000.",
        "Show me options that are within ₹60,000 with at least 16GB RAM.",
    ]
    selected_preset = st.radio("Choose a template prompt:", presets, index=0)

    st.divider()
    st.caption("Black Box Phase 1 | Observability & ML Analysis")

# Main tabs
tab_agent, tab_ml, tab_catalog = st.tabs(["🤖 Agent Execution", "🧠 Scikit-Learn ML Recommender", "📦 Laptop Catalogue"])

# ----------------- TAB 1: Agent Execution -----------------
with tab_agent:
    user_query = st.text_input("Enter your laptop requirement:", value=selected_preset)
    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)

    if run_btn:
        if not user_query.strip():
            st.warning("Please enter a valid request.")
        else:
            with st.spinner("Agent is reasoning and executing tools..."):
                try:
                    agent = LaptopAgent()
                    result = agent.run(user_query)

                    st.subheader("📋 Final Recommendation")
                    st.success(result.final_response)

                    # Observable Timeline
                    st.subheader("⏱️ Observable Execution Timeline")
                    events_list = [e.to_dict() for e in result.events]

                    for idx, ev in enumerate(result.events, start=1):
                        ev_type = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                        with st.expander(f"[{idx}] {ev_type.upper()} {f'— Tool: {ev.tool_name}' if ev.tool_name else ''}", expanded=True):
                            st.write(f"**Status:** `{ev.status.value if hasattr(ev.status, 'value') else ev.status}`")
                            if ev.summary:
                                st.info(f"**Summary:** {ev.summary}")
                            if ev.input:
                                st.json({"input": ev.input})
                            if ev.output:
                                st.json({"output": ev.output})

                    # Raw Events Export
                    st.download_button(
                        label="📥 Download Trace Events (JSON)",
                        data=json.dumps(events_list, indent=2),
                        file_name="agent_events.json",
                        mime="application/json",
                    )

                except ConfigError as ce:
                    st.error(f"Configuration Error: {ce}")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

# ----------------- TAB 2: Scikit-Learn Recommender -----------------
with tab_ml:
    st.subheader("🧠 Scikit-Learn Nearest Neighbors Laptop Similarity")
    st.write("Using Scikit-Learn's `StandardScaler` and `NearestNeighbors` (Cosine Distance) to calculate spec similarities across laptops.")

    if products:
        product_names = [p["name"] for p in products]
        selected_laptop = st.selectbox("Select a base laptop to inspect similar ML matches:", product_names)

        col_base, col_sim = st.columns([1, 2])
        base_item = next(p for p in products if p["name"] == selected_laptop)

        with col_base:
            st.markdown(f"### 🎯 Selected Laptop\n**{base_item['name']}**")
            st.metric("Price", f"₹{base_item['price']:,}")
            st.metric("RAM / Storage", f"{base_item['ram_gb']}GB / {base_item['storage_gb']}GB")
            st.write(f"🎮 **Gaming Suitability:** {base_item.get('gaming_suitability', 'N/A')}/5")
            st.write(f"💻 **Programming Suitability:** {base_item.get('programming_suitability', 'N/A')}/5")
            st.write(f"⚡ **Dedicated GPU:** {'Yes' if base_item.get('dedicated_gpu') else 'No'}")

        with col_sim:
            st.markdown("### 🔍 Top Similar Alternatives (Scikit-Learn ML)")
            matches = get_similar_laptops(selected_laptop, products, nn_model, scaler)
            for m in matches:
                p = m["product"]
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.markdown(f"**{p['name']}**")
                        st.caption(f"{p['processor']} | {p['ram_gb']}GB RAM | {p['storage_gb']}GB SSD | {p.get('gpu', 'Integrated')}")
                    with c2:
                        st.markdown(f"**₹{p['price']:,}**")
                        st.caption(f"Category: {p.get('category', 'general')}")
                    with c3:
                        st.metric("Similarity", f"{m['similarity_score']}%")

# ----------------- TAB 3: Catalogue -----------------
with tab_catalog:
    st.subheader("📦 Available Catalogue (`data/products.json`)")
    if products:
        df = pd.DataFrame(products)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No products found.")
