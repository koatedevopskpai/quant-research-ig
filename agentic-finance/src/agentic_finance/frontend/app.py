"""
frontend/app.py — Streamlit dashboard for the quant-research-ig REST API.

USAGE:
    streamlit run agentic_finance/frontend/app.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="quant-research-ig",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 quant-research-ig · Agentic Trading Dashboard")


# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.header("API Status")
try:
    r = requests.get(f"{API_BASE}/risk", timeout=3)
    st.sidebar.success(f"API reachable ({r.status_code})")
except requests.ConnectionError:
    st.sidebar.error("API not running — start with:\n\n  `uvicorn agentic_finance.api.server:app`")
    st.stop()


st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", "Research", "Execution", "Risk", "Portfolio", "Audit",
])


# ── Helpers ─────────────────────────────────────────────────────────

def api_get(path: str) -> Any:
    try:
        return requests.get(f"{API_BASE}{path}", timeout=5).json()
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return None


def api_post(path: str, body: dict | None = None) -> Any:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return None


# ── Pages ───────────────────────────────────────────────────────────

if page == "Overview":

    col1, col2, col3, col4 = st.columns(4)
    risk = api_get("/risk") or {}
    col1.metric("VaR (parametric)", f"{risk.get('var_parametric', 0):.2%}")
    col2.metric("Drawdown", f"{risk.get('current_drawdown', 0):.2%}")
    col3.metric("Positions", risk.get("n_positions", 0))
    col4.metric("Daily P&L", f"{risk.get('daily_pnl', 0):.2f}")

    st.subheader("Recent Audit Decisions")
    audit = api_get("/audit?limit=10") or []
    if audit:
        cols = st.columns([3, 2, 2, 5])
        for e in audit:
            cols[0].write(e["decision_type"])
            cols[1].write(e["agent"])
            cols[2].write(e["status"])
            cols[3].write(e["description"][:80])
    else:
        st.info("No audit entries yet")

elif page == "Research":

    st.subheader("Run Research Model")
    col1, col2 = st.columns(2)
    epic = col1.text_input("Epic", "IX.D.SPTRD.IFD.IP")
    model = col2.selectbox("Model", ["regime", "forecast", "lstm", "sentiment"])

    if st.button("Run Research"):
        with st.spinner("Running model..."):
            resp = api_post(f"/research/{epic}", {"model_type": model})
        if resp:
            st.json(resp["model_output"])
            st.markdown(f"**Summary:** {resp['summary']}")

elif page == "Execution":

    st.subheader("Submit Signal")
    col1, col2, col3 = st.columns(3)
    epic = col1.text_input("Epic", "IX.D.SPTRD.IFD.IP", key="exec_epic")
    direction = col2.slider("Direction", -1.0, 1.0, 0.5)
    confidence = col3.slider("Confidence", 0.0, 1.0, 0.7)
    price = st.number_input("Current Price", min_value=0.01, value=100.0)

    if st.button("Submit Signal"):
        with st.spinner("Processing signal..."):
            resp = api_post("/signal", {
                "epic": epic,
                "direction": direction,
                "confidence": confidence,
                "source": "streamlit",
                "current_price": price,
            })
        if resp and "detail" not in resp:
            st.success(f"Order created: {resp['side']} {resp['size']} @ {resp.get('limit_price', 'market')}")
            st.json(resp)
        else:
            detail = resp.get("detail", "Unknown error") if resp else "API unreachable"
            st.error(detail)

    st.subheader("Positions")
    positions = api_get("/positions") or {}
    if positions:
        for epic_, size_ in positions.items():
            st.write(f"{epic_}: {size_:.2f}")
    else:
        st.info("No open positions")

elif page == "Risk":

    st.subheader("Risk Metrics")
    if st.button("Refresh"):
        st.rerun()

    risk = api_get("/risk") or {}
    if risk:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Historical VaR", f"{risk.get('var_historical', 0):.2%}")
            st.metric("Parametric VaR", f"{risk.get('var_parametric', 0):.2%}")
            st.metric("Monte Carlo VaR", f"{risk.get('var_monte_carlo', 0):.2%}")
        with col2:
            st.metric("Drawdown", f"{risk.get('current_drawdown', 0):.2%}")
            st.metric("Daily P&L", f"{risk.get('daily_pnl', 0):.2f}")
            st.metric("Total Exposure", f"{risk.get('total_exposure', 0):.2f}")

        if risk.get("is_breached"):
            st.error("🚨 Risk limit breached!")
            for b in risk.get("breaches", []):
                st.warning(b)
        else:
            st.success("All limits within tolerance")

    if st.button("Check Risk (force audit)"):
        result = api_post("/risk/check")
        if result:
            st.json(result)

    if st.button("Kill Switch"):
        result = api_post("/risk/kill")
        if result:
            st.warning(f"Kill switch activated: {result}")

elif page == "Portfolio":

    st.subheader("Current Allocation")
    portfolio = api_get("/portfolio") or {}
    if portfolio:
        st.metric("Total Capital", f"${portfolio.get('total_capital', 0):,.2f}")
        st.metric("Cash", f"${portfolio.get('cash', 0):,.2f}")
        for alloc in portfolio.get("allocations", []):
            with st.container(border=True):
                st.write(f"**{alloc['strategy_id']}**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Weight", f"{alloc['weight']:.1%}")
                col2.metric("Capital", f"${alloc['capital']:,.2f}")
                col3.write(alloc.get("reason", ""))

    st.subheader("Rebalance")
    method = st.selectbox("Method", ["equal", "risk_parity", "mean_variance"])
    if st.button("Rebalance"):
        with st.spinner("Rebalancing..."):
            resp = api_post(f"/allocate/{method}")
        if resp:
            st.success("Rebalanced!")
            st.json(resp)

    st.subheader("Registered Strategies")
    strategies = api_get("/portfolio/strategies") or {}
    st.json(strategies)

elif page == "Audit":

    st.subheader("Audit Trail")
    col1, col2, col3 = st.columns(3)
    agent_filter = col1.text_input("Agent filter")
    type_filter = col2.text_input("Decision type filter")
    limit = col3.number_input("Limit", min_value=1, max_value=500, value=50)

    params = {"limit": limit}
    if agent_filter:
        params["agent"] = agent_filter
    if type_filter:
        params["decision_type"] = type_filter

    audit = api_get(f"/audit?{'&'.join(f'{k}={v}' for k, v in params.items())}") or []
    if audit:
        for e in audit:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{e['decision_type']}** · {e['agent']}")
                col2.write(e["status"])
                st.write(f"_{e['description'][:120]}_")
                if e["market_context"]:
                    with st.expander("Context"):
                        st.json(e["market_context"])
    else:
        st.info("No audit entries match the filters")

    if st.button("Export All as JSON"):
        export = requests.get(f"{API_BASE}/audit/export", timeout=5)
        if export.ok:
            st.download_button(
                "Download audit.json",
                export.text,
                file_name="audit.json",
            )
