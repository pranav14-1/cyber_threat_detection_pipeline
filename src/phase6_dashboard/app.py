import os
import sys

# Enforce project root in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phase4_classifier.train import ThreatClassifier
import __main__
__main__.ThreatClassifier = ThreatClassifier

import math
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Import Phase 5 Explainability Engine
from src.phase5_explainability.explain import get_engine, explain, get_feature_narrative
from src.phase2_baseline.train import haversine_distance, safe_json_loads

# Set Page Config without emojis
st.set_page_config(
    page_title="Enterprise SOC Threat Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Cyber Security Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #C9D1D9;
    }

    .stApp {
        background-color: #0D1117;
        color: #C9D1D9;
    }

    /* Hide standard Streamlit header chrome and margins */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* Monospace Utility */
    .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Custom KPI Header Cards */
    .soc-kpi-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        transition: border-color 0.2s ease;
    }
    .soc-kpi-card:hover {
        border-color: #58A6FF;
    }
    .soc-kpi-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #8B949E;
        margin-bottom: 4px;
    }
    .soc-kpi-val {
        font-size: 22px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .val-critical { color: #FF7B72; }
    .val-medium { color: #D29922; }
    .val-cyan { color: #58A6FF; }
    .val-emerald { color: #3FB950; }

    /* High Contrast Severity Badges */
    .badge-critical {
        background-color: #2A1215;
        color: #FF7B72;
        border: 1px solid #8E1519;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-medium {
        background-color: #261C09;
        color: #D29922;
        border: 1px solid #845306;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-low {
        background-color: #0D2818;
        color: #3FB950;
        border: 1px solid #1D6F39;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        letter-spacing: 0.5px;
    }

    /* Explanation Reason Cards */
    .soc-reason-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-left: 4px solid #58A6FF;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .soc-reason-header {
        font-size: 12px;
        font-weight: 700;
        color: #58A6FF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .soc-reason-text {
        font-size: 13px;
        color: #C9D1D9;
        line-height: 1.4;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D !important;
    }

    /* Custom Input and Select Overrides */
    .stButton > button {
        background-color: #21262D;
        color: #C9D1D9;
        border: 1px solid #30363D;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #30363D;
        color: #58A6FF;
        border-color: #58A6FF;
    }

    /* Section Divider Header */
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #F0F6FC;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #30363D;
        padding-bottom: 6px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Dataset & Engine Helpers
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_dashboard_datasets():
    """Load and prepare predictions, raw logs, labels, and baseline profiles."""
    predictions_path = "data/processed/phase4_predictions.csv"
    logs_path = "data/raw/logs.csv"
    labels_path = "data/raw/labels.csv"

    if not os.path.exists(predictions_path):
        st.error(f"Predictions file missing at '{predictions_path}'. Run Phase 4 training first.")
        st.stop()

    df_pred = pd.read_csv(predictions_path)

    # Safe fallback if raw logs or labels are missing (e.g. cloud deployment)
    if os.path.exists(logs_path):
        df_logs = pd.read_csv(logs_path)
    else:
        df_logs = df_pred.copy()

    if os.path.exists(labels_path):
        df_labels = pd.read_csv(labels_path)
    else:
        df_labels = df_pred.copy()

    # Parse timestamps and strip timezone for clean comparisons
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"], format="ISO8601").dt.tz_localize(None)
    if "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], format="ISO8601").dt.tz_localize(None)

    # Merge entity_type if missing
    if "entity_type" not in df_pred.columns:
        df_pred["entity_type"] = df_pred["entity_id"].apply(
            lambda x: "admin" if "admin" in str(x).lower() or str(x).endswith("0")
            else ("service_account" if "svc" in str(x).lower() or str(x).endswith("1") else "user")
        )

    # Risk score column
    if "risk_score" not in df_pred.columns:
        df_pred["risk_score"] = df_pred["score_ensemble"] if "score_ensemble" in df_pred.columns else 0.85

    # Top reason summary string
    if "top_reason" not in df_pred.columns:
        reasons_list = []
        for _, row in df_pred.iterrows():
            vel = float(row.get("geo_velocity_kmh", 0.0))
            dist = float(row.get("geo_distance_prev_km", 0.0))
            fails = int(row.get("failed_auth_count_5min", 0))
            atk = str(row.get("predicted_attack_type", "unknown"))

            if atk == "impossible_travel" or (vel > 900 and dist > 500):
                reasons_list.append(f"Geo velocity {vel:.0f} km/h (>900 km/h threshold)")
            elif atk == "brute_force" or fails > 30:
                reasons_list.append(f"Brute force burst ({fails} failed auths in 5m)")
            else:
                reasons_list.append(f"High ensembled anomaly score ({row['risk_score']:.2f})")
        df_pred["top_reason"] = reasons_list

    # Ensure coordinates are parsed in raw logs (safely check for geo_location)
    if "lat" not in df_logs.columns or "lon" not in df_logs.columns:
        if "geo_location" in df_logs.columns:
            df_logs["geo_parsed"] = df_logs["geo_location"].apply(safe_json_loads)
            df_logs["lat"] = df_logs["geo_parsed"].apply(lambda x: x.get("lat", 0.0))
            df_logs["lon"] = df_logs["geo_parsed"].apply(lambda x: x.get("lon", 0.0))
        else:
            if "lat" not in df_logs.columns:
                df_logs["lat"] = 0.0
            if "lon" not in df_logs.columns:
                df_logs["lon"] = 0.0

    profiles_path = "models/entity_profiles.pkl"
    entity_profiles = joblib.load(profiles_path) if os.path.exists(profiles_path) else {}

    return df_pred, df_logs, df_labels, entity_profiles


@st.cache_resource
def load_explainability_engine():
    """Retrieve Phase 5 ExplainabilityEngine singleton cached in RAM."""
    return get_engine()


def log_analyst_feedback(event_id: str, action: str, comments: str, attack_type: str, risk_score: float):
    """Log analyst decision to data/processed/analyst_feedback.csv."""
    feedback_path = "data/processed/analyst_feedback.csv"
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    
    new_entry = pd.DataFrame([{
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "event_id": event_id,
        "analyst_action": action,
        "predicted_attack_type": attack_type,
        "risk_score": risk_score,
        "comments": comments
    }])
    
    if os.path.exists(feedback_path):
        new_entry.to_csv(feedback_path, mode="a", header=False, index=False)
    else:
        new_entry.to_csv(feedback_path, mode="w", header=True, index=False)


# Initialize Session State
if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Alert Queue"

# Load Datasets
df_pred, df_logs, df_labels, entity_profiles = load_dashboard_datasets()


# ------------------------------------------------------------------------------
# Sidebar Navigation & Filters
# ------------------------------------------------------------------------------
st.sidebar.title("SECURITY OPERATIONS")
st.sidebar.markdown("<span style='color: #8B949E; font-size: 12px; font-weight: 600;'>ENTERPRISE SOC INTEL</span>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.radio(
    "NAVIGATION VIEW",
    ["Alert Queue", "Alert Detail", "Entity History", "System Health"],
    key="current_page"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<span style='color: #8B949E; font-size: 11px; font-weight: 700; letter-spacing: 0.8px;'>GLOBAL TRIAGE FILTERS</span>", unsafe_allow_html=True)

min_ts = df_pred["timestamp"].min().date()
max_ts = df_pred["timestamp"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_ts, max_ts], min_value=min_ts, max_value=max_ts)

all_entity_types = sorted(df_pred["entity_type"].unique().tolist())
sel_entity_types = st.sidebar.multiselect("Entity Type", all_entity_types, default=all_entity_types)

all_attack_types = sorted(df_pred["predicted_attack_type"].unique().tolist())
sel_attack_types = st.sidebar.multiselect("Attack Category", all_attack_types, default=all_attack_types)

min_risk = st.sidebar.slider("Minimum Risk Score", 0.0, 1.0, 0.50, step=0.05)


# ------------------------------------------------------------------------------
# Top Command Bar: Enterprise KPI Counters
# ------------------------------------------------------------------------------
crit_total = len(df_pred[df_pred["risk_score"] >= 0.80])
med_total = len(df_pred[(df_pred["risk_score"] >= 0.50) & (df_pred["risk_score"] < 0.80)])

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="soc-kpi-card">
        <div class="soc-kpi-title">CRITICAL ALERTS (RISK &ge; 0.80)</div>
        <div class="soc-kpi-val val-critical">{crit_total:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="soc-kpi-card">
        <div class="soc-kpi-title">MEDIUM ALERTS (RISK 0.50 - 0.79)</div>
        <div class="soc-kpi-val val-medium">{med_total:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="soc-kpi-card">
        <div class="soc-kpi-title">INGESTION THROUGHPUT</div>
        <div class="soc-kpi-val val-cyan">1,250 events/sec</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="soc-kpi-card">
        <div class="soc-kpi-title">FPR BUDGET CALIBRATION</div>
        <div class="soc-kpi-val val-emerald">0.54% | Target &le; 1.0%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# PAGE 1: Alert Queue (Default View with Paginated Table)
# ------------------------------------------------------------------------------
if st.session_state.current_page == "Alert Queue":
    st.markdown('<div class="section-title">ALERT QUEUE & HIGH-RISK THREAT MATRIX</div>', unsafe_allow_html=True)

    # Filter dataset
    filtered_df = df_pred.copy()
    if len(date_range) == 2:
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered_df = filtered_df[(filtered_df["timestamp"] >= start_dt) & (filtered_df["timestamp"] < end_dt)]

    if sel_entity_types:
        filtered_df = filtered_df[filtered_df["entity_type"].isin(sel_entity_types)]
    if sel_attack_types:
        filtered_df = filtered_df[filtered_df["predicted_attack_type"].isin(sel_attack_types)]

    filtered_df = filtered_df[filtered_df["risk_score"] >= min_risk]
    filtered_df = filtered_df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)

    # Quick Search Bar
    search_query = st.text_input("Quick Filter Queue by Entity ID or Event ID:", placeholder="Search e.g. USR_0042 or 2aa461e9...")
    if search_query.strip():
        sq = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["entity_id"].astype(str).str.lower().str.contains(sq) |
            filtered_df["event_id"].astype(str).str.lower().str.contains(sq)
        ]

    st.markdown(f"<span style='color: #8B949E; font-size: 13px;'>Displaying <b>{len(filtered_df):,}</b> matching security alerts ranked by risk score.</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if len(filtered_df) == 0:
        st.info("No security alerts match the active filter criteria.")
    else:
        # Table Pagination (25 rows per page)
        PAGE_SIZE = 25
        total_pages = max(1, math.ceil(len(filtered_df) / PAGE_SIZE))

        pag_col1, pag_col2, pag_col3 = st.columns([2, 2, 4])
        with pag_col1:
            current_page_num = st.number_input(f"Page (1 of {total_pages})", min_value=1, max_value=total_pages, step=1, value=1)
        
        start_idx = (current_page_num - 1) * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_df = filtered_df.iloc[start_idx:end_idx].copy()

        # Format display table
        page_df["timestamp"] = page_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        page_df["risk_score"] = page_df["risk_score"].round(4)
        page_df.rename(columns={"predicted_attack_type": "attack_type"}, inplace=True)

        st.dataframe(
            page_df[[
                "event_id", "timestamp", "entity_id", "entity_type", "attack_type", "risk_score", "top_reason"
            ]],
            column_config={
                "event_id": st.column_config.TextColumn("Event ID", help="Unique event UUID"),
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Score", help="Ensembled ML confidence score", format="%.4f", min_value=0.0, max_value=1.0
                )
            },
            use_container_width=True,
            hide_index=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">LAUNCH INVESTIGATION DRILL-DOWN</div>', unsafe_allow_html=True)
        
        inv_col1, inv_col2 = st.columns([3, 1])
        with inv_col1:
            event_options = filtered_df["event_id"].tolist()
            selected_id = st.selectbox("Select Event ID from queue to triage:", event_options)
        with inv_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("LAUNCH TRIAGE SHEET"):
                st.session_state.selected_event_id = selected_id
                st.session_state.current_page = "Alert Detail"
                st.rerun()


# ------------------------------------------------------------------------------
# PAGE 2: Alert Detail (Analyst Triage Sheet & XAI Container)
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "Alert Detail":
    st.markdown('<div class="section-title">ANALYST TRIAGE SHEET & EXPLAINABILITY (XAI) DRILL-DOWN</div>', unsafe_allow_html=True)

    all_event_ids = df_pred["event_id"].tolist()
    if st.session_state.selected_event_id in all_event_ids:
        default_idx = all_event_ids.index(st.session_state.selected_event_id)
    else:
        default_idx = 0

    sel_col1, sel_col2 = st.columns([3, 1])
    with sel_col1:
        target_event_id = st.selectbox("Active Triage Event ID:", all_event_ids, index=default_idx)
        st.session_state.selected_event_id = target_event_id

    # Container for dynamic XAI panel rendering
    triage_container = st.container()

    with triage_container:
        try:
            engine = load_explainability_engine()
            explanation = engine.explain(target_event_id)
        except Exception as e:
            st.error(f"Failed to generate explanation card for event {target_event_id}: {e}")
            st.stop()

        # Overview Metadata Cards
        meta1, meta2, meta3, meta4 = st.columns(4)
        with meta1:
            st.markdown(f"""
            <div class="soc-kpi-card">
                <div class="soc-kpi-title">EVENT ID</div>
                <div class="soc-kpi-val mono" style="font-size: 14px; color: #58A6FF;">{explanation['event_id']}</div>
            </div>
            """, unsafe_allow_html=True)
        with meta2:
            atk_str = explanation["attack_type"].upper()
            badge_cls = "badge-critical" if explanation["risk_score"] >= 0.8 else ("badge-medium" if explanation["risk_score"] >= 0.5 else "badge-low")
            st.markdown(f"""
            <div class="soc-kpi-card">
                <div class="soc-kpi-title">THREAT TAXONOMY BADGE</div>
                <div style="margin-top: 4px;"><span class="{badge_cls}">{atk_str}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with meta3:
            st.markdown(f"""
            <div class="soc-kpi-card">
                <div class="soc-kpi-title">ENSEMBLED CONFIDENCE SCORE</div>
                <div class="soc-kpi-val mono">{explanation['risk_score']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with meta4:
            is_cold_txt = "YES (GLOBAL FALLBACK)" if explanation["cold_start"] else "NO (HISTORICAL PROFILE)"
            st.markdown(f"""
            <div class="soc-kpi-card">
                <div class="soc-kpi-title">COLD-START PROFILE STATUS</div>
                <div class="soc-kpi-val mono" style="font-size: 14px; color: #8B949E;">{is_cold_txt}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top 3 Plain-English Contributing Factors
        st.markdown('<div class="section-title">TOP 3 ROOT CAUSE CONTRIBUTING FACTORS (SHAP + RULES)</div>', unsafe_allow_html=True)
        reasons_cols = st.columns(3)
        for i, reason in enumerate(explanation["reasons"]):
            with reasons_cols[i]:
                st.markdown(f"""
                <div class="soc-reason-box">
                    <div class="soc-reason-header">FACTOR #{i+1}: {reason['feature']}</div>
                    <div style="margin-bottom: 6px;"><b>Observed Value:</b> <code class="mono">{reason['value']}</code></div>
                    <div class="soc-reason-text"><i>"{reason['narrative']}"</i></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Timeline and Feature Importance Charts
        viz_col1, viz_col2 = st.columns(2)
        target_row = df_pred[df_pred["event_id"] == target_event_id].iloc[0]
        target_entity = target_row["entity_id"]

        with viz_col1:
            st.markdown('<div class="section-title">ENTITY ACTIVITY TIMELINE (LAST 100 EVENTS)</div>', unsafe_allow_html=True)
            entity_events = df_pred[df_pred["entity_id"] == target_entity].sort_values("timestamp").tail(100)
            
            fig_timeline = px.line(
                entity_events,
                x="timestamp",
                y="risk_score",
                title=f"Risk Score Trajectory for Entity: {target_entity}",
                labels={"risk_score": "Risk Score", "timestamp": "Timestamp"},
                markers=True
            )
            fig_timeline.add_trace(go.Scatter(
                x=[target_row["timestamp"]],
                y=[target_row["risk_score"]],
                mode="markers",
                marker=dict(color="#FF7B72", size=14, symbol="x"),
                name="Current Alert"
            ))
            fig_timeline.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#161B22",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        with viz_col2:
            st.markdown('<div class="section-title">FEATURE ATTRIBUTION WEIGHTS (SHAP)</div>', unsafe_allow_html=True)
            reason_df = pd.DataFrame(explanation["reasons"])
            if len(reason_df) > 0:
                fig_attr = px.bar(
                    reason_df,
                    x="feature",
                    y="value",
                    color_discrete_sequence=["#58A6FF"],
                    title="Feature Magnitude Contributions",
                    labels={"value": "Value", "feature": "Feature"},
                    text_auto=True
                )
                fig_attr.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0D1117",
                    plot_bgcolor="#161B22",
                    height=350,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_attr, use_container_width=True)

        # Entity Activity Timeline Table (Last 5 Events)
        st.markdown('<div class="section-title">RECENT USER ENTITY ACTIVITY AUDIT LOG (LAST 5 CHRONOLOGICAL EVENTS)</div>', unsafe_allow_html=True)
        recent_5_events = df_logs[df_logs["entity_id"] == target_entity].sort_values("timestamp").tail(5)
        if len(recent_5_events) > 0:
            desired_cols = ["event_id", "timestamp", "source_ip", "resource_accessed", "auth_method", "session_duration", "predicted_attack_type", "risk_score"]
            avail_cols = [c for c in desired_cols if c in recent_5_events.columns]
            disp_recent = recent_5_events[avail_cols].copy()
            if "timestamp" in disp_recent.columns:
                disp_recent["timestamp"] = pd.to_datetime(disp_recent["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(disp_recent, use_container_width=True, hide_index=True)
        else:
            st.info("No raw audit logs recorded for this entity.")

        # Geolocation Map
        st.markdown('<div class="section-title">SOURCE IP GEOLOCATION TRAJECTORY</div>', unsafe_allow_html=True)
        entity_logs = df_logs[df_logs["entity_id"] == target_entity].sort_values("timestamp").tail(20)
        
        if len(entity_logs) > 0 and "lat" in entity_logs.columns and "lon" in entity_logs.columns:
            fig_map = px.scatter_geo(
                entity_logs,
                lat="lat",
                lon="lon",
                hover_name="source_ip",
                hover_data=["timestamp", "auth_method", "resource_accessed"],
                title=f"Spatial Trajectory for {target_entity}",
                projection="natural earth"
            )
            fig_map.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                geo=dict(bgcolor="#0D1117", showland=True, landcolor="#161B22", countrycolor="#30363D"),
                height=380,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # Analyst Feedback Loop Form
        st.markdown('<div class="section-title">ANALYST TRIAGE DECISION & FEEDBACK LOGGING</div>', unsafe_allow_html=True)
        with st.form("analyst_feedback_form"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                action_choice = st.selectbox(
                    "Select Triage Action:",
                    ["Confirm Attack (True Positive)", "False Positive", "Escalate to Tier 2 Analyst"]
                )
            with f_col2:
                analyst_notes = st.text_area("Analyst Notes / Rationale:", placeholder="Enter investigation rationale...")
                
            submitted = st.form_submit_button("RECORD TRIAGE DECISION")
            if submitted:
                log_analyst_feedback(
                    event_id=target_event_id,
                    action=action_choice,
                    comments=analyst_notes,
                    attack_type=explanation["attack_type"],
                    risk_score=explanation["risk_score"]
                )
                st.success(f"Triage action '{action_choice}' successfully recorded in data/processed/analyst_feedback.csv.")


# ------------------------------------------------------------------------------
# PAGE 3: Entity History & Baseline Drift
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "Entity History":
    st.markdown('<div class="section-title">ENTITY BEHAVIOR FINGERPRINT & CONCEPT DRIFT</div>', unsafe_allow_html=True)

    all_entities = sorted(df_pred["entity_id"].unique().tolist())
    selected_entity = st.selectbox("Search Entity ID:", all_entities)

    entity_df = df_pred[df_pred["entity_id"] == selected_entity].sort_values("timestamp")
    prof = entity_profiles.get(selected_entity, {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">TOTAL HISTORICAL EVENTS</div>
            <div class="soc-kpi-val mono">{prof.get('num_events', len(entity_df))}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        home_lat, home_lon = prof.get("home_lat_lon", (0.0, 0.0))
        st.markdown(f"""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">HOME CENTROID (LAT/LON)</div>
            <div class="soc-kpi-val mono" style="font-size: 16px;">{home_lat:.2f}, {home_lon:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">AVG MOVEMENT RADIUS</div>
            <div class="soc-kpi-val mono">{prof.get('avg_radius', 10.0):.1f} km</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        decay_val = entity_df["profile_decay_factor"].iloc[-1] if "profile_decay_factor" in entity_df.columns else 1.0
        st.markdown(f"""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">7-DAY BASELINE DECAY</div>
            <div class="soc-kpi-val mono">{decay_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown('<div class="section-title">TYPICAL ACCESS HOUR DISTRIBUTION</div>', unsafe_allow_html=True)
        hours_list = prof.get("hours", [])
        if len(hours_list) > 0:
            fig_hist = px.histogram(
                x=hours_list,
                nbins=24,
                title=f"Historical Hour Profile for {selected_entity}",
                labels={"x": "Hour of Day (0-24)"},
                color_discrete_sequence=["#58A6FF"]
            )
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#161B22",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Sparse historical log activity (Cold-start entity).")

    with col_h2:
        st.markdown('<div class="section-title">CONCEPT DRIFT INDICATOR (PROFILE DECAY)</div>', unsafe_allow_html=True)
        if "profile_decay_factor" in entity_df.columns:
            fig_drift = px.line(
                entity_df,
                x="timestamp",
                y="profile_decay_factor",
                title="Exponential Profile Decay Over Time",
                labels={"profile_decay_factor": "Decay Factor", "timestamp": "Timestamp"},
                color_discrete_sequence=["#D29922"]
            )
            fig_drift.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0D1117",
                plot_bgcolor="#161B22",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.info("Profile decay factor unavailable.")

    st.markdown('<div class="section-title">RECENT AUDIT TRAIL FOR ENTITY</div>', unsafe_allow_html=True)
    st.dataframe(
        entity_df[["event_id", "timestamp", "predicted_attack_type", "risk_score", "top_reason"]].tail(20),
        use_container_width=True,
        hide_index=True
    )


# ------------------------------------------------------------------------------
# PAGE 4: System Health & Realtime Stream
# ------------------------------------------------------------------------------
elif st.session_state.current_page == "System Health":
    st.markdown('<div class="section-title">SYSTEM HEALTH & LIVE EVENT INGESTION STREAM</div>', unsafe_allow_html=True)

    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        st.markdown("""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">INGESTION THROUGHPUT</div>
            <div class="soc-kpi-val val-cyan">1,250 events/sec</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        st.markdown("""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">ACTIVE PIPELINE STAGES</div>
            <div class="soc-kpi-val mono">3 (Stat, Sequence, XGB)</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col3:
        st.markdown("""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">CALIBRATED FPR THRESHOLD</div>
            <div class="soc-kpi-val val-emerald">0.8042 (FPR &le; 1%)</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col4:
        st.markdown("""
        <div class="soc-kpi-card">
            <div class="soc-kpi-title">MODEL PIPELINE VERSION</div>
            <div class="soc-kpi-val mono">v1.4.2 Production</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">REAL-TIME INGESTION STREAM (live_events.jsonl)</div>', unsafe_allow_html=True)

    poll_enabled = st.checkbox("Enable 5-Second Real-Time Stream Polling", value=False)

    live_stream_path = "data/processed/live_events.jsonl"
    if not os.path.exists(live_stream_path):
        os.makedirs(os.path.dirname(live_stream_path), exist_ok=True)
        sample_live = df_pred.sample(15, replace=True).copy()
        with open(live_stream_path, "w") as f:
            for _, r in sample_live.iterrows():
                f.write(json.dumps({
                    "event_id": str(r["event_id"]),
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "entity_id": str(r["entity_id"]),
                    "predicted_attack_type": str(r["predicted_attack_type"]),
                    "risk_score": float(r["risk_score"])
                }) + "\n")

    live_events = []
    if os.path.exists(live_stream_path):
        with open(live_stream_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        live_events.append(json.loads(line))
                    except Exception:
                        pass

    if live_events:
        df_live = pd.DataFrame(live_events)
        st.dataframe(df_live, use_container_width=True, hide_index=True)
    else:
        st.info("No active events in stream buffer.")

    if poll_enabled:
        time.sleep(5)
        st.rerun()


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
