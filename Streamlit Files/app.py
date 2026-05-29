from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fishing Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── File paths ────────────────────────────────────────────────────
# Resolves to the Data folder relative to app.py regardless of
# where Streamlit launches the process from
DATA_DIR = Path(__file__).parent.parent / "Python Files" / "Datasets"

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stApp { background-color: #0d1117; color: #c9d1d9; }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    .section-header {
        font-size: 11px;
        font-weight: 500;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 32px;
    }

    .page-title {
        font-size: 22px;
        font-weight: 400;
        color: #e6edf3;
        letter-spacing: -0.01em;
    }

    .page-sub {
        font-size: 13px;
        color: #8b949e;
        margin-top: 4px;
        line-height: 1.5;
    }

    .stRadio label { font-size: 13px; color: #c9d1d9; }

    .stSelectbox label, .stSlider label, .stMultiSelect label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 12px 16px;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #58a6ff;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .stDataFrame { border: 1px solid #21262d; border-radius: 4px; }
    hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────

# Public Carto basemap — no Mapbox token required
BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

GEAR_LABELS = {
    "drifting_longlines": "Drifting longlines",
    "trawlers":           "Trawlers",
    "fixed_gear":         "Fixed gear",
    "purse_seines":       "Purse seines",
    "trollers":           "Trollers",
    "pole_and_line":      "Pole and line",
    "unknown":            "Unknown",
}


def label_gear(raw):
    return GEAR_LABELS.get(raw, raw.replace("_", " ").title())


# ── Plotly theme helper ───────────────────────────────────────────

def apply_theme(fig, title=None, height=None, xaxis_title=None, yaxis_title=None,
                xaxis_fmt=None, barmode=None, legend_inside=False):
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Sans", color="#c9d1d9", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
            x=0.6 if legend_inside else None,
            y=0.95 if legend_inside else None,
        ),
    )
    fig.update_xaxes(gridcolor="#21262d", linecolor="#21262d", tickcolor="#8b949e")
    fig.update_yaxes(gridcolor="#21262d", linecolor="#21262d", tickcolor="#8b949e")
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color="#e6edf3")))
    if height:
        fig.update_layout(height=height)
    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)
    if xaxis_fmt:
        fig.update_xaxes(tickformat=xaxis_fmt)
    if barmode:
        fig.update_layout(barmode=barmode)
    return fig


# ── Data loaders ──────────────────────────────────────────────────

@st.cache_data
def load_windows():
    df = pd.read_csv(DATA_DIR / "windows_features_slim.csv")
    df["win_start"]  = pd.to_datetime(df["win_start"])
    df["gear_label"] = df["shiptype"].apply(label_gear)
    return df


@st.cache_data
def load_predictions():
    df = pd.read_csv(DATA_DIR / "unlabelled_predictions_slim.csv")
    df["win_start"]  = pd.to_datetime(df["win_start"])
    df["gear_label"] = df["shiptype"].apply(label_gear)
    return df


@st.cache_data
def load_ping_sample():
    """
    Loads the pre-sampled ping map data.
    Sampling and label creation were done in the notebook to keep app load fast.
    """
    df = pd.read_csv(DATA_DIR / "ping_map_sample.csv")
    df["gear_label"] = df["shiptype"].apply(label_gear)
    df["colour"] = df["label"].apply(
        lambda x: [248, 81, 73, 180] if x == 1 else [88, 166, 255, 120]
    )
    return df


@st.cache_data
def load_risk():
    df = pd.read_csv(DATA_DIR / "gfw_risk_scored.csv")
    df["event_start"] = pd.to_datetime(df["event_start"])
    return df

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="page-title">Fishing Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Two-component system for detecting and scoring '
        'fishing activity from AIS vessel tracks.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Fishing Classifier", "IUU Risk Scoring"],
        index=0,
        help=(
            "Classifier: kinematic model trained on labelled AIS windows.\n"
            "Risk Scoring: spatial risk analysis of confirmed GFW fishing events."
        )
    )

    st.markdown("---")

    if mode == "Fishing Classifier":
        st.markdown("**Filters**")
        gear_display        = ["All"] + sorted(GEAR_LABELS.values())
        selected_gear_label = st.selectbox("Gear type", gear_display)
        selected_gear_raw   = (
            None if selected_gear_label == "All"
            else {v: k for k, v in GEAR_LABELS.items()}.get(
                selected_gear_label, selected_gear_label
            )
        )
        prob_threshold = st.slider("Min fishing probability", 0.0, 1.0, 0.5, 0.05)
        show_mode      = st.radio(
            "Show windows",
            ["Predicted fishing", "All", "Predicted not fishing"]
        )
        st.markdown("---")
        st.markdown("**Map**")
        map_class = st.radio(
            "Ping class",
            ["Both", "Fishing only", "Not fishing only"],
            help="Filter which ping class is shown on the training data map."
        )

    else:
        st.markdown("**Filters**")
        min_risk       = st.slider("Min risk score", 1.0, 4.0, 1.0, 0.5)
        show_mpa_only  = st.checkbox("MPA events only")
        show_high_seas = st.checkbox("High seas only")


# ── Mode 1: Fishing Classifier ────────────────────────────────────
if mode == "Fishing Classifier":

    st.markdown('<div class="page-title">Fishing Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">LightGBM classifier trained on 160k windowed AIS segments '
        'from 258 vessels. Features are derived from raw ping kinematics — speed variance, '
        'heading entropy, and displacement ratio — without using any pre-existing fishing '
        'labels as inputs.</div>',
        unsafe_allow_html=True
    )

    windows_df = load_windows()
    predict_df = load_predictions()
    ping_df    = load_ping_sample()

    # Apply gear filter
    if selected_gear_raw:
        windows_df = windows_df[windows_df["shiptype"] == selected_gear_raw]
        predict_df = predict_df[predict_df["shiptype"] == selected_gear_raw]
        ping_df    = ping_df[ping_df["shiptype"] == selected_gear_raw]

    # Apply prediction filter
    if show_mode == "Predicted fishing":
        predict_df = predict_df[predict_df["fishing_pred"] == 1]
    elif show_mode == "Predicted not fishing":
        predict_df = predict_df[predict_df["fishing_pred"] == 0]
    predict_df = predict_df[predict_df["fishing_prob"] >= prob_threshold]

    # Apply map class filter
    if map_class == "Fishing only":
        map_ping_df = ping_df[ping_df["label"] == 1]
    elif map_class == "Not fishing only":
        map_ping_df = ping_df[ping_df["label"] == 0]
    else:
        map_ping_df = ping_df

    # ── Training data overview ────────────────────────────────────
    st.markdown('<div class="section-header">Training Data Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total windows",   f"{len(windows_df):,}")
    c2.metric("Fishing windows", f"{windows_df['label'].sum():,}")
    c3.metric("Vessels",         f"{windows_df['mmsi'].nunique()}")
    c4.metric("Gear types",      f"{windows_df['shiptype'].nunique()}")
    c5.metric("Model accuracy",  "85%")

    # ── Ping location map ─────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Training Ping Locations</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub" style="margin-bottom:14px">50,000 sampled pings from the '
        'labelled training dataset. Red indicates confirmed fishing activity. '
        'Blue indicates confirmed non-fishing. Filter by gear type or ping class '
        'using the sidebar controls.</div>',
        unsafe_allow_html=True
    )

    ping_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_ping_df,
        get_position="[lon, lat]",
        get_fill_color="colour",
        get_radius=18000,
        radius_min_pixels=2,
        radius_max_pixels=8,
        pickable=True,
        opacity=0.7,
    )

    ping_tooltip = {
        "html": (
            "<div style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            "background:#0d1117;border:1px solid #21262d;padding:8px 12px;"
            "border-radius:4px;color:#c9d1d9'>"
            "<b style='color:#e6edf3'>{gear_label}</b><br>"
            "Speed: {speed} kn<br>"
            "<span style='color:#8b949e'>Label: {label}</span>"
            "</div>"
        )
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[ping_layer],
            initial_view_state=pdk.ViewState(
                latitude=20, longitude=0, zoom=1.2, pitch=0
            ),
            tooltip=ping_tooltip,
            map_style=BASEMAP,
        ),
        use_container_width=True
    )

    st.markdown(
        '<div style="display:flex;gap:20px;margin-top:6px;font-size:11px;color:#8b949e">'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        'background:#f85149;margin-right:5px;vertical-align:middle"></span>Fishing</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        'background:#58a6ff;margin-right:5px;vertical-align:middle"></span>Not fishing</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Feature distributions ─────────────────────────────────────
    st.markdown(
        '<div class="section-header">Feature Distributions — Fishing vs Not Fishing</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub" style="margin-bottom:16px">Each histogram compares the '
        'distribution of a kinematic feature across labelled fishing and non-fishing windows. '
        'Separation between the two classes indicates predictive value.</div>',
        unsafe_allow_html=True
    )

    features_to_plot = [
        ("net_displacement_ratio", "Net displacement ratio",
         "Bounding box diagonal / total path length. Near 0 = vessel looped; near 1 = straight transit."),
        ("traj_entropy",           "Trajectory entropy",
         "Shannon entropy of heading distribution across 8 compass bins. High = many directions = looping."),
        ("speed_mean",             "Mean speed (knots)",
         "Average vessel speed across the window. Fishing trawlers operate at 2–4 knots."),
        ("speed_std",              "Speed std dev (knots)",
         "Standard deviation of speed. High variance indicates transitions between fishing and repositioning."),
        ("dist_port_km",           "Distance from port (km)",
         "Mean distance from nearest port during the window. Fishing activity concentrates far offshore."),
        ("turn_rate",              "Turn rate (deg/min)",
         "Mean absolute heading change per minute. Fishing vessels change direction frequently."),
    ]

    for row_start in range(0, len(features_to_plot), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(features_to_plot):
                break
            feat, feat_label, desc = features_to_plot[idx]
            if feat not in windows_df.columns:
                continue

            not_fish = windows_df[windows_df["label"] == 0][feat].dropna()
            fishing  = windows_df[windows_df["label"] == 1][feat].dropna()
            clip_val = not_fish.quantile(0.99)
            not_fish = not_fish.clip(upper=clip_val)
            fishing  = fishing.clip(upper=clip_val)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=not_fish, name="Not fishing", opacity=0.65,
                marker_color="#58a6ff", nbinsx=60,
                hovertemplate="%{x:.3f}: %{y}<extra>Not fishing</extra>"
            ))
            fig.add_trace(go.Histogram(
                x=fishing, name="Fishing", opacity=0.65,
                marker_color="#f85149", nbinsx=60,
                hovertemplate="%{x:.3f}: %{y}<extra>Fishing</extra>"
            ))
            apply_theme(
                fig, title=feat_label, height=240,
                barmode="overlay", legend_inside=True
            )
            col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            col.markdown(
                f'<div style="font-size:11px;color:#8b949e;margin-top:-8px;'
                f'margin-bottom:16px">{desc}</div>',
                unsafe_allow_html=True
            )

    # ── Gear type breakdown ───────────────────────────────────────
    st.markdown(
        '<div class="section-header">Gear Type Distribution</div>',
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)

    with col_a:
        gear_counts = (
            windows_df.groupby("gear_label")["label"]
            .agg(
                fishing=lambda x: (x == 1).sum(),
                not_fishing=lambda x: (x == 0).sum()
            )
            .reset_index()
            .sort_values("fishing", ascending=True)
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=gear_counts["gear_label"], x=gear_counts["not_fishing"],
            name="Not fishing", orientation="h",
            marker_color="#21262d", marker_line_color="#30363d", marker_line_width=1
        ))
        fig.add_trace(go.Bar(
            y=gear_counts["gear_label"], x=gear_counts["fishing"],
            name="Fishing", orientation="h",
            marker_color="#f85149"
        ))
        apply_theme(fig, title="Windows by gear type", height=280, barmode="stack")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        rate = (
            windows_df.groupby("gear_label")["label"]
            .mean().reset_index()
        )
        rate.columns = ["gear_label", "fishing_rate"]
        rate = rate.sort_values("fishing_rate", ascending=True)

        fig = go.Figure(go.Bar(
            y=rate["gear_label"], x=rate["fishing_rate"],
            orientation="h",
            marker=dict(
                color=rate["fishing_rate"],
                colorscale=[[0, "#21262d"], [0.5, "#e3b341"], [1, "#f85149"]],
                showscale=False
            ),
            text=[f"{v:.0%}" for v in rate["fishing_rate"]],
            textfont=dict(family="IBM Plex Mono", size=11, color="#e6edf3"),
            textposition="outside"
        ))
        apply_theme(
            fig, title="Fishing rate by gear type",
            height=280, xaxis_fmt=".0%"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Inference predictions ─────────────────────────────────────
    st.markdown(
        '<div class="section-header">Inference Results — Unlabelled Vessels</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-sub" style="margin-bottom:16px">Predictions from applying the '
        'trained model to 354k windows from 46 vessels with no existing labels. '
        'A sample of the results is shown below.</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Windows shown",        f"{len(predict_df):,}")
    m2.metric("Predicted fishing",     f"{(predict_df['fishing_pred'] == 1).sum():,}")
    m3.metric("Predicted not fishing", f"{(predict_df['fishing_pred'] == 0).sum():,}")
    m4.metric("Mean fishing prob",     f"{predict_df['fishing_prob'].mean():.2f}")

    col_hist, col_table = st.columns([1, 1])

    with col_hist:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predict_df["fishing_prob"],
            nbinsx=50, marker_color="#58a6ff", opacity=0.8,
            hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra></extra>"
        ))
        apply_theme(
            fig,
            title="Fishing probability distribution",
            height=280,
            xaxis_title="Fishing probability",
            yaxis_title="Window count"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_table:
        sample = (
            predict_df[[
                "gear_label", "win_start", "fishing_prob",
                "speed_mean", "traj_entropy", "net_displacement_ratio"
            ]]
            .rename(columns={
                "gear_label":             "Gear type",
                "win_start":              "Window start",
                "fishing_prob":           "Fishing prob",
                "speed_mean":             "Speed mean",
                "traj_entropy":           "Traj entropy",
                "net_displacement_ratio": "Displacement ratio",
            })
            .head(200)
        )
        st.dataframe(
            sample.style.format({
                "Fishing prob":       "{:.3f}",
                "Speed mean":         "{:.2f}",
                "Traj entropy":       "{:.3f}",
                "Displacement ratio": "{:.3f}",
            }).background_gradient(subset=["Fishing prob"], cmap="RdYlGn_r"),
            height=280,
            use_container_width=True
        )


# ── Mode 2: IUU Risk Scoring ──────────────────────────────────────
else:

    st.markdown('<div class="page-title">IUU Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Composite risk scores applied to confirmed GFW fishing events. '
        'Score is a product of MPA tier weight, vessel authorisation status, high seas flag, '
        'and GFW potential risk indicator. Events are sourced from the GFW public fishing '
        'events API.</div>',
        unsafe_allow_html=True
    )

    risk_df  = load_risk()
    filtered = risk_df[risk_df["risk_score"] >= min_risk].copy()
    if show_mpa_only:
        filtered = filtered[filtered["in_mpa"] == 1]
    if show_high_seas:
        filtered = filtered[filtered["on_high_seas"] == 1]

    # ── Summary metrics ───────────────────────────────────────────
    st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Events shown",     f"{len(filtered):,}")
    c2.metric("High risk (≥3.0)", f"{(filtered['risk_score'] >= 3.0).sum()}")
    c3.metric("In MPA",           f"{filtered['in_mpa'].sum()}")
    c4.metric("Auth violations",  f"{(filtered['auth_multiplier'] > 1).sum()}")
    c5.metric("High seas",        f"{filtered['on_high_seas'].sum()}")

    # ── Map ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Event Map</div>', unsafe_allow_html=True)

    map_df = filtered.dropna(subset=["lat", "lon"]).copy()

    def score_to_rgb(score, max_score=3.9):
        ratio = min(max((score - 1.0) / (max_score - 1.0), 0.0), 1.0)
        return [
            int(80  + 175 * ratio),
            int(160 - 130 * ratio),
            int(200 - 180 * ratio),
            200
        ]

    map_df["colour"] = map_df["risk_score"].apply(score_to_rgb)
    map_df["radius"] = (map_df["risk_score"] * 35000).astype(int)

    risk_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_fill_color="colour",
        get_radius="radius",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 40],
        line_width_min_pixels=1,
    )

    risk_tooltip = {
        "html": (
            "<div style='font-family:IBM Plex Mono,monospace;font-size:12px;"
            "background:#0d1117;border:1px solid #21262d;padding:10px 14px;"
            "border-radius:4px;color:#c9d1d9;min-width:200px'>"
            "<b style='color:#e6edf3'>{vessel_name}</b> &nbsp;"
            "<span style='color:#8b949e'>{vessel_flag}</span><br>"
            "<span style='color:#58a6ff;font-size:15px'>{risk_score}</span>"
            "&nbsp;<span style='color:#8b949e;font-size:11px'>risk score</span><br><br>"
            "<span style='color:#8b949e'>Auth status</span><br>"
            "<span style='font-size:11px'>{auth_status}</span><br><br>"
            "<span style='color:#8b949e'>"
            "High seas: {on_high_seas}&nbsp;&nbsp;"
            "MPA: {in_mpa}&nbsp;&nbsp;"
            "No-take: {in_mpa_no_take}"
            "</span>"
            "</div>"
        )
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[risk_layer],
            initial_view_state=pdk.ViewState(
                latitude=10, longitude=10, zoom=1.2, pitch=0
            ),
            tooltip=risk_tooltip,
            map_style=BASEMAP,
        ),
        use_container_width=True
    )

    st.markdown(
        '<div style="font-size:11px;color:#8b949e;margin-top:4px">'
        'Point size and colour scale with risk score. Hover a point for vessel details.</div>',
        unsafe_allow_html=True
    )

    # ── Risk breakdown ────────────────────────────────────────────
    st.markdown('<div class="section-header">Risk Breakdown</div>', unsafe_allow_html=True)

    col_chart, col_ref = st.columns([1, 1])

    with col_chart:
        score_counts = (
            filtered["risk_score"]
            .value_counts()
            .sort_index(ascending=False)
            .reset_index()
        )
        score_counts.columns = ["risk_score", "count"]

        bar_colours = [
            "#f85149" if s >= 3.0 else "#e3b341" if s >= 2.0 else "#58a6ff"
            for s in score_counts["risk_score"]
        ]

        fig = go.Figure(go.Bar(
            x=score_counts["risk_score"].apply(lambda x: f"{x:.2f}"),
            y=score_counts["count"],
            marker_color=bar_colours,
            hovertemplate="Score %{x}: %{y} events<extra></extra>"
        ))
        apply_theme(
            fig,
            title="Events by risk score",
            height=300,
            xaxis_title="Risk score",
            yaxis_title="Events"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_ref:
        st.markdown("""
        <div style="font-size:12px;color:#8b949e;line-height:1.9;margin-top:8px">
        <b style="color:#e6edf3;display:block;margin-bottom:10px">Score components</b>

        <b style="color:#c9d1d9">MPA tier weight</b><br>
        1.0 &mdash; no MPA<br>
        1.5 &mdash; general MPA<br>
        2.0 &mdash; partial no-take zone<br>
        3.0 &mdash; strict no-take (IUCN I&ndash;II)<br><br>

        <b style="color:#c9d1d9">Authorisation multiplier</b><br>
        1.0 &mdash; valid authorisation on record<br>
        1.5 &mdash; partial match<br>
        2.0 &mdash; no matching authorisation<br><br>

        <b style="color:#c9d1d9">High seas multiplier</b><br>
        1.3 &mdash; fishing outside any EEZ<br><br>

        <b style="color:#c9d1d9">Potential risk flag</b><br>
        1.5 &mdash; GFW flagged as potential risk
        </div>
        """, unsafe_allow_html=True)

    # ── Alert table ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Alert Feed</div>', unsafe_allow_html=True)

    display = (
        filtered[[
            "vessel_name", "vessel_flag", "risk_score",
            "in_mpa", "in_mpa_no_take", "on_high_seas",
            "potential_risk_flag", "auth_status",
            "avg_speed_knots", "duration_hrs", "event_start"
        ]]
        .sort_values("risk_score", ascending=False)
        .rename(columns={
            "vessel_name":         "Vessel",
            "vessel_flag":         "Flag",
            "risk_score":          "Risk",
            "in_mpa":              "MPA",
            "in_mpa_no_take":      "No-take",
            "on_high_seas":        "High seas",
            "potential_risk_flag": "GFW flag",
            "auth_status":         "Auth status",
            "avg_speed_knots":     "Speed (kn)",
            "duration_hrs":        "Duration (hr)",
            "event_start":         "Event start",
        })
    )

    st.dataframe(
        display.style.format({
            "Risk":          "{:.2f}",
            "Speed (kn)":    "{:.2f}",
            "Duration (hr)": "{:.1f}",
        }).background_gradient(subset=["Risk"], cmap="RdYlGn_r"),
        height=400,
        use_container_width=True
    )

    st.download_button(
        label="Download as CSV",
        data=filtered.to_csv(index=False),
        file_name="iuu_risk_scores.csv",
        mime="text/csv"
    )