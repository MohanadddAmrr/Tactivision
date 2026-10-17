"""
TactiVision Pro — Public Showcase (Streamlit Cloud)
====================================================

A read-only visualisation of pre-computed TactiVision pipeline output. Designed
to let reviewers and recruiters explore the project without a local setup.

The full pipeline (detection, tracking, feature extraction) requires GPU +
heavy ML dependencies and runs locally. This showcase serves the real JSON
output and heatmap PNGs the pipeline wrote after analysing broadcast footage.

Author:  Mohanad Amr
Project: TactiVision Pro — graduation project, Misr International University, 2026
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TactiVision Pro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/MohanadddAmrr/TactiVision",
        "Report a bug": "https://github.com/MohanadddAmrr/TactiVision/issues",
        "About": (
            "TactiVision Pro — End-to-end computer vision pipeline for football "
            "analytics. Graduation project, Misr International University, 2026."
        ),
    },
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
DEMO_OUTPUTS = REPO_ROOT / "demo" / "demo_outputs"
GITHUB_URL = "https://github.com/MohanadddAmrr/TactiVision"
LINKEDIN_URL = "https://www.linkedin.com/in/mohanad-amr-5a50952ab/"

# Match-eligibility rules: only matches that have real pipeline output are shown
MIN_PLAYERS = 15
MIN_FRAMES = 1500
REQUIRE_POSSESSION = True

# ---------------------------------------------------------------------------
# Design system (serious, data-forward, no decoration)
# ---------------------------------------------------------------------------
# Core palette: deep navy base, slate panels, cool blue accents.
BG = "#0a0e1a"           # Page background
SURFACE = "#131b2b"      # Card background
SURFACE_2 = "#1a2436"    # Hover / nested surface
BORDER = "#223049"       # Hairline borders
TEXT = "#e8ecf1"         # Primary text
TEXT_MUTED = "#8a97ad"   # Secondary text
TEXT_DIM = "#5c6a82"     # Labels
ACCENT = "#4b7bec"       # Primary accent (links, highlights)
ACCENT_SOFT = "#2d4878"  # Accent muted

# Semantic data colours (used only in visualisations, never as decoration)
TEAM_A = "#e74c3c"   # Home team (Arsenal in the sample match)
TEAM_B = "#3498db"   # Away team
REF = "#95a5a6"      # Referees
POSITIVE = "#2ecc71"
NEUTRAL = "#8395a7"

CUSTOM_CSS = f"""
<style>
    /* Reset Streamlit defaults */
    .main > div {{ padding-top: 0.5rem; }}
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }}

    /* Serious typography */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                     "Helvetica Neue", Arial, sans-serif;
        letter-spacing: -0.005em;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT};
        font-weight: 600;
        letter-spacing: -0.015em;
    }}

    h2 {{ font-size: 1.5rem; margin-top: 0.5rem; margin-bottom: 1rem; }}
    h3 {{ font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0.6rem; color: {TEXT_MUTED}; }}

    /* Hero banner — flat, no gradient */
    .hero {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.75rem;
    }}
    .hero .wordmark {{
        color: {TEXT};
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .hero .subtitle {{
        color: {TEXT_MUTED};
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        line-height: 1.5;
    }}
    .hero .meta {{
        margin-top: 1rem;
        color: {TEXT_DIM};
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .hero .meta span {{ margin-right: 1.5rem; }}

    /* KPI tile — clinical */
    .kpi {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        padding: 1.1rem 1.25rem;
        height: 100%;
    }}
    .kpi .label {{
        color: {TEXT_DIM};
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }}
    .kpi .value {{
        color: {TEXT};
        font-size: 1.85rem;
        font-weight: 600;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }}
    .kpi .sub {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        margin-top: 0.4rem;
        line-height: 1.4;
    }}

    /* Section divider */
    .divider {{
        height: 1px;
        background: {BORDER};
        margin: 1.5rem 0;
    }}

    /* Sidebar polish — compact + full-width text */
    section[data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }}
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {{
        color: {TEXT};
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.5rem 0;
    }}

    /* Links */
    a {{ color: {ACCENT}; text-decoration: none; }}
    a:hover {{ color: {ACCENT}; text-decoration: underline; }}

    /* Tabs — no bold colour on active */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.88rem;
        font-weight: 500;
        padding: 0.6rem 1.1rem;
        color: {TEXT_MUTED};
        border-bottom: 2px solid transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT} !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
    }}

    /* Info/warning boxes — restrained */
    .stAlert {{
        border-left: 2px solid {ACCENT_SOFT};
        background: {SURFACE};
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_matches() -> list[dict]:
    """Return matches with real pipeline output.

    Filters out thin test runs: a match needs enough players, enough frames,
    and populated possession data.
    """
    if not DEMO_OUTPUTS.exists():
        return []
    matches: list[dict] = []
    for match_dir in sorted(DEMO_OUTPUTS.iterdir()):
        if not match_dir.is_dir():
            continue
        metrics_path = match_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                data = json.load(f)
        except Exception:
            continue
        if int(data.get("num_players", 0)) < MIN_PLAYERS:
            continue
        if int(data.get("frame", 0)) < MIN_FRAMES:
            continue
        if REQUIRE_POSSESSION:
            tpp = data.get("possession", {}).get("team_possession_percentage", {})
            if not tpp or (float(tpp.get("A", 0)) + float(tpp.get("B", 0))) < 10:
                continue
        team_names = data.get("team_names", {})
        a = team_names.get("A", "Team A")
        b = team_names.get("B", "Team B")
        matches.append({
            "dir": match_dir,
            "name": match_dir.name,
            "display": f"{a} vs {b}",
            "team_a": a,
            "team_b": b,
            "frames": int(data.get("frame", 0)),
            "players": int(data.get("num_players", 0)),
        })
    matches.sort(key=lambda m: (m["name"] != "arsenal", -m["frames"]))
    return matches


@st.cache_data(show_spinner=False)
def load_metrics(match_dir_str: str) -> dict:
    with open(Path(match_dir_str) / "metrics.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def list_player_heatmaps(match_dir_str: str) -> list[str]:
    match_dir = Path(match_dir_str)
    files = sorted(
        match_dir.glob("heatmap_player_*.png"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return [p.name for p in files]


def safe_num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plotly_dark(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="-apple-system, Segoe UI, Inter, sans-serif",
                  size=12),
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=0),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def kpi(label: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def team_colour_map(team_a_name: str, team_b_name: str) -> dict:
    return {
        team_a_name: TEAM_A,
        team_b_name: TEAM_B,
        "Referee": REF,
        "unknown": NEUTRAL,
    }


def team_name_from_code(code: str, team_a_name: str, team_b_name: str) -> str:
    if code == "A":
        return team_a_name
    if code == "B":
        return team_b_name
    if code == "REF":
        return "Referee"
    return code or "unknown"


def simple_xg(angle_deg: float, distance_px: float, pitch_scale: float = 7.0) -> float:
    """Lightweight xG proxy.

    This is a transparent, logistic-regression-style approximation using only
    the two features present in the pre-computed shot events (shot angle to
    goal and distance to goal). Not a trained model — intended as a plausible
    difficulty indicator for display purposes.

    Pitch scale (px per metre) is a rough calibration; adjust for your match.
    """
    distance_m = max(distance_px / pitch_scale, 1.0)
    # Narrow angle hurts; distance hurts (inverse); mild intercept.
    linear = 1.8 - 0.14 * distance_m - 0.025 * (90.0 - min(angle_deg, 90.0))
    try:
        return 1.0 / (1.0 + math.exp(-linear))
    except OverflowError:
        return 0.0 if linear < 0 else 1.0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<div style='font-size:1.15rem;font-weight:600;color:{TEXT};"
        f"letter-spacing:-0.02em;'>TactiVision Pro</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{TEXT_MUTED};font-size:0.82rem;margin-bottom:1rem;'>"
        f"Football analytics showcase</div>",
        unsafe_allow_html=True,
    )

    matches = list_matches()
    if not matches:
        st.error(
            "No pre-computed matches with full data found. "
            "Expected `demo/demo_outputs/arsenal/metrics.json` with populated "
            "possession, shots, and sprints."
        )
        st.stop()

    if len(matches) == 1:
        selected_match = matches[0]
        metrics = load_metrics(str(selected_match["dir"]))
        st.markdown("#### Match")
        st.markdown(
            f"<div style='color:{TEXT};font-weight:500;padding:0.4rem 0;'>"
            f"{selected_match['display']}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{selected_match['frames']:,} frames · "
            f"{selected_match['players']} players"
        )
    else:
        st.markdown("#### Match")
        selected_idx = st.selectbox(
            "Select match",
            range(len(matches)),
            format_func=lambda i: matches[i]["display"],
            index=0,
            label_visibility="collapsed",
        )
        selected_match = matches[selected_idx]
        metrics = load_metrics(str(selected_match["dir"]))

    st.markdown("#### About this showcase")
    st.markdown(
        f"<div style='color:{TEXT_MUTED};font-size:0.82rem;line-height:1.5;'>"
        "A read-only view of pipeline output. The full detection + tracking + "
        "feature-extraction pipeline requires GPU and runs locally — this page "
        "serves the pre-computed JSON and heatmap PNGs it wrote."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Links")
    st.markdown(f"[Full source — GitHub]({GITHUB_URL})")
    st.markdown(f"[Mohanad Amr — LinkedIn]({LINKEDIN_URL})")

    st.markdown(
        f"<div style='position:absolute;bottom:1rem;color:{TEXT_DIM};"
        f"font-size:0.7rem;'>Graduation project · MIU 2026</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
TEAM_A_NAME = selected_match["team_a"]
TEAM_B_NAME = selected_match["team_b"]
TEAM_COLOURS = team_colour_map(TEAM_A_NAME, TEAM_B_NAME)

st.markdown(
    f"""
    <div class="hero">
        <h1 class="wordmark">TactiVision Pro</h1>
        <p class="subtitle">
            End-to-end computer vision pipeline converting broadcast football
            video into 70+ tactical features. Built by a six-person team led
            by Mohanad Amr as the 2026 graduation project at Misr International
            University.
        </p>
        <div class="meta">
            <span>YOLOv8n</span>
            <span>RT-DETR</span>
            <span>ByteTrack</span>
            <span>PyTorch</span>
            <span>OpenCV</span>
            <span>Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs — serious, no icons, title case
# ---------------------------------------------------------------------------
tab_labels = [
    "Overview",
    "Match Summary",
    "Possession",
    "Shots",
    "xG Model",
    "Physical",
    "Team Heatmaps",
    "Player Heatmaps",
    "Player Tracks",
    "Pipeline Architecture",
    "Technical Notes",
    "About",
]
tabs = st.tabs(tab_labels)


# ---------------------------------------------------------------------------
# TAB — Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("## Project overview")
    st.markdown(
        f"""
TactiVision Pro ingests raw broadcast football footage and outputs a dense set
of tactical features: player and ball tracking, possession, shots, sprints,
heatmaps, formation templates, pressing intensity, player identity resolution
and more. The system was built end-to-end over seven months by a six-person
engineering team, with Mohanad Amr as lead.

The showcase on this page runs the pipeline's output through a lightweight
dashboard — the heavy inference pipeline itself (YOLOv8n, RT-DETR, ByteTrack,
EasyOCR) requires GPU resources and runs locally. The sample match displayed
here is **{TEAM_A_NAME} vs {TEAM_B_NAME}**, selected from the sidebar.
"""
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### The hardest problem: ball detection")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(
            """
In broadcast football footage the ball is small, fast, frequently occluded,
blurred by motion and often cropped out of frame. A vanilla object detector
struggles to produce a usable detection rate.

The pipeline stacks three detectors:

1. **YOLOv8n**, fine-tuned on custom-annotated football frames, handles the
   common case.
2. **RT-DETR** runs as a second pass specifically on frames YOLO missed.
3. **Colour and short-horizon prediction** fill the remaining gaps when the
   ball is occluded or temporarily off-screen.
"""
        )
    with c2:
        bt = metrics.get("ball_tracking", {})
        total = int(safe_num(bt.get("total_detections", 0)))
        yolo = int(safe_num(bt.get("yolo_detections", 0)))
        color_d = int(safe_num(bt.get("color_detections", 0)))
        pred = int(safe_num(bt.get("predicted_detections", 0)))
        rate = safe_num(bt.get("detection_rate", 0)) * 100

        source_df = pd.DataFrame({
            "Source": ["YOLOv8n (primary)",
                       "Colour & HSV fallback",
                       "Short-horizon predict"],
            "Detections": [yolo, color_d, pred],
        })
        fig = px.bar(
            source_df,
            x="Detections",
            y="Source",
            orientation="h",
            color_discrete_sequence=[ACCENT],
            text="Detections",
        )
        fig.update_traces(textposition="outside", marker_color=ACCENT)
        fig.update_layout(
            showlegend=False,
            title=f"Ball-detection source breakdown — {selected_match['display']}",
        )
        st.plotly_chart(plotly_dark(fig, height=300), width="stretch")
        st.caption(
            f"Across {int(safe_num(metrics.get('frame', 0))):,} frames the dual-model "
            f"pipeline produced {total:,} ball detections (per-frame detection rate "
            f"{rate:.1f}%). YOLO alone contributed {yolo:,}; the fallback and "
            f"predict passes recovered the remaining {color_d + pred:,}."
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Pipeline capabilities")
    cols = st.columns(4)
    capabilities = [
        ("Detection & tracking",
         "YOLOv8n fine-tune · RT-DETR fallback · ByteTrack ID continuity · ReID module"),
        ("Tactical analysis",
         "17 formation templates · PPDA pressing · Possession zones · xG estimation"),
        ("Player identity",
         "EasyOCR jersey numbers · SQLite roster matching · Team classification"),
        ("Outputs",
         "16-tab local dashboard · Positional heatmaps · PDF reports · Highlights"),
    ]
    for col, (title, body) in zip(cols, capabilities):
        with col:
            st.markdown(
                f"<div style='color:{TEXT};font-weight:600;font-size:0.95rem;"
                f"margin-bottom:0.4rem;'>{title}</div>"
                f"<div style='color:{TEXT_MUTED};font-size:0.82rem;"
                f"line-height:1.5;'>{body}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# TAB — Match Summary
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"## {TEAM_A_NAME} vs {TEAM_B_NAME} — headline numbers")

    bt = metrics.get("ball_tracking", {})
    pos = metrics.get("possession", {})
    tpp = pos.get("team_possession_percentage", {})
    shots = metrics.get("shot_detection", {})
    sprints = metrics.get("sprint_detection", {})

    row1 = st.columns(4)
    with row1[0]:
        frames = int(safe_num(metrics.get("frame", 0)))
        kpi("Frames tracked", f"{frames:,}", f"approx. {frames/25:.0f} seconds at 25 FPS")
    with row1[1]:
        kpi("Unique players", f"{int(safe_num(metrics.get('num_players', 0)))}",
            "including both teams and referees")
    with row1[2]:
        kpi("Ball detections", f"{int(safe_num(bt.get('total_detections', 0))):,}",
            f"per-frame detection rate {safe_num(bt.get('detection_rate', 0))*100:.1f}%")
    with row1[3]:
        kpi("Possession changes",
            f"{int(safe_num(pos.get('total_possession_changes', 0)))}",
            "complete turnovers detected")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    row2 = st.columns(4)
    with row2[0]:
        kpi(f"{TEAM_A_NAME} possession",
            f"{safe_num(tpp.get('A', 0)):.1f} percent",
            "share of ball-in-play time")
    with row2[1]:
        kpi(f"{TEAM_B_NAME} possession",
            f"{safe_num(tpp.get('B', 0)):.1f} percent",
            "share of ball-in-play time")
    with row2[2]:
        kpi("Shots detected", f"{int(safe_num(shots.get('total_shots', 0)))}",
            "high-velocity ball events near goal")
    with row2[3]:
        kpi("Sprints detected", f"{int(safe_num(sprints.get('total_sprints', 0)))}",
            f"top speed {safe_num(sprints.get('top_speed_mps', 0)):.1f} metres per second")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Tracking quality")
    tq = metrics.get("tracking_quality", {})
    row3 = st.columns(4)
    with row3[0]:
        kpi("Canonical players",
            f"{int(safe_num(tq.get('canonical_players', 0)))}",
            "after cross-track deduplication")
    with row3[1]:
        kpi("Raw YOLO tracks", f"{int(safe_num(tq.get('raw_yolo_tracks', 0)))}",
            "before deduplication")
    with row3[2]:
        kpi("Deduplication ratio", f"{safe_num(tq.get('dedup_ratio', 0)):.1f}x",
            "raw to canonical")
    with row3[3]:
        kpi("Speed samples", f"{int(safe_num(tq.get('speed_readings_capped', 0))):,}",
            "frame-to-frame velocity readings")


# ---------------------------------------------------------------------------
# TAB — Possession
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## Possession analysis")
    pos = metrics.get("possession", {})
    tpp = pos.get("team_possession_percentage", {})

    if not tpp:
        st.warning("No possession data for this match.")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("### Team possession share")
            pie_df = pd.DataFrame({
                "Team": [TEAM_A_NAME, TEAM_B_NAME],
                "Possession": [safe_num(tpp.get("A", 0)),
                               safe_num(tpp.get("B", 0))],
            })
            fig = go.Figure(data=[
                go.Pie(
                    labels=pie_df["Team"],
                    values=pie_df["Possession"],
                    hole=0.6,
                    marker=dict(colors=[TEAM_A, TEAM_B], line=dict(color=BG, width=2)),
                    textinfo="label+percent",
                    textfont=dict(color=TEXT, size=13),
                )
            ])
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_dark(fig, height=360), width="stretch")

        with c2:
            st.markdown("### Possession by pitch zone")
            zone_stats = pos.get("zone_stats", {})
            if zone_stats:
                zone_rows = []
                for team_code, zones in zone_stats.items():
                    team_name = team_name_from_code(team_code, TEAM_A_NAME, TEAM_B_NAME)
                    for zone, pct in zones.items():
                        zone_rows.append({"Team": team_name,
                                          "Zone": zone,
                                          "Percent": safe_num(pct)})
                zone_df = pd.DataFrame(zone_rows)
                fig = px.bar(
                    zone_df, x="Zone", y="Percent", color="Team",
                    barmode="group",
                    color_discrete_map=TEAM_COLOURS,
                    labels={"Percent": "Percent of possession in zone"},
                )
                st.plotly_chart(plotly_dark(fig, height=360), width="stretch")
            else:
                st.info("No zone-level breakdown available for this match.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        dur = pos.get("duration_stats", {})
        if dur:
            st.markdown("### Possession duration distribution")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                kpi("Average duration",
                    f"{safe_num(dur.get('avg_duration', 0)):.2f} s",
                    "per possession")
            with d2:
                kpi("Short (under 1 s)",
                    f"{safe_num(dur.get('short_pct', 0)):.1f} %",
                    f"{int(safe_num(dur.get('short_possessions', 0)))} possessions")
            with d3:
                kpi("Medium (1 to 5 s)",
                    f"{safe_num(dur.get('medium_pct', 0)):.1f} %",
                    f"{int(safe_num(dur.get('medium_possessions', 0)))} possessions")
            with d4:
                kpi("Long (over 5 s)",
                    f"{safe_num(dur.get('long_pct', 0)):.1f} %",
                    f"{int(safe_num(dur.get('long_possessions', 0)))} possessions")

        history = pos.get("possession_history", [])
        if history:
            st.markdown("### Possession timeline")
            hist_df = pd.DataFrame(history)
            if {"team", "start_time", "duration"}.issubset(hist_df.columns):
                hist_df["team_name"] = hist_df["team"].apply(
                    lambda c: team_name_from_code(c, TEAM_A_NAME, TEAM_B_NAME))
                fig = px.scatter(
                    hist_df,
                    x="start_time",
                    y="team_name",
                    size="duration",
                    color="team_name",
                    color_discrete_map=TEAM_COLOURS,
                    hover_data={"duration": ":.2f",
                                "player_id": True,
                                "team_name": False},
                    labels={"start_time": "Time (seconds)", "team_name": ""},
                )
                fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0)))
                fig.update_layout(showlegend=False)
                st.plotly_chart(plotly_dark(fig, height=200), width="stretch")


# ---------------------------------------------------------------------------
# TAB — Shots
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## Shot detection")
    shot_events = metrics.get("shot_events", [])
    total_shots = int(safe_num(metrics.get("shot_detection", {}).get("total_shots", 0)))

    if not shot_events:
        st.info("No shot events detected in this match's pipeline run.")
    else:
        st.markdown(
            f"The pipeline detected **{total_shots} shot event(s)** — high-velocity "
            "ball movements originating in the attacking third."
        )

        shots_df = pd.DataFrame(shot_events)
        display_cols = [c for c in [
            "timestamp", "shooter_id", "shooter_team",
            "velocity_mps", "angle_to_goal_deg", "distance_to_goal_px", "frame",
        ] if c in shots_df.columns]
        pretty = shots_df[display_cols].copy()
        if "shooter_team" in pretty.columns:
            pretty["shooter_team"] = pretty["shooter_team"].apply(
                lambda c: team_name_from_code(c, TEAM_A_NAME, TEAM_B_NAME))

        pretty = pretty.rename(columns={
            "timestamp": "Time (s)",
            "shooter_id": "Player ID",
            "shooter_team": "Team",
            "velocity_mps": "Ball velocity (m/s)",
            "angle_to_goal_deg": "Angle to goal (deg)",
            "distance_to_goal_px": "Distance to goal (px)",
            "frame": "Frame",
        })

        st.dataframe(
            pretty.style.format({
                "Time (s)": "{:.2f}",
                "Ball velocity (m/s)": "{:.1f}",
                "Angle to goal (deg)": "{:.1f}",
                "Distance to goal (px)": "{:.0f}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if {"velocity_mps", "angle_to_goal_deg"}.issubset(shots_df.columns):
            st.markdown("### Shot characteristics")
            scatter_df = shots_df.copy()
            scatter_df["team_name"] = scatter_df["shooter_team"].apply(
                lambda c: team_name_from_code(c, TEAM_A_NAME, TEAM_B_NAME))
            fig = px.scatter(
                scatter_df,
                x="angle_to_goal_deg",
                y="velocity_mps",
                color="team_name",
                size="distance_to_goal_px",
                hover_data=["shooter_id", "timestamp"],
                color_discrete_map=TEAM_COLOURS,
                labels={
                    "angle_to_goal_deg": "Angle to goal (degrees)",
                    "velocity_mps": "Ball velocity (metres per second)",
                },
            )
            st.plotly_chart(plotly_dark(fig, height=400), width="stretch")


# ---------------------------------------------------------------------------
# TAB — xG Model
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## Expected goals (xG)")
    shot_events = metrics.get("shot_events", [])

    st.markdown(
        """
A transparent xG estimate computed from the shot angle to goal and distance to
goal that the pipeline recorded for each shot event. This is **not a trained
production model** — it is a two-feature logistic approximation intended to
indicate shot difficulty.

The full pipeline includes a trained xG model (see `services/xg_calculator.py`
in the repository). This showcase uses a simplified version so you can see how
the two inputs combine without requiring the trained weights.
"""
    )

    if not shot_events:
        st.info("No shot events detected in this match's pipeline run.")
    else:
        xg_df = pd.DataFrame(shot_events).copy()
        xg_df["team_name"] = xg_df["shooter_team"].apply(
            lambda c: team_name_from_code(c, TEAM_A_NAME, TEAM_B_NAME))
        xg_df["xG"] = xg_df.apply(
            lambda r: simple_xg(
                safe_num(r.get("angle_to_goal_deg", 0)),
                safe_num(r.get("distance_to_goal_px", 0)),
            ),
            axis=1,
        )

        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("### xG totals by team")
            agg = (xg_df.groupby("team_name")
                        .agg(shots=("xG", "size"), total_xG=("xG", "sum"))
                        .reset_index())
            fig = px.bar(
                agg, x="team_name", y="total_xG",
                color="team_name",
                color_discrete_map=TEAM_COLOURS,
                text=agg["total_xG"].round(2),
                labels={"team_name": "Team", "total_xG": "Total xG"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_dark(fig, height=320), width="stretch")

        with c2:
            st.markdown("### Shot quality distribution")
            fig = px.scatter(
                xg_df,
                x="distance_to_goal_px",
                y="angle_to_goal_deg",
                size="xG",
                color="team_name",
                color_discrete_map=TEAM_COLOURS,
                hover_data={"xG": ":.3f",
                            "velocity_mps": ":.1f",
                            "timestamp": ":.1f",
                            "team_name": False},
                labels={
                    "distance_to_goal_px": "Distance to goal (pixels)",
                    "angle_to_goal_deg": "Angle to goal (degrees)",
                },
            )
            st.plotly_chart(plotly_dark(fig, height=320), width="stretch")

        st.markdown("### Per-shot xG table")
        pretty = xg_df[["timestamp", "team_name", "shooter_id",
                        "angle_to_goal_deg", "distance_to_goal_px",
                        "velocity_mps", "xG"]].copy()
        pretty = pretty.rename(columns={
            "timestamp": "Time (s)",
            "team_name": "Team",
            "shooter_id": "Player ID",
            "angle_to_goal_deg": "Angle (deg)",
            "distance_to_goal_px": "Distance (px)",
            "velocity_mps": "Velocity (m/s)",
            "xG": "xG",
        })
        st.dataframe(
            pretty.style.format({
                "Time (s)": "{:.2f}",
                "Angle (deg)": "{:.1f}",
                "Distance (px)": "{:.0f}",
                "Velocity (m/s)": "{:.1f}",
                "xG": "{:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "xG approximation: 1 / (1 + exp(-(1.8 - 0.14 x distance_m - "
            "0.025 x (90 - angle_deg)))). Distance in metres using a 7 px/m "
            "calibration, angle clamped to [0, 90]."
        )


# ---------------------------------------------------------------------------
# TAB — Physical
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("## Physical and sprint analysis")
    sprint_events = metrics.get("sprint_events", [])
    sprint_summary = metrics.get("sprint_detection", {})

    if not sprint_events:
        st.info("No sprint events detected for this match.")
    else:
        c = st.columns(4)
        with c[0]:
            kpi("Sprints",
                f"{int(safe_num(sprint_summary.get('total_sprints', 0)))}",
                "across all players")
        with c[1]:
            kpi("High intensity",
                f"{int(safe_num(sprint_summary.get('high_intensity_sprints', 0)))}",
                "qualifying under the threshold")
        with c[2]:
            top_mps = safe_num(sprint_summary.get("top_speed_mps", 0))
            kpi("Top speed", f"{top_mps:.1f} m/s",
                f"approx. {top_mps*3.6:.1f} km/h")
        with c[3]:
            kpi("Average max speed",
                f"{safe_num(sprint_summary.get('avg_max_speed_mps', 0)):.1f} m/s",
                "mean across sprint events")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        sprint_df = pd.DataFrame(sprint_events)
        sprint_df["team_name"] = sprint_df["team"].apply(
            lambda c: team_name_from_code(c, TEAM_A_NAME, TEAM_B_NAME))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Sprint top-speed distribution")
            fig = px.histogram(
                sprint_df, x="max_speed_mps", color="team_name",
                color_discrete_map=TEAM_COLOURS,
                nbins=20,
                labels={"max_speed_mps": "Maximum sprint speed (metres per second)"},
            )
            fig.update_layout(barmode="overlay")
            fig.update_traces(opacity=0.75)
            st.plotly_chart(plotly_dark(fig, height=320), width="stretch")

        with c2:
            st.markdown("### Sprint duration distribution")
            fig = px.histogram(
                sprint_df, x="duration_sec", color="team_name",
                color_discrete_map=TEAM_COLOURS,
                nbins=20,
                labels={"duration_sec": "Sprint duration (seconds)"},
            )
            fig.update_layout(barmode="overlay")
            fig.update_traces(opacity=0.75)
            st.plotly_chart(plotly_dark(fig, height=320), width="stretch")

        st.markdown("### Top sprinters")
        top_sprints = sprint_df.sort_values("max_speed_mps", ascending=False).head(10)
        display = top_sprints[["player_id", "team_name", "duration_sec",
                               "max_speed_mps", "avg_speed_mps",
                               "is_high_intensity"]].copy()
        display = display.rename(columns={
            "player_id": "Player",
            "team_name": "Team",
            "duration_sec": "Duration (s)",
            "max_speed_mps": "Max speed (m/s)",
            "avg_speed_mps": "Average speed (m/s)",
            "is_high_intensity": "High intensity",
        })
        st.dataframe(
            display.style.format({
                "Duration (s)": "{:.2f}",
                "Max speed (m/s)": "{:.2f}",
                "Average speed (m/s)": "{:.2f}",
            }),
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# TAB — Team Heatmaps
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("## Team and global heatmaps")
    st.markdown(
        f"Generated by the pipeline from frame-by-frame player positions, "
        f"projected onto a normalised pitch coordinate system."
    )

    match_dir = selected_match["dir"]
    team_cols = st.columns(3)
    team_heatmap_list = [
        (f"{TEAM_A_NAME}", "heatmap_team_A.png"),
        (f"{TEAM_B_NAME}", "heatmap_team_B.png"),
        ("Global (all players)", "heatmap_global.png"),
    ]
    for col, (name, filename) in zip(team_cols, team_heatmap_list):
        img_path = match_dir / filename
        with col:
            st.markdown(
                f"<div style='color:{TEXT};font-weight:600;font-size:0.95rem;"
                f"margin-bottom:0.5rem;'>{name}</div>",
                unsafe_allow_html=True,
            )
            if img_path.exists():
                st.image(str(img_path), width="stretch")
            else:
                st.info(f"Not available: {filename}")


# ---------------------------------------------------------------------------
# TAB — Player Heatmaps
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("## Per-player heatmap browser")
    st.markdown(
        f"Positional density for each tracked player, drawn from the full "
        f"frame set the pipeline analysed."
    )

    match_dir = selected_match["dir"]
    player_files = list_player_heatmaps(str(match_dir))

    if not player_files:
        st.info("No per-player heatmaps for this match.")
    else:
        player_ids = [f.replace("heatmap_player_", "").replace(".png", "")
                      for f in player_files]
        c1, c2 = st.columns([1, 2])
        with c1:
            selected_pid = st.selectbox(
                f"Player ID ({len(player_ids)} tracked)",
                options=player_ids,
                index=0,
            )
            identities = metrics.get("player_identities", {})
            identity = identities.get(selected_pid, {})
            team = identity.get("team", "unknown")
            display = identity.get("display", f"Player {selected_pid}")
            jersey = identity.get("number")
            team_label = team_name_from_code(team, TEAM_A_NAME, TEAM_B_NAME)
            st.markdown(f"**Display name:** {display}")
            st.markdown(f"**Team:** {team_label}")
            if jersey:
                st.markdown(f"**Jersey number:** {jersey}")
        with c2:
            img_path = match_dir / f"heatmap_player_{selected_pid}.png"
            if img_path.exists():
                st.image(str(img_path), width="stretch")
            else:
                st.warning("Heatmap file missing.")


# ---------------------------------------------------------------------------
# TAB — Player Tracks
# ---------------------------------------------------------------------------
with tabs[8]:
    st.markdown("## Player tracks and workload")
    tracks = metrics.get("tracks", [])
    if not tracks:
        st.info("No per-player track data in this match's metrics.")
    else:
        tracks_df = pd.DataFrame(tracks)
        tracks_df["team_name"] = tracks_df["team"].apply(
            lambda c: team_name_from_code(c or "unknown", TEAM_A_NAME, TEAM_B_NAME))

        c1, c2 = st.columns([2, 1])
        with c1:
            team_filter = st.multiselect(
                "Filter by team",
                options=sorted(tracks_df["team_name"].unique()),
                default=sorted(tracks_df["team_name"].unique()),
            )
        with c2:
            min_distance = st.number_input(
                "Minimum distance (m)", min_value=0.0, value=50.0, step=10.0
            )

        filtered = tracks_df[
            tracks_df["team_name"].isin(team_filter)
            & (tracks_df.get("total_distance_m", 0) >= min_distance)
        ].copy()

        st.markdown(
            f"<div style='color:{TEXT_MUTED};font-size:0.85rem;'>"
            f"Showing <strong>{len(filtered)}</strong> of {len(tracks_df)} "
            f"tracked players</div>",
            unsafe_allow_html=True,
        )

        if not filtered.empty and "total_distance_m" in filtered.columns:
            top20 = filtered.nlargest(20, "total_distance_m")
            fig = px.bar(
                top20,
                x="total_distance_m",
                y="display_name",
                color="team_name",
                orientation="h",
                color_discrete_map=TEAM_COLOURS,
                hover_data=["player_id", "avg_speed_mps", "max_speed_mps"],
                labels={"total_distance_m": "Total distance (metres)",
                        "display_name": "Player"},
                title="Top 20 by distance covered",
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(plotly_dark(fig, height=500), width="stretch")

        display_cols = [c for c in [
            "player_id", "display_name", "team_name", "jersey_number",
            "total_distance_m", "avg_speed_mps", "max_speed_mps",
            "workload_score", "involvement_index",
        ] if c in filtered.columns]
        pretty = filtered[display_cols].rename(columns={
            "player_id": "ID",
            "display_name": "Display",
            "team_name": "Team",
            "jersey_number": "Jersey",
            "total_distance_m": "Distance (m)",
            "avg_speed_mps": "Average speed (m/s)",
            "max_speed_mps": "Max speed (m/s)",
            "workload_score": "Workload",
            "involvement_index": "Involvement",
        })
        st.dataframe(
            pretty.style.format({
                "Distance (m)": "{:.1f}",
                "Average speed (m/s)": "{:.2f}",
                "Max speed (m/s)": "{:.2f}",
                "Workload": "{:.1f}",
                "Involvement": "{:.2f}",
            }),
            width="stretch",
            hide_index=True,
            height=420,
        )


# ---------------------------------------------------------------------------
# TAB — Pipeline Architecture
# ---------------------------------------------------------------------------
with tabs[9]:
    st.markdown("## Pipeline architecture")
    st.markdown(
        """
Broadcast video is processed frame by frame through a staged pipeline. Each
stage emits structured output that the next stage consumes, with intermediate
results persisted to JSON for the dashboard layer to render.
"""
    )

    st.markdown(
        """
```
Broadcast video (mp4)
        |
        v
+----------------------+
|  Frame extraction    |  OpenCV
+----------------------+
        |
        v
+----------------------+
|  YOLOv8n detection   |  Primary: players + ball
+----------------------+
        |
        v
+----------------------+
|  RT-DETR second pass |  Ball recovery on YOLO-miss frames
+----------------------+
        |
        v
+----------------------+
|  ByteTrack           |  Identity continuity across frames
+----------------------+
        |
        v
+----------------------+
|  Feature extractors  |  70+ tactical features
+----------------------+
        |
        v
+----------------------+
|  Streamlit dashboard |  16 interactive tabs (local)
+----------------------+
```
"""
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Component stack")
        st.markdown(
            """
- **Language:** Python 3.10+
- **Deep learning:** PyTorch, Ultralytics (YOLOv8), RT-DETR
- **Tracking:** ByteTrack (with BoT-SORT as baseline comparison)
- **Computer vision:** OpenCV
- **OCR for jersey numbers:** EasyOCR
- **Dashboard:** Streamlit, Plotly
- **Data layer:** SQLite (roster matching), pandas, NumPy
- **Analytics:** SciPy, scikit-learn
"""
        )
    with c2:
        st.markdown("### Full dashboard scope")
        st.markdown(
            """
The local `demo/dashboard_final.py` runs sixteen interactive tabs covering:

- Overview, Shooting, Passing, Physical, Tactical
- xG and advanced analytics, Heatmaps, Highlights
- Database integration, Settings, AI recommendations
- Player performance, Match comparison
- Live match data, Report generation, Roster management

This public showcase focuses on the subset that can run on free Streamlit
Cloud without GPU or heavy ML dependencies.
"""
        )


# ---------------------------------------------------------------------------
# TAB — Technical Notes
# ---------------------------------------------------------------------------
with tabs[10]:
    st.markdown("## Why this showcase is read-only")
    st.markdown(
        """
The full pipeline requires:

- GPU-accelerated inference (YOLOv8n and RT-DETR are prohibitively slow on CPU
  for broadcast-length video)
- Multi-gigabyte ML dependencies (PyTorch, TensorFlow, Ultralytics,
  OpenCV-contrib, EasyOCR)
- One- to five-gigabyte input video files
- Roughly one to two gigabytes of RAM per concurrent user

None of this fits inside a free Streamlit Community Cloud instance. The
showcase therefore serves the pre-computed JSON and heatmap PNGs that the
pipeline wrote after running on broadcast footage locally. Everything you see
is the real algorithm's output on real footage — just loaded from disk rather
than recomputed on each page view.

### Running the full pipeline locally
"""
    )
    st.code(
        f"""git clone {GITHUB_URL}.git
cd TactiVision
pip install -r requirements_full.txt
streamlit run demo/dashboard_final.py""",
        language="bash",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Resource profile for this showcase")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        kpi("Memory used", "approx. 300 MB", "well under the 1 GB free-tier cap")
    with rc2:
        kpi("Cold-start time", "15 to 25 s", "first request after idle")
    with rc3:
        kpi("Payload", "approx. 7 MB", "pre-computed heatmap PNGs")


# ---------------------------------------------------------------------------
# TAB — About
# ---------------------------------------------------------------------------
with tabs[11]:
    st.markdown("## About the project and team")
    st.markdown(
        """
TactiVision Pro was built as the graduation project for the BSc Data Science
programme at **Misr International University (MIU)**, class of 2026. The
system was developed over seven months by a six-person engineering team.

### Team composition

- **Mohanad Amr** — team lead, ML and CV engineer, dashboard architecture
- **Five co-engineers** across detection, tracking, analytics, reporting,
  identity resolution, and QA

### Contact
"""
    )
    st.markdown(f"- GitHub: [MohanadddAmrr]({GITHUB_URL})")
    st.markdown(f"- LinkedIn: [Mohanad Amr]({LINKEDIN_URL})")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Acknowledgements")
    st.markdown(
        """
Thanks to the TactiVision engineering team for seven months of focused work,
and to the Data Science faculty at Misr International University for
supervising the project. External libraries and tools used in the pipeline
include Ultralytics YOLOv8, RT-DETR, ByteTrack, EasyOCR, OpenCV, PyTorch,
Streamlit, Plotly, and pandas.
"""
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    f"<div style='color:{TEXT_DIM};font-size:0.78rem;text-align:center;"
    f"padding:0.5rem 0;'>"
    f"TactiVision Pro showcase &middot; data: {selected_match['display']} &middot; "
    f"<a href='{GITHUB_URL}' style='color:{TEXT_DIM};'>Source on GitHub</a>"
    f"</div>",
    unsafe_allow_html=True,
)
