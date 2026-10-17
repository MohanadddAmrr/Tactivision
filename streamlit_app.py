"""
TactiVision Pro — Live Showcase (Streamlit Cloud)
==================================================

A lightweight, read-only dashboard that showcases the output of the TactiVision Pro
football analytics pipeline using pre-computed match data.

This is NOT the full pipeline (which requires GPU + heavy ML dependencies and runs
locally). This showcase visualises real pre-computed results from the Arsenal match
analysis so recruiters and reviewers can explore the project without a local setup.

The full source code and heavy pipeline live at the TactiVision repository root.
Run the full dashboard locally with:  streamlit run demo/dashboard_final.py

Author:  Mohanad Amr
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TactiVision Pro — Football Analytics Showcase",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/MohanadddAmrr/TactiVision",
        "Report a bug": "https://github.com/MohanadddAmrr/TactiVision/issues",
        "About": (
            "TactiVision Pro — End-to-end computer vision pipeline for football "
            "analytics. Built as a graduation project by Mohanad Amr (team lead, "
            "6 members) at Misr International University."
        ),
    },
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
DEMO_OUTPUTS = REPO_ROOT / "demo" / "demo_outputs"
GITHUB_URL = "https://github.com/MohanadddAmrr/TactiVision"
LINKEDIN_URL = "https://www.linkedin.com/in/mohanad-amr-5a50952ab/"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main > div { padding-top: 1rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1300px; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 2rem 2rem 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        margin: 0 0 .4rem 0;
        background: linear-gradient(90deg, #f1f5f9 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero .tag { color: #64748b; font-size: .95rem; margin: 0; }

    /* KPI tile */
    .kpi {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        height: 100%;
    }
    .kpi .label {
        color: #64748b;
        font-size: .75rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: .3rem;
    }
    .kpi .value {
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: 600;
        line-height: 1.1;
    }
    .kpi .sub { color: #94a3b8; font-size: .78rem; margin-top: .25rem; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: .22rem .6rem;
        margin-right: .4rem;
        margin-bottom: .4rem;
        border-radius: 999px;
        background: #1e293b;
        color: #cbd5e1;
        font-size: .72rem;
        font-weight: 500;
        border: 1px solid #334155;
    }
    .badge.hot { background: #7f1d1d; color: #fecaca; border-color: #991b1b; }

    /* Section divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e293b, transparent);
        margin: 1.5rem 0;
    }

    /* Section headings */
    h2 { color: #e2e8f0; font-weight: 600; }
    h3 { color: #cbd5e1; font-weight: 500; }

    /* Links */
    a { color: #60a5fa; text-decoration: none; }
    a:hover { color: #93c5fd; text-decoration: underline; }

    /* Streamlit tweaks */
    .stTabs [data-baseweb="tab"] {
        font-size: .95rem;
        padding: .5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
    }

    /* Table polish */
    [data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
TEAM_A_COLOR = "#DA291C"       # Arsenal red
TEAM_B_COLOR = "#60a5fa"       # Opposition blue
NEUTRAL = "#94a3b8"
ACCENT = "#34d399"
BG = "#0f172a"
GRID = "#1e293b"

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_matches() -> list[dict]:
    """Return matches with a non-trivial metrics.json."""
    matches = []
    if not DEMO_OUTPUTS.exists():
        return matches
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
        # Only include matches that have real data
        if data.get("num_players", 0) < 5:
            continue
        team_names = data.get("team_names", {})
        display_name = match_dir.name
        if team_names:
            a = team_names.get("A", "Team A")
            b = team_names.get("B", "Team B")
            display_name = f"{a} vs {b}"
        matches.append({
            "dir": match_dir,
            "name": match_dir.name,
            "display": display_name,
            "frames": data.get("frame", 0),
            "players": data.get("num_players", 0),
        })
    # Sort: arsenal first (richest), then by frames desc
    matches.sort(key=lambda m: (m["name"] != "arsenal", -m["frames"]))
    return matches


@st.cache_data(show_spinner=False)
def load_metrics(match_dir_str: str) -> dict:
    """Load match metrics JSON."""
    with open(Path(match_dir_str) / "metrics.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def list_player_heatmaps(match_dir_str: str) -> list[str]:
    """Return sorted list of player heatmap filenames."""
    match_dir = Path(match_dir_str)
    files = sorted(match_dir.glob("heatmap_player_*.png"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return [p.name for p in files]


def safe_num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plotly_dark(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
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


# ---------------------------------------------------------------------------
# Sidebar — match selector + credits
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚽ TactiVision Pro")
    st.caption("Football Analytics Showcase")

    matches = list_matches()
    if not matches:
        st.error("No pre-computed matches found. Check demo/demo_outputs/.")
        st.stop()

    match_names = [m["display"] for m in matches]
    selected_idx = st.selectbox(
        "Select match",
        range(len(matches)),
        format_func=lambda i: match_names[i],
        index=0,
    )
    selected_match = matches[selected_idx]
    metrics = load_metrics(str(selected_match["dir"]))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("#### About this showcase")
    st.markdown(
        "This is a **read-only visualisation** of pre-computed results from the "
        "TactiVision Pro pipeline. The full pipeline (detection, tracking, feature "
        "extraction) runs locally and requires GPU + ML dependencies."
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("#### Links")
    st.markdown(f"[📂 Full source on GitHub]({GITHUB_URL})")
    st.markdown(f"[💼 Mohanad on LinkedIn]({LINKEDIN_URL})")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption("Graduation project · 6-person team · MIU 2026")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
team_names = metrics.get("team_names", {})
TEAM_A_NAME = team_names.get("A", "Team A")
TEAM_B_NAME = team_names.get("B", "Team B")

st.markdown(
    f"""
    <div class="hero">
        <h1>TactiVision Pro — Football Analytics Pipeline</h1>
        <p class="tag">End-to-end computer vision converting broadcast football video
        into 70+ tactical features · YOLOv8 · RT-DETR · ByteTrack · Streamlit</p>
        <div style="margin-top:.8rem">
            <span class="badge hot">Graduation Project 2026</span>
            <span class="badge">Team lead · 6 members</span>
            <span class="badge">Misr International University</span>
            <span class="badge">Python · PyTorch · OpenCV</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🏠 Overview",
    "📊 Match Stats",
    "⚽ Possession",
    "🥅 Shots",
    "💨 Physical",
    "🔥 Heatmaps",
    "👥 Players",
    "🏗️ How It Works",
])

# ---------------------------------------------------------------------------
# TAB 1 — Overview / Project pitch
# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("## What is TactiVision Pro?")
    st.markdown(
        f"""
TactiVision Pro is an end-to-end computer vision pipeline that converts raw
broadcast football video into **70+ tactical features**: passes, shots, xG, formations,
pressing, player identities, heatmaps and more.

It was built as my graduation project at Misr International University, leading a
six-person engineering team through 7 months of iteration. This showcase displays
the pipeline's output on the **{TEAM_A_NAME} vs {TEAM_B_NAME}** match (plus
other test runs — pick from the sidebar).
"""
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### The hardest problem: ball detection")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(
            """
In broadcast football footage, the ball is:
- **Small** — a few pixels wide on wide shots
- **Fast** — hard for frame-by-frame detectors
- **Occluded** — hidden by players constantly
- **Blurred** — motion blur on strikes and passes
- **Off-screen** — the camera doesn't always follow it

A vanilla YOLOv8n detector struggles to produce a usable detection rate.
We built a **dual-model system** that stacks:

1. **YOLOv8n** fine-tuned on custom-annotated football frames
2. **RT-DETR second-pass** recovery on frames YOLO missed
3. **Color-based fallback** and short-horizon prediction for occlusion gaps
"""
        )
    with c2:
        bt = metrics.get("ball_tracking", {})
        total = int(safe_num(bt.get("total_detections", 0)))
        yolo = int(safe_num(bt.get("yolo_detections", 0)))
        color_d = int(safe_num(bt.get("color_detections", 0)))
        pred = int(safe_num(bt.get("predicted_detections", 0)))
        rate = safe_num(bt.get("detection_rate", 0)) * 100

        # Stacked bar: breakdown of detection sources on this match
        source_df = pd.DataFrame({
            "Source": ["YOLOv8n", "Color/HSV fallback", "Short-horizon predict"],
            "Detections": [yolo, color_d, pred],
        })
        fig = px.bar(
            source_df,
            x="Detections",
            y="Source",
            orientation="h",
            color="Source",
            color_discrete_map={
                "YOLOv8n": "#f87171",
                "Color/HSV fallback": "#60a5fa",
                "Short-horizon predict": "#34d399",
            },
            text="Detections",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, title=f"Ball detection breakdown — {selected_match['display']}")
        st.plotly_chart(plotly_dark(fig, height=320), width='stretch')
        st.caption(
            f"Across **{int(safe_num(metrics.get('frame', 0)))} frames**, the dual-model system "
            f"produced **{total:,} ball detections** (detection rate: **{rate:.1f}%**). "
            f"YOLO alone contributed {yolo:,}; the fallback and predict passes recovered "
            f"the remaining {color_d + pred:,}."
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### What's in the pipeline")
    c = st.columns(4)
    with c[0]:
        st.markdown("**Detection & tracking**")
        st.markdown("YOLOv8n · RT-DETR · ByteTrack · ReID module")
    with c[1]:
        st.markdown("**Tactical analysis**")
        st.markdown("Formations (17 templates) · PPDA pressing · Possession zones · xG")
    with c[2]:
        st.markdown("**Player identity**")
        st.markdown("EasyOCR jersey numbers · SQLite roster matching · Team classification")
    with c[3]:
        st.markdown("**Outputs**")
        st.markdown("16-tab dashboard · Heatmaps · PDF reports · Highlight clips")

    st.info(
        "💡 This showcase runs on free Streamlit Cloud (CPU, 1 GB RAM). "
        "The full pipeline requires GPU + heavy ML dependencies — "
        f"run it locally from [the TactiVision repo]({GITHUB_URL}): `streamlit run demo/dashboard_final.py`"
    )


# ---------------------------------------------------------------------------
# TAB 2 — Match Stats (KPI tiles)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"## {TEAM_A_NAME} vs {TEAM_B_NAME} — Headline Numbers")

    bt = metrics.get("ball_tracking", {})
    pos = metrics.get("possession", {})
    tpp = pos.get("team_possession_percentage", {})
    shots = metrics.get("shot_detection", {})
    sprints = metrics.get("sprint_detection", {})

    row1 = st.columns(4)
    with row1[0]:
        kpi("Frames tracked", f"{int(safe_num(metrics.get('frame', 0))):,}",
            f"≈ {safe_num(metrics.get('frame', 0))/25:.1f} s @ 25 FPS")
    with row1[1]:
        kpi("Unique players", f"{int(safe_num(metrics.get('num_players', 0)))}",
            "incl. referees + both teams")
    with row1[2]:
        kpi("Ball detections", f"{int(safe_num(bt.get('total_detections', 0))):,}",
            f"detection rate: {safe_num(bt.get('detection_rate', 0))*100:.1f}%")
    with row1[3]:
        kpi("Possession changes", f"{int(safe_num(pos.get('total_possession_changes', 0)))}",
            "complete turnovers")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    row2 = st.columns(4)
    with row2[0]:
        kpi(f"{TEAM_A_NAME} possession",
            f"{safe_num(tpp.get('A', 0)):.1f}%",
            "share of ball-in-play time")
    with row2[1]:
        kpi(f"{TEAM_B_NAME} possession",
            f"{safe_num(tpp.get('B', 0)):.1f}%",
            "share of ball-in-play time")
    with row2[2]:
        kpi("Shots detected", f"{int(safe_num(shots.get('total_shots', 0)))}",
            "high-velocity ball events near goal")
    with row2[3]:
        kpi("Sprints detected", f"{int(safe_num(sprints.get('total_sprints', 0)))}",
            f"top speed: {safe_num(sprints.get('top_speed_mps', 0)):.1f} m/s")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Tracking quality")
    tq = metrics.get("tracking_quality", {})
    row3 = st.columns(4)
    with row3[0]:
        kpi("Canonical players",
            f"{int(safe_num(tq.get('canonical_players', 0)))}",
            "after dedup across tracks")
    with row3[1]:
        kpi("Raw YOLO tracks",
            f"{int(safe_num(tq.get('raw_yolo_tracks', 0)))}",
            "before dedup")
    with row3[2]:
        kpi("Dedup ratio",
            f"{safe_num(tq.get('dedup_ratio', 0)):.1f}x",
            "raw → canonical")
    with row3[3]:
        kpi("Speed readings",
            f"{int(safe_num(tq.get('speed_readings_capped', 0))):,}",
            "velocity samples")


# ---------------------------------------------------------------------------
# TAB 3 — Possession
# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("## Possession Analysis")
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
                "Possession": [safe_num(tpp.get("A", 0)), safe_num(tpp.get("B", 0))],
            })
            fig = px.pie(
                pie_df, values="Possession", names="Team",
                color="Team",
                color_discrete_map={TEAM_A_NAME: TEAM_A_COLOR, TEAM_B_NAME: TEAM_B_COLOR},
                hole=0.55,
            )
            fig.update_traces(textinfo="label+percent", textfont_size=14)
            st.plotly_chart(plotly_dark(fig, height=380), width='stretch')

        with c2:
            st.markdown("### Possession by pitch zone")
            zone_stats = pos.get("zone_stats", {})
            if zone_stats:
                zone_rows = []
                for team_code, zones in zone_stats.items():
                    team_name = TEAM_A_NAME if team_code == "A" else TEAM_B_NAME
                    for zone, pct in zones.items():
                        zone_rows.append({
                            "Team": team_name,
                            "Zone": zone,
                            "Pct": safe_num(pct),
                        })
                zone_df = pd.DataFrame(zone_rows)
                fig = px.bar(
                    zone_df, x="Zone", y="Pct", color="Team",
                    barmode="group",
                    color_discrete_map={TEAM_A_NAME: TEAM_A_COLOR, TEAM_B_NAME: TEAM_B_COLOR},
                    labels={"Pct": "% of possession in zone"},
                )
                st.plotly_chart(plotly_dark(fig, height=380), width='stretch')
            else:
                st.info("No zone-level breakdown available for this match.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Duration distribution
        dur = pos.get("duration_stats", {})
        if dur:
            st.markdown("### Possession duration distribution")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                kpi("Avg duration", f"{safe_num(dur.get('avg_duration', 0)):.2f}s", "per possession")
            with d2:
                kpi("Short (<1s)",
                    f"{safe_num(dur.get('short_pct', 0)):.1f}%",
                    f"{int(safe_num(dur.get('short_possessions', 0)))} possessions")
            with d3:
                kpi("Medium (1-5s)",
                    f"{safe_num(dur.get('medium_pct', 0)):.1f}%",
                    f"{int(safe_num(dur.get('medium_possessions', 0)))} possessions")
            with d4:
                kpi("Long (>5s)",
                    f"{safe_num(dur.get('long_pct', 0)):.1f}%",
                    f"{int(safe_num(dur.get('long_possessions', 0)))} possessions")

        # Possession history timeline
        history = pos.get("possession_history", [])
        if history:
            st.markdown("### Possession timeline")
            hist_df = pd.DataFrame(history)
            if "team" in hist_df.columns and "start_time" in hist_df.columns and "duration" in hist_df.columns:
                hist_df["team_name"] = hist_df["team"].map(
                    {"A": TEAM_A_NAME, "B": TEAM_B_NAME, "REF": "Referee"}
                ).fillna(hist_df["team"])
                fig = px.bar(
                    hist_df,
                    x="start_time", y=[1] * len(hist_df),
                    color="team_name",
                    color_discrete_map={
                        TEAM_A_NAME: TEAM_A_COLOR,
                        TEAM_B_NAME: TEAM_B_COLOR,
                        "Referee": NEUTRAL,
                    },
                    hover_data={"duration": ":.2f", "player_id": True, "team_name": True},
                    labels={"start_time": "Time (seconds)", "y": ""},
                )
                fig.update_layout(showlegend=True, yaxis=dict(showticklabels=False))
                st.plotly_chart(plotly_dark(fig, height=200), width='stretch')


# ---------------------------------------------------------------------------
# TAB 4 — Shots
# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("## Shot Detection")
    shot_events = metrics.get("shot_events", [])
    total_shots = int(safe_num(metrics.get("shot_detection", {}).get("total_shots", 0)))

    if not shot_events:
        st.info("No shot events detected in this match's pipeline run.")
    else:
        st.markdown(
            f"The pipeline detected **{total_shots} shot event(s)** — high-velocity "
            "ball movements originating in the attacking third."
        )

        # Shot table
        shots_df = pd.DataFrame(shot_events)
        if "ball_position" in shots_df.columns:
            shots_df["ball_x"] = shots_df["ball_position"].apply(lambda p: p[0] if isinstance(p, list) and len(p) > 0 else None)
            shots_df["ball_y"] = shots_df["ball_position"].apply(lambda p: p[1] if isinstance(p, list) and len(p) > 1 else None)

        display_cols = [
            c for c in ["timestamp", "shooter_id", "shooter_team",
                        "velocity_mps", "angle_to_goal_deg", "distance_to_goal_px", "frame"]
            if c in shots_df.columns
        ]
        pretty = shots_df[display_cols].copy()
        if "shooter_team" in pretty.columns:
            pretty["shooter_team"] = pretty["shooter_team"].map(
                {"A": TEAM_A_NAME, "B": TEAM_B_NAME, "REF": "Referee"}
            ).fillna(pretty["shooter_team"])

        pretty = pretty.rename(columns={
            "timestamp": "Time (s)",
            "shooter_id": "Player ID",
            "shooter_team": "Team",
            "velocity_mps": "Ball velocity (m/s)",
            "angle_to_goal_deg": "Angle to goal (°)",
            "distance_to_goal_px": "Distance to goal (px)",
            "frame": "Frame",
        })

        st.dataframe(pretty.style.format({
            "Time (s)": "{:.2f}",
            "Ball velocity (m/s)": "{:.1f}",
            "Angle to goal (°)": "{:.1f}",
            "Distance to goal (px)": "{:.0f}",
        }), width='stretch', hide_index=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Shot velocity vs angle scatter
        if "ball_velocity_mps" in shots_df.columns or "velocity_mps" in shots_df.columns:
            vel_col = "velocity_mps" if "velocity_mps" in shots_df.columns else "ball_velocity_mps"
            if vel_col in shots_df.columns and "angle_to_goal_deg" in shots_df.columns:
                st.markdown("### Shot characteristics")
                scatter_df = shots_df.copy()
                scatter_df["team_name"] = scatter_df["shooter_team"].map(
                    {"A": TEAM_A_NAME, "B": TEAM_B_NAME, "REF": "Referee"}
                ).fillna(scatter_df["shooter_team"])
                fig = px.scatter(
                    scatter_df,
                    x="angle_to_goal_deg",
                    y=vel_col,
                    color="team_name",
                    size="distance_to_goal_px",
                    hover_data=["shooter_id", "timestamp"],
                    color_discrete_map={
                        TEAM_A_NAME: TEAM_A_COLOR,
                        TEAM_B_NAME: TEAM_B_COLOR,
                        "Referee": NEUTRAL,
                    },
                    labels={
                        "angle_to_goal_deg": "Angle to goal (degrees)",
                        vel_col: "Ball velocity (m/s)",
                    },
                )
                st.plotly_chart(plotly_dark(fig, height=400), width='stretch')


# ---------------------------------------------------------------------------
# TAB 5 — Physical / Sprints
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("## Physical & Sprint Analysis")
    sprint_events = metrics.get("sprint_events", [])
    sprint_summary = metrics.get("sprint_detection", {})

    if not sprint_events:
        st.info("No sprint events detected for this match.")
    else:
        c = st.columns(4)
        with c[0]:
            kpi("Sprints", f"{int(safe_num(sprint_summary.get('total_sprints', 0)))}",
                "across all players")
        with c[1]:
            kpi("High-intensity",
                f"{int(safe_num(sprint_summary.get('high_intensity_sprints', 0)))}",
                "qualifying as high-intensity")
        with c[2]:
            kpi("Top speed",
                f"{safe_num(sprint_summary.get('top_speed_mps', 0)):.1f} m/s",
                f"≈ {safe_num(sprint_summary.get('top_speed_mps', 0))*3.6:.1f} km/h")
        with c[3]:
            kpi("Avg max speed",
                f"{safe_num(sprint_summary.get('avg_max_speed_mps', 0)):.1f} m/s",
                "across sprints")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        sprint_df = pd.DataFrame(sprint_events)
        sprint_df["team_name"] = sprint_df["team"].map(
            {"A": TEAM_A_NAME, "B": TEAM_B_NAME, "REF": "Referee"}
        ).fillna(sprint_df["team"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Sprint top-speed distribution")
            fig = px.histogram(
                sprint_df, x="max_speed_mps", color="team_name",
                color_discrete_map={
                    TEAM_A_NAME: TEAM_A_COLOR,
                    TEAM_B_NAME: TEAM_B_COLOR,
                    "Referee": NEUTRAL,
                },
                nbins=20,
                labels={"max_speed_mps": "Max sprint speed (m/s)"},
            )
            st.plotly_chart(plotly_dark(fig, height=360), width='stretch')

        with c2:
            st.markdown("### Sprint duration distribution")
            fig = px.histogram(
                sprint_df, x="duration_sec", color="team_name",
                color_discrete_map={
                    TEAM_A_NAME: TEAM_A_COLOR,
                    TEAM_B_NAME: TEAM_B_COLOR,
                    "Referee": NEUTRAL,
                },
                nbins=20,
                labels={"duration_sec": "Sprint duration (seconds)"},
            )
            st.plotly_chart(plotly_dark(fig, height=360), width='stretch')

        st.markdown("### Top sprinters")
        top_sprints = sprint_df.sort_values("max_speed_mps", ascending=False).head(10)
        display = top_sprints[["player_id", "team_name", "duration_sec",
                               "max_speed_mps", "avg_speed_mps", "is_high_intensity"]].copy()
        display = display.rename(columns={
            "player_id": "Player",
            "team_name": "Team",
            "duration_sec": "Duration (s)",
            "max_speed_mps": "Max speed (m/s)",
            "avg_speed_mps": "Avg speed (m/s)",
            "is_high_intensity": "High intensity",
        })
        st.dataframe(display.style.format({
            "Duration (s)": "{:.2f}",
            "Max speed (m/s)": "{:.2f}",
            "Avg speed (m/s)": "{:.2f}",
        }), width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# TAB 6 — Heatmaps
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("## Positional Heatmaps")
    st.markdown(
        "Generated by the pipeline from frame-by-frame player positions, projected "
        "onto a normalised pitch coordinate system."
    )

    match_dir = selected_match["dir"]

    # Team / global heatmaps
    st.markdown("### Team-level & global")
    team_cols = st.columns(3)
    for col, (name, filename) in zip(
        team_cols,
        [
            (f"{TEAM_A_NAME} (Team A)", "heatmap_team_A.png"),
            (f"{TEAM_B_NAME} (Team B)", "heatmap_team_B.png"),
            ("Global (all players)", "heatmap_global.png"),
        ],
    ):
        img_path = match_dir / filename
        with col:
            st.markdown(f"**{name}**")
            if img_path.exists():
                st.image(str(img_path), width='stretch')
            else:
                st.info(f"Not available: {filename}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Per-player heatmap browser
    st.markdown("### Per-player heatmap")
    player_files = list_player_heatmaps(str(match_dir))
    if not player_files:
        st.info("No per-player heatmaps for this match.")
    else:
        player_ids = [f.replace("heatmap_player_", "").replace(".png", "") for f in player_files]
        c1, c2 = st.columns([1, 2])
        with c1:
            selected_pid = st.selectbox(
                f"Pick a player ID ({len(player_ids)} available)",
                options=player_ids,
                index=0,
            )
            # Try to pull identity info
            identities = metrics.get("player_identities", {})
            identity = identities.get(selected_pid, {})
            team = identity.get("team", "unknown")
            display = identity.get("display", f"Player {selected_pid}")
            jersey = identity.get("number")
            team_name_map = {"A": TEAM_A_NAME, "B": TEAM_B_NAME, "REF": "Referee"}
            team_label = team_name_map.get(team, team)
            st.markdown(f"**Display:** {display}")
            st.markdown(f"**Team:** {team_label}")
            if jersey:
                st.markdown(f"**Jersey #:** {jersey}")
        with c2:
            img_path = match_dir / f"heatmap_player_{selected_pid}.png"
            if img_path.exists():
                st.image(str(img_path), width='stretch')
            else:
                st.warning("Heatmap file missing.")


# ---------------------------------------------------------------------------
# TAB 7 — Players
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("## Player Tracking Summary")
    tracks = metrics.get("tracks", [])
    if not tracks:
        st.info("No per-player track data in this match's metrics.")
    else:
        tracks_df = pd.DataFrame(tracks)
        tracks_df["team_name"] = tracks_df["team"].map(
            {"A": TEAM_A_NAME, "B": TEAM_B_NAME, "REF": "Referee"}
        ).fillna(tracks_df["team"].fillna("unknown"))

        c1, c2 = st.columns([2, 1])
        with c1:
            team_filter = st.multiselect(
                "Filter by team",
                options=sorted(tracks_df["team_name"].unique()),
                default=sorted(tracks_df["team_name"].unique()),
            )
        with c2:
            min_distance = st.number_input(
                "Min distance (m)", min_value=0.0, value=50.0, step=10.0
            )

        filtered = tracks_df[
            tracks_df["team_name"].isin(team_filter)
            & (tracks_df.get("total_distance_m", 0) >= min_distance)
        ]

        st.markdown(f"Showing **{len(filtered)}** of {len(tracks_df)} tracked players")

        # Top distance / speed chart
        if not filtered.empty and "total_distance_m" in filtered.columns:
            top20 = filtered.nlargest(20, "total_distance_m")
            fig = px.bar(
                top20,
                x="total_distance_m",
                y="display_name",
                color="team_name",
                orientation="h",
                color_discrete_map={
                    TEAM_A_NAME: TEAM_A_COLOR,
                    TEAM_B_NAME: TEAM_B_COLOR,
                    "Referee": NEUTRAL,
                    "unknown": "#475569",
                },
                hover_data=["player_id", "avg_speed_mps", "max_speed_mps"],
                labels={
                    "total_distance_m": "Total distance (m)",
                    "display_name": "Player",
                },
                title="Top 20 by distance covered",
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(plotly_dark(fig, height=500), width='stretch')

        # Table
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
            "avg_speed_mps": "Avg speed (m/s)",
            "max_speed_mps": "Max speed (m/s)",
            "workload_score": "Workload",
            "involvement_index": "Involvement",
        })
        st.dataframe(pretty.style.format({
            "Distance (m)": "{:.1f}",
            "Avg speed (m/s)": "{:.2f}",
            "Max speed (m/s)": "{:.2f}",
            "Workload": "{:.1f}",
            "Involvement": "{:.2f}",
        }), width='stretch', hide_index=True, height=400)


# ---------------------------------------------------------------------------
# TAB 8 — How It Works
# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("## How TactiVision Pro works")

    st.markdown(
        """
### The pipeline
```
Broadcast video (mp4)
        │
        ▼
┌──────────────────────┐
│  Frame extraction    │  OpenCV
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  YOLOv8n detection   │  Primary: players + ball
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  RT-DETR second pass │  Ball recovery on YOLO-miss frames
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  ByteTrack           │  Identity continuity across frames
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Feature extractors  │  70+ tactical features
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Streamlit dashboard │  16 interactive tabs (local)
└──────────────────────┘
```
"""
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Tech stack")
        st.markdown(
            """
- **Language:** Python 3.10+
- **Deep learning:** PyTorch, Ultralytics (YOLOv8), RT-DETR
- **Tracking:** ByteTrack (with BoT-SORT comparison baseline)
- **Computer vision:** OpenCV
- **OCR (jersey numbers):** EasyOCR
- **Dashboard:** Streamlit + Plotly
- **Data:** SQLite (roster matching), pandas, NumPy
- **Analytics:** SciPy, scikit-learn
"""
        )

    with c2:
        st.markdown("### What the full dashboard includes")
        st.markdown(
            """
The local `demo/dashboard_final.py` has **16 tabs**:
- Overview · Shooting · Passing · Physical · Tactical
- xG & Analytics · Heatmaps · Highlights
- Database · Settings · AI Recommendations
- Player Performance · Match Comparison
- Live Match Data · Generate Report · Roster Management

This public showcase focuses on the subset that can run on free Streamlit Cloud
without GPU or heavy ML dependencies.
"""
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Why this showcase is read-only")
    st.markdown(
        """
Running the full pipeline requires:
- **GPU-accelerated inference** (YOLOv8 + RT-DETR on broadcast video is CPU-slow)
- **Multi-GB ML dependencies** (PyTorch, TensorFlow, Ultralytics, OpenCV-contrib)
- **1-5 GB match video files**
- **~1-2 GB RAM per concurrent user**

None of those fit on a free Streamlit Cloud instance. This showcase instead
visualises the **pre-computed JSON output** that the pipeline wrote after
analysing matches locally. Everything you see here is the real algorithm's output
on real broadcast footage — just loaded from disk instead of re-computed on each page view.

**To run the full pipeline yourself:**
"""
    )
    st.code(
        f"""git clone {GITHUB_URL}.git
cd TactiVision
pip install -r requirements.txt
streamlit run demo/dashboard_final.py""",
        language="bash",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### The team")
    st.markdown(
        """
Built by a six-person engineering team as the graduation project for the
BSc Data Science programme at **Misr International University (MIU)**, 2026.

- **Mohanad Amr** — Team lead, ML/CV engineer, dashboard architecture
- **5 co-engineers** — detection, tracking, analytics, PDF reporting, QA

For questions or collaboration, reach out:
"""
    )
    st.markdown(f"- GitHub: [MohanadddAmrr]({GITHUB_URL})")
    st.markdown(f"- LinkedIn: [Mohanad Amr]({LINKEDIN_URL})")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    f"TactiVision Pro · Showcase build · Data: {selected_match['display']} · "
    f"[Source on GitHub]({GITHUB_URL})"
)
