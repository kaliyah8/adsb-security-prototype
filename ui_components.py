from __future__ import annotations

import html

import pandas as pd
import streamlit as st

FRIENDLY_COLUMN_MAP = {
    "icao": "Aircraft ID",
    "icao24": "Aircraft ID",
    "callsign": "Call Sign",
    "timestamp": "Time",
    "frame_idx": "Frame",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "altitude": "Altitude (m)",
    "baro_altitude": "Barometric Altitude (m)",
    "heading": "Heading (deg)",
    "velocity": "Reported Speed (m/s)",
    "speed_mps": "Reported Speed (m/s)",
    "distance_m": "Distance Moved (m)",
    "delta_t": "Time Gap (s)",
    "implied_speed_mps": "Implied Speed (m/s)",
    "accel_mps2": "Acceleration (m/s²)",
    "vert_rate_mps": "Vertical Rate (m/s)",
    "vertical_rate": "Vertical Rate (m/s)",
    "delta_heading_deg": "Heading Change (deg)",
    "turn_rate_dps": "Turn Rate (deg/s)",
    "speed_mismatch_mps": "Speed Mismatch (m/s)",
    "data_quality_score": "Data Quality",
    "bad_point": "Bad Point",
    "attack_window": "Attack Active",
    "scenario_label": "Scenario",
    "human_alert_label": "Alert State",
    "prev_latitude": "Previous Latitude",
    "prev_longitude": "Previous Longitude",
    "prev_velocity": "Previous Speed",
    "prev_heading": "Previous Heading",
    "prev_altitude": "Previous Altitude",
    "prev_timestamp": "Previous Time",
    "has_prev": "Has Previous Point",
}

_COLUMN_HELP_ROWS = [
    ["Aircraft ID", "The unique identifier used to follow one aircraft across time."],
    ["Reported Speed (m/s)", "The speed value reported in the aircraft data stream."],
    ["Implied Speed (m/s)", "The speed inferred from how far the aircraft moved between updates."],
    ["Speed Mismatch (m/s)", "Difference between reported speed and motion-implied speed."],
    ["Acceleration (m/s²)", "How quickly the aircraft speed is changing."],
    ["Vertical Rate (m/s)", "How fast the aircraft is climbing or descending."],
    ["Turn Rate (deg/s)", "How quickly the aircraft heading is changing."],
    ["Data Quality", "A trust score estimating how reliable a point is."],
    ["Attack Active", "Whether that row falls inside the injected anomaly window."],
    ["Alert State", "Human-readable anomaly label shown in the dashboard."],
]


def _safe(text: object) -> str:
    return html.escape("" if text is None else str(text))


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #08111f;
            --bg-main-2: #0b1220;
            --bg-panel: rgba(15, 23, 42, 0.92);
            --bg-panel-strong: rgba(17, 24, 39, 0.98);
            --bg-soft: rgba(30, 41, 59, 0.72);
            --line: rgba(148, 163, 184, 0.16);
            --line-soft: rgba(148, 163, 184, 0.10);
            --text: #e5edf7;
            --text-strong: #f8fafc;
            --muted: #94a3b8;
            --accent: #60a5fa;
            --accent-2: #38bdf8;
            --success: #22c55e;
            --warn: #f59e0b;
            --danger: #ef4444;
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 12px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.08), transparent 28%),
                linear-gradient(180deg, var(--bg-main) 0%, var(--bg-main-2) 100%);
            color: var(--text);
        }

        .main .block-container {
            padding-top: 0.85rem;
            padding-bottom: 1.4rem;
            max-width: 1460px;
        }

        [data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        .sidebar-caption {
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.45;
            margin-bottom: 0.9rem;
        }

        .hero-wrap {
            padding: 0.95rem 1.1rem 0.9rem 1.1rem;
            border: 1px solid var(--line);
            border-radius: var(--radius-lg);
            background: linear-gradient(
                180deg,
                rgba(17,24,39,0.98) 0%,
                rgba(15,23,42,0.94) 100%
            );
            box-shadow: var(--shadow);
            margin-bottom: 0.7rem;
        }

        .hero-kicker {
            color: var(--accent);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: 1.55rem;
            font-weight: 850;
            line-height: 1.08;
            color: var(--text-strong);
            margin-bottom: 0.22rem;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
            max-width: 900px;
        }

        .section-title {
            font-size: 1.16rem;
            font-weight: 800;
            color: var(--text-strong);
            margin-top: 0.15rem;
            margin-bottom: 0.08rem;
            letter-spacing: 0.01em;
        }

        .section-subtitle {
            color: var(--muted);
            margin-bottom: 0.75rem;
            font-size: 0.93rem;
            line-height: 1.45;
        }

        .summary-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-bottom: 1rem;
        }

        .summary-chip {
            border: 1px solid var(--line-soft);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(12,19,32,0.96));
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.14);
        }

        .summary-chip-label {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            margin-bottom: 0.32rem;
        }

        .summary-chip-value {
            color: var(--text-strong);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .summary-chip-subtle {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.32rem;
            line-height: 1.35;
        }

        .focus-panel {
            border: 1px solid var(--line-soft);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, rgba(17,24,39,0.97), rgba(15,23,42,0.94));
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
            margin-bottom: 0.85rem;
        }

        .focus-title {
            color: var(--text-strong);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .focus-text {
            color: var(--text);
            font-size: 1rem;
            line-height: 1.5;
        }

        .focus-text strong {
            color: #ffffff;
        }

        .compact-note {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
        }

        @media (max-width: 1100px) {
            .summary-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 640px) {
            .summary-strip {
                grid-template-columns: 1fr;
            }
        }

        .metric-card {
            border: 1px solid var(--line-soft);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            background: linear-gradient(
                180deg,
                rgba(17,24,39,0.99) 0%,
                rgba(15,23,42,0.95) 100%
            );
            min-height: 116px;
            margin-bottom: 0.5rem;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.42rem;
            font-weight: 800;
        }

        .metric-value {
            color: #ffffff;
            font-size: 1.75rem;
            font-weight: 850;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }

        .metric-caption {
            color: var(--muted);
            font-size: 0.84rem;
            margin-top: 0.42rem;
            line-height: 1.38;
        }

        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.22rem 0.6rem;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            border: 1px solid transparent;
        }

        .status-ok {
            background: rgba(34, 197, 94, 0.14);
            color: #bbf7d0;
            border-color: rgba(34, 197, 94, 0.22);
        }

        .status-warn {
            background: rgba(245, 158, 11, 0.14);
            color: #fde68a;
            border-color: rgba(245, 158, 11, 0.22);
        }

        .status-danger {
            background: rgba(239, 68, 68, 0.14);
            color: #fecaca;
            border-color: rgba(239, 68, 68, 0.22);
        }

        .status-neutral {
            background: rgba(148, 163, 184, 0.12);
            color: #cbd5e1;
            border-color: rgba(148, 163, 184, 0.18);
        }

        .info-banner {
            border-radius: 14px;
            padding: 0.82rem 0.95rem;
            border: 1px solid var(--line);
            margin-top: 0.35rem;
            margin-bottom: 0.9rem;
            font-weight: 650;
            line-height: 1.45;
        }

        .info-success {
            background: rgba(34, 197, 94, 0.12);
            color: #bbf7d0;
        }

        .info-info {
            background: rgba(59, 130, 246, 0.12);
            color: #bfdbfe;
        }

        .info-warn {
            background: rgba(245, 158, 11, 0.12);
            color: #fde68a;
        }

        .info-danger {
            background: rgba(239, 68, 68, 0.12);
            color: #fecaca;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(148,163,184,0.18);
            background: linear-gradient(180deg, #172033 0%, #111827 100%);
            color: white;
            font-weight: 800;
            padding: 0.48rem 0.9rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.14);
        }

        .stButton > button:hover {
            border-color: rgba(96,165,250,0.32);
            color: #ffffff;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stNumberInput > div > div,
        .stTextInput > div > div {
            background: rgba(15,23,42,0.88);
            border-radius: 12px;
        }

        .stDataFrame, .stTable {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.10);
            background: rgba(10, 18, 30, 0.82);
        }

        div[data-baseweb="tab-list"] {
            gap: 0.4rem;
            margin-bottom: 0.5rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.55rem 0.95rem;
            background: rgba(15,23,42,0.72);
            color: #cbd5e1;
            border: 1px solid rgba(148,163,184,0.10);
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(30,41,59,0.96);
            color: #ffffff;
            border-color: rgba(96,165,250,0.18);
        }

        .streamlit-expanderHeader {
            font-weight: 700;
            color: var(--text-strong);
        }

        [data-testid="stExpander"] {
            border: 1px solid rgba(148,163,184,0.10);
            border-radius: 14px;
            background: rgba(10, 18, 30, 0.58);
        }

        [data-testid="stExpander"] details {
            border-radius: 14px;
        }

        [data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
            position: fixed;
        }

        hr {
            border-color: rgba(148,163,184,0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, kicker: str = "ADS-B Security Monitoring") -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-kicker">{_safe(kicker)}</div>
            <div class="hero-title">{_safe(title)}</div>
            <div class="hero-subtitle">{_safe(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{_safe(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{_safe(subtitle)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: object, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{_safe(label)}</div>
            <div class="metric-value">{_safe(value)}</div>
            <div class="metric-caption">{_safe(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_banner(text: str, kind: str = "info") -> None:
    cls = {
        "success": "info-success",
        "info": "info-info",
        "warn": "info-warn",
        "warning": "info-warn",
        "danger": "info-danger",
        "error": "info-danger",
    }.get(kind, "info-info")

    st.markdown(f'<div class="info-banner {cls}">{_safe(text)}</div>', unsafe_allow_html=True)


def render_sidebar_caption(text: str) -> None:
    st.markdown(f'<div class="sidebar-caption">{_safe(text)}</div>', unsafe_allow_html=True)


def render_status_pill(status: str) -> str:
    s = (status or "").strip().lower()

    if s in {"ok", "normal", "healthy", "active", "stable", "ready", "online", "clear"}:
        css = "status-ok"
    elif s in {"warn", "warning", "degraded", "caution", "elevated"}:
        css = "status-warn"
    elif s in {"alert", "critical", "danger", "offline", "error", "anomaly"}:
        css = "status-danger"
    else:
        css = "status-neutral"

    return f'<span class="status-pill {css}">{_safe(status)}</span>'


def simplify_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    preferred = [
        "icao",
        "callsign",
        "timestamp",
        "frame_idx",
        "scenario_label",
        "human_alert_label",
        "data_quality_score",
        "latitude",
        "longitude",
        "altitude",
        "baro_altitude",
        "speed_mps",
        "velocity",
        "implied_speed_mps",
        "speed_mismatch_mps",
        "accel_mps2",
        "vert_rate_mps",
        "vertical_rate",
        "turn_rate_dps",
        "heading",
        "distance_m",
        "delta_t",
        "attack_window",
        "bad_point",
    ]

    ordered = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[ordered]
    out = out.rename(columns={c: FRIENDLY_COLUMN_MAP.get(c, c) for c in out.columns})

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(3)

    return out


def column_help_table() -> pd.DataFrame:
    return pd.DataFrame(_COLUMN_HELP_ROWS, columns=["Column", "Meaning"])
