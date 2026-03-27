from __future__ import annotations

import html
import json
import hashlib
import inspect
import os
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from anomaly_detection import detect_anomalies
from data_fetcher import fetch_live_adsb_data
from data_processing import process_adsb_data
from data_simulation import (
    generate_aircraft_data,
    inject_ghost_aircraft_attack,
    inject_gps_spoofing_attack,
    inject_teleportation_attack,
)
from ml_features import build_feature_frame
from ui_components import (
    column_help_table,
    inject_global_css,
    metric_card,
    render_hero,
    render_info_banner,
    render_section_title,
    render_sidebar_caption,
    render_status_pill,
    simplify_dataframe_for_display,
)

# Optional ML imports
try:
    from ml_pipeline import load_artifacts, score_sequences, train_lstm_autoencoder
    from ml_evaluation import (
        EVAL_METRICS_PATH,
        EVAL_ROW_PATH,
        EVAL_SCENARIO_PATH,
        EVAL_SEQ_PATH,
        run_evaluation,
    )

    ML_AVAILABLE = True
    ML_IMPORT_ERROR: Optional[str] = None
except Exception as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = repr(e)
    EVAL_METRICS_PATH = "artifacts/evaluation_metrics.json"
    EVAL_ROW_PATH = "artifacts/evaluation_row_details.csv"
    EVAL_SCENARIO_PATH = "artifacts/evaluation_scenario_summary.csv"
    EVAL_SEQ_PATH = "artifacts/evaluation_seq_details.csv"


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="ADS-B Security Monitoring",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


# -----------------------------
# Avatar / custom theme
# -----------------------------
def inject_avatar_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --avatar-bg-0: #020814;
            --avatar-bg-1: #041121;
            --avatar-bg-2: #07203a;
            --avatar-bg-3: #0b2d4d;
            --avatar-blue: #55d6c5;
            --avatar-blue-soft: #b8fff0;
            --avatar-text: #eaf6ff;
            --avatar-muted: #9db8cc;
            --avatar-border: rgba(85, 214, 197, 0.18);
            --avatar-red: #ff6363;
            --avatar-yellow: #ffc857;
            --avatar-purple: #b889ff;
            --avatar-green: #34d399;
        }

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(85, 214, 197, 0.12), transparent 24%),
                radial-gradient(circle at top right, rgba(255, 200, 87, 0.08), transparent 22%),
                linear-gradient(180deg, var(--avatar-bg-0) 0%, var(--avatar-bg-1) 42%, #01060e 100%);
            color: var(--avatar-text);
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(7, 32, 58, 0.98), rgba(2, 8, 20, 0.99));
            border-right: 1px solid var(--avatar-border);
        }

        [data-testid="stSidebar"] * {
            color: var(--avatar-text) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
            background: transparent;
            border: none;
            padding: 0;
            border-radius: 0;
            margin: 0.35rem 0 1rem 0;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            height: 44px;
            padding: 0 1rem;
            font-weight: 700;
            color: var(--avatar-muted);
            border: 1px solid rgba(85, 214, 197, 0.12);
            background: rgba(8, 28, 48, 0.38);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(85,214,197,0.22), rgba(85,214,197,0.10));
            color: var(--avatar-text) !important;
            border-color: rgba(85, 214, 197, 0.24);
        }

        .stButton button, .stDownloadButton button {
            background: linear-gradient(180deg, #15505a, #0d3340);
            color: var(--avatar-text);
            border: 1px solid rgba(85, 214, 197, 0.20);
            border-radius: 12px;
            font-weight: 700;
        }

        .stButton button:hover, .stDownloadButton button:hover {
            border-color: rgba(255, 200, 87, 0.40);
            color: white;
        }

        .stDataFrame, div[data-testid="stTable"] {
            border: 1px solid var(--avatar-border);
            border-radius: 16px;
            overflow: hidden;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(8, 28, 48, 0.92), rgba(4, 17, 33, 0.96));
            border: 1px solid var(--avatar-border);
            border-radius: 18px;
            padding: 0.7rem 0.9rem;
        }

        .avatar-panel {
            background: linear-gradient(180deg, rgba(8, 28, 48, 0.92), rgba(4, 17, 33, 0.96));
            border: 1px solid var(--avatar-border);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
        }

        .avatar-subtle {
            color: var(--avatar-muted);
            font-size: 0.84rem;
        }

        .avatar-chip-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.35rem;
        }

        .avatar-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            border: 1px solid rgba(85, 214, 197, 0.14);
            background: rgba(85, 214, 197, 0.08);
            color: var(--avatar-blue-soft);
        }

        .avatar-title {
            font-family: "Orbitron", sans-serif;
            letter-spacing: 0.06em;
            font-weight: 800;
        }

        .mini-stat-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.65rem;
        }

        .mini-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            border: 1px solid rgba(85, 214, 197, 0.10);
            background: rgba(85, 214, 197, 0.05);
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
        }

        .mini-stat .label {
            color: var(--avatar-muted);
            font-size: 0.84rem;
        }

        .mini-stat .value {
            color: var(--avatar-text);
            font-weight: 700;
        }

        .legend-inline {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 0.6rem;
        }

        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--avatar-muted);
            font-size: 0.8rem;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }

        .dashboard-panel {
            background: linear-gradient(180deg, rgba(8, 28, 48, 0.92), rgba(4, 17, 33, 0.96));
            border: 1px solid var(--avatar-border);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
            margin-bottom: 0.7rem;
        }

        .panel-eyebrow {
            color: var(--avatar-blue-soft);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-bottom: 0.35rem;
        }

        .panel-title {
            color: var(--avatar-text);
            font-family: "Orbitron", sans-serif;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }

        .panel-subtitle {
            color: var(--avatar-muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-bottom: 0.8rem;
        }

        .detail-grid {
            display: grid;
            gap: 0.6rem;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            border: 1px solid rgba(85, 214, 197, 0.10);
            background: rgba(85, 214, 197, 0.045);
        }

        .detail-row .label {
            color: var(--avatar-muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
        }

        .detail-row .value {
            color: var(--avatar-text);
            font-size: 0.95rem;
            font-weight: 700;
            text-align: right;
            line-height: 1.35;
        }

        .insight-note {
            border-radius: 16px;
            border: 1px solid rgba(85, 214, 197, 0.10);
            background: rgba(85, 214, 197, 0.055);
            color: var(--avatar-text);
            padding: 0.85rem 0.95rem;
            line-height: 1.55;
            margin-top: 0.8rem;
        }

        .chart-shell {
            padding: 0.8rem 0.95rem 0.15rem 0.95rem;
            border-radius: 18px;
            border: 1px solid rgba(85, 214, 197, 0.10);
            background: rgba(8, 28, 48, 0.48);
            margin-bottom: 0.45rem;
        }

        .chart-title {
            color: var(--avatar-text);
            font-size: 0.94rem;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin-bottom: 0.14rem;
        }

        .chart-subtitle {
            color: var(--avatar-muted);
            font-size: 0.83rem;
            line-height: 1.45;
            margin-bottom: 0.6rem;
        }

        .landing-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
            gap: 0.95rem;
            margin-bottom: 0.95rem;
        }

        .story-card {
            background: linear-gradient(180deg, rgba(8, 28, 48, 0.94), rgba(4, 17, 33, 0.98));
            border: 1px solid rgba(85, 214, 197, 0.14);
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1.05rem 1.15rem;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
        }

        .run-snapshot-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.2rem 0 0.95rem 0;
        }

        .run-snapshot-card {
            background: linear-gradient(180deg, rgba(8, 28, 48, 0.88), rgba(4, 17, 33, 0.96));
            border: 1px solid rgba(85, 214, 197, 0.12);
            border-radius: 20px;
            padding: 0.95rem 1rem;
        }

        .run-snapshot-label {
            color: var(--avatar-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .run-snapshot-value {
            color: var(--avatar-text);
            font-size: 1.15rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.18rem;
        }

        .run-snapshot-meta {
            color: var(--avatar-muted);
            font-size: 0.83rem;
            line-height: 1.42;
        }

        @media (max-width: 980px) {
            .run-snapshot-grid {
                grid-template-columns: 1fr;
            }
        }

        .story-kicker {
            color: var(--avatar-blue-soft);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-size: 0.73rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .story-title {
            color: var(--avatar-text);
            font-family: "Orbitron", sans-serif;
            font-size: 1.2rem;
            line-height: 1.18;
            margin-bottom: 0.4rem;
        }

        .story-body {
            color: var(--avatar-text);
            font-size: 0.96rem;
            line-height: 1.62;
        }

        .story-body p {
            margin: 0 0 0.75rem 0;
        }

        .story-list {
            display: grid;
            gap: 0.58rem;
            margin-top: 0.7rem;
        }

        .story-item {
            border: 1px solid rgba(85, 214, 197, 0.10);
            background: rgba(85, 214, 197, 0.05);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
        }

        .story-item strong {
            display: block;
            color: var(--avatar-text);
            font-size: 0.9rem;
            margin-bottom: 0.18rem;
        }

        .story-item span {
            color: var(--avatar-muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .stat-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.55rem 0 0.8rem 0;
        }

        .stat-tile {
            border: 1px solid rgba(85, 214, 197, 0.10);
            background: rgba(85, 214, 197, 0.05);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
        }

        .stat-tile-label {
            color: var(--avatar-muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 800;
            margin-bottom: 0.26rem;
        }

        .stat-tile-value {
            color: var(--avatar-text);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.3;
        }

        @media (max-width: 980px) {
            .stat-strip {
                grid-template-columns: 1fr;
            }
        }

        .capability-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.2rem;
            margin-bottom: 0.9rem;
        }

        .capability-card {
            background: linear-gradient(180deg, rgba(8, 28, 48, 0.90), rgba(4, 17, 33, 0.96));
            border: 1px solid rgba(85, 214, 197, 0.10);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            min-height: 175px;
        }

        .capability-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: rgba(85, 214, 197, 0.14);
            color: #d8f1ff;
            font-weight: 800;
            margin-bottom: 0.7rem;
        }

        .capability-title {
            color: var(--avatar-text);
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.32rem;
        }

        .capability-body {
            color: var(--avatar-muted);
            font-size: 0.87rem;
            line-height: 1.5;
        }

        .page-note {
            border-radius: 18px;
            border: 1px solid rgba(85, 214, 197, 0.10);
            background: rgba(85, 214, 197, 0.055);
            color: var(--avatar-text);
            padding: 0.95rem 1rem;
            line-height: 1.6;
            margin-bottom: 0.95rem;
        }

        .page-note strong {
            color: #ffffff;
        }

        @media (max-width: 1100px) {
            .landing-grid,
            .capability-grid {
                grid-template-columns: 1fr;
            }
        }

        .stSelectbox label,
        .stRadio label,
        .stSlider label,
        .stNumberInput label,
        .stCheckbox label,
        .stToggle label {
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_avatar_theme()


# -----------------------------
# Environment loading
# -----------------------------
def _load_env_local() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                os.environ[k.strip()] = v.strip()
    except Exception:
        pass


_load_env_local()


# -----------------------------
# Generic helpers
# -----------------------------
def as_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen: dict[str, int] = {}
    new_cols: list[str] = []

    for col in out.columns:
        name = str(col)
        if name not in seen:
            seen[name] = 0
            new_cols.append(name)
        else:
            seen[name] += 1
            new_cols.append(f"{name} ({seen[name]})")

    out.columns = new_cols
    return out


def prettify_column_name(name: str) -> str:
    pretty = str(name).replace("_", " ").replace("__", " ")
    pretty = " ".join(pretty.split()).strip()
    return pretty.title()


def prepare_df_for_streamlit(df: pd.DataFrame, simplify: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if simplify:
        try:
            simplified = simplify_dataframe_for_display(out)
            if isinstance(simplified, pd.DataFrame):
                out = simplified
        except Exception:
            pass

    out = make_columns_unique(out)
    out.columns = [prettify_column_name(c) for c in out.columns]
    return out


def escape_text(value: object) -> str:
    return html.escape("—" if value is None or value == "" else str(value))


def format_display_value(value: Any, *, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            text = value.strip()
            return text or "—"
    else:
        text = str(value).strip()
        return text or "—"

    if not np.isfinite(numeric):
        return "—"

    return f"{numeric:.{decimals}f}{suffix}"


def show_dataframe(
    df: pd.DataFrame,
    *,
    max_rows: int = 250,
    simplify: bool = False,
    hide_index: bool = True,
) -> None:
    safe_df = prepare_df_for_streamlit(df, simplify=simplify)
    st.dataframe(safe_df.head(max_rows), width="stretch", hide_index=hide_index)


def safe_call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    sig = inspect.signature(fn)
    accepted: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(*args, **accepted)


def safe_generate_sim_data(num_aircraft: int, num_steps: int) -> pd.DataFrame:
    gen_fn: Any = generate_aircraft_data

    attempts = [
        lambda: safe_call_with_supported_kwargs(
            gen_fn,
            num_aircraft=num_aircraft,
            time_steps=num_steps,
            aircraft_count=num_aircraft,
            n_aircraft=num_aircraft,
            num_steps=num_steps,
            n_steps=num_steps,
        ),
        lambda: gen_fn(num_aircraft=num_aircraft, time_steps=num_steps),
        lambda: gen_fn(num_aircraft, num_steps),
        lambda: gen_fn(),
    ]

    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, pd.DataFrame):
                return result
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not generate simulation data. Last error: {last_error}")


def safe_apply_attack(
    df: pd.DataFrame,
    attack_name: str,
    start_step: int,
    end_step: int,
    lat_shift: float,
    lon_shift: float,
) -> pd.DataFrame:
    if df.empty or attack_name == "None":
        return df.copy()

    d = df.copy()

    aircraft_ids = sorted(d["icao"].astype(str).unique().tolist()) if "icao" in d.columns else []
    if not aircraft_ids:
        return d

    target_icao = aircraft_ids[0]
    ghost_icao = "GHOST0001"

    try:
        if attack_name == "Teleportation":
            return inject_teleportation_attack(
                d,
                target_icao=target_icao,
                start_step=int(start_step),
                end_step=int(end_step),
            )

        if attack_name == "GPS Spoofing":
            return inject_gps_spoofing_attack(
                d,
                target_icao=target_icao,
                start_step=int(start_step),
                end_step=int(end_step),
                lat_shift=float(lat_shift),
                lon_shift=float(lon_shift),
            )

        if attack_name == "Ghost Aircraft":
            return inject_ghost_aircraft_attack(
                d,
                ghost_icao=ghost_icao,
                start_step=int(start_step),
                end_step=int(end_step),
            )
    except Exception:
        return d

    return d


def normalize_detection_output(base_df: pd.DataFrame, result: Any) -> pd.DataFrame:
    if result is None:
        return base_df.copy()

    candidate: Optional[pd.DataFrame] = None

    if isinstance(result, pd.DataFrame):
        candidate = result.copy()

    elif isinstance(result, tuple):
        for item in result:
            if isinstance(item, pd.DataFrame):
                candidate = item.copy()
                break

    elif isinstance(result, dict):
        for key in ["df", "data", "processed_df", "result_df", "anomalies"]:
            value = result.get(key)
            if isinstance(value, pd.DataFrame):
                candidate = value.copy()
                break

    if candidate is None or candidate.empty:
        return base_df.copy()

    has_geo = {"icao", "timestamp", "latitude", "longitude"}.issubset(candidate.columns)
    if has_geo:
        return candidate

    out = base_df.copy()
    if {"icao", "timestamp"}.issubset(candidate.columns):
        key_cols = ["icao", "timestamp"]
        extra_cols = [c for c in candidate.columns if c not in out.columns]
        if extra_cols:
            merged = candidate[key_cols + extra_cols].drop_duplicates(subset=key_cols, keep="last")
            out = out.merge(merged, on=key_cols, how="left")
        return out

    return out


def ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        out["timestamp"] = np.arange(len(out))
    return out


def ensure_aircraft_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "icao" not in out.columns:
        if "icao24" in out.columns:
            out["icao"] = out["icao24"].astype(str)
        elif "callsign" in out.columns:
            out["icao"] = out["callsign"].fillna("UNKNOWN").astype(str)
        else:
            out["icao"] = "UNKNOWN"
    out["icao"] = out["icao"].astype(str)
    return out


def ensure_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "latitude" not in out.columns and "lat" in out.columns:
        out["latitude"] = out["lat"]

    if "longitude" not in out.columns:
        if "lon" in out.columns:
            out["longitude"] = out["lon"]
        elif "lng" in out.columns:
            out["longitude"] = out["lng"]

    if "latitude" in out.columns:
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    else:
        out["latitude"] = pd.Series(np.nan, index=out.index, dtype=float)

    if "longitude" in out.columns:
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    else:
        out["longitude"] = pd.Series(np.nan, index=out.index, dtype=float)

    return out


def add_frame_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = ensure_aircraft_id_column(ensure_timestamp_column(out))
    out = out.sort_values(["icao", "timestamp"]).reset_index(drop=True)
    out["frame_idx"] = out.groupby("icao").cumcount()
    return out


def add_user_friendly_columns(
    df: pd.DataFrame,
    attack_name: str,
    attack_start: int,
    attack_end: int,
) -> pd.DataFrame:
    out = add_frame_index(df)

    if "speed_mps" not in out.columns:
        if "velocity" in out.columns:
            out["speed_mps"] = pd.to_numeric(out["velocity"], errors="coerce")
        else:
            out["speed_mps"] = np.nan

    if "altitude" not in out.columns:
        if "baro_altitude" in out.columns:
            out["altitude"] = pd.to_numeric(out["baro_altitude"], errors="coerce")
        else:
            out["altitude"] = np.nan

    if "implied_speed_mps" not in out.columns:
        if "distance_m" in out.columns and "delta_t" in out.columns:
            distance_s = as_numeric_series(out, "distance_m", 0.0)
            dt_s = as_numeric_series(out, "delta_t", 1.0).replace(0, np.nan)
            out["implied_speed_mps"] = (distance_s / dt_s).replace([np.inf, -np.inf], np.nan)
        else:
            out["implied_speed_mps"] = out["speed_mps"]

    if "speed_mismatch_mps" not in out.columns:
        reported = as_numeric_series(out, "speed_mps", 0.0)
        implied = as_numeric_series(out, "implied_speed_mps", 0.0)
        out["speed_mismatch_mps"] = (reported - implied).abs()

    if "data_quality_score" not in out.columns:
        if "bad_point" in out.columns:
            bad = pd.to_numeric(out["bad_point"], errors="coerce").fillna(0).astype(int)
            out["data_quality_score"] = np.where(bad == 1, 0.35, 1.0)
        else:
            out["data_quality_score"] = 1.0

    if "attack_window" not in out.columns:
        out["attack_window"] = out["frame_idx"].between(attack_start, attack_end)

    out["scenario_label"] = np.where(out["attack_window"], attack_name, "Normal Flight")
    out["human_alert_label"] = "Normal"

    speed_mismatch = as_numeric_series(out, "speed_mismatch_mps", 0.0)
    distance_m = as_numeric_series(out, "distance_m", 0.0)
    accel = as_numeric_series(out, "accel_mps2", 0.0)
    turn_rate = as_numeric_series(out, "turn_rate_dps", 0.0)

    teleport_mask = (distance_m > 5000) | (speed_mismatch > 200)
    spoof_mask = speed_mismatch > 20
    ghost_mask = (turn_rate.abs() > 12) | (accel.abs() > 15)

    out.loc[teleport_mask, "human_alert_label"] = "Impossible Jump"
    out.loc[spoof_mask, "human_alert_label"] = "Position Mismatch"
    out.loc[ghost_mask, "human_alert_label"] = "Ghost / Fake Track"

    if attack_name != "None":
        label_map = {
            "Teleportation": "Impossible Jump",
            "GPS Spoofing": "Position Mismatch",
            "Ghost Aircraft": "Ghost / Fake Track",
        }
        out.loc[out["attack_window"], "human_alert_label"] = label_map.get(attack_name, attack_name)

    return out


def latest_state_per_aircraft(df: pd.DataFrame) -> pd.DataFrame:
    temp = add_frame_index(df)
    latest = (
        temp.sort_values(["icao", "timestamp"])
        .groupby("icao", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest


def build_paths(df: pd.DataFrame) -> pd.DataFrame:
    temp = add_frame_index(ensure_lat_lon(df))
    required = {"icao", "latitude", "longitude"}
    if not required.issubset(temp.columns):
        return pd.DataFrame(columns=["icao", "path", "path_color", "path_width"])

    valid = temp.dropna(subset=["latitude", "longitude"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["icao", "path", "path_color", "path_width"])

    valid = valid.sort_values(["icao", "timestamp"]).copy()

    rows: list[dict[str, Any]] = []
    for icao, grp in valid.groupby("icao"):
        g = grp.tail(18).copy()
        if len(g) < 2:
            continue
        latest_alert = str(g.iloc[-1].get("human_alert_label", "Normal"))
        rows.append(
            {
                "icao": str(icao),
                "path": g[["longitude", "latitude"]].values.tolist(),
                "path_color": build_map_colors(latest_alert, 190 if latest_alert != "Normal" else 85),
                "path_width": 4 if latest_alert != "Normal" else 2,
            }
        )

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "aircraft_tracked": 0,
            "active_alerts": 0,
            "avg_speed_mps": None,
            "avg_quality": None,
            "map_ready_points": 0,
        }

    latest = latest_state_per_aircraft(df)
    latest_alert = latest["human_alert_label"].fillna("Normal").astype(str)
    active_alerts = int((latest_alert != "Normal").sum())

    avg_speed = None
    if "speed_mps" in latest.columns:
        speed = pd.to_numeric(latest["speed_mps"], errors="coerce")
        if not speed.dropna().empty:
            avg_speed = float(speed.mean())

    avg_quality = None
    if "data_quality_score" in latest.columns:
        qual = pd.to_numeric(latest["data_quality_score"], errors="coerce")
        if not qual.dropna().empty:
            avg_quality = float(qual.mean())

    map_ready_points = 0
    if {"latitude", "longitude"}.issubset(latest.columns):
        map_ready_points = int(latest.dropna(subset=["latitude", "longitude"]).shape[0])

    return {
        "aircraft_tracked": int(latest["icao"].nunique()),
        "active_alerts": active_alerts,
        "avg_speed_mps": avg_speed,
        "avg_quality": avg_quality,
        "map_ready_points": map_ready_points,
    }


def compute_df_signature(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"

    cols_to_use = [
        c
        for c in ["icao", "timestamp", "latitude", "longitude", "velocity", "baro_altitude", "heading"]
        if c in df.columns
    ]
    sample = df[cols_to_use].tail(250).copy() if cols_to_use else df.tail(250).copy()

    try:
        payload = sample.to_json(date_format="iso", orient="split")
    except Exception:
        payload = repr((list(df.columns), df.shape))

    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def compute_ml_settings_signature(
    seq_len: int,
    max_dt_sec: int,
    min_quality: float,
    threshold_percentile: float,
    persistence_k: int,
    persistence_m: int,
) -> str:
    payload = f"{seq_len}|{max_dt_sec}|{min_quality:.4f}|{threshold_percentile:.4f}|{persistence_k}|{persistence_m}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def compute_run_context_signature(
    source_mode: str,
    attack_name: str,
    attack_start: int,
    attack_end: int,
    aircraft_count: int,
    time_steps: int,
    lat_shift: float,
    lon_shift: float,
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
) -> str:
    payload = "|".join(
        [
            source_mode,
            attack_name,
            str(int(attack_start)),
            str(int(attack_end)),
            str(int(aircraft_count)),
            str(int(time_steps)),
            f"{float(lat_shift):.4f}",
            f"{float(lon_shift):.4f}",
            f"{float(lat_min):.4f}",
            f"{float(lon_min):.4f}",
            f"{float(lat_max):.4f}",
            f"{float(lon_max):.4f}",
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# -----------------------------
# Refresh helpers
# -----------------------------
def schedule_browser_refresh(interval_seconds: int) -> None:
    interval_ms = max(5, int(interval_seconds)) * 1000
    components.html(
        f"""
        <script>
        const delay = {interval_ms};
        if (!window.__adsbRefreshScheduled) {{
            window.__adsbRefreshScheduled = true;
            setTimeout(function() {{
                window.parent.location.reload();
            }}, delay);
        }}
        </script>
        """,
        height=0,
    )


def enable_live_rerun_timer(interval_seconds: int, enabled: bool) -> str:
    if not enabled:
        return "paused"

    if hasattr(st, "fragment"):
        run_every_value = f"{max(5, int(interval_seconds))}s"

        @st.fragment(run_every=run_every_value)
        def _live_tick() -> None:
            st.session_state["_live_tick_counter"] = st.session_state.get("_live_tick_counter", 0) + 1
            st.rerun()

        _live_tick()
        return "fragment"

    schedule_browser_refresh(interval_seconds)
    return "browser"


# -----------------------------
# ML cache helpers
# -----------------------------
def get_cached_ml_artifacts() -> tuple[Optional[Any], Optional[str]]:
    if not ML_AVAILABLE:
        return None, f"ML unavailable: {ML_IMPORT_ERROR}"

    if "ml_artifacts_cache" in st.session_state:
        cached = st.session_state["ml_artifacts_cache"]
        cached_error = st.session_state.get("ml_artifacts_error")
        return cached, cached_error

    try:
        result = load_artifacts()
        if result is None:
            st.session_state["ml_artifacts_cache"] = None
            st.session_state["ml_artifacts_error"] = "ML artifacts not found. Train the model first."
            return None, "ML artifacts not found. Train the model first."

        st.session_state["ml_artifacts_cache"] = result
        st.session_state["ml_artifacts_error"] = None
        return result, None
    except Exception as e:
        err = f"Could not load ML artifacts: {e}"
        st.session_state["ml_artifacts_cache"] = None
        st.session_state["ml_artifacts_error"] = err
        return None, err


def clear_cached_ml_artifacts() -> None:
    st.session_state.pop("ml_artifacts_cache", None)
    st.session_state.pop("ml_artifacts_error", None)
    st.session_state.pop("ml_scores_cache", None)
    st.session_state.pop("ml_anoms_cache", None)
    st.session_state.pop("ml_scores_sig", None)
    st.session_state.pop("ml_meta_cache", None)


# -----------------------------
# History / incident helpers
# -----------------------------
def empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "icao",
            "timestamp",
            "latitude",
            "longitude",
            "velocity",
            "heading",
            "baro_altitude",
            "vertical_rate",
            "speed_mps",
            "altitude",
            "human_alert_label",
            "scenario_label",
            "attack_window",
            "data_quality_score",
            "distance_m",
            "delta_t",
            "implied_speed_mps",
            "speed_mismatch_mps",
            "accel_mps2",
            "turn_rate_dps",
            "frame_idx",
        ]
    )


def init_history_state() -> None:
    if "flight_history_df" not in st.session_state:
        st.session_state["flight_history_df"] = empty_history_df()


def update_flight_history(source_mode: str, processed_df: pd.DataFrame, max_rows_per_aircraft: int = 400) -> pd.DataFrame:
    init_history_state()

    if processed_df is None or processed_df.empty:
        return cast(pd.DataFrame, st.session_state["flight_history_df"])

    current = ensure_lat_lon(ensure_aircraft_id_column(ensure_timestamp_column(processed_df))).copy()

    keep_cols = list(empty_history_df().columns)
    for col in keep_cols:
        if col not in current.columns:
            current[col] = np.nan
    current = current[keep_cols].copy()

    if source_mode == "Simulation":
        history_df = (
            current.sort_values(["icao", "timestamp"])
            .drop_duplicates(subset=["icao", "timestamp"], keep="last")
            .reset_index(drop=True)
        )
        st.session_state["flight_history_df"] = history_df
        return history_df

    existing = cast(pd.DataFrame, st.session_state["flight_history_df"]).copy()

    merged = pd.concat([existing, current], ignore_index=True)
    merged = merged.drop_duplicates(subset=["icao", "timestamp"], keep="last")
    merged = merged.sort_values(["icao", "timestamp"]).reset_index(drop=True)
    merged = (
        merged.groupby("icao", group_keys=False)
        .tail(max_rows_per_aircraft)
        .reset_index(drop=True)
    )

    st.session_state["flight_history_df"] = merged
    return merged


def get_aircraft_history(history_df: pd.DataFrame, icao: str) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    d = ensure_lat_lon(ensure_aircraft_id_column(ensure_timestamp_column(history_df))).copy()
    d = d[d["icao"].astype(str) == str(icao)].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values("timestamp").reset_index(drop=True)
    d = add_frame_index(d)
    return d


def anomaly_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(False, index=df.index)

    if "human_alert_label" in df.columns:
        mask = mask | df["human_alert_label"].fillna("Normal").astype(str).ne("Normal")

    if "attack_window" in df.columns:
        attack_series = pd.to_numeric(df["attack_window"], errors="coerce").fillna(0).astype(bool)
        mask = mask | attack_series

    return mask


def incident_aircraft_options(history_df: pd.DataFrame) -> list[str]:
    if history_df is None or history_df.empty:
        return []

    d = ensure_aircraft_id_column(history_df).copy()
    d = d.sort_values(["icao", "timestamp"]).reset_index(drop=True)

    latest = latest_state_per_aircraft(d)
    latest_alert_mask = anomaly_mask(latest)
    latest_alert_aircraft = latest.loc[latest_alert_mask, "icao"].astype(str).tolist()

    any_alert_aircraft = d.loc[anomaly_mask(d), "icao"].astype(str).drop_duplicates().tolist()

    merged = []
    for icao in latest_alert_aircraft + any_alert_aircraft:
        if icao not in merged:
            merged.append(icao)
    return merged


def alert_explanation(alert_label: str) -> str:
    mapping = {
        "Impossible Jump": (
            "This track shows an unrealistic spatial jump between consecutive updates. "
            "The aircraft appears to move farther than physically possible for the elapsed time."
        ),
        "Position Mismatch": (
            "This track’s reported position does not align well with its expected motion. "
            "That pattern is consistent with spoofing or position displacement behavior."
        ),
        "Ghost / Fake Track": (
            "This track appears suspicious because it shows unrealistic motion or an abrupt emergence "
            "without believable continuity."
        ),
        "Normal": "No anomaly is currently flagged for this aircraft.",
    }
    return mapping.get(alert_label, "This aircraft is under review.")


def alert_severity(alert_label: str) -> str:
    if alert_label in {"Impossible Jump", "Ghost / Fake Track"}:
        return "High"
    if alert_label == "Position Mismatch":
        return "Medium"
    return "Low"


def detect_source_label(use_ml: bool, ml_anoms_df: Optional[pd.DataFrame], selected_icao: str, alert_label: str) -> str:
    if not use_ml or not isinstance(ml_anoms_df, pd.DataFrame) or ml_anoms_df.empty:
        return "Rule-Based"

    if "icao" in ml_anoms_df.columns:
        has_ml_match = ml_anoms_df["icao"].astype(str).eq(str(selected_icao)).any()
        if has_ml_match and alert_label != "Normal":
            return "Rule-Based + ML"
        if has_ml_match:
            return "ML"

    return "Rule-Based"


def build_segment_paths(aircraft_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if aircraft_df.empty:
        empty = pd.DataFrame(columns=["icao", "path"])
        return empty, empty, empty, pd.DataFrame()

    d = ensure_lat_lon(aircraft_df).dropna(subset=["latitude", "longitude"]).copy()
    if d.empty:
        empty = pd.DataFrame(columns=["icao", "path"])
        return empty, empty, empty, pd.DataFrame()

    d = d.sort_values("timestamp").reset_index(drop=True)
    mask = anomaly_mask(d)

    all_coords = d[["longitude", "latitude"]].values.tolist()
    full_df = pd.DataFrame([{"icao": str(d.iloc[0]["icao"]), "path": all_coords}])

    anomaly_points = d.loc[mask].copy()

    if anomaly_points.empty:
        empty = pd.DataFrame(columns=["icao", "path"])
        return full_df, full_df, empty, pd.DataFrame()

    first_anom_idx = int(anomaly_points.index[0])

    before_df = d.iloc[: first_anom_idx + 1].copy()
    after_df = d.iloc[first_anom_idx:].copy()

    before_path = pd.DataFrame(columns=["icao", "path"])
    after_path = pd.DataFrame(columns=["icao", "path"])

    if len(before_df) >= 2:
        before_path = pd.DataFrame(
            [{"icao": str(d.iloc[0]["icao"]), "path": before_df[["longitude", "latitude"]].values.tolist()}]
        )

    if len(after_df) >= 2:
        after_path = pd.DataFrame(
            [{"icao": str(d.iloc[0]["icao"]), "path": after_df[["longitude", "latitude"]].values.tolist()}]
        )

    return full_df, before_path, after_path, anomaly_points


def render_incident_map(
    aircraft_df: pd.DataFrame,
    attack_name: str,
    map_height: int,
    point_radius: int,
) -> None:
    if aircraft_df.empty:
        render_info_banner("No history is available for this aircraft.", "info")
        return

    d = ensure_lat_lon(aircraft_df).dropna(subset=["latitude", "longitude"]).copy()
    if d.empty:
        render_info_banner("No valid coordinates are available for this aircraft.", "info")
        return

    d = d.sort_values("timestamp").reset_index(drop=True)
    full_path_df, before_path_df, after_path_df, anomaly_points = build_segment_paths(d)
    latest_point = d.tail(1).copy()

    center_lat = float(pd.to_numeric(d["latitude"], errors="coerce").mean())
    center_lon = float(pd.to_numeric(d["longitude"], errors="coerce").mean())

    layers: list[Any] = []

    if not full_path_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=full_path_df,
                get_path="path",
                get_color=[71, 85, 105, 95],
                width_scale=2,
                width_min_pixels=2,
                pickable=False,
            )
        )

    if not before_path_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=before_path_df,
                get_path="path",
                get_color=[56, 189, 248, 225],
                width_scale=4,
                width_min_pixels=4,
                pickable=False,
            )
        )

    if not after_path_df.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=after_path_df,
                get_path="path",
                get_color=[251, 191, 36, 225],
                width_scale=4,
                width_min_pixels=4,
                pickable=False,
            )
        )

    base_points = d.copy()
    base_points["tooltip_text"] = (
        "Aircraft: " + base_points["icao"].astype(str)
        + "\nTimestamp: " + base_points["timestamp"].astype(str)
        + "\nAlert: " + base_points["human_alert_label"].fillna("Normal").astype(str)
    )
    base_points["fill_color"] = [[148, 163, 184, 85] for _ in range(len(base_points))]

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=base_points,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_radius=max(250, int(point_radius * 0.42)),
            pickable=True,
            stroked=False,
            filled=True,
        )
    )

    if not anomaly_points.empty:
        anomaly_points = anomaly_points.copy()
        anomaly_points["tooltip_text"] = (
            "Aircraft: " + anomaly_points["icao"].astype(str)
            + "\nTimestamp: " + anomaly_points["timestamp"].astype(str)
            + "\nAnomaly: " + anomaly_points["human_alert_label"].fillna("Normal").astype(str)
            + "\nScenario: " + anomaly_points["scenario_label"].fillna("Normal Flight").astype(str)
        )
        anomaly_points["fill_color"] = [[255, 99, 99, 245] for _ in range(len(anomaly_points))]

        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=anomaly_points,
                get_position="[longitude, latitude]",
                get_fill_color="fill_color",
                get_radius=max(650, int(point_radius * 0.90)),
                pickable=True,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                get_line_color=[255, 255, 255, 180],
            )
        )

    latest_point["tooltip_text"] = "Aircraft: " + latest_point["icao"].astype(str) + "\nLatest position"
    latest_point["fill_color"] = [[52, 211, 153, 245]]

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=latest_point,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_radius=max(750, int(point_radius * 1.02)),
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            get_line_color=[255, 255, 255, 180],
        )
    )

    tooltip_value: Any = {
        "text": "{tooltip_text}",
        "style": {
            "backgroundColor": "rgba(5, 17, 33, 0.97)",
            "color": "white",
            "fontSize": "13px",
        },
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=7.0,
            pitch=38,
            bearing=12,
        ),
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip=cast(Any, tooltip_value),
    )

    st.pydeck_chart(deck, use_container_width=True, height=map_height)

    st.markdown(
        """
        <div class="legend-inline">
            <div class="legend-pill"><span class="legend-dot" style="background:#38bdf8;"></span>Path before anomaly</div>
            <div class="legend-pill"><span class="legend-dot" style="background:#ff6363;"></span>Flagged anomaly point</div>
            <div class="legend-pill"><span class="legend-dot" style="background:#fbbf24;"></span>Path after anomaly</div>
            <div class="legend-pill"><span class="legend-dot" style="background:#34d399;"></span>Latest aircraft position</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_incident_center(
    history_df: pd.DataFrame,
    selected_icao: Optional[str],
    attack_name: str,
    source_mode: str,
    point_radius: int,
    map_height: int,
    use_ml: bool,
    ml_anoms_df: Optional[pd.DataFrame],
) -> None:
    render_section_title("Incident Visualizer", "Full aircraft path with exact anomaly location")

    if history_df is None or history_df.empty:
        render_info_banner("Run monitoring first so aircraft history is available.", "info")
        return

    options = incident_aircraft_options(history_df)
    if not options:
        render_info_banner(
            "No anomalous aircraft are currently available. Run a scenario with an injected attack or wait for a live alert.",
            "success",
        )
        return

    if selected_icao is None or selected_icao not in options:
        selected_icao = options[0]

    aircraft_df = get_aircraft_history(history_df, selected_icao)
    if aircraft_df.empty:
        render_info_banner("No history found for the selected incident aircraft.", "info")
        return

    aircraft_df = aircraft_df.sort_values("timestamp").reset_index(drop=True)
    mask = anomaly_mask(aircraft_df)
    anom_rows = aircraft_df.loc[mask].copy()
    last_row = aircraft_df.iloc[-1]

    active_alert_label = "Normal"
    detected_time = "—"
    anomaly_count = 0

    if not anom_rows.empty:
        first_anom = anom_rows.iloc[0]
        active_alert_label = str(first_anom.get("human_alert_label", "Normal"))
        detected_time = str(first_anom.get("timestamp", "—"))
        anomaly_count = int(len(anom_rows))
    else:
        active_alert_label = str(last_row.get("human_alert_label", "Normal"))

    severity = alert_severity(active_alert_label)
    source_label = detect_source_label(use_ml, ml_anoms_df, selected_icao, active_alert_label)
    explanation = alert_explanation(active_alert_label)

    callsign_text = "N/A"
    if "callsign" in aircraft_df.columns:
        callsign_series = aircraft_df["callsign"].dropna().astype(str)
        if not callsign_series.empty:
            callsign_text = str(callsign_series.iloc[-1])

    left, right = st.columns([0.95, 1.75])

    with left:
        speed_val = pd.to_numeric(pd.Series([last_row.get("speed_mps")]), errors="coerce").iloc[0]
        alt_val = pd.to_numeric(pd.Series([last_row.get("altitude")]), errors="coerce").iloc[0]
        heading_val = pd.to_numeric(pd.Series([last_row.get("heading")]), errors="coerce").iloc[0]
        q_val = pd.to_numeric(pd.Series([last_row.get("data_quality_score")]), errors="coerce").iloc[0]

        render_detail_panel(
            "Active Incident",
            [
                ("Aircraft", selected_icao),
                ("Callsign", callsign_text),
                ("Anomaly Type", active_alert_label),
                ("Detected At", detected_time),
                ("Severity", severity),
                ("Detection Source", source_label),
                ("Data Source", source_mode),
                ("Scenario", attack_name if source_mode == "Simulation" else "Live Traffic"),
                ("Flagged Points", anomaly_count),
                ("Latest Speed", format_display_value(speed_val, suffix=" m/s")),
                ("Latest Altitude", format_display_value(alt_val, suffix=" m")),
                ("Latest Heading", format_display_value(heading_val, suffix="°")),
                ("Quality Score", format_display_value(q_val)),
            ],
            eyebrow="Incident",
            subtitle="Current incident identity, severity, and most recent aircraft state.",
            footer=explanation,
        )

    with right:
        render_chart_shell(
            "Incident Map",
            "The full aircraft path is split around the anomaly so the suspicious transition is easy to isolate in demos and reviews.",
        )
        render_incident_map(
            aircraft_df=aircraft_df,
            attack_name=attack_name,
            map_height=map_height,
            point_radius=point_radius,
        )

    st.divider()
    render_section_title("Incident Timeline", "Ordered history for the selected aircraft")

    timeline_cols = [
        c
        for c in [
            "timestamp",
            "icao",
            "latitude",
            "longitude",
            "altitude",
            "speed_mps",
            "heading",
            "human_alert_label",
            "scenario_label",
            "data_quality_score",
            "distance_m",
            "implied_speed_mps",
            "speed_mismatch_mps",
            "turn_rate_dps",
            "accel_mps2",
        ]
        if c in aircraft_df.columns
    ]

    show_dataframe(aircraft_df[timeline_cols], max_rows=800, simplify=True, hide_index=True)


# -----------------------------
# Dashboard view helpers
# -----------------------------
def current_scenario_name(source_mode: str, attack_name: str) -> str:
    if source_mode == "Live OpenSky Snapshot":
        return "Live Traffic"
    return "Normal Flight" if attack_name == "None" else attack_name


def current_status_label(source_mode: str, attack_name: str, active_alerts: int) -> str:
    if source_mode == "Live OpenSky Snapshot":
        return "Online" if active_alerts == 0 else "Warning"
    if attack_name == "None":
        return "Normal"
    return "Anomaly"


def render_detail_panel(
    title: str,
    rows: list[tuple[str, object]],
    *,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    footer: str | None = None,
) -> None:
    row_html = "".join(
        f"""
        <div class="detail-row">
            <span class="label">{escape_text(label)}</span>
            <span class="value">{escape_text(value)}</span>
        </div>
        """
        for label, value in rows
    )

    eyebrow_html = f'<div class="panel-eyebrow">{escape_text(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f'<div class="panel-subtitle">{escape_text(subtitle)}</div>' if subtitle else ""
    footer_html = f'<div class="insight-note">{escape_text(footer)}</div>' if footer else ""

    st.markdown(
        f"""
        <div class="dashboard-panel">
            {eyebrow_html}
            <div class="panel-title">{escape_text(title)}</div>
            {subtitle_html}
            <div class="detail-grid">{row_html}</div>
            {footer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_shell(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="chart-shell">
            <div class="chart-title">{escape_text(title)}</div>
            <div class="chart-subtitle">{escape_text(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_card(kicker: str, title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-kicker">{escape_text(kicker)}</div>
            <div class="story-title">{escape_text(title)}</div>
            <div class="story-body">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_strip(items: list[tuple[str, object]]) -> None:
    if not items:
        return

    cols_per_row = min(3, len(items))
    for start in range(0, len(items), cols_per_row):
        row_items = items[start : start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (label, value) in zip(cols, row_items):
            with col:
                st.markdown(
                    (
                        f'<div class="stat-tile">'
                        f'<div class="stat-tile-label">{escape_text(label)}</div>'
                        f'<div class="stat-tile-value">{escape_text(value)}</div>'
                        f"</div>"
                    ),
                    unsafe_allow_html=True,
                )


def render_capability_grid(cards: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, (icon, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="capability-card">
                    <div class="capability-icon">{escape_text(icon)}</div>
                    <div class="capability-title">{escape_text(title)}</div>
                    <div class="capability-body">{escape_text(body)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_page_note(text: str) -> None:
    st.markdown(f'<div class="page-note">{text}</div>', unsafe_allow_html=True)


def build_scenario_rollup_table(scenario_df: pd.DataFrame) -> pd.DataFrame:
    if scenario_df.empty or "scenario" not in scenario_df.columns:
        return pd.DataFrame()

    grouped = scenario_df.groupby("scenario", dropna=False)
    rows: list[dict[str, Any]] = []
    for scenario_name, g in grouped:
        positive_series = (
            pd.to_numeric(g["positive_sequences"], errors="coerce").fillna(0)
            if "positive_sequences" in g.columns
            else pd.Series([0] * len(g))
        )
        detected_series = (
            pd.to_numeric(g["detected_positive_sequences"], errors="coerce").fillna(0)
            if "detected_positive_sequences" in g.columns
            else pd.Series([0] * len(g))
        )
        any_detection_series = (
            pd.to_numeric(g["any_detection"], errors="coerce").fillna(0)
            if "any_detection" in g.columns
            else pd.Series([0] * len(g))
        )

        positive_sequences = int(positive_series.sum())
        detected_sequences = int(detected_series.sum())
        run_count = int(len(g))
        detection_rate = detected_sequences / positive_sequences if positive_sequences > 0 else float(
            any_detection_series.mean()
        )
        rows.append(
            {
                "Scenario": scenario_name,
                "Runs": run_count,
                "Positive Sequences": positive_sequences,
                "Detected Sequences": detected_sequences,
                "Detection Rate": round(float(detection_rate), 3),
            }
        )

    return pd.DataFrame(rows).sort_values("Scenario").reset_index(drop=True)


def render_score_distribution_chart(seq_df: pd.DataFrame) -> None:
    required = {"anomaly_score", "seq_label"}
    if seq_df.empty or not required.issubset(seq_df.columns):
        render_info_banner("Score-distribution data is not available yet.", "info")
        return

    dist_df = seq_df[["anomaly_score", "seq_label"]].copy()
    dist_df["anomaly_score"] = pd.to_numeric(dist_df["anomaly_score"], errors="coerce")
    dist_df["seq_label"] = pd.to_numeric(dist_df["seq_label"], errors="coerce").fillna(0).astype(int)
    dist_df = dist_df.dropna()
    if dist_df.empty:
        render_info_banner("Score-distribution data is empty after cleaning.", "info")
        return

    dist_df["label_name"] = np.where(dist_df["seq_label"] == 1, "Anomalous", "Normal")

    score_min = float(dist_df["anomaly_score"].min())
    score_max = float(dist_df["anomaly_score"].max())
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_min == score_max:
        render_info_banner("Score-distribution data does not span a usable range.", "info")
        return

    bins = np.linspace(score_min, score_max, 32).tolist()
    if len(np.unique(bins)) <= 1:
        render_info_banner("Unable to create score bins for distribution rendering.", "info")
        return

    dist_df["score_bin"] = pd.cut(dist_df["anomaly_score"], bins=bins, include_lowest=True)
    hist_df = (
        dist_df.groupby(["score_bin", "label_name"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )

    label_totals = hist_df.groupby("label_name")["count"].transform("sum").replace(0, np.nan)
    hist_df["share_pct"] = (hist_df["count"] / label_totals) * 100.0
    hist_df = hist_df.dropna(subset=["share_pct"])
    hist_df["score_mid"] = hist_df["score_bin"].apply(lambda interval: float(interval.mid) if interval is not None else np.nan)
    hist_df = hist_df.dropna(subset=["score_mid"])

    if hist_df.empty:
        render_info_banner("No usable score-density data is available for rendering.", "info")
        return

    st.caption("Each line is normalized within its own class so rare anomalous sequences remain visible.")

    st.vega_lite_chart(
        hist_df,
        {
            "mark": {"type": "line", "strokeWidth": 3, "point": False},
            "encoding": {
                "x": {
                    "field": "score_mid",
                    "type": "quantitative",
                    "title": "Anomaly score",
                    "axis": {"format": ".2f", "labelColor": "#9db8cc", "titleColor": "#9db8cc"},
                },
                "y": {
                    "field": "share_pct",
                    "type": "quantitative",
                    "title": "Share within class (%)",
                    "axis": {"format": ".1f", "labelColor": "#9db8cc", "titleColor": "#9db8cc"},
                },
                "color": {
                    "field": "label_name",
                    "type": "nominal",
                    "scale": {"domain": ["Normal", "Anomalous"], "range": ["#38bdf8", "#fbbf24"]},
                    "legend": {"title": None, "labelColor": "#eaf6ff", "orient": "bottom"},
                },
                "tooltip": [
                    {"field": "label_name", "type": "nominal", "title": "Class"},
                    {"field": "score_mid", "type": "quantitative", "title": "Score", "format": ".3f"},
                    {"field": "share_pct", "type": "quantitative", "title": "Share %", "format": ".2f"},
                    {"field": "count", "type": "quantitative", "title": "Sequences"},
                ],
            },
            "width": "container",
            "height": 280,
            "config": {
                "background": "#111318",
                "view": {"stroke": None},
                "axis": {"gridColor": "rgba(157,184,204,0.14)", "domain": False, "tickColor": "rgba(157,184,204,0.14)"},
            },
        },
        use_container_width=True,
    )

    stats_rows: list[dict[str, Any]] = []
    for label_name, g in dist_df.groupby("label_name"):
        stats_rows.append(
            {
                "Class": label_name,
                "Count": int(len(g)),
                "Median": round(float(g["anomaly_score"].median()), 3),
                "P95": round(float(g["anomaly_score"].quantile(0.95)), 3),
                "P99": round(float(g["anomaly_score"].quantile(0.99)), 3),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        st.dataframe(stats_df, width="stretch", hide_index=True)


def render_system_status_block(
    source_mode: str,
    attack_name: str,
    use_ml: bool,
    ml_artifacts: Any,
    active_alerts: int,
) -> None:
    render_section_title("System Status", "Current surveillance condition")
    st.markdown(
        render_status_pill(current_status_label(source_mode, attack_name, active_alerts)),
        unsafe_allow_html=True,
    )

    render_detail_panel(
        "Surveillance Snapshot",
        [
            ("Source", source_mode),
            ("Scenario", current_scenario_name(source_mode, attack_name)),
            ("ML", "Enabled" if use_ml else "Disabled"),
            ("Artifacts", "Loaded" if ml_artifacts is not None else "Unavailable"),
            ("Active Alerts", active_alerts),
        ],
        eyebrow="Overview",
        subtitle="A compact readout of system state before drilling into operations or incident review.",
    )


def render_alert_summary_panel(latest_df: pd.DataFrame) -> None:
    render_section_title("Alert Queue", "Current alert output")

    if latest_df.empty or "human_alert_label" not in latest_df.columns:
        render_info_banner("No processed alert data available.", "info")
        return

    alerts_df = latest_df.copy()
    alerts_df["human_alert_label"] = alerts_df["human_alert_label"].fillna("Normal").astype(str)
    active_alerts_df = alerts_df[alerts_df["human_alert_label"] != "Normal"].copy()

    if active_alerts_df.empty:
        render_info_banner("No active alerts detected.", "success")
        return

    display_cols = [
        c
        for c in [
            "icao",
            "human_alert_label",
            "scenario_label",
            "speed_mps",
            "altitude",
            "data_quality_score",
        ]
        if c in active_alerts_df.columns
    ]

    critical_count = int(
        active_alerts_df["human_alert_label"].isin(["Impossible Jump", "Ghost / Fake Track"]).sum()
    )
    aircraft_count = int(active_alerts_df["icao"].astype(str).nunique()) if "icao" in active_alerts_df.columns else 0
    top_alert = active_alerts_df["human_alert_label"].value_counts().idxmax()

    render_detail_panel(
        "Queue Summary",
        [
            ("Active Rows", len(active_alerts_df)),
            ("Affected Aircraft", aircraft_count),
            ("Critical Alerts", critical_count),
            ("Dominant Alert", top_alert),
        ],
        eyebrow="Alerts",
        subtitle="Use this queue for the fastest read on what needs attention right now.",
    )

    st.dataframe(
        prepare_df_for_streamlit(active_alerts_df[display_cols], simplify=True),
        width="stretch",
        hide_index=True,
    )


def render_product_overview_section(
    source_mode: str,
    attack_name: str,
    use_ml: bool,
    critical_alerts: int,
    summary: dict[str, Any],
    ml_eval_metrics: dict[str, Any],
) -> None:
    current_context = "Live airspace monitoring" if source_mode == "Live OpenSky Snapshot" else attack_name
    seq_metrics = cast(dict[str, Any], ml_eval_metrics.get("sequence_metrics", {})) if ml_eval_metrics else {}
    benchmark_text = (
        f"Held-out benchmark F1 {seq_metrics.get('f1', float('nan')):.3f} with false-positive rate "
        f"{seq_metrics.get('false_positive_rate', float('nan')):.3f}."
        if seq_metrics
        else "Evaluation artifacts can be generated to show benchmark precision, recall, and false-positive rate."
    )

    render_page_note(
        "<strong>What this system does:</strong> It watches aircraft movement data, looks for suspicious patterns such as "
        "impossible jumps, spoofed positions, and ghost aircraft, and explains what deserves attention in plain language."
    )

    left_body = (
        "<p>This dashboard is designed to feel like a real monitoring product: approachable for first-time viewers, but "
        "credible enough for technical audiences who want to inspect the details.</p>"
        "<div class=\"story-list\">"
        f"<div class=\"story-item\"><strong>Current context</strong><span>{escape_text(current_context)}</span></div>"
        f"<div class=\"story-item\"><strong>Tracked aircraft</strong><span>{escape_text(summary.get('aircraft_tracked', 0))} currently visible in the run.</span></div>"
        f"<div class=\"story-item\"><strong>Critical alerts</strong><span>{escape_text(critical_alerts)} high-severity conditions need immediate review.</span></div>"
        "</div>"
    )
    right_body = (
        "<p>People who are new to ADS-B or anomaly detection should still be able to understand the system quickly.</p>"
        "<div class=\"story-list\">"
        "<div class=\"story-item\"><strong>Plain-English guidance</strong><span>Each page explains what the system sees, why it matters, and where to look next.</span></div>"
        f"<div class=\"story-item\"><strong>ML status</strong><span>{'Enabled and ready' if use_ml else 'Currently disabled'}.</span></div>"
        f"<div class=\"story-item\"><strong>Benchmark health</strong><span>{escape_text(benchmark_text)}</span></div>"
        "</div>"
    )

    left_col, right_col = st.columns([1.15, 1.0])
    with left_col:
        render_story_card("Mission", "Aircraft anomaly detection that feels understandable at first glance.", left_body)
    with right_col:
        render_story_card("Why It Matters", "Clearer visibility into suspicious aircraft behavior without overwhelming the viewer.", right_body)

    render_capability_grid(
        [
            (
                "01",
                "Live Airspace Awareness",
                "Shows aircraft positions, movement, and alert state in a way that feels operational instead of academic.",
            ),
            (
                "02",
                "Hybrid Threat Detection",
                "Combines rule-based checks with ML scoring so the system can catch both obvious and subtle anomalies.",
            ),
            (
                "03",
                "Clear Explanations",
                "Translates technical signals into language that makes sense for non-experts, judges, and demo audiences.",
            ),
        ]
    )


# -----------------------------
# Animated attack visuals
# -----------------------------
def build_attack_animation_html(scene: str) -> str:
    scene_key = (scene or "normal").strip().lower()

    title_map = {
        "normal": "Normal Flight",
        "teleportation": "Teleportation Attack",
        "gps": "GPS Spoofing Attack",
        "ghost": "Ghost Aircraft Attack",
    }
    subtitle_map = {
        "normal": "The aircraft follows its intended path normally.",
        "teleportation": "The aircraft suddenly jumps to an impossible new position.",
        "gps": "False position data gradually pulls the aircraft away from its intended route.",
        "ghost": "A false second aircraft appears alongside the legitimate flight path.",
    }

    scene_title = title_map.get(scene_key, "Scenario")
    scene_subtitle = subtitle_map.get(scene_key, "")

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <style>
        :root {{
          --bg1: #020814;
          --bg2: #041121;
          --panel-border: rgba(56, 189, 248, 0.18);
          --grid: rgba(255,255,255,0.045);
          --text: #eaf6ff;
          --muted: #9db8cc;
          --blue: #38bdf8;
          --blue-soft: rgba(56, 189, 248, 0.28);
          --red: #ff6363;
          --yellow: #fbbf24;
          --purple: #b889ff;
          --white-soft: rgba(255,255,255,0.92);
        }}

        * {{
          box-sizing: border-box;
        }}

        html, body {{
          margin: 0;
          padding: 0;
          width: 100%;
          height: 100%;
          overflow: hidden;
          font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: var(--text);
          background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
            radial-gradient(circle at bottom right, rgba(56, 189, 248, 0.08), transparent 24%),
            linear-gradient(180deg, var(--bg2) 0%, var(--bg1) 100%);
        }}

        .wrap {{
          width: 100%;
          height: 100%;
          padding: 16px;
        }}

        .card {{
          width: 100%;
          height: 100%;
          border-radius: 24px;
          background: linear-gradient(180deg, rgba(3, 17, 43, 0.95), rgba(4, 21, 52, 0.95));
          border: 1px solid var(--panel-border);
          box-shadow:
            inset 0 0 0 1px rgba(255,255,255,0.03),
            0 18px 50px rgba(0,0,0,0.30);
          padding: 22px 24px 18px 24px;
          display: flex;
          flex-direction: column;
          gap: 14px;
        }}

        .header {{
          flex: 0 0 auto;
        }}

        .title {{
          margin: 0;
          font-size: 26px;
          font-weight: 800;
          line-height: 1.05;
          letter-spacing: -0.03em;
        }}

        .subtitle {{
          margin-top: 8px;
          font-size: 14px;
          color: var(--muted);
          line-height: 1.45;
        }}

        .stage {{
          position: relative;
          flex: 1 1 auto;
          min-height: 360px;
          border-radius: 20px;
          overflow: hidden;
          border: 1px solid rgba(56, 189, 248, 0.14);
          background:
            linear-gradient(180deg, rgba(6, 24, 61, 0.82), rgba(3, 16, 42, 0.92)),
            repeating-linear-gradient(
              to right,
              transparent 0,
              transparent 54px,
              var(--grid) 54px,
              var(--grid) 56px
            ),
            repeating-linear-gradient(
              to bottom,
              transparent 0,
              transparent 54px,
              var(--grid) 54px,
              var(--grid) 56px
            );
        }}

        svg {{
          width: 100%;
          height: 100%;
          display: block;
        }}

        .route-intended {{
          fill: none;
          stroke: var(--blue);
          stroke-width: 6;
          stroke-linecap: round;
          stroke-linejoin: round;
          stroke-dasharray: 10 10;
          opacity: 0.95;
        }}

        .route-intended-faded {{
          fill: none;
          stroke: var(--blue-soft);
          stroke-width: 6;
          stroke-linecap: round;
          stroke-linejoin: round;
          stroke-dasharray: 10 10;
          opacity: 0.55;
        }}

        .route-alert {{
          fill: none;
          stroke: var(--red);
          stroke-width: 7;
          stroke-linecap: round;
          stroke-linejoin: round;
          stroke-dasharray: 8 10;
          opacity: 0.98;
          filter: drop-shadow(0 0 10px rgba(255,99,99,0.28));
        }}

        .route-gps {{
          fill: none;
          stroke: var(--yellow);
          stroke-width: 7;
          stroke-linecap: round;
          stroke-linejoin: round;
          stroke-dasharray: 8 10;
          opacity: 0.95;
          filter: drop-shadow(0 0 8px rgba(251,191,36,0.20));
        }}

        .route-ghost {{
          fill: none;
          stroke: var(--purple);
          stroke-width: 7;
          stroke-linecap: round;
          stroke-linejoin: round;
          stroke-dasharray: 8 10;
          opacity: 0.94;
          filter: drop-shadow(0 0 8px rgba(184,137,255,0.20));
        }}

        .plane {{
          font-size: 36px;
          dominant-baseline: middle;
          text-anchor: middle;
          filter: drop-shadow(0 0 10px rgba(255,255,255,0.22));
          pointer-events: none;
        }}

        .plane-main {{
          fill: var(--white-soft);
        }}

        .plane-secondary {{
          fill: #eddcff;
        }}

        .pulse {{
          fill: rgba(255, 99, 99, 0.18);
          stroke: rgba(255, 99, 99, 0.55);
          stroke-width: 2;
        }}

        .legend-row {{
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 14px;
          flex-wrap: wrap;
        }}

        .legend {{
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }}

        .pill {{
          display: inline-flex;
          align-items: center;
          gap: 10px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.04);
          color: var(--muted);
          border-radius: 999px;
          padding: 10px 16px;
          font-size: 13px;
          line-height: 1;
          white-space: nowrap;
        }}

        .dot {{
          width: 14px;
          height: 14px;
          border-radius: 50%;
          flex: 0 0 14px;
        }}

        .note {{
          color: var(--muted);
          font-size: 13px;
          line-height: 1.35;
          max-width: 460px;
          text-align: right;
        }}

        @media (max-width: 900px) {{
          .title {{
            font-size: 23px;
          }}
          .note {{
            text-align: left;
            max-width: 100%;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="card">
          <div class="header">
            <h1 class="title">{scene_title}</h1>
            <div class="subtitle">{scene_subtitle}</div>
          </div>

          <div class="stage">
            <svg id="scene-svg" viewBox="0 0 1200 540" preserveAspectRatio="none">
              <g id="scene-layer"></g>
              <circle id="jump-pulse" class="pulse" r="0" opacity="0"></circle>
              <text id="plane-main" class="plane plane-main">✈</text>
              <text id="plane-secondary" class="plane plane-secondary" opacity="0">✈</text>
            </svg>
          </div>

          <div class="legend-row">
            <div class="legend" id="legend"></div>
            <div class="note">
              animation loop showing the effects of each attack.
            </div>
          </div>
        </div>
      </div>

      <script>
        (function() {{
          const scene = {scene_key!r};
          const layer = document.getElementById("scene-layer");
          const planeMain = document.getElementById("plane-main");
          const planeSecondary = document.getElementById("plane-secondary");
          const jumpPulse = document.getElementById("jump-pulse");
          const legend = document.getElementById("legend");

          function setLegend(items) {{
            legend.innerHTML = items.map(item => `
              <div class="pill">
                <span class="dot" style="background:${{item.color}}"></span>
                <span>${{item.label}}</span>
              </div>
            `).join("");
          }}

          function createPath(d, cls, id) {{
            const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
            p.setAttribute("d", d);
            p.setAttribute("class", cls);
            if (id) p.setAttribute("id", id);
            layer.appendChild(p);
            return p;
          }}

          function createCircle(x, y, r, fill, opacity = 1) {{
            const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            c.setAttribute("cx", x);
            c.setAttribute("cy", y);
            c.setAttribute("r", r);
            c.setAttribute("fill", fill);
            c.setAttribute("opacity", opacity);
            layer.appendChild(c);
            return c;
          }}

          function pointAndAngle(path, t) {{
            const len = path.getTotalLength();
            const clamped = Math.max(0, Math.min(1, t));
            const p = path.getPointAtLength(len * clamped);
            const delta = 2;
            const p2 = path.getPointAtLength(Math.min(len, len * clamped + delta));
            const angle = Math.atan2(p2.y - p.y, p2.x - p.x) * 180 / Math.PI;
            return {{ x: p.x, y: p.y, angle }};
          }}

          function placePlane(el, x, y, angle) {{
            el.setAttribute("x", x);
            el.setAttribute("y", y);
            el.setAttribute("transform", `rotate(${{angle}} ${{x}} ${{y}})`);
          }}

          function pulseAt(x, y) {{
            jumpPulse.setAttribute("cx", x);
            jumpPulse.setAttribute("cy", y);
            jumpPulse.setAttribute("opacity", "1");
            jumpPulse.setAttribute("r", "0");

            let start = null;
            function pulse(ts) {{
              if (!start) start = ts;
              const e = ts - start;
              const dur = 520;
              const p = Math.min(e / dur, 1);
              jumpPulse.setAttribute("r", String(8 + p * 34));
              jumpPulse.setAttribute("opacity", String(1 - p));
              if (p < 1) requestAnimationFrame(pulse);
            }}
            requestAnimationFrame(pulse);
          }}

          function animateNormal() {{
            const path = createPath(
              "M 110 390 C 230 300, 360 285, 470 320 S 700 395, 830 350 S 1030 240, 1110 280",
              "route-intended",
              "normal-route"
            );

            setLegend([
              {{ color: "#38bdf8", label: "Expected route" }}
            ]);

            function frame(ts) {{
              const cycle = 7200;
              const t = (ts % cycle) / cycle;
              const pos = pointAndAngle(path, t);
              placePlane(planeMain, pos.x, pos.y, pos.angle);
              requestAnimationFrame(frame);
            }}
            requestAnimationFrame(frame);
          }}

          function animateTeleportation() {{
            const prePath = createPath(
              "M 110 390 C 225 315, 335 300, 430 318",
              "route-intended",
              "teleport-pre"
            );

            const postPath = createPath(
              "M 760 180 C 825 205, 900 255, 975 285 S 1080 325, 1110 305",
              "route-intended",
              "teleport-post"
            );

            createPath(
              "M 110 390 C 230 300, 360 285, 470 320 S 700 395, 830 350 S 1030 240, 1110 280",
              "route-intended-faded",
              "teleport-intended-full"
            );

            createPath(
              "M 430 318 L 760 180",
              "route-alert",
              "teleport-jump"
            );

            createCircle(430, 318, 6, "rgba(56,189,248,0.95)", 1);
            createCircle(760, 180, 6, "rgba(255,99,99,0.95)", 1);

            setLegend([
              {{ color: "#38bdf8", label: "Intended route" }},
              {{ color: "#ff6363", label: "Impossible position jump" }}
            ]);

            function frame(ts) {{
              const cycle = 7600;
              const t = (ts % cycle) / cycle;

              if (t < 0.46) {{
                const local = t / 0.46;
                const pos = pointAndAngle(prePath, local);
                placePlane(planeMain, pos.x, pos.y, pos.angle);
              }} else if (t < 0.54) {{
                const dest = pointAndAngle(postPath, 0.0);
                placePlane(planeMain, dest.x, dest.y, dest.angle);
                if (!frame._jumped) {{
                  pulseAt(430, 318);
                  pulseAt(760, 180);
                  frame._jumped = true;
                }}
              }} else {{
                const local = (t - 0.54) / 0.46;
                const pos = pointAndAngle(postPath, local);
                placePlane(planeMain, pos.x, pos.y, pos.angle);
              }}

              if (t < 0.44) frame._jumped = false;
              requestAnimationFrame(frame);
            }}

            requestAnimationFrame(frame);
          }}

          function animateGps() {{
            createPath(
              "M 110 385 C 250 300, 400 285, 550 315 S 840 390, 1110 260",
              "route-intended-faded",
              "gps-intended"
            );

            const spoofed = createPath(
              "M 110 385 C 250 300, 400 285, 520 320 S 760 450, 955 430 S 1080 355, 1110 250",
              "route-gps",
              "gps-spoofed"
            );

            setLegend([
              {{ color: "#38bdf8", label: "Intended route" }},
              {{ color: "#fbbf24", label: "Spoofed displayed route" }}
            ]);

            function frame(ts) {{
              const cycle = 7600;
              const t = (ts % cycle) / cycle;
              const pos = pointAndAngle(spoofed, t);
              placePlane(planeMain, pos.x, pos.y, pos.angle);
              requestAnimationFrame(frame);
            }}

            requestAnimationFrame(frame);
          }}

          function animateGhost() {{
            const realPath = createPath(
              "M 110 390 C 240 315, 370 290, 515 315 S 835 400, 1110 275",
              "route-intended",
              "ghost-real"
            );

            const ghostPath = createPath(
              "M 100 410 C 226 344, 354 324, 496 336 S 806 420, 1084 294",
              "route-ghost",
              "ghost-fake"
            );

            setLegend([
              {{ color: "#38bdf8", label: "Legitimate aircraft" }},
              {{ color: "#b889ff", label: "Ghost aircraft track" }}
            ]);

            planeSecondary.setAttribute("opacity", "1");

            function frame(ts) {{
              const cycle = 7800;
              const t = (ts % cycle) / cycle;

              const realPos = pointAndAngle(realPath, t);
              placePlane(planeMain, realPos.x, realPos.y, realPos.angle);

              const ghostT = Math.max(0, t - 0.11);
              const ghostPos = pointAndAngle(ghostPath, ghostT);
              placePlane(planeSecondary, ghostPos.x, ghostPos.y, ghostPos.angle);

              requestAnimationFrame(frame);
            }}

            requestAnimationFrame(frame);
          }}

          if (scene === "teleportation") {{
            animateTeleportation();
          }} else if (scene === "gps") {{
            animateGps();
          }} else if (scene === "ghost") {{
            animateGhost();
          }} else {{
            animateNormal();
          }}
        }})();
      </script>
    </body>
    </html>
    """


def render_attack_animation_scene(scene: str, caption: str) -> None:
    st.caption(caption)
    components.html(build_attack_animation_html(scene), height=575, scrolling=False)


def render_attack_visual_gallery() -> None:
    render_section_title(
        "Attack Visual Gallery",
        "Animated scenario loops for presentation and explanation",
    )

    normal_tab, tp_tab, gps_tab, ghost_tab = st.tabs(
        ["Normal Flight", "Teleportation", "GPS Spoofing", "Ghost Aircraft"]
    )

    with normal_tab:
        render_attack_animation_scene(
            "normal",
            "Baseline aircraft behavior with continuous motion and no injected anomaly.",
        )

    with tp_tab:
        render_attack_animation_scene(
            "teleportation",
            "Shows a physically impossible jump between distant positions.",
        )

    with gps_tab:
        render_attack_animation_scene(
            "gps",
            "Shows the displayed aircraft gradually drifting away from its intended route.",
        )

    with ghost_tab:
        render_attack_animation_scene(
            "ghost",
            "Shows a second false aircraft appearing and diverging from the legitimate track.",
        )


def render_active_scenario_animation(attack_name: str) -> None:
    scene_lookup = {
        "Teleportation": (
            "teleportation",
            "Teleportation preview: the aircraft is shown making an impossible jump between distant positions.",
        ),
        "GPS Spoofing": (
            "gps",
            "GPS spoofing preview: the displayed path is gradually pulled away from the aircraft's intended route.",
        ),
        "Ghost Aircraft": (
            "ghost",
            "Ghost aircraft preview: a false aircraft appears without a believable history in the airspace.",
        ),
    }

    scene = scene_lookup.get(attack_name)
    if not scene:
        return

    render_chart_shell(
        "Attack Playback",
        "A visual explanation of the selected scenario so the audience can connect the map behavior to the attack concept.",
    )
    render_attack_animation_scene(scene[0], scene[1])


def render_threat_library_page() -> None:
    render_section_title("Threat Library", "Plain-English explainers for the attacks this project is built to detect")
    render_page_note(
        "<strong>How to use this page:</strong> Start here if your audience is new to ADS-B security. Each threat type explains "
        "what the attack looks like, why it is suspicious, and how the system helps surface it."
    )
    render_capability_grid(
        [
            (
                "TP",
                "Teleportation",
                "An aircraft appears to jump an impossible distance in too little time, creating a path that cannot happen physically.",
            ),
            (
                "GS",
                "GPS Spoofing",
                "False position data slowly drags the aircraft away from its true route, making the displayed track unreliable.",
            ),
            (
                "GH",
                "Ghost Aircraft",
                "A false aircraft appears in the airspace without a believable history or motion pattern.",
            ),
        ]
    )
    render_page_note(
        "<strong>Quick glossary:</strong> <strong>ADS-B</strong> is the aircraft broadcast data feed used for tracking. "
        "<strong>Spoofing</strong> means false position data is injected into the feed. <strong>Ghost aircraft</strong> means "
        "a fake track appears as if it were a real aircraft."
    )
    render_attack_visual_gallery()


# -----------------------------
# ML helpers
# -----------------------------
def _artifact_obj_to_meta(artifact_obj: Any) -> dict[str, Any]:
    if artifact_obj is None:
        return {}

    if isinstance(artifact_obj, dict):
        if "meta" in artifact_obj and isinstance(artifact_obj["meta"], dict):
            return artifact_obj["meta"]
        return {
            k: artifact_obj.get(k)
            for k in [
                "seq_len",
                "threshold",
                "feature_cols",
                "min_quality",
                "persistence_k",
                "persistence_m",
                "max_dt_sec",
            ]
            if k in artifact_obj
        }

    meta: dict[str, Any] = {}
    for key in [
        "seq_len",
        "threshold",
        "feature_cols",
        "min_quality",
        "persistence_k",
        "persistence_m",
        "max_dt_sec",
    ]:
        if hasattr(artifact_obj, key):
            meta[key] = getattr(artifact_obj, key)

    return meta


def friendly_ml_summary(meta: Optional[dict[str, Any]]) -> str:
    if not meta:
        return "ML artifacts loaded."

    seq_len = meta.get("seq_len", "?")
    threshold = meta.get("threshold", "?")
    min_quality = meta.get("min_quality", "?")
    persistence_k = meta.get("persistence_k", meta.get("k", "?"))
    persistence_m = meta.get("persistence_m", meta.get("m", "?"))
    max_dt_sec = meta.get("max_dt_sec", "?")

    threshold_text = f"{threshold:.6f}" if isinstance(threshold, (int, float)) else str(threshold)

    return (
        f"Artifacts loaded "
        f"(seq_len={seq_len}, threshold={threshold_text}, min_quality={min_quality}, "
        f"persistence={persistence_k}/{persistence_m}, max_dt_sec={max_dt_sec})"
    )


def safe_score_sequences(
    df: pd.DataFrame,
    artifacts: Any,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    if not ML_AVAILABLE:
        return None, None, "ML unavailable."

    try:
        result = score_sequences(df, artifacts)

        if isinstance(result, tuple):
            score_df = result[0] if len(result) > 0 and isinstance(result[0], pd.DataFrame) else pd.DataFrame()
            anom_df = result[1] if len(result) > 1 and isinstance(result[1], pd.DataFrame) else pd.DataFrame()
            return score_df, anom_df, None

        if isinstance(result, pd.DataFrame):
            return result, pd.DataFrame(), None

        if isinstance(result, dict):
            scores_df = result.get("scores") if isinstance(result.get("scores"), pd.DataFrame) else pd.DataFrame()
            anoms_df = result.get("anoms") if isinstance(result.get("anoms"), pd.DataFrame) else pd.DataFrame()
            if not scores_df.empty or not anoms_df.empty:
                return scores_df, anoms_df, None

        return None, None, "ML scoring returned no dataframe output."
    except Exception as e:
        return None, None, f"ML scoring failed: {e}"


def safe_train_lstm(train_df: pd.DataFrame, params: dict[str, Any]) -> str:
    if not ML_AVAILABLE:
        return f"ML unavailable: {ML_IMPORT_ERROR}"

    try:
        train_fn: Any = train_lstm_autoencoder
        sig = inspect.signature(train_fn)

        kwargs: dict[str, Any] = {}

        aliases = {
            "seq_len": ["seq_len", "sequence_length"],
            "max_dt_sec": ["max_dt_sec"],
            "min_quality": ["min_quality"],
            "epochs": ["epochs", "n_epochs"],
            "threshold_percentile": ["threshold_percentile", "percentile"],
            "persistence_k": ["persistence_k", "k"],
            "persistence_m": ["persistence_m", "m"],
        }

        for user_key, fn_keys in aliases.items():
            for fn_key in fn_keys:
                if fn_key in sig.parameters and user_key in params:
                    kwargs[fn_key] = params[user_key]

        train_fn(train_df, **kwargs)
        return "Training completed. Artifacts were refreshed."
    except Exception as e:
        return f"Training failed: {e}"


def safe_run_ml_evaluation(runs_per_scenario: int = 5) -> str:
    if not ML_AVAILABLE:
        return f"ML unavailable: {ML_IMPORT_ERROR}"

    try:
        result = run_evaluation(runs_per_scenario=int(runs_per_scenario), save=True)
        seq_metrics = result["metrics"]["sequence_metrics"]
        return (
            "Evaluation completed. "
            f"F1={seq_metrics.get('f1', float('nan')):.3f}, "
            f"ROC-AUC={seq_metrics.get('roc_auc', float('nan')):.3f}, "
            f"PR-AUC={seq_metrics.get('pr_auc', float('nan')):.3f}."
        )
    except Exception as e:
        return f"Evaluation failed: {e}"


def load_ml_evaluation_artifacts() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: dict[str, Any] = {}
    seq_df = pd.DataFrame()
    row_df = pd.DataFrame()
    scenario_df = pd.DataFrame()

    try:
        if os.path.exists(EVAL_METRICS_PATH):
            with open(EVAL_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
    except Exception:
        metrics = {}

    try:
        if os.path.exists(EVAL_SEQ_PATH):
            seq_df = pd.read_csv(EVAL_SEQ_PATH)
    except Exception:
        seq_df = pd.DataFrame()

    try:
        if os.path.exists(EVAL_ROW_PATH):
            row_df = pd.read_csv(EVAL_ROW_PATH)
    except Exception:
        row_df = pd.DataFrame()

    try:
        if os.path.exists(EVAL_SCENARIO_PATH):
            scenario_df = pd.read_csv(EVAL_SCENARIO_PATH)
    except Exception:
        scenario_df = pd.DataFrame()

    return metrics, seq_df, row_df, scenario_df


# -----------------------------
# Map helpers
# -----------------------------
def build_map_colors(label: str, opacity: int) -> list[int]:
    mapping = {
        "Normal": [52, 211, 153, opacity],
        "Impossible Jump": [255, 99, 99, opacity],
        "Position Mismatch": [251, 191, 36, opacity],
        "Ghost / Fake Track": [184, 137, 255, opacity],
    }
    return mapping.get(label, [52, 211, 153, opacity])


def render_air_traffic_map(
    latest_df: pd.DataFrame,
    paths_df: pd.DataFrame,
    point_radius: int,
    point_opacity: int,
    map_height: int,
) -> None:
    latest_df = ensure_lat_lon(latest_df)

    if latest_df.empty:
        render_info_banner("No map-ready rows available.", "warn")
        return

    if not {"latitude", "longitude"}.issubset(latest_df.columns):
        render_info_banner("Latitude/longitude columns are missing.", "warn")
        return

    valid_points = latest_df.dropna(subset=["latitude", "longitude"]).copy()
    if valid_points.empty:
        render_info_banner("No valid coordinates available for map display.", "warn")
        return

    center_lat = float(pd.to_numeric(valid_points["latitude"], errors="coerce").mean())
    center_lon = float(pd.to_numeric(valid_points["longitude"], errors="coerce").mean())

    valid_points["fill_color"] = (
        valid_points["human_alert_label"]
        .fillna("Normal")
        .astype(str)
        .apply(lambda x: build_map_colors(x, point_opacity))
    )

    valid_points["tooltip_text"] = (
        "Aircraft: " + valid_points["icao"].astype(str)
        + "\nAlert: " + valid_points["human_alert_label"].fillna("Normal").astype(str)
        + "\nScenario: " + valid_points["scenario_label"].fillna("Normal Flight").astype(str)
    )

    layers: list[Any] = []

    if not paths_df.empty and "path" in paths_df.columns:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=paths_df,
                get_path="path",
                get_color="path_color",
                get_width="path_width",
                width_scale=1,
                width_min_pixels=2,
                pickable=False,
                opacity=0.55,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=valid_points,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_radius=point_radius,
            pickable=True,
            stroked=True,
            filled=True,
            line_width_min_pixels=1,
            get_line_color=[255, 255, 255, 100],
        )
    )

    tooltip_value: Any = {
        "text": "{tooltip_text}",
        "style": {
            "backgroundColor": "rgba(5, 17, 33, 0.95)",
            "color": "white",
            "fontSize": "13px",
        },
    }

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=7.0,
        pitch=38,
        bearing=12,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip=cast(Any, tooltip_value),
    )

    st.pydeck_chart(deck, use_container_width=True, height=map_height)

    st.markdown(
        """
        <div class="legend-inline">
            <div class="legend-pill"><span class="legend-dot" style="background:#34d399;"></span>Normal aircraft</div>
            <div class="legend-pill"><span class="legend-dot" style="background:#ff6363;"></span>Impossible jump</div>
            <div class="legend-pill"><span class="legend-dot" style="background:#fbbf24;"></span>Position mismatch</div>
            <div class="legend-pill"><span class="legend-dot" style="background:#b889ff;"></span>Ghost / fake track</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Aircraft inspector
# -----------------------------
def anomaly_inspector(df: pd.DataFrame, selected_aircraft: Optional[str]) -> None:
    if not selected_aircraft:
        render_info_banner("Select an aircraft to inspect.", "info")
        return

    aircraft_df = df[df["icao"].astype(str) == str(selected_aircraft)].copy()
    if aircraft_df.empty:
        render_info_banner("No aircraft data found for the selected aircraft.", "info")
        return

    aircraft_df = aircraft_df.sort_values("timestamp")
    last = aircraft_df.iloc[-1]
    speed_val = pd.to_numeric(pd.Series([last.get("speed_mps")]), errors="coerce").iloc[0]
    alt_val = pd.to_numeric(pd.Series([last.get("altitude")]), errors="coerce").iloc[0]
    head_val = pd.to_numeric(pd.Series([last.get("heading")]), errors="coerce").iloc[0]
    q_val = pd.to_numeric(pd.Series([last.get("data_quality_score")]), errors="coerce").iloc[0]

    render_chart_shell(
        f"Aircraft {selected_aircraft}",
        "Latest known aircraft state for the selected track, followed by trend charts for the same aircraft.",
    )
    render_stat_strip(
        [
            ("Current Alert", last.get("human_alert_label", "Normal")),
            ("Scenario", last.get("scenario_label", "Normal Flight")),
            ("Speed", format_display_value(speed_val, suffix=" m/s")),
            ("Altitude", format_display_value(alt_val, suffix=" m")),
            ("Heading", format_display_value(head_val, suffix="°")),
            ("Quality Score", format_display_value(q_val)),
        ]
    )

    tab_speed, tab_alt, tab_turn, tab_quality = st.tabs(["Speed", "Altitude", "Turn Rate", "Quality"])

    with tab_speed:
        if "speed_mps" in aircraft_df.columns:
            render_chart_shell("Speed Profile", "Reported speed over time for the selected aircraft.")
            chart_df = aircraft_df[["timestamp", "speed_mps"]].copy()
            chart_df["speed_mps"] = pd.to_numeric(chart_df["speed_mps"], errors="coerce")
            st.line_chart(chart_df.set_index("timestamp"), height=280)
        else:
            st.info("Speed data not available.")

    with tab_alt:
        if "altitude" in aircraft_df.columns:
            render_chart_shell("Altitude Profile", "Altitude trend across the available surveillance window.")
            chart_df = aircraft_df[["timestamp", "altitude"]].copy()
            chart_df["altitude"] = pd.to_numeric(chart_df["altitude"], errors="coerce")
            st.line_chart(chart_df.set_index("timestamp"), height=280)
        else:
            st.info("Altitude data not available.")

    with tab_turn:
        if "turn_rate_dps" in aircraft_df.columns:
            render_chart_shell("Turn Dynamics", "Heading-change intensity used to spot unusual maneuvering.")
            chart_df = aircraft_df[["timestamp", "turn_rate_dps"]].copy()
            chart_df["turn_rate_dps"] = pd.to_numeric(chart_df["turn_rate_dps"], errors="coerce")
            st.line_chart(chart_df.set_index("timestamp"), height=280)
        else:
            st.info("Turn-rate data not available.")

    with tab_quality:
        if "data_quality_score" in aircraft_df.columns:
            render_chart_shell("Data Quality", "Trust score for the incoming points over time.")
            chart_df = aircraft_df[["timestamp", "data_quality_score"]].copy()
            chart_df["data_quality_score"] = pd.to_numeric(chart_df["data_quality_score"], errors="coerce")
            st.line_chart(chart_df.set_index("timestamp"), height=280)
        else:
            st.info("Quality-score data not available.")


# -----------------------------
# Session defaults
# -----------------------------
def init_control_state() -> None:
    defaults: dict[str, Any] = {
        "source_mode": "Simulation",
        "use_ml": ML_AVAILABLE,
        "lat_min": 35.80,
        "lon_min": -87.20,
        "lat_max": 36.60,
        "lon_max": -86.20,
        "point_radius": 1400,
        "point_opacity": 210,
        "map_height": 680,
        "aircraft_count": 25,
        "time_steps": 120,
        "attack_name": "None",
        "attack_start": 20,
        "attack_end": 25,
        "lat_shift": 0.10,
        "lon_shift": 0.10,
        "show_ml_engineering": False,
        "seq_len": 20,
        "max_dt_sec": 10,
        "min_quality": 0.80,
        "epochs": 24,
        "threshold_percentile": 99.30,
        "persistence_k": 5,
        "persistence_m": 6,
        "evaluation_runs_per_scenario": 5,
        "live_auto_refresh_enabled": True,
        "refresh_interval_sec": 15,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_control_state()


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown('<div class="avatar-title" style="font-size:1.08rem;">CONTROL PANEL</div>', unsafe_allow_html=True)
    render_sidebar_caption("Cleaner operator view. Run the system, choose the scenario, and only change what matters.")

    run_clicked = st.button("Run Monitoring", type="primary", width="stretch", key="run_monitoring_btn")

    with st.container():
        st.markdown("### System")
        st.radio(
            "Mode",
            ["Simulation", "Live OpenSky Snapshot"],
            key="source_mode",
        )
        st.toggle("Enable ML", key="use_ml")

    if st.session_state["source_mode"] == "Live OpenSky Snapshot":
        with st.container():
            st.markdown("### Live Coverage Area")
            st.number_input("Lat Min", value=float(st.session_state["lat_min"]), format="%.4f", key="lat_min")
            st.number_input("Lon Min", value=float(st.session_state["lon_min"]), format="%.4f", key="lon_min")
            st.number_input("Lat Max", value=float(st.session_state["lat_max"]), format="%.4f", key="lat_max")
            st.number_input("Lon Max", value=float(st.session_state["lon_max"]), format="%.4f", key="lon_max")

            st.markdown("### Live Refresh")
            st.toggle("Play Live Refresh", key="live_auto_refresh_enabled")
            st.slider("Refresh Interval (sec)", 5, 120, key="refresh_interval_sec")

    if st.session_state["source_mode"] == "Simulation":
        with st.container():
            st.markdown("### Scenario")
            st.selectbox(
                "Attack Type",
                ["None", "Teleportation", "GPS Spoofing", "Ghost Aircraft"],
                key="attack_name",
            )
            st.slider("Aircraft Count", 5, 100, key="aircraft_count")
            st.slider("Simulation Frames", 30, 300, step=5, key="time_steps")
            max_attack_frame = max(0, int(st.session_state["time_steps"]) - 1)
            st.slider("Attack Start", 0, max_attack_frame, key="attack_start")
            if int(st.session_state["attack_end"]) < int(st.session_state["attack_start"]):
                st.session_state["attack_end"] = int(st.session_state["attack_start"])
            st.slider("Attack End", int(st.session_state["attack_start"]), max_attack_frame, key="attack_end")

            if st.session_state["attack_name"] == "GPS Spoofing":
                st.slider("Latitude Shift", 0.00, 1.00, step=0.01, key="lat_shift")
                st.slider("Longitude Shift", 0.00, 1.00, step=0.01, key="lon_shift")
            else:
                st.session_state["lat_shift"] = 0.10
                st.session_state["lon_shift"] = 0.10

    with st.expander("Display", expanded=False):
        st.slider("Marker Size", 200, 5000, step=100, key="point_radius")
        st.slider("Map Height", 450, 1000, step=50, key="map_height")
        st.slider("Marker Opacity", 60, 255, step=5, key="point_opacity")

    with st.expander("Advanced ML", expanded=False):
        st.toggle("Show Advanced ML Details", key="show_ml_engineering")
        st.number_input("Sequence Length", min_value=5, max_value=120, step=1, key="seq_len")
        st.number_input("Max Gap (sec)", min_value=1, max_value=60, step=1, key="max_dt_sec")
        st.slider("Min Quality", 0.0, 1.0, step=0.05, key="min_quality")
        st.number_input("Epochs", min_value=1, max_value=100, step=1, key="epochs")
        st.number_input(
            "Threshold %",
            min_value=90.0,
            max_value=99.99,
            step=0.01,
            key="threshold_percentile",
        )
        st.number_input("Persistence k", min_value=1, max_value=20, step=1, key="persistence_k")
        st.number_input("Persistence m", min_value=1, max_value=20, step=1, key="persistence_m")
        st.number_input(
            "Eval Runs / Scenario",
            min_value=1,
            max_value=20,
            step=1,
            key="evaluation_runs_per_scenario",
        )
        train_clicked = st.button("Train / Retrain Model", width="stretch", key="train_retrain_btn")
        eval_clicked = st.button("Run ML Evaluation", width="stretch", key="run_eval_btn")


# -----------------------------
# Locals from session state
# -----------------------------
source_mode = cast(str, st.session_state["source_mode"])
use_ml = bool(st.session_state["use_ml"])

lat_min = float(st.session_state["lat_min"])
lon_min = float(st.session_state["lon_min"])
lat_max = float(st.session_state["lat_max"])
lon_max = float(st.session_state["lon_max"])

point_radius = int(st.session_state["point_radius"])
point_opacity = int(st.session_state["point_opacity"])
map_height = int(st.session_state["map_height"])

aircraft_count = int(st.session_state["aircraft_count"])
time_steps = int(st.session_state["time_steps"])
attack_name = cast(str, st.session_state["attack_name"])
attack_start = int(st.session_state["attack_start"])
attack_end = int(st.session_state["attack_end"])
lat_shift = float(st.session_state["lat_shift"])
lon_shift = float(st.session_state["lon_shift"])

show_ml_engineering = bool(st.session_state["show_ml_engineering"])
seq_len = int(st.session_state["seq_len"])
max_dt_sec = int(st.session_state["max_dt_sec"])
min_quality = float(st.session_state["min_quality"])
epochs = int(st.session_state["epochs"])
threshold_percentile = float(st.session_state["threshold_percentile"])
persistence_k = int(st.session_state["persistence_k"])
persistence_m = int(st.session_state["persistence_m"])
evaluation_runs_per_scenario = int(st.session_state["evaluation_runs_per_scenario"])

live_auto_refresh_enabled = bool(st.session_state["live_auto_refresh_enabled"])
refresh_interval_sec = int(st.session_state["refresh_interval_sec"])

current_run_context_sig = compute_run_context_signature(
    source_mode=source_mode,
    attack_name=attack_name,
    attack_start=attack_start,
    attack_end=attack_end,
    aircraft_count=aircraft_count,
    time_steps=time_steps,
    lat_shift=lat_shift,
    lon_shift=lon_shift,
    lat_min=lat_min,
    lon_min=lon_min,
    lat_max=lat_max,
    lon_max=lon_max,
)


# -----------------------------
# Session state bootstrap
# -----------------------------
if "first_load_complete" not in st.session_state:
    st.session_state["first_load_complete"] = True
    run_clicked = True

init_history_state()


# -----------------------------
# Refresh mode
# -----------------------------
refresh_backend = "off"
if source_mode == "Live OpenSky Snapshot":
    refresh_backend = enable_live_rerun_timer(refresh_interval_sec, live_auto_refresh_enabled)


# -----------------------------
# Data pipeline
# -----------------------------
raw_df = pd.DataFrame()
processed_df = pd.DataFrame()
latest_df = pd.DataFrame()
paths_df = pd.DataFrame()
history_df = cast(pd.DataFrame, st.session_state.get("flight_history_df", empty_history_df()))

should_auto_run_live = source_mode == "Live OpenSky Snapshot" and live_auto_refresh_enabled
should_run_pipeline = run_clicked or should_auto_run_live

if should_run_pipeline:
    try:
        if source_mode == "Simulation":
            sim_df = safe_generate_sim_data(aircraft_count, time_steps)
            sim_df = safe_apply_attack(sim_df, attack_name, attack_start, attack_end, lat_shift, lon_shift)
            raw_df = sim_df.copy()
            effective_attack_name = attack_name
            effective_attack_start = attack_start
            effective_attack_end = attack_end
        else:
            fetch_result = fetch_live_adsb_data(
                lamin=float(lat_min),
                lomin=float(lon_min),
                lamax=float(lat_max),
                lomax=float(lon_max),
            )

            fetch_err: Optional[str] = None

            if isinstance(fetch_result, tuple):
                raw_df = fetch_result[0]
                if len(fetch_result) > 1:
                    fetch_err = fetch_result[1]
            elif isinstance(fetch_result, pd.DataFrame):
                raw_df = fetch_result
            else:
                raise RuntimeError("Live fetch returned an unexpected result.")

            if fetch_err:
                st.error(fetch_err)
                st.stop()

            effective_attack_name = "None"
            effective_attack_start = 0
            effective_attack_end = 0

        raw_df = ensure_lat_lon(ensure_aircraft_id_column(ensure_timestamp_column(raw_df)))

        base_processed = process_adsb_data(raw_df)
        detection_result = detect_anomalies(base_processed)
        processed_df = normalize_detection_output(base_processed, detection_result)

        processed_df = ensure_lat_lon(ensure_aircraft_id_column(ensure_timestamp_column(processed_df)))
        processed_df = add_user_friendly_columns(
            processed_df,
            effective_attack_name,
            effective_attack_start,
            effective_attack_end,
        )

        latest_df = latest_state_per_aircraft(processed_df)
        paths_df = build_paths(processed_df)
        history_df = update_flight_history(source_mode, processed_df)

        st.session_state["raw_df"] = raw_df
        st.session_state["processed_df"] = processed_df
        st.session_state["latest_df"] = latest_df
        st.session_state["paths_df"] = paths_df
        st.session_state["history_df_current_run"] = history_df
        st.session_state["processed_df_sig"] = compute_df_signature(processed_df)
        st.session_state["last_run_context_sig"] = current_run_context_sig
        st.session_state["last_run_source_mode"] = source_mode
        st.session_state["last_run_attack_name"] = effective_attack_name
        st.session_state["last_run_attack_start"] = effective_attack_start
        st.session_state["last_run_attack_end"] = effective_attack_end

    except Exception as e:
        st.error(f"Dashboard run failed: {e}")

if "processed_df" in st.session_state:
    raw_df = cast(pd.DataFrame, st.session_state.get("raw_df", pd.DataFrame()))
    processed_df = cast(pd.DataFrame, st.session_state.get("processed_df", pd.DataFrame()))
    latest_df = cast(pd.DataFrame, st.session_state.get("latest_df", pd.DataFrame()))
    paths_df = cast(pd.DataFrame, st.session_state.get("paths_df", pd.DataFrame()))
    history_df = cast(
        pd.DataFrame,
        st.session_state.get("history_df_current_run", st.session_state.get("flight_history_df", empty_history_df())),
    )

display_source_mode = cast(str, st.session_state.get("last_run_source_mode", source_mode))
display_attack_name = cast(
    str,
    st.session_state.get(
        "last_run_attack_name",
        "None" if display_source_mode == "Simulation" else "Live Traffic",
    ),
)
display_attack_start = int(st.session_state.get("last_run_attack_start", attack_start))
display_attack_end = int(st.session_state.get("last_run_attack_end", attack_end))
controls_changed_since_run = (
    "last_run_context_sig" in st.session_state
    and cast(str, st.session_state["last_run_context_sig"]) != current_run_context_sig
    and not should_auto_run_live
)


# -----------------------------
# ML load / score / train
# -----------------------------
ml_artifacts: Optional[Any] = None
ml_meta: dict[str, Any] = {}
ml_status: Optional[str] = None
ml_scores_df: Optional[pd.DataFrame] = None
ml_anoms_df: Optional[pd.DataFrame] = None
ml_eval_metrics: dict[str, Any] = {}
ml_eval_seq_df = pd.DataFrame()
ml_eval_row_df = pd.DataFrame()
ml_eval_scenario_df = pd.DataFrame()

current_processed_sig = compute_df_signature(processed_df)
current_ml_settings_sig = compute_ml_settings_signature(
    int(seq_len),
    int(max_dt_sec),
    float(min_quality),
    float(threshold_percentile),
    int(persistence_k),
    int(persistence_m),
)

if use_ml and not processed_df.empty:
    ml_artifacts, ml_error = get_cached_ml_artifacts()

    if ml_error:
        ml_status = ml_error
    elif ml_artifacts is not None:
        ml_meta = _artifact_obj_to_meta(ml_artifacts)
        cache_sig = st.session_state.get("ml_scores_sig")
        combined_sig = f"{current_processed_sig}|{current_ml_settings_sig}"

        if cache_sig == combined_sig:
            ml_scores_df = cast(Optional[pd.DataFrame], st.session_state.get("ml_scores_cache"))
            ml_anoms_df = cast(Optional[pd.DataFrame], st.session_state.get("ml_anoms_cache"))
        else:
            scores_df, anoms_df, score_error = safe_score_sequences(processed_df, ml_artifacts)
            ml_scores_df = scores_df
            ml_anoms_df = anoms_df

            st.session_state["ml_scores_cache"] = ml_scores_df
            st.session_state["ml_anoms_cache"] = ml_anoms_df
            st.session_state["ml_scores_sig"] = combined_sig
            st.session_state["ml_meta_cache"] = ml_meta

            if score_error:
                ml_status = score_error

        if ml_status is None:
            ml_status = friendly_ml_summary(ml_meta)

if train_clicked and not processed_df.empty:
    train_message = safe_train_lstm(
        processed_df,
        {
            "seq_len": int(seq_len),
            "max_dt_sec": int(max_dt_sec),
            "min_quality": float(min_quality),
            "epochs": int(epochs),
            "threshold_percentile": float(threshold_percentile),
            "persistence_k": int(persistence_k),
            "persistence_m": int(persistence_m),
        },
    )
    clear_cached_ml_artifacts()

    if "failed" in train_message.lower():
        st.error(train_message)
    else:
        st.success(train_message)

if eval_clicked:
    eval_message = safe_run_ml_evaluation(evaluation_runs_per_scenario)
    if "failed" in eval_message.lower():
        st.error(eval_message)
    else:
        st.success(eval_message)

ml_eval_metrics, ml_eval_seq_df, ml_eval_row_df, ml_eval_scenario_df = load_ml_evaluation_artifacts()


# -----------------------------
# Main page
# -----------------------------
render_hero(
    "Aircraft Security Monitoring Platform",
    "An interactive aviation security dashboard for tracking aircraft behavior, spotting suspicious movement, and explaining anomalies clearly.",
)

if controls_changed_since_run:
    pending_scenario = attack_name if source_mode == "Simulation" else "Live monitoring"
    current_scenario = display_attack_name if display_source_mode == "Simulation" else "Live monitoring"
    render_info_banner(
        "The sidebar controls have changed since the last run. "
        f"The dashboard is still showing the previous run ({current_scenario}) until you press Run Monitoring. "
        f"Your pending selection is {pending_scenario}.",
        "warn",
    )

if source_mode == "Live OpenSky Snapshot":
    if live_auto_refresh_enabled:
        backend_label = "Streamlit fragment timer" if refresh_backend == "fragment" else "browser fallback"
        render_info_banner(
            f"Live refresh is running continuously every {refresh_interval_sec} seconds using {backend_label}. Live mode stays locked unless you manually switch it.",
            "info",
        )
    else:
        render_info_banner(
            "Live refresh is paused. Press Run Monitoring for a manual snapshot.",
            "warn",
        )

summary = compute_summary(processed_df)

critical_alerts = 0
if not latest_df.empty and "human_alert_label" in latest_df.columns:
    critical_alerts = int(
        latest_df["human_alert_label"].fillna("Normal").isin(
            ["Impossible Jump", "Ghost / Fake Track"]
        ).sum()
    )

mode_subtle = "Live feed" if display_source_mode == "Live OpenSky Snapshot" else "Simulation run"
scenario_value = display_attack_name if display_source_mode == "Simulation" else "Live monitoring"
refresh_value = (
    f"{refresh_interval_sec}s auto-refresh" if source_mode == "Live OpenSky Snapshot" and live_auto_refresh_enabled else "Manual refresh"
)
ml_state = "Online" if (use_ml and ml_artifacts is not None) else "Offline"
ml_eval_f1 = "Not run"
if ml_eval_metrics:
    seq_metrics = cast(dict[str, Any], ml_eval_metrics.get("sequence_metrics", {}))
    if seq_metrics:
        ml_eval_f1 = f"F1 {seq_metrics.get('f1', float('nan')):.3f} / FPR {seq_metrics.get('false_positive_rate', float('nan')):.3f}"

st.markdown(
    f"""
    <div class="run-snapshot-grid">
        <div class="run-snapshot-card">
            <div class="run-snapshot-label">Run Context</div>
            <div class="run-snapshot-value">{display_source_mode} / {scenario_value}</div>
            <div class="run-snapshot-meta">{mode_subtle}. This is the scenario currently shown in the dashboard.</div>
        </div>
        <div class="run-snapshot-card">
            <div class="run-snapshot-label">Airspace Readout</div>
            <div class="run-snapshot-value">{summary["aircraft_tracked"]} aircraft, {summary["active_alerts"]} alerts</div>
            <div class="run-snapshot-meta">{critical_alerts} high-priority conditions currently need review.</div>
        </div>
        <div class="run-snapshot-card">
            <div class="run-snapshot-label">Model + Refresh</div>
            <div class="run-snapshot-value">{ml_state}</div>
            <div class="run-snapshot-meta">{ml_eval_f1}. {refresh_value}.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_ops, tab_ml, tab_threats, tab_settings = st.tabs(
    ["Overview", "Operations", "ML Intelligence", "Threat Library", "Settings"]
)

with tab_overview:
    render_section_title("Overview", "A cleaner first look at the map, the active scenario, and what deserves attention")
    render_page_note(
        "<strong>Start here:</strong> Look at the map first, then read the current scenario card and the alert summary. "
        "If an attack is selected, use the attack playback to understand what the anomaly should look like."
    )
    left, right = st.columns([1.7, 0.9])

    with left:
        render_section_title("Live Situation Overview", "The main airspace view for the current run")
        render_chart_shell(
            "Airspace Snapshot",
            "A large map preview so you can quickly understand where aircraft are and whether something looks suspicious.",
        )
        render_air_traffic_map(latest_df, paths_df, point_radius, point_opacity, max(760, map_height))

        if display_source_mode == "Simulation" and display_attack_name != "None":
            render_active_scenario_animation(display_attack_name)

    with right:
        render_detail_panel(
            "Current Scenario",
            [
                ("Mode", display_source_mode),
                ("Scenario", display_attack_name if display_source_mode == "Simulation" else "Live Traffic"),
                ("Aircraft In View", summary["aircraft_tracked"]),
                ("Active Alerts", summary["active_alerts"]),
                ("High-Priority Alerts", critical_alerts),
                ("Model Status", ml_state),
            ],
            eyebrow="Overview",
            subtitle="This is the quickest summary of what you are looking at in the current run.",
        )
        render_system_status_block(
            source_mode=display_source_mode,
            attack_name=display_attack_name,
            use_ml=use_ml,
            ml_artifacts=ml_artifacts,
            active_alerts=summary["active_alerts"],
        )
        if not latest_df.empty:
            render_alert_summary_panel(latest_df)
        else:
            render_info_banner("Run monitoring to populate the alert queue and airspace summary.", "info")

    render_page_note(
        "<strong>Need deeper analysis?</strong> Use <strong>Operations</strong> for aircraft-by-aircraft review and incident maps. "
        "Use <strong>ML Intelligence</strong> when you want benchmark metrics and model output."
    )

with tab_ops:
    render_section_title("Operations", "Mission-control workspace for live tracking, aircraft review, and incident analysis")
    render_page_note(
        "<strong>Operations is the working page.</strong> Use it when you want to inspect a specific aircraft, review alerts, "
        "or walk through an incident in detail."
    )
    render_chart_shell(
        "Operations Map",
        "Use this as the main operating picture for live or simulated traffic before drilling into a single aircraft.",
    )
    render_air_traffic_map(latest_df, paths_df, point_radius, point_opacity, map_height + 40)

    st.divider()
    render_section_title("Aircraft Inspector", "Selected aircraft details and aircraft-level trends")
    selector_left, selector_right = st.columns([1.35, 0.9])

    with selector_left:
        if not latest_df.empty:
            aircraft_options = sorted(latest_df["icao"].astype(str).unique().tolist())
            selected_aircraft = st.selectbox("Aircraft Track", aircraft_options, key="map_aircraft")
        else:
            selected_aircraft = None
            render_info_banner("Run monitoring first so aircraft detail becomes available.", "info")

    with selector_right:
        incident_options = incident_aircraft_options(history_df)
        if incident_options:
            selected_incident_aircraft = st.selectbox(
                "Incident Aircraft",
                incident_options,
                key="incident_aircraft_selector",
            )
        else:
            selected_incident_aircraft = None
            render_info_banner(
                "No incident aircraft are available yet. Run an attack scenario or wait for a live alert.",
                "info",
            )

    render_alert_summary_panel(latest_df)

    if selected_aircraft is not None:
        anomaly_inspector(processed_df, selected_aircraft)

    st.divider()
    render_section_title("Incident Review", "Focused playback and explanation for a selected anomalous aircraft")
    render_incident_center(
        history_df=history_df,
        selected_icao=selected_incident_aircraft,
        attack_name=display_attack_name,
        source_mode=display_source_mode,
        point_radius=point_radius,
        map_height=map_height + 40,
        use_ml=use_ml,
        ml_anoms_df=ml_anoms_df,
    )

    with st.expander("Alert Breakdown", expanded=False):
        if not latest_df.empty and "human_alert_label" in latest_df.columns:
            counts = (
                latest_df["human_alert_label"]
                .fillna("Normal")
                .value_counts()
                .rename_axis("Alert")
                .reset_index(name="Count")
            )
            st.dataframe(counts, width="stretch", hide_index=True)

with tab_ml:
    render_section_title("ML Intelligence", "Model performance, current run behavior, and benchmark-backed evidence")
    st.markdown(
        """
        <div class="insight-note" style="margin-top:0; margin-bottom:0.9rem;">
            Read this tab in two passes: first check benchmark health to judge whether the model is behaving well,
            then review current-run scores and alerts to see what it is flagging right now.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not ML_AVAILABLE:
        render_info_banner(f"ML import failed: {ML_IMPORT_ERROR}", "danger")
    elif processed_df.empty:
        render_info_banner("Run a scenario first so ML can score processed sequences.", "info")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("ML Enabled", "Yes" if use_ml else "No", "")
        with c2:
            metric_card("Artifacts", "Loaded" if ml_artifacts is not None else "Missing", "")
        with c3:
            metric_card(
                "Scored Sequences",
                0 if not isinstance(ml_scores_df, pd.DataFrame) else len(ml_scores_df),
                "",
            )
        with c4:
            metric_card(
                "Persistent Alerts",
                0 if not isinstance(ml_anoms_df, pd.DataFrame) else len(ml_anoms_df),
                "",
            )

        if ml_status:
            render_info_banner(ml_status, "info")

        if ml_eval_metrics:
            render_section_title("Benchmark Health", "Held-out benchmark summary")
            seq_metrics = cast(dict[str, Any], ml_eval_metrics.get("sequence_metrics", {}))

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                metric_card("Seq F1", f"{seq_metrics.get('f1', float('nan')):.3f}", "Held-out sequence F1")
            with e2:
                metric_card("ROC-AUC", f"{seq_metrics.get('roc_auc', float('nan')):.3f}", "Sequence ROC-AUC")
            with e3:
                metric_card("PR-AUC", f"{seq_metrics.get('pr_auc', float('nan')):.3f}", "Sequence PR-AUC")
            with e4:
                metric_card(
                    "Normal FPR",
                    f"{seq_metrics.get('false_positive_rate', float('nan')):.3f}",
                    "False-positive rate on held-out data",
                )

            if not ml_eval_scenario_df.empty:
                left_eval, right_eval = st.columns([1.0, 1.25])
                with left_eval:
                    render_chart_shell("Scenario Summary", "A compact benchmark rollup by attack family instead of a long per-run sheet.")
                    render_section_title("Scenario Summary", "Coverage by attack family")
                    scenario_rollup_df = build_scenario_rollup_table(ml_eval_scenario_df)
                    st.dataframe(
                        prepare_df_for_streamlit(
                            scenario_rollup_df if not scenario_rollup_df.empty else ml_eval_scenario_df,
                            simplify=False,
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                with right_eval:
                    render_chart_shell(
                        "Score Distribution",
                        "Normalized score curves make the rare anomalous class visible instead of letting normal traffic dominate the chart.",
                    )
                    render_section_title("Score Distribution", "Normal vs anomalous sequence-score spread")
                    render_score_distribution_chart(ml_eval_seq_df)

            if not ml_eval_row_df.empty and {"scenario", "label", "pred_label"}.issubset(ml_eval_row_df.columns):
                render_chart_shell("Row-Level Recall", "Point-wise coverage helps show how well hybrid logic catches sparse or short-lived attacks.")
                render_section_title("Row-Level Recall", "Point-wise coverage for hybrid and persistent alerts")
                row_summary_rows: list[dict[str, Any]] = []
                for scenario_name, g in ml_eval_row_df.groupby("scenario", dropna=False):
                    labels = pd.to_numeric(g["label"], errors="coerce").fillna(0).astype(int)
                    preds = pd.to_numeric(g["pred_label"], errors="coerce").fillna(0).astype(int)
                    positive_rows = int(labels.sum())
                    detected_rows = int(((labels == 1) & (preds == 1)).sum())
                    row_summary_rows.append(
                        {
                            "scenario": scenario_name,
                            "positive_rows": positive_rows,
                            "detected_rows": detected_rows,
                            "row_recall": float(detected_rows / max(positive_rows, 1)),
                        }
                    )
                row_summary = pd.DataFrame(row_summary_rows)
                st.dataframe(prepare_df_for_streamlit(row_summary), width="stretch", hide_index=True)

        current_left, current_right = st.columns([1.2, 1.0])
        with current_left:
            if isinstance(ml_scores_df, pd.DataFrame) and not ml_scores_df.empty:
                if "anomaly_score" in ml_scores_df.columns and "timestamp" in ml_scores_df.columns:
                    chart_df = ml_scores_df[["timestamp", "anomaly_score"]].copy()
                    chart_df["timestamp"] = pd.to_numeric(chart_df["timestamp"], errors="coerce")
                    chart_df["anomaly_score"] = pd.to_numeric(chart_df["anomaly_score"], errors="coerce")
                    chart_df = chart_df.dropna().sort_values("timestamp")

                    if not chart_df.empty:
                        render_chart_shell("Current Run Trend", "Sequence anomaly score over time for the active traffic run.")
                        render_section_title("Current Run Trend", "Sequence anomaly score over time")
                        st.line_chart(chart_df.set_index("timestamp"), height=250)
            else:
                render_info_banner("Run a scenario to populate current-run ML trends.", "info")

        with current_right:
            if isinstance(ml_anoms_df, pd.DataFrame) and not ml_anoms_df.empty:
                render_chart_shell("Current ML Alerts", "Persistent anomalies that survived thresholding and temporal smoothing.")
                render_section_title("Current ML Alerts", "Persistent anomaly output")
                st.dataframe(
                    prepare_df_for_streamlit(ml_anoms_df, simplify=True),
                    width="stretch",
                    hide_index=True,
                )
            elif use_ml and ml_artifacts is not None:
                render_chart_shell("Current ML Alerts", "Persistent anomalies that survived thresholding and temporal smoothing.")
                render_section_title("Current ML Alerts", "Persistent anomaly output")
                render_info_banner("No persistent ML anomalies for the current run.", "success")

        with st.expander("Advanced ML Details", expanded=False):
            if ml_meta:
                st.json(ml_meta)

            try:
                feature_df = build_feature_frame(processed_df)
            except Exception as e:
                feature_df = None
                st.info(f"Feature frame could not be built: {e}")

            if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
                st.markdown("#### Feature Sample")
                show_dataframe(feature_df, max_rows=100, simplify=True, hide_index=True)

            if isinstance(ml_scores_df, pd.DataFrame) and not ml_scores_df.empty:
                st.markdown("#### Sequence Scores")
                show_dataframe(ml_scores_df, max_rows=100, simplify=False, hide_index=True)

            if show_ml_engineering and ml_artifacts is not None:
                details: dict[str, Any] = {
                    "working_dir": os.getcwd(),
                    "artifacts_dir_exists": os.path.exists(os.path.join(os.getcwd(), "artifacts")),
                    "model_file_exists": os.path.exists(os.path.join(os.getcwd(), "artifacts", "lstm_autoencoder.pt")),
                    "scaler_file_exists": os.path.exists(os.path.join(os.getcwd(), "artifacts", "scaler.pkl")),
                    "meta_file_exists": os.path.exists(os.path.join(os.getcwd(), "artifacts", "meta.pkl")),
                    "artifact_type": type(ml_artifacts).__name__,
                }

                if hasattr(ml_artifacts, "__dataclass_fields__"):
                    details["artifact_fields"] = list(getattr(ml_artifacts, "__dataclass_fields__").keys())
                elif isinstance(ml_artifacts, dict):
                    details["artifact_fields"] = list(ml_artifacts.keys())

                st.json(details)

with tab_threats:
    render_threat_library_page()

with tab_settings:
    render_section_title("Settings", "Advanced configuration, diagnostics, and reference data")
    render_page_note(
        "<strong>Settings is the technical back room.</strong> It keeps the advanced controls and raw reference material available "
        "without forcing non-technical viewers to see them first."
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("#### Last Executed Run")
        st.write(f"**Mode:** {display_source_mode}")
        if display_source_mode == "Live OpenSky Snapshot":
            st.write(f"**BBox:** lat [{lat_min}, {lat_max}] / lon [{lon_min}, {lon_max}]")
            st.write(f"**Live Refresh:** {'Running' if live_auto_refresh_enabled else 'Paused'}")
            st.write(f"**Refresh Interval:** {refresh_interval_sec} sec")
            st.write(f"**Refresh Backend:** {refresh_backend}")
        else:
            st.write(f"**Aircraft Count:** {aircraft_count}")
            st.write(f"**Time Steps:** {time_steps}")
            st.write(f"**Scenario:** {display_attack_name}")
            st.write(f"**Window:** {display_attack_start} to {display_attack_end}")

        if controls_changed_since_run:
            st.markdown("#### Pending Controls")
            st.write(f"**Mode:** {source_mode}")
            if source_mode == "Live OpenSky Snapshot":
                st.write(f"**BBox:** lat [{lat_min}, {lat_max}] / lon [{lon_min}, {lon_max}]")
            else:
                st.write(f"**Scenario:** {attack_name}")
                st.write(f"**Window:** {attack_start} to {attack_end}")

    with s2:
        st.markdown("#### Display")
        st.write(f"**Marker Size:** {point_radius}")
        st.write(f"**Marker Opacity:** {point_opacity}")
        st.write(f"**Map Height:** {map_height}")
        st.write(f"**ML Enabled:** {'Yes' if use_ml else 'No'}")

    with st.expander("Advanced Data Tables", expanded=False):
        if not processed_df.empty:
            st.markdown("#### Processed Data")
            show_dataframe(processed_df, max_rows=200, simplify=True, hide_index=True)

        if not history_df.empty:
            st.markdown("#### Flight History")
            show_dataframe(history_df, max_rows=300, simplify=True, hide_index=True)

        if not raw_df.empty:
            st.markdown("#### Raw Data")
            show_dataframe(raw_df, max_rows=200, simplify=False, hide_index=True)

        st.markdown("#### Column Help")
        st.dataframe(column_help_table(), width="stretch", hide_index=True)

    with st.expander("ML Training Controls", expanded=False):
        st.write(f"**Sequence Length:** {seq_len}")
        st.write(f"**Max Time Gap:** {max_dt_sec}")
        st.write(f"**Min Quality:** {min_quality}")
        st.write(f"**Epochs:** {epochs}")
        st.write(f"**Threshold Percentile:** {threshold_percentile}")
        st.write(f"**Persistence:** {persistence_k}/{persistence_m}")

        if train_clicked and processed_df.empty:
            render_info_banner("Run a scenario before training.", "warn")

        if train_clicked and not processed_df.empty:
            render_info_banner("Training request executed.", "success")
