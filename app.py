# app.py
import os
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

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

# Optional ML
try:
    from ml_pipeline import load_artifacts, score_sequences, train_lstm_autoencoder

    ML_AVAILABLE = True
    ML_IMPORT_ERROR = None
except Exception as e:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = repr(e)


# ---- .env loading (force override) ----
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
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ[k] = v
    except Exception:
        pass


_load_env_local()

MAPBOX_KEY = (os.getenv("MAPBOX_API_KEY") or "").strip()
if MAPBOX_KEY:
    setattr(pdk.settings, "mapbox_api_key", MAPBOX_KEY)

st.set_page_config(page_title="ADS-B Security Monitoring Prototype", layout="wide")
st.title("ADS-B Security Monitoring Prototype")
st.caption("Simulated + Live OpenSky snapshot refresh, physics-based detection, optional ML scoring (quality gated + persistent).")

DEFAULT_VIEW = {"lat": 34.0522, "lon": -118.2437, "zoom": 7, "pitch": 35}
DEF_LAMIN, DEF_LAMAX = 33.5, 34.5
DEF_LOMIN, DEF_LOMAX = -119.5, -117.5
LAX_LAT, LAX_LON = 33.9416, -118.4085


def _ensure_state() -> None:
    st.session_state.setdefault("df_raw", pd.DataFrame())
    st.session_state.setdefault("df_proc", pd.DataFrame())
    st.session_state.setdefault("df_feat", pd.DataFrame())
    st.session_state.setdefault("anoms", pd.DataFrame())
    st.session_state.setdefault("scores", pd.DataFrame())
    st.session_state.setdefault("attacked", None)
    st.session_state.setdefault("attack_window", None)
    st.session_state.setdefault("last_live_epoch", 0)


_ensure_state()


def _standardize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes either OpenSky state vectors or simulator output into:
      icao, timestamp (int), latitude, longitude, velocity (m/s),
      heading (deg), baro_altitude (m), vertical_rate (m/s)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # OpenSky -> internal names
    if "icao24" in d.columns and "icao" not in d.columns:
        d = d.rename(columns={"icao24": "icao"})
    if "lat" in d.columns and "latitude" not in d.columns:
        d = d.rename(columns={"lat": "latitude"})
    if "lon" in d.columns and "longitude" not in d.columns:
        d = d.rename(columns={"lon": "longitude"})
    if "true_track" in d.columns and "heading" not in d.columns:
        d = d.rename(columns={"true_track": "heading"})

    # timestamp
    if "timestamp" not in d.columns:
        if "last_contact" in d.columns:
            d["timestamp"] = d["last_contact"]
        elif "time_position" in d.columns:
            d["timestamp"] = d["time_position"]
        else:
            d["timestamp"] = np.nan

    # legacy aliases
    if "altitude" in d.columns and "baro_altitude" not in d.columns:
        d = d.rename(columns={"altitude": "baro_altitude"})
    if "speed" in d.columns and "velocity" not in d.columns:
        d = d.rename(columns={"speed": "velocity"})

    d["icao"] = d["icao"].astype(str).str.strip()
    for c in ["timestamp", "latitude", "longitude", "velocity", "baro_altitude", "heading", "vertical_rate"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["icao", "timestamp", "latitude", "longitude"]).copy()
    d = d[(d["latitude"].between(-90, 90)) & (d["longitude"].between(-180, 180))].copy()
    d["timestamp"] = d["timestamp"].astype(int)

    # ensure optional columns exist (still keep as 0 here, but ML feature builder will track missingness later)
    for c in ["velocity", "baro_altitude", "heading", "vertical_rate"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # If an older simulator format used knots/feet labels, convert to SI.
    is_sim_like = ("speed" in df.columns) or ("altitude" in df.columns)
    if is_sim_like:
        d["velocity"] = d["velocity"] * 0.514444  # knots -> m/s
        d["baro_altitude"] = d["baro_altitude"] * 0.3048  # ft -> m

    return d


@st.cache_data(ttl=10)
def _live_fetch_cached(lamin: float, lomin: float, lamax: float, lomax: float):
    return fetch_live_adsb_data(lamin, lomin, lamax, lomax)


with st.sidebar:
    st.header("Data Source")
    source = st.radio("Source", ["Simulated", "Live (OpenSky)"], index=0)

    st.header("Detection")
    det_choices = ["Physics Rules"] + (["ML (LSTM)"] if ML_AVAILABLE else [])
    mode = st.radio("Mode", det_choices, index=0)

    st.divider()
    st.header("Map Display")
    point_size = st.slider("Point size", 400, 4000, 2000, 200)
    point_opacity = st.slider("Point opacity", 50, 255, 235, 5)

    st.divider()
    st.header("Parameters")

    if source == "Simulated":
        num_aircraft = st.slider("Aircraft", 5, 80, 25, 5)
        time_steps = st.slider("Time steps", 20, 400, 120, 10)
        attack = st.selectbox("Attack", ["None", "Teleportation", "Ghost Aircraft", "GPS Spoofing"], 0)
        start_step = st.slider("Attack start step", 0, time_steps - 1, min(20, time_steps - 1), 1)
        end_step = st.slider("Attack end step", start_step, time_steps - 1, min(start_step + 5, time_steps - 1), 1)
        lat_shift = st.slider("Lat shift (deg)", 0.0, 1.0, 0.10, 0.01)
        lon_shift = st.slider("Lon shift (deg)", 0.0, 1.0, 0.10, 0.01)
        auto_refresh = False
    else:
        lamin = st.number_input("lamin", value=float(DEF_LAMIN), format="%.4f")
        lamax = st.number_input("lamax", value=float(DEF_LAMAX), format="%.4f")
        lomin = st.number_input("lomin", value=float(DEF_LOMIN), format="%.4f")
        lomax = st.number_input("lomax", value=float(DEF_LOMAX), format="%.4f")
        refresh_sec = st.slider("Refresh seconds", 5, 60, 10, 1)
        auto_refresh = st.toggle("Auto refresh", value=False)

    st.divider()

    train_clicked = False
    if mode == "ML (LSTM)" and ML_AVAILABLE:
        train_clicked = st.button("Train / Retrain LSTM")

    run_clicked = st.button("Run")

    st.caption(f"Mapbox key loaded: {bool(MAPBOX_KEY)}")


def run_pipeline() -> None:
    attacked = None
    attack_window = None

    if source == "Simulated":
        raw = generate_aircraft_data(
            num_aircraft=num_aircraft,
            time_steps=time_steps,
            center_lat=LAX_LAT,
            center_lon=LAX_LON,
        )
        icaos = sorted(raw["icao"].astype(str).unique().tolist())
        target = icaos[0] if icaos else None

        if attack == "Teleportation" and target:
            raw = inject_teleportation_attack(raw, target, start_step, end_step)
            attacked, attack_window = target, (start_step, end_step)
        elif attack == "Ghost Aircraft":
            raw = inject_ghost_aircraft_attack(raw, "GHOST1", start_step, end_step)
            attacked, attack_window = "GHOST1", (start_step, end_step)
        elif attack == "GPS Spoofing" and target:
            raw = inject_gps_spoofing_attack(raw, target, start_step, end_step, lat_shift=lat_shift, lon_shift=lon_shift)
            attacked, attack_window = target, (start_step, end_step)

        df_raw = _standardize_raw(raw)

    else:
        df_live, err = _live_fetch_cached(lamin, lomin, lamax, lomax)
        st.session_state["last_live_epoch"] = int(time.time())
        if err:
            st.error(err)
            st.session_state["df_raw"] = pd.DataFrame()
            st.session_state["df_proc"] = pd.DataFrame()
            st.session_state["df_feat"] = pd.DataFrame()
            st.session_state["anoms"] = pd.DataFrame()
            st.session_state["scores"] = pd.DataFrame()
            st.session_state["attacked"] = None
            st.session_state["attack_window"] = None
            return
        df_raw = _standardize_raw(df_live)

    if df_raw.empty:
        st.error("No data returned.")
        return

    df_proc = process_adsb_data(df_raw)

    # Build ML feature frame always (useful for debugging even in physics mode)
    df_feat = build_feature_frame(df_proc, id_col="icao", ts_col="timestamp", smooth=True, smooth_window=3, impute="median", add_quality=True)

    if mode == "Physics Rules":
        anoms = detect_anomalies(df_proc)
        scores = pd.DataFrame()
    else:
        if not ML_AVAILABLE:
            st.error("ML unavailable.")
            return

        if train_clicked:
            train_lstm_autoencoder(df_proc, seq_len=30, threshold_percentile=99.5, epochs=12, min_quality=0.7, persistence_k=3, persistence_m=5)
            st.success("Model trained and saved.")

        arts = load_artifacts()
        if arts is None:
            st.info("Model hasn’t been trained yet. Use the sidebar to train it.")
            return

        scores, anoms = score_sequences(df_proc, arts)

    st.session_state["df_raw"] = df_raw
    st.session_state["df_proc"] = df_proc
    st.session_state["df_feat"] = df_feat
    st.session_state["anoms"] = anoms if anoms is not None else pd.DataFrame()
    st.session_state["scores"] = scores if scores is not None else pd.DataFrame()
    st.session_state["attacked"] = attacked
    st.session_state["attack_window"] = attack_window


if run_clicked:
    run_pipeline()

if source == "Live (OpenSky)" and auto_refresh:
    time.sleep(int(refresh_sec))
    _live_fetch_cached.clear()
    run_pipeline()
    st.rerun()


df_proc = st.session_state.get("df_proc", pd.DataFrame())
df_feat = st.session_state.get("df_feat", pd.DataFrame())
anoms = st.session_state.get("anoms", pd.DataFrame())
scores = st.session_state.get("scores", pd.DataFrame())
attacked = st.session_state.get("attacked", None)

tab_overview, tab_map, tab_data, tab_model = st.tabs(["Overview", "Map", "Data", "Model"])

with tab_overview:
    st.subheader("System Overview")
    if df_proc.empty:
        st.info("Click Run in the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df_proc):,}")
        c2.metric("Aircraft", f"{df_proc['icao'].nunique():,}")
        c3.metric("Anomalies", f"{len(anoms):,}")
        c4.metric("Highlighted", attacked or "None")

with tab_map:
    st.subheader("Map View (High Contrast)")

    if df_proc.empty:
        st.info("Run the pipeline first.")
    else:
        ts_vals = np.sort(pd.to_numeric(df_proc["timestamp"], errors="coerce").dropna().unique())
        selected_ts = int(ts_vals.max())
        if len(ts_vals) > 1:
            selected_ts = st.slider("Timestamp", int(ts_vals.min()), int(ts_vals.max()), int(ts_vals.max()), 1)

        frame = df_proc[df_proc["timestamp"] == selected_ts].copy()
        anoms_t = (
            anoms[anoms["timestamp"] == selected_ts].copy()
            if (not anoms.empty and "timestamp" in anoms.columns)
            else pd.DataFrame()
        )

        if frame.empty:
            st.warning("No points for that timestamp.")
        else:
            plot = frame[["icao", "latitude", "longitude"]].copy()
            plot["label"] = "Normal"

            # Normal: GREEN
            plot["r"], plot["g"], plot["b"], plot["a"] = 0, 200, 0, int(point_opacity)

            # Anomaly: RED
            if not anoms_t.empty and "icao" in anoms_t.columns:
                bad = set(anoms_t["icao"].astype(str).unique().tolist())
                m = plot["icao"].astype(str).isin(bad)
                plot.loc[m, ["r", "g", "b", "a"]] = [255, 0, 0, int(point_opacity)]
                plot.loc[m, "label"] = "Anomaly"

            # Highlighted (attack target in sim): YELLOW
            if attacked:
                m2 = plot["icao"].astype(str) == str(attacked)
                plot.loc[m2, ["r", "g", "b", "a"]] = [255, 255, 0, int(point_opacity)]
                plot.loc[m2, "label"] = "Highlighted"

            view = pdk.ViewState(
                latitude=float(plot["latitude"].mean()),
                longitude=float(plot["longitude"].mean()),
                zoom=float(DEFAULT_VIEW["zoom"]),
                pitch=float(DEFAULT_VIEW["pitch"]),
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=plot,
                get_position=["longitude", "latitude"],
                get_fill_color=["r", "g", "b", "a"],
                get_radius=int(point_size),
                pickable=True,
                stroked=True,
                get_line_color=[0, 0, 0, 220],
                line_width_min_pixels=1,
            )

            map_style = "mapbox://styles/mapbox/light-v11"

            tooltip_obj: Any = {"text": "{icao}\n{label}"}
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                map_style=map_style,
                tooltip=cast(Any, tooltip_obj),
            )

            st.pydeck_chart(deck, use_container_width=True)

with tab_data:
    st.subheader("Tables")
    if df_proc.empty:
        st.info("Run first.")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Anomalies (top 200)**")
            if anoms.empty:
                st.write("None.")
            else:
                cols = [c for c in ["timestamp", "icao", "anomaly_type", "anomaly_score", "anom_count_last_m"] if c in anoms.columns]
                st.dataframe(anoms[cols].head(200), use_container_width=True)
        with right:
            st.markdown("**Processed (top 200)**")
            st.dataframe(df_proc.head(200), use_container_width=True)

with tab_model:
    st.subheader("Model / ML")
    if not ML_AVAILABLE:
        st.warning("ML unavailable.")
        st.code(ML_IMPORT_ERROR or "Unknown")
    else:
        arts = load_artifacts()
        if arts is None:
            st.info("Model hasn’t been trained yet. Use the sidebar to train it.")
        else:
            st.success(
                f"Artifacts loaded (seq_len={arts.seq_len}, threshold={arts.threshold:.6f}, "
                f"min_quality={arts.min_quality}, persistence={arts.persistence_k}/{arts.persistence_m})"
            )
            st.write("Feature columns:", arts.feature_cols)

        if not df_feat.empty:
            st.markdown("**Feature / Quality sample (top 50)**")
            show_cols = [c for c in ["timestamp", "icao", "data_quality_score", "bad_point"] if c in df_feat.columns] + [
                c for c in df_feat.columns if c in ["implied_speed_mps", "velocity", "accel_mps2", "vert_rate_mps", "turn_rate_dps", "speed_mismatch_mps"]
            ]
            st.dataframe(df_feat[show_cols].head(50), use_container_width=True)

        if not scores.empty:
            st.markdown("**Sequence scores (top 200 by anomaly_score)**")
            st.dataframe(scores.sort_values("anomaly_score", ascending=False).head(200), use_container_width=True)