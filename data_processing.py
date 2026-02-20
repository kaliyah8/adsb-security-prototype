# data_processing.py
import numpy as np
import pandas as pd
from utils import haversine_distance

def _wrap_deg(d: float) -> float:
    """Wrap delta heading into [-180, 180]."""
    x = (d + 180.0) % 360.0 - 180.0
    return float(x)

def process_adsb_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expects standardized columns:
      icao, timestamp (int), latitude, longitude, velocity (m/s), heading (deg),
      baro_altitude (m), vertical_rate (m/s) optional.
    Produces per-aircraft deltas and implied kinematics.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    d = df_raw.copy()

    # required cols
    for c in ["icao", "timestamp", "latitude", "longitude"]:
        if c not in d.columns:
            raise ValueError(f"Missing required column: {c}")

    # fill optional
    for c in ["velocity", "heading", "baro_altitude", "vertical_rate"]:
        if c not in d.columns:
            d[c] = 0.0

    d["icao"] = d["icao"].astype(str).str.strip()
    d["timestamp"] = pd.to_numeric(d["timestamp"], errors="coerce").astype("Int64")
    d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
    d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
    d["velocity"] = pd.to_numeric(d["velocity"], errors="coerce").fillna(0.0)
    d["heading"] = pd.to_numeric(d["heading"], errors="coerce").fillna(0.0)
    d["baro_altitude"] = pd.to_numeric(d["baro_altitude"], errors="coerce").fillna(0.0)
    d["vertical_rate"] = pd.to_numeric(d["vertical_rate"], errors="coerce").fillna(0.0)

    d = d.dropna(subset=["icao", "timestamp", "latitude", "longitude"]).copy()
    d["timestamp"] = d["timestamp"].astype(int)

    d = d.sort_values(["icao", "timestamp"]).reset_index(drop=True)

    # previous values per icao
    d["prev_latitude"] = d.groupby("icao")["latitude"].shift(1)
    d["prev_longitude"] = d.groupby("icao")["longitude"].shift(1)
    d["prev_velocity"] = d.groupby("icao")["velocity"].shift(1)
    d["prev_heading"] = d.groupby("icao")["heading"].shift(1)
    d["prev_altitude"] = d.groupby("icao")["baro_altitude"].shift(1)
    d["prev_timestamp"] = d.groupby("icao")["timestamp"].shift(1)

    d["delta_t"] = (d["timestamp"] - d["prev_timestamp"]).astype(float)
    d.loc[d["delta_t"] <= 0, "delta_t"] = np.nan

    # distance moved (meters)
    d["distance_m"] = d.apply(
        lambda r: haversine_distance(r["prev_latitude"], r["prev_longitude"], r["latitude"], r["longitude"])
        if pd.notna(r["prev_latitude"]) and pd.notna(r["delta_t"])
        else np.nan,
        axis=1,
    )

    # implied ground speed (m/s)
    d["implied_speed_mps"] = d["distance_m"] / d["delta_t"]

    # implied acceleration (m/s^2)
    d["accel_mps2"] = (d["velocity"] - d["prev_velocity"]) / d["delta_t"]

    # implied vertical speed (m/s)
    d["vert_rate_mps"] = (d["baro_altitude"] - d["prev_altitude"]) / d["delta_t"]

    # turn rate (deg/s)
    d["delta_heading_deg"] = (d["heading"] - d["prev_heading"]).apply(lambda x: _wrap_deg(x) if pd.notna(x) else np.nan)
    d["turn_rate_dps"] = d["delta_heading_deg"] / d["delta_t"]

    # Quality mask: only rows with previous point
    d["has_prev"] = d["prev_timestamp"].notna()

    return d