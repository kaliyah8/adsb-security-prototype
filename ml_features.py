# ml_features.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

# Canonical ML features (fixed schema)
FEATURE_COLS: List[str] = [
    "implied_speed_mps",      # derived from distance/dt
    "velocity",               # feed speed (must be m/s)
    "accel_mps2",             # derived
    "vert_rate_mps",          # vertical rate (m/s)
    "turn_rate_dps",          # heading rate (deg/sec)
    "speed_mismatch_mps",     # |implied - velocity|
]

# Conservative sanity caps to prevent receiver glitches from dominating scaling/model
SANITY_CAPS = {
    "implied_speed_mps": (0.0, 400.0),       # ~0–777 knots
    "velocity":          (0.0, 400.0),
    "accel_mps2":        (-30.0, 30.0),
    "vert_rate_mps":     (-80.0, 80.0),      # ~ +/- 15700 ft/min
    "turn_rate_dps":     (-30.0, 30.0),
    "speed_mismatch_mps": (0.0, 300.0),
}

# Default column name expectations (you can override via args)
DEFAULT_ID_COL = "icao"
DEFAULT_TS_COL = "timestamp"
DEFAULT_LAT_COL = "latitude"
DEFAULT_LON_COL = "longitude"
DEFAULT_HDG_COL = "heading"


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    R = 6371000.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def _wrap_angle_deg(d: pd.Series) -> pd.Series:
    """Wrap to [-180, 180] for smallest signed difference."""
    return (d + 180.0) % 360.0 - 180.0


def _coalesce_columns(df: pd.DataFrame, targets: List[str]) -> Optional[str]:
    """Return first column name in targets that exists in df."""
    for c in targets:
        if c in df.columns:
            return c
    return None


def _clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


# FIX: accept Optional[str] in cols to satisfy type checker
def _ensure_numeric(df: pd.DataFrame, cols: List[Optional[str]]) -> None:
    for c in cols:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def build_feature_frame(
    df_proc: pd.DataFrame,
    *,
    id_col: str = DEFAULT_ID_COL,
    ts_col: str = DEFAULT_TS_COL,
    lat_col: str = DEFAULT_LAT_COL,
    lon_col: str = DEFAULT_LON_COL,
    hdg_col: str = DEFAULT_HDG_COL,
    smooth: bool = True,
    smooth_window: int = 3,
    impute: str = "median",          # "median" or "zero"
    add_quality: bool = True,
) -> pd.DataFrame:
    """
    Build a robust ML-ready feature frame from df_proc.

    Guarantees:
    - Does NOT silently create missing feature columns as 0 without tracking missingness
    - Adds *_is_missing indicators
    - Adds bad_point + data_quality_score (optional)
    - Clips to sanity caps to prevent massive spikes from dominating scaler/AE
    - Optional rolling median smoothing per aircraft
    - speed_mismatch_mps is computed AFTER smoothing/imputation so it matches final values

    Output: original df_proc columns PLUS canonical FEATURE_COLS and quality columns.
    """
    if df_proc is None or df_proc.empty:
        return pd.DataFrame()

    d = df_proc.copy()

    # Required identifiers
    if id_col not in d.columns:
        raise ValueError(f"build_feature_frame: missing id_col '{id_col}'")
    if ts_col not in d.columns:
        raise ValueError(f"build_feature_frame: missing ts_col '{ts_col}'")

    d[id_col] = d[id_col].astype(str)
    d[ts_col] = pd.to_numeric(d[ts_col], errors="coerce")
    d = d.dropna(subset=[id_col, ts_col]).copy()
    d[ts_col] = d[ts_col].astype(int)

    # Try to locate likely source columns from df_proc
    vel_src = _coalesce_columns(d, ["velocity", "speed_mps", "gs_mps"])
    vr_src = _coalesce_columns(d, ["vert_rate_mps", "vs_mps", "vertical_rate", "vertical_rate_mps"])
    accel_src = _coalesce_columns(d, ["accel_mps2"])
    turn_src = _coalesce_columns(d, ["turn_rate_dps"])
    implied_src = _coalesce_columns(d, ["implied_speed_mps"])
    hdg_src = _coalesce_columns(d, [hdg_col, "true_track", "track", "heading_deg"])
    lat_src = _coalesce_columns(d, [lat_col, "lat"])
    lon_src = _coalesce_columns(d, [lon_col, "lon"])

    # Ensure numeric types where relevant
    _ensure_numeric(d, [vel_src, vr_src, accel_src, turn_src, implied_src, hdg_src, lat_src, lon_src])

    # ---- Build canonical columns (initialize as NaN, not 0) ----
    for c in FEATURE_COLS:
        if c not in d.columns:
            d[c] = np.nan

    # Copy sources into canonical columns where available
    if vel_src:
        d["velocity"] = d[vel_src]
    if vr_src:
        d["vert_rate_mps"] = d[vr_src]
    if accel_src:
        d["accel_mps2"] = d[accel_src]
    if turn_src:
        d["turn_rate_dps"] = d[turn_src]
    if implied_src:
        d["implied_speed_mps"] = d[implied_src]

    # Sort once for per-aircraft differencing
    d = d.sort_values([id_col, ts_col]).reset_index(drop=True)

    # If implied_speed_mps missing, compute from lat/lon + dt
    if d["implied_speed_mps"].isna().any():
        if lat_src and lon_src:
            lat = d[lat_src].to_numpy(dtype=float)
            lon = d[lon_src].to_numpy(dtype=float)
            t = d[ts_col].to_numpy(dtype=float)
            ids = d[id_col].to_numpy()

            lat_prev = np.roll(lat, 1)
            lon_prev = np.roll(lon, 1)
            t_prev = np.roll(t, 1)
            ids_prev = np.roll(ids, 1)

            same_id = ids == ids_prev
            dt = np.where(same_id, t - t_prev, np.nan)

            finite_ll = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(lat_prev) & np.isfinite(lon_prev)
            dist_m = np.where(same_id & finite_ll, _haversine_m(lat_prev, lon_prev, lat, lon), np.nan)
            implied = np.where((dt > 0) & np.isfinite(dist_m), dist_m / dt, np.nan)

            missing_mask = d["implied_speed_mps"].isna().to_numpy()
            d.loc[missing_mask, "implied_speed_mps"] = implied[missing_mask]

    # If accel missing, compute from velocity + dt
    if d["accel_mps2"].isna().any():
        v = d["velocity"].to_numpy(dtype=float)
        t = d[ts_col].to_numpy(dtype=float)
        ids = d[id_col].to_numpy()

        v_prev = np.roll(v, 1)
        t_prev = np.roll(t, 1)
        ids_prev = np.roll(ids, 1)

        same_id = ids == ids_prev
        dt = np.where(same_id, t - t_prev, np.nan)
        accel = np.where((dt > 0) & np.isfinite(v) & np.isfinite(v_prev), (v - v_prev) / dt, np.nan)

        missing_mask = d["accel_mps2"].isna().to_numpy()
        d.loc[missing_mask, "accel_mps2"] = accel[missing_mask]

    # If turn_rate missing, compute from heading + dt with wrap-around
    if d["turn_rate_dps"].isna().any():
        if hdg_src:
            h = d[hdg_src].to_numpy(dtype=float)
            t = d[ts_col].to_numpy(dtype=float)
            ids = d[id_col].to_numpy()

            h_prev = np.roll(h, 1)
            t_prev = np.roll(t, 1)
            ids_prev = np.roll(ids, 1)

            same_id = ids == ids_prev
            dt = np.where(same_id, t - t_prev, np.nan)
            dh = _wrap_angle_deg(pd.Series(h - h_prev)).to_numpy(dtype=float)
            tr = np.where((dt > 0) & np.isfinite(dh) & np.isfinite(h) & np.isfinite(h_prev), dh / dt, np.nan)

            missing_mask = d["turn_rate_dps"].isna().to_numpy()
            d.loc[missing_mask, "turn_rate_dps"] = tr[missing_mask]

    # Missingness indicators BEFORE imputation (exclude mismatch; computed later)
    for c in FEATURE_COLS:
        if c == "speed_mismatch_mps":
            continue
        d[f"{c}_is_missing"] = d[c].isna().astype(np.float32)

    # Clip to sanity caps (prevents extreme spikes from dominating scaler/AE) + track true clipping
    was_clipped = {}
    for c in FEATURE_COLS:
        if c == "speed_mismatch_mps":
            continue
        lo, hi = SANITY_CAPS[c]
        pre = d[c]
        was_clipped[c] = ((pre < lo) | (pre > hi)) & pre.notna()
        d[c] = _clip_series(pre, lo, hi)

    # Optional smoothing: rolling median by aircraft
    if smooth and smooth_window >= 3:
        for c in FEATURE_COLS:
            if c == "speed_mismatch_mps":
                continue
            d[c] = d.groupby(id_col, dropna=False)[c].transform(
                lambda s: s.rolling(window=smooth_window, min_periods=1).median()
            )

    # Impute AFTER recording missingness and clipping
    if impute == "median":
        # per-aircraft median first, then global
        for c in FEATURE_COLS:
            if c == "speed_mismatch_mps":
                continue
            d[c] = d.groupby(id_col, dropna=False)[c].transform(lambda s: s.fillna(s.median()))
        for c in FEATURE_COLS:
            if c == "speed_mismatch_mps":
                continue
            med = d[c].median()
            med_val = float(med) if np.isfinite(med) else 0.0
            d[c] = d[c].fillna(med_val)
    elif impute == "zero":
        for c in FEATURE_COLS:
            if c == "speed_mismatch_mps":
                continue
            d[c] = d[c].fillna(0.0)
    else:
        raise ValueError("impute must be 'median' or 'zero'")

    # Compute mismatch LAST so it matches the FINAL (clipped/smoothed/imputed) values
    d["speed_mismatch_mps"] = (d["implied_speed_mps"] - d["velocity"]).abs()
    lo, hi = SANITY_CAPS["speed_mismatch_mps"]
    d["speed_mismatch_mps"] = d["speed_mismatch_mps"].clip(lower=lo, upper=hi)
    d["speed_mismatch_mps_is_missing"] = d["speed_mismatch_mps"].isna().astype(np.float32)

    # Quality scoring + bad point flag
    if add_quality:
        miss_cols = [f"{c}_is_missing" for c in FEATURE_COLS]
        for mc in miss_cols:
            if mc not in d.columns:
                d[mc] = 0.0

        d["missing_feature_count"] = d[miss_cols].sum(axis=1).astype(np.float32)
        d["missing_feature_frac"] = (d["missing_feature_count"] / float(len(FEATURE_COLS))).astype(np.float32)

        clipped_any = pd.Series(False, index=d.index)
        for c in FEATURE_COLS:
            if c == "speed_mismatch_mps":
                continue
            if c in was_clipped:
                clipped_any = clipped_any | was_clipped[c].fillna(False)

        d["bad_point"] = (clipped_any | (d["missing_feature_frac"] > 0.5)).astype(np.int8)

        # data_quality_score: 1 good, 0 bad
        d["data_quality_score"] = (1.0 - d["missing_feature_frac"]) * (1.0 - 0.7 * d["bad_point"])
        d["data_quality_score"] = d["data_quality_score"].clip(0.0, 1.0).astype(np.float32)

    # Final numeric guarantee + count coercions that create new NaNs
    d["post_numeric_coerce_count"] = 0
    for c in FEATURE_COLS:
        before_na = int(d[c].isna().sum())
        d[c] = pd.to_numeric(d[c], errors="coerce")
        after_na = int(d[c].isna().sum())
        new_na = max(0, after_na - before_na)
        if new_na:
            d["post_numeric_coerce_count"] += new_na

        # Model-safe fill
        d[c] = d[c].fillna(0.0)

    return d