from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

FEATURE_COLS: List[str] = [
    "implied_speed_mps",
    "velocity",
    "accel_mps2",
    "vert_rate_mps",
    "turn_rate_dps",
    "speed_mismatch_mps",
]

SANITY_CAPS = {
    "implied_speed_mps": (0.0, 400.0),
    "velocity": (0.0, 400.0),
    "accel_mps2": (-30.0, 30.0),
    "vert_rate_mps": (-80.0, 80.0),
    "turn_rate_dps": (-30.0, 30.0),
    "speed_mismatch_mps": (0.0, 300.0),
}

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


def _ensure_numeric(df: pd.DataFrame, cols: List[Optional[str]]) -> None:
    """Coerce selected columns to numeric, skipping Nones."""
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
    impute: str = "median",  # "median" or "zero"
    add_quality: bool = True,
) -> pd.DataFrame:
    """
    ML-ready feature frame.

    Key point:
    - If df_proc has only single snapshots per aircraft (common in Live snapshot mode),
      derived quantities will be NaN and must be imputed.
      Live mode MUST accumulate history across refreshes to make dt/derived kinematics real.
    """
    if df_proc is None or df_proc.empty:
        return pd.DataFrame()

    d = df_proc.copy()

    if id_col not in d.columns:
        raise ValueError(f"build_feature_frame: missing id_col '{id_col}'")
    if ts_col not in d.columns:
        raise ValueError(f"build_feature_frame: missing ts_col '{ts_col}'")

    d[id_col] = d[id_col].astype(str)
    d[ts_col] = pd.to_numeric(d[ts_col], errors="coerce")
    d = d.dropna(subset=[id_col, ts_col]).copy()
    d[ts_col] = d[ts_col].astype(int)

    # locate likely columns
    vel_src = _coalesce_columns(d, ["velocity", "speed_mps", "gs_mps"])
    vr_src = _coalesce_columns(d, ["vert_rate_mps", "vs_mps", "vertical_rate", "vertical_rate_mps"])
    accel_src = _coalesce_columns(d, ["accel_mps2"])
    turn_src = _coalesce_columns(d, ["turn_rate_dps"])
    implied_src = _coalesce_columns(d, ["implied_speed_mps"])
    hdg_src = _coalesce_columns(d, [hdg_col, "true_track", "track", "heading_deg"])
    lat_src = _coalesce_columns(d, [lat_col, "lat"])
    lon_src = _coalesce_columns(d, [lon_col, "lon"])

    _ensure_numeric(d, [vel_src, vr_src, accel_src, turn_src, implied_src, hdg_src, lat_src, lon_src])

    # init canonical columns as NaN
    for c in FEATURE_COLS:
        if c not in d.columns:
            d[c] = np.nan

    # copy sources into canonical columns
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

    d = d.sort_values([id_col, ts_col]).reset_index(drop=True)

    # compute implied_speed if missing and lat/lon available
    if d["implied_speed_mps"].isna().any() and lat_src and lon_src:
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

        m = d["implied_speed_mps"].isna().to_numpy()
        d.loc[m, "implied_speed_mps"] = implied[m]

    # compute accel if missing
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

        m = d["accel_mps2"].isna().to_numpy()
        d.loc[m, "accel_mps2"] = accel[m]

    # compute turn rate if missing and heading exists
    if d["turn_rate_dps"].isna().any() and hdg_src:
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

        m = d["turn_rate_dps"].isna().to_numpy()
        d.loc[m, "turn_rate_dps"] = tr[m]

    # missingness indicators BEFORE imputation (exclude mismatch; computed later)
    for c in FEATURE_COLS:
        if c == "speed_mismatch_mps":
            continue
        d[f"{c}_is_missing"] = d[c].isna().astype(np.float32)

    # clip + track clipping
    was_clipped = {}
    for c in FEATURE_COLS:
        if c == "speed_mismatch_mps":
            continue
        lo, hi = SANITY_CAPS[c]
        pre = d[c]
        was_clipped[c] = ((pre < lo) | (pre > hi)) & pre.notna()
        d[c] = _clip_series(pre, lo, hi)

    # smoothing
    if smooth and smooth_window >= 3:
        for c in FEATURE_COLS:
            if c == "speed_mismatch_mps":
                continue
            d[c] = d.groupby(id_col, dropna=False)[c].transform(
                lambda s: s.rolling(window=smooth_window, min_periods=1).median()
            )

    # impute
    if impute == "median":
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

    # compute mismatch AFTER imputation so it matches final values
    d["speed_mismatch_mps"] = (d["implied_speed_mps"] - d["velocity"]).abs()
    lo, hi = SANITY_CAPS["speed_mismatch_mps"]
    d["speed_mismatch_mps"] = d["speed_mismatch_mps"].clip(lower=lo, upper=hi)

    # mismatch missingness is not meaningful after imputation
    d["speed_mismatch_mps_is_missing"] = 0.0

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

        d["data_quality_score"] = (1.0 - d["missing_feature_frac"]) * (1.0 - 0.7 * d["bad_point"])
        d["data_quality_score"] = d["data_quality_score"].clip(0.0, 1.0).astype(np.float32)

    # final numeric safety (model-safe fill)
    for c in FEATURE_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    return d