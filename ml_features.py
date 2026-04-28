from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

FEATURE_COLS: List[str] = [
    "implied_speed_mps",
    "velocity",
    "accel_mps2",
    "vert_rate_mps",
    "turn_rate_dps",
    "speed_mismatch_mps",
    "altitude_m",
    "heading_delta_deg",
    "yaw_rate_dps",
    "speed_delta_mps",
    "relative_speed_to_median_mps",
    "velocity_window_mean",
    "velocity_window_std",
    "accel_window_mean",
    "accel_window_std",
]

SANITY_CAPS = {
    "implied_speed_mps": (0.0, 400.0),
    "velocity": (0.0, 400.0),
    "accel_mps2": (-30.0, 30.0),
    "vert_rate_mps": (-80.0, 80.0),
    "turn_rate_dps": (-30.0, 30.0),
    "speed_mismatch_mps": (0.0, 300.0),
    "altitude_m": (0.0, 15000.0),
    "heading_delta_deg": (-180.0, 180.0),
    "yaw_rate_dps": (-30.0, 30.0),
    "speed_delta_mps": (-80.0, 80.0),
    "relative_speed_to_median_mps": (0.0, 250.0),
    "velocity_window_mean": (0.0, 400.0),
    "velocity_window_std": (0.0, 120.0),
    "accel_window_mean": (-30.0, 30.0),
    "accel_window_std": (0.0, 30.0),
    "track_age_sec": (0.0, 3600.0),
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


def _safe_group_median_fill(series: pd.Series) -> pd.Series:
    """
    Fill NaNs with that group's median.
    If the entire group is NaN, leave as NaN for later global fill.
    This avoids runtime warnings from nanmedian on empty groups.
    """
    non_na = series.dropna()
    if non_na.empty:
        return series
    med = float(non_na.median())
    return series.fillna(med)


def _rolling_group_stat(
    d: pd.DataFrame,
    id_col: str,
    value_col: str,
    window: int,
    stat: str,
) -> pd.Series:
    grouped = d.groupby(id_col, dropna=False)[value_col]
    roll = grouped.rolling(window=window, min_periods=1)
    if stat == "mean":
        return roll.mean().reset_index(level=0, drop=True)
    if stat == "std":
        return roll.std().reset_index(level=0, drop=True).fillna(0.0)
    raise ValueError(f"Unsupported rolling stat: {stat}")


def _impute_feature_block(
    d: pd.DataFrame,
    id_col: str,
    cols: List[str],
    strategy: str,
    neighbors: int,
) -> pd.DataFrame:
    if not cols:
        return d

    if strategy == "median":
        for c in cols:
            d[c] = d.groupby(id_col, dropna=False)[c].transform(_safe_group_median_fill)
            non_na = d[c].dropna()
            med_val = float(non_na.median()) if not non_na.empty else 0.0
            d[c] = d[c].fillna(med_val)
        return d

    if strategy == "zero":
        d.loc[:, cols] = d.loc[:, cols].fillna(0.0)
        return d

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=max(2, int(neighbors)), weights="distance")
    elif strategy == "iterative":
        imputer = IterativeImputer(random_state=42, max_iter=10, sample_posterior=False)
    else:
        raise ValueError("impute must be 'median', 'zero', 'knn', or 'iterative'")

    d.loc[:, cols] = imputer.fit_transform(d.loc[:, cols].to_numpy(dtype=float))

    for c in cols:
        non_na = pd.to_numeric(d[c], errors="coerce").dropna()
        fallback = float(non_na.median()) if not non_na.empty else 0.0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(fallback)

    return d


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
    impute: str = "median",  # "median", "zero", "knn", or "iterative"
    impute_neighbors: int = 5,
    add_quality: bool = True,
) -> pd.DataFrame:
    """
    ML-ready feature frame.

    Key points:
    - Derived motion features are built from per-aircraft time history.
    - track_age_sec is added as an auxiliary explainability / hybrid-detection field.
      It is NOT part of FEATURE_COLS by default.
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
    alt_src = _coalesce_columns(d, ["baro_altitude", "altitude_m", "geo_altitude", "altitude"])
    lat_src = _coalesce_columns(d, [lat_col, "lat"])
    lon_src = _coalesce_columns(d, [lon_col, "lon"])

    _ensure_numeric(d, [vel_src, vr_src, accel_src, turn_src, implied_src, hdg_src, alt_src, lat_src, lon_src])

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
    if alt_src:
        d["altitude_m"] = d[alt_src]

    d = d.sort_values([id_col, ts_col]).reset_index(drop=True)
    epsilon = 1e-6

    # auxiliary feature for explainability / ghost handling
    first_seen_ts = d.groupby(id_col, dropna=False)[ts_col].transform("min")
    d["track_age_sec"] = (d[ts_col] - first_seen_ts).astype(float)
    d["track_age_sec"] = d["track_age_sec"].clip(
        lower=SANITY_CAPS["track_age_sec"][0],
        upper=SANITY_CAPS["track_age_sec"][1],
    )

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
        implied = np.full_like(dt, np.nan, dtype=float)
        valid = (dt > epsilon) & np.isfinite(dist_m)
        np.divide(dist_m, dt, out=implied, where=valid)

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
        accel = np.full_like(dt, np.nan, dtype=float)
        valid = (dt > epsilon) & np.isfinite(v) & np.isfinite(v_prev)
        np.divide(v - v_prev, dt, out=accel, where=valid)

        m = d["accel_mps2"].isna().to_numpy()
        d.loc[m, "accel_mps2"] = accel[m]

    if d["speed_delta_mps"].isna().any():
        v = d["velocity"].to_numpy(dtype=float)
        t = d[ts_col].to_numpy(dtype=float)
        ids = d[id_col].to_numpy()

        v_prev = np.roll(v, 1)
        t_prev = np.roll(t, 1)
        ids_prev = np.roll(ids, 1)

        same_id = ids == ids_prev
        dt = np.where(same_id, t - t_prev, np.nan)
        speed_delta = np.where((dt > epsilon) & np.isfinite(v) & np.isfinite(v_prev), v - v_prev, np.nan)

        m = d["speed_delta_mps"].isna().to_numpy()
        d.loc[m, "speed_delta_mps"] = speed_delta[m]

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
        tr = np.full_like(dt, np.nan, dtype=float)
        valid = (dt > epsilon) & np.isfinite(dh) & np.isfinite(h) & np.isfinite(h_prev)
        np.divide(dh, dt, out=tr, where=valid)

        m = d["turn_rate_dps"].isna().to_numpy()
        d.loc[m, "turn_rate_dps"] = tr[m]

    if hdg_src:
        h = d[hdg_src].to_numpy(dtype=float)
        ids = d[id_col].to_numpy()
        h_prev = np.roll(h, 1)
        ids_prev = np.roll(ids, 1)
        same_id = ids == ids_prev
        heading_delta = np.where(
            same_id & np.isfinite(h) & np.isfinite(h_prev),
            _wrap_angle_deg(pd.Series(h - h_prev)).to_numpy(dtype=float),
            np.nan,
        )
        m = d["heading_delta_deg"].isna().to_numpy()
        d.loc[m, "heading_delta_deg"] = heading_delta[m]

    if d["yaw_rate_dps"].isna().any():
        d["yaw_rate_dps"] = d["turn_rate_dps"]

    if d["relative_speed_to_median_mps"].isna().any():
        speed_median = d.groupby(ts_col, dropna=False)["velocity"].transform("median")
        d["relative_speed_to_median_mps"] = (d["velocity"] - speed_median).abs()

    window = max(3, int(smooth_window) + 2)
    d["velocity_window_mean"] = _rolling_group_stat(d, id_col, "velocity", window, "mean")
    d["velocity_window_std"] = _rolling_group_stat(d, id_col, "velocity", window, "std")
    d["accel_window_mean"] = _rolling_group_stat(d, id_col, "accel_mps2", window, "mean")
    d["accel_window_std"] = _rolling_group_stat(d, id_col, "accel_mps2", window, "std")

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
    impute_cols = [c for c in FEATURE_COLS if c != "speed_mismatch_mps"]
    d = _impute_feature_block(d, id_col, impute_cols, impute, impute_neighbors)

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

    # final numeric safety
    for c in FEATURE_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    d["track_age_sec"] = pd.to_numeric(d["track_age_sec"], errors="coerce").fillna(0.0)

    return d
