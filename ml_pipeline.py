from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, List, cast, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_processing import process_adsb_data
from data_simulation import generate_aircraft_data
from lstm_autoencoder import LSTMAutoencoder
from ml_features import FEATURE_COLS, build_feature_frame

ARTIFACT_DIR = "artifacts"
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "lstm_autoencoder.pt")
META_PATH = os.path.join(ARTIFACT_DIR, "meta.pkl")


@dataclass
class Artifacts:
    seq_len: int
    threshold: float
    feature_cols: List[str]
    scaler: StandardScaler
    model: LSTMAutoencoder

    # gating + persistence metadata
    min_quality: float
    persistence_k: int
    persistence_m: int
    max_dt_sec: int

    # hybrid ghost handling metadata
    ghost_birth_grace_sec: int
    ghost_age_window_sec: int

    # preprocessing + scoring metadata
    smooth_window: int
    impute_strategy: str
    impute_neighbors: int
    feature_score_center: float
    feature_score_scale: float
    combined_score_weight: float
    reconstruction_threshold: float
    feature_context_threshold: float


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _require_df(obj: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return obj


def _select_cols_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.loc[:, cols].copy()


def _fit_scaler(df_in: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    df = _require_df(df_in)
    X = _select_cols_df(df, feature_cols).to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def _transform_with_scaler(df_in: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> pd.DataFrame:
    df = _require_df(df_in).copy()
    X = _select_cols_df(df, feature_cols).to_numpy(dtype=np.float32)
    Xs = scaler.transform(X)

    out = df.copy()
    for j, c in enumerate(feature_cols):
        out[c] = Xs[:, j]
    return out


def _make_sequences(
    df_in: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    id_col: str = "icao",
    ts_col: str = "timestamp",
    max_dt_sec: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences per aircraft sorted by timestamp.

    Returns:
        X: (N, seq_len, F)
        keys: (N, 2) with [icao, last_timestamp] for each sequence

    Rules:
    - Drops duplicate (icao, timestamp)
    - Only builds sequences where all internal gaps satisfy 0 < dt <= max_dt_sec
    - For a sequence of length seq_len, there are seq_len - 1 internal dt gaps
    """
    df = _require_df(df_in).copy()

    for c in [id_col, ts_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing ML feature column: {c}")

    df[id_col] = df[id_col].astype(str)
    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce").fillna(0).astype(int)

    df = df.sort_values([id_col, ts_col]).reset_index(drop=True)
    df = df.drop_duplicates(subset=[id_col, ts_col], keep="last").reset_index(drop=True)

    sequences: List[np.ndarray] = []
    keys: List[Tuple[str, int]] = []

    for icao, g in df.groupby(id_col, sort=False):
        g = cast(pd.DataFrame, g).copy()
        if len(g) < seq_len:
            continue

        g["dt"] = g[ts_col].diff().astype(float)
        g["dt_ok"] = ((g["dt"] > 0) & (g["dt"] <= float(max_dt_sec))).astype(int)

        feats = _select_cols_df(g, feature_cols).to_numpy(dtype=np.float32)
        dt_ok = g["dt_ok"].to_numpy(dtype=int)
        ts_vals = g[ts_col].to_numpy(dtype=int)

        for i in range(seq_len - 1, len(g)):
            start = i - seq_len + 1
            window = feats[start : i + 1]

            internal_dt_ok = dt_ok[start + 1 : i + 1]

            if internal_dt_ok.shape[0] != seq_len - 1:
                continue
            if not np.all(internal_dt_ok == 1):
                continue

            sequences.append(window)
            keys.append((str(icao), int(ts_vals[i])))

    if not sequences:
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.zeros((0, 2), dtype=object),
        )

    X = np.stack(sequences, axis=0)
    K = np.array(keys, dtype=object)
    return X, K


def _time_split(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values(ts_col).copy()
    if len(d) < 10:
        return d, d.iloc[0:0].copy()

    split_idx = int(len(d) * train_frac)
    train = d.iloc[:split_idx].copy()
    val = d.iloc[split_idx:].copy()
    return train, val


def _robust_threshold(errs: np.ndarray, percentile: float, mad_scale: float) -> Tuple[float, float, float]:
    errs = np.asarray(errs, dtype=float)
    if errs.size == 0:
        raise ValueError("Cannot compute threshold from empty error array.")

    percentile_thr = float(np.percentile(errs, percentile))

    med = float(np.median(errs))
    mad = float(np.median(np.abs(errs - med)))
    robust_sigma = 1.4826 * mad
    mad_thr = med + float(mad_scale) * robust_sigma

    final_thr = max(percentile_thr, mad_thr)
    return percentile_thr, float(mad_thr), float(final_thr)


def _terminal_feature_score(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3 or X.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.mean(np.abs(X[:, -1, :]), axis=1).astype(np.float32)


def _combine_scores(
    recon_err: np.ndarray,
    feature_score: np.ndarray,
    center: float,
    scale: float,
    weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    recon = np.asarray(recon_err, dtype=np.float32)
    feat = np.asarray(feature_score, dtype=np.float32)
    denom = max(float(scale), 1e-6)
    context = np.clip((feat - float(center)) / denom, a_min=0.0, a_max=None).astype(np.float32)
    combined = recon + float(weight) * context
    return combined.astype(np.float32), context


def _calibration_scores(
    scaler: StandardScaler,
    model: LSTMAutoencoder,
    feature_cols: List[str],
    *,
    seq_len: int,
    max_dt_sec: int,
    min_quality: float,
    smooth_window: int,
    impute: str,
    impute_neighbors: int,
    feature_score_center: float,
    feature_score_scale: float,
    combined_score_weight: float,
    calibration_seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined_scores: List[np.ndarray] = []
    recon_scores: List[np.ndarray] = []
    context_scores: List[np.ndarray] = []

    for seed in calibration_seeds:
        df_raw = generate_aircraft_data(seed=int(seed))
        df_proc = process_adsb_data(df_raw)
        df_feat = build_feature_frame(
            df_proc,
            id_col="icao",
            ts_col="timestamp",
            smooth=True,
            smooth_window=smooth_window,
            impute=impute,
            impute_neighbors=impute_neighbors,
            add_quality=True,
        )
        if "data_quality_score" in df_feat.columns:
            df_feat = df_feat[
                (df_feat["data_quality_score"] >= float(min_quality)) & (df_feat.get("bad_point", 0) == 0)
            ].copy()
        if df_feat.empty:
            continue

        df_scaled = _transform_with_scaler(df_feat, scaler, feature_cols)
        X, _ = _make_sequences(
            df_scaled,
            feature_cols,
            seq_len,
            id_col="icao",
            ts_col="timestamp",
            max_dt_sec=max_dt_sec,
        )
        if X.shape[0] == 0:
            continue

        recon = model.reconstruction_error(X).astype(np.float32)
        raw_feature = _terminal_feature_score(X)
        combined, context = _combine_scores(
            recon,
            raw_feature,
            center=feature_score_center,
            scale=feature_score_scale,
            weight=combined_score_weight,
        )
        combined_scores.append(combined)
        recon_scores.append(recon)
        context_scores.append(context)

    if not combined_scores:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.concatenate(combined_scores),
        np.concatenate(recon_scores),
        np.concatenate(context_scores),
    )


def train_lstm_autoencoder(
    df_proc_in: pd.DataFrame,
    *,
    seq_len: int = 20,
    threshold_percentile: float = 99.3,
    threshold_mad_scale: float = 6.0,
    epochs: int = 24,
    batch_size: int = 64,
    lr: float = 1e-4,
    min_quality: float = 0.75,
    persistence_k: int = 3,
    persistence_m: int = 5,
    max_dt_sec: int = 5,
    ghost_birth_grace_sec: int = 20,
    ghost_age_window_sec: int = 20,
    smooth_window: int = 5,
    impute: str = "knn",
    impute_neighbors: int = 5,
    combined_score_weight: float = 0.20,
    calibration_percentile: float = 99.7,
    calibration_seeds: Optional[List[int]] = None,
    print_dt_stats: bool = True,
) -> Artifacts:
    df_proc = _require_df(df_proc_in).copy()

    print("\n================ ML TRAINING START ================\n")
    print(f"Input processed rows: {len(df_proc)}")

    df_feat = build_feature_frame(
        df_proc,
        id_col="icao",
        ts_col="timestamp",
        smooth=True,
        smooth_window=smooth_window,
        impute=impute,
        impute_neighbors=impute_neighbors,
        add_quality=True,
    )

    print(f"Rows after feature build: {len(df_feat)}")

    if print_dt_stats:
        df_debug = df_feat.sort_values(["icao", "timestamp"]).copy()
        df_debug["dt"] = df_debug.groupby("icao")["timestamp"].diff()
        print("\n==== DT STATISTICS (seconds) ====")
        print(df_debug["dt"].describe())
        print("================================\n")

    if "data_quality_score" in df_feat.columns:
        before_gate = len(df_feat)
        df_feat = df_feat[
            (df_feat["data_quality_score"] >= float(min_quality)) & (df_feat.get("bad_point", 0) == 0)
        ].copy()
        after_gate = len(df_feat)
        print(f"Rows after quality gating: {after_gate} / {before_gate}")
    else:
        print("No data_quality_score column found. Skipping quality gating.")

    if df_feat.empty:
        raise ValueError("After quality gating, no data remains for training. Lower min_quality or check preprocessing.")

    feature_cols = list(FEATURE_COLS)

    train_df, val_df = _time_split(df_feat, ts_col="timestamp", train_frac=0.8)
    if val_df.empty:
        val_df = train_df.copy()

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    _ensure_dir(ARTIFACT_DIR)

    scaler = _fit_scaler(train_df, feature_cols)
    train_scaled = _transform_with_scaler(train_df, scaler, feature_cols)
    val_scaled = _transform_with_scaler(val_df, scaler, feature_cols)

    X_train, _ = _make_sequences(
        train_scaled,
        feature_cols,
        seq_len,
        id_col="icao",
        ts_col="timestamp",
        max_dt_sec=max_dt_sec,
    )

    print(f"X_train shape: {X_train.shape}")

    if X_train.shape[0] == 0:
        raise ValueError(
            "Not enough training sequences.\n"
            "- Reduce seq_len, OR\n"
            "- Accumulate more history (Live mode), OR\n"
            "- Increase max_dt_sec if timestamps update slower."
        )

    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_size=64)
    model.fit(X_train, epochs=epochs, batch_size=batch_size, lr=lr)

    X_val, _ = _make_sequences(
        val_scaled,
        feature_cols,
        seq_len,
        id_col="icao",
        ts_col="timestamp",
        max_dt_sec=max_dt_sec,
    )

    print(f"X_val shape: {X_val.shape}")

    if X_val.shape[0] == 0:
        print("WARNING: No validation sequences were created. Threshold is being set from training error.")

    threshold_source = X_val if X_val.shape[0] else X_train
    recon_errs = model.reconstruction_error(threshold_source).astype(np.float32)
    train_feature_score = _terminal_feature_score(X_train)
    train_center = float(np.median(train_feature_score))
    train_mad = float(np.median(np.abs(train_feature_score - train_center)))
    train_scale = max(1.4826 * train_mad, 1e-6)
    threshold_feature_score = _terminal_feature_score(threshold_source)
    errs, feature_context = _combine_scores(
        recon_errs,
        threshold_feature_score,
        center=train_center,
        scale=train_scale,
        weight=combined_score_weight,
    )

    percentile_thr, mad_thr, threshold = _robust_threshold(
        errs,
        percentile=threshold_percentile,
        mad_scale=threshold_mad_scale,
    )
    _, _, reconstruction_threshold = _robust_threshold(
        recon_errs,
        percentile=threshold_percentile,
        mad_scale=threshold_mad_scale,
    )
    _, _, feature_context_threshold = _robust_threshold(
        feature_context,
        percentile=threshold_percentile,
        mad_scale=threshold_mad_scale,
    )

    calibration_seed_list = calibration_seeds or [101, 202, 303]
    calib_combined, calib_recon, calib_context = _calibration_scores(
        scaler,
        model,
        feature_cols,
        seq_len=seq_len,
        max_dt_sec=max_dt_sec,
        min_quality=min_quality,
        smooth_window=smooth_window,
        impute=impute,
        impute_neighbors=impute_neighbors,
        feature_score_center=train_center,
        feature_score_scale=train_scale,
        combined_score_weight=combined_score_weight,
        calibration_seeds=calibration_seed_list,
    )
    if calib_combined.size:
        threshold = max(threshold, float(np.percentile(calib_combined, calibration_percentile)))
    if calib_recon.size:
        reconstruction_threshold = max(
            reconstruction_threshold,
            float(np.percentile(calib_recon, calibration_percentile)),
        )
    if calib_context.size:
        feature_context_threshold = max(
            feature_context_threshold,
            float(np.percentile(calib_context, calibration_percentile)),
        )

    print(f"Threshold percentile: {threshold_percentile}")
    print(f"Percentile threshold: {percentile_thr:.6f}")
    print(f"MAD threshold: {mad_thr:.6f}")
    print(f"Feature score center: {train_center:.6f}")
    print(f"Feature score scale: {train_scale:.6f}")
    print(f"Mean feature context contribution: {float(np.mean(feature_context)):.6f}")
    print(f"Calibration percentile: {calibration_percentile}")
    print(f"Calibration seeds: {calibration_seed_list}")
    print(f"Reconstruction threshold: {reconstruction_threshold:.6f}")
    print(f"Feature context threshold: {feature_context_threshold:.6f}")
    print(f"Final threshold: {threshold:.6f}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    model.save(MODEL_PATH)

    meta = {
        "seq_len": seq_len,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "min_quality": float(min_quality),
        "persistence_k": int(persistence_k),
        "persistence_m": int(persistence_m),
        "threshold_percentile": float(threshold_percentile),
        "threshold_mad_scale": float(threshold_mad_scale),
        "max_dt_sec": int(max_dt_sec),
        "ghost_birth_grace_sec": int(ghost_birth_grace_sec),
        "ghost_age_window_sec": int(ghost_age_window_sec),
        "smooth_window": int(smooth_window),
        "impute_strategy": str(impute),
        "impute_neighbors": int(impute_neighbors),
        "feature_score_center": train_center,
        "feature_score_scale": train_scale,
        "combined_score_weight": float(combined_score_weight),
        "calibration_percentile": float(calibration_percentile),
        "calibration_seeds": list(calibration_seed_list),
        "reconstruction_threshold": float(reconstruction_threshold),
        "feature_context_threshold": float(feature_context_threshold),
        "version": "v8_multifeature_knn_hybrid_ghost_context_score",
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print("\nArtifacts saved:")
    print(f"- {SCALER_PATH}")
    print(f"- {MODEL_PATH}")
    print(f"- {META_PATH}")
    print("\n================= ML TRAINING END =================\n")

    return Artifacts(
        seq_len=seq_len,
        threshold=threshold,
        feature_cols=feature_cols,
        scaler=scaler,
        model=model,
        min_quality=float(min_quality),
        persistence_k=int(persistence_k),
        persistence_m=int(persistence_m),
        max_dt_sec=int(max_dt_sec),
        ghost_birth_grace_sec=int(ghost_birth_grace_sec),
        ghost_age_window_sec=int(ghost_age_window_sec),
        smooth_window=int(smooth_window),
        impute_strategy=str(impute),
        impute_neighbors=int(impute_neighbors),
        feature_score_center=train_center,
        feature_score_scale=train_scale,
        combined_score_weight=float(combined_score_weight),
        reconstruction_threshold=float(reconstruction_threshold),
        feature_context_threshold=float(feature_context_threshold),
    )


def load_artifacts() -> Optional[Artifacts]:
    if not (os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        return None

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    feature_cols = cast(List[str], meta["feature_cols"])
    seq_len = int(meta["seq_len"])
    threshold = float(meta["threshold"])

    min_quality = float(meta.get("min_quality", 0.7))
    persistence_k = int(meta.get("persistence_k", 3))
    persistence_m = int(meta.get("persistence_m", 5))
    max_dt_sec = int(meta.get("max_dt_sec", 5))
    ghost_birth_grace_sec = int(meta.get("ghost_birth_grace_sec", 20))
    ghost_age_window_sec = int(meta.get("ghost_age_window_sec", 20))
    smooth_window = int(meta.get("smooth_window", 3))
    impute_strategy = str(meta.get("impute_strategy", "median"))
    impute_neighbors = int(meta.get("impute_neighbors", 5))
    feature_score_center = float(meta.get("feature_score_center", 0.0))
    feature_score_scale = float(meta.get("feature_score_scale", 1.0))
    combined_score_weight = float(meta.get("combined_score_weight", 0.0))
    reconstruction_threshold = float(meta.get("reconstruction_threshold", threshold))
    feature_context_threshold = float(meta.get("feature_context_threshold", threshold))

    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_size=64)
    model.load(MODEL_PATH)

    return Artifacts(
        seq_len=seq_len,
        threshold=threshold,
        feature_cols=feature_cols,
        scaler=scaler,
        model=model,
        min_quality=min_quality,
        persistence_k=persistence_k,
        persistence_m=persistence_m,
        max_dt_sec=max_dt_sec,
        ghost_birth_grace_sec=ghost_birth_grace_sec,
        ghost_age_window_sec=ghost_age_window_sec,
        smooth_window=smooth_window,
        impute_strategy=impute_strategy,
        impute_neighbors=impute_neighbors,
        feature_score_center=feature_score_center,
        feature_score_scale=feature_score_scale,
        combined_score_weight=combined_score_weight,
        reconstruction_threshold=reconstruction_threshold,
        feature_context_threshold=feature_context_threshold,
    )


def _apply_persistence(scores: pd.DataFrame, k: int, m: int, flag_col: str = "is_seq_anom") -> pd.DataFrame:
    s = scores.sort_values(["icao", "timestamp"]).copy()
    s["seq_anom_int"] = s[flag_col].astype(int)

    s["anom_count_last_m"] = (
        s.groupby("icao", dropna=False)["seq_anom_int"]
        .transform(lambda x: x.rolling(window=m, min_periods=1).sum())
    )
    s["is_persistent_anom"] = s["anom_count_last_m"] >= int(k)
    return s


def _build_hybrid_ghost_flags(df_feat: pd.DataFrame, artifacts: Artifacts) -> pd.DataFrame:
    """
    Flags tracks that appear late in the scenario and are still very young.
    This works even if the ghost track is too short to form a full LSTM sequence.
    """
    if df_feat.empty:
        return pd.DataFrame(columns=["icao", "timestamp", "is_hybrid_ghost"])

    d = df_feat[
        [
            "icao",
            "timestamp",
            "track_age_sec",
            "velocity",
            "implied_speed_mps",
            "relative_speed_to_median_mps",
            "velocity_window_std",
        ]
    ].copy()
    d["icao"] = d["icao"].astype(str)
    d["timestamp"] = pd.to_numeric(d["timestamp"], errors="coerce").fillna(0).astype(int)
    d["track_age_sec"] = pd.to_numeric(d["track_age_sec"], errors="coerce").fillna(0.0)
    d["velocity"] = pd.to_numeric(d["velocity"], errors="coerce").fillna(0.0)
    d["implied_speed_mps"] = pd.to_numeric(d["implied_speed_mps"], errors="coerce").fillna(0.0)
    d["relative_speed_to_median_mps"] = pd.to_numeric(d["relative_speed_to_median_mps"], errors="coerce").fillna(0.0)
    d["velocity_window_std"] = pd.to_numeric(d["velocity_window_std"], errors="coerce").fillna(0.0)

    global_start = int(d["timestamp"].min())
    first_seen = d.groupby("icao", dropna=False)["timestamp"].min().rename("first_seen_ts")
    d = d.merge(first_seen, on="icao", how="left")

    d["late_birth"] = (d["first_seen_ts"] - global_start) >= int(artifacts.ghost_birth_grace_sec)
    d["young_track"] = d["track_age_sec"] <= float(artifacts.ghost_age_window_sec)
    d["slow_track"] = (d["velocity"] <= 35.0) & (d["implied_speed_mps"] <= 35.0)
    d["speed_outlier"] = d["relative_speed_to_median_mps"] >= 120.0
    d["motionless_profile"] = d["velocity_window_std"] <= 2.0
    d["is_hybrid_ghost"] = (
        d["late_birth"] & d["young_track"] & (d["slow_track"] | d["speed_outlier"] | d["motionless_profile"])
    ).astype(bool)

    return d[["icao", "timestamp", "is_hybrid_ghost"]].drop_duplicates(subset=["icao", "timestamp"])


def score_sequences(df_proc_in: pd.DataFrame, artifacts: Artifacts) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score sequences with quality gating + persistence + direct hybrid ghost handling.

    Returns:
      scores: per-sequence table
      anoms: combined anomaly table
    """
    df_proc = _require_df(df_proc_in).copy()

    df_feat = build_feature_frame(
        df_proc,
        id_col="icao",
        ts_col="timestamp",
        smooth=True,
        smooth_window=int(artifacts.smooth_window),
        impute=str(artifacts.impute_strategy),
        impute_neighbors=int(artifacts.impute_neighbors),
        add_quality=True,
    )

    if "data_quality_score" in df_feat.columns:
        df_feat = df_feat[
            (df_feat["data_quality_score"] >= float(artifacts.min_quality)) & (df_feat.get("bad_point", 0) == 0)
        ].copy()

    if df_feat.empty:
        return pd.DataFrame(), pd.DataFrame()

    hybrid_ghost = _build_hybrid_ghost_flags(df_feat, artifacts)

    df_scaled = _transform_with_scaler(df_feat, artifacts.scaler, artifacts.feature_cols)
    X, K = _make_sequences(
        df_scaled,
        artifacts.feature_cols,
        artifacts.seq_len,
        id_col="icao",
        ts_col="timestamp",
        max_dt_sec=int(artifacts.max_dt_sec),
    )

    if X.shape[0] == 0:
        scores = pd.DataFrame(
            columns=[
                "icao",
                "timestamp",
                "anomaly_score",
                "reconstruction_score",
                "feature_context_score",
                "threshold",
                "reconstruction_threshold",
                "feature_context_threshold",
                "is_seq_anom",
                "is_strong_seq_anom",
                "seq_anom_int",
                "anom_count_last_m",
                "is_persistent_anom",
                "is_hybrid_ghost",
                "is_final_anom",
            ]
        )
    else:
        recon_errs = artifacts.model.reconstruction_error(X).astype(np.float32)
        raw_feature_score = _terminal_feature_score(X)
        errs, feature_context = _combine_scores(
            recon_errs,
            raw_feature_score,
            center=artifacts.feature_score_center,
            scale=artifacts.feature_score_scale,
            weight=artifacts.combined_score_weight,
        )

        scores = pd.DataFrame(
            {
                "icao": K[:, 0].astype(str),
                "timestamp": K[:, 1].astype(int),
                "anomaly_score": errs,
                "reconstruction_score": recon_errs,
                "feature_context_score": feature_context,
            }
        )

        scores["threshold"] = float(artifacts.threshold)
        scores["reconstruction_threshold"] = float(artifacts.reconstruction_threshold)
        scores["feature_context_threshold"] = float(artifacts.feature_context_threshold)
        scores["is_seq_anom"] = scores["anomaly_score"] > float(artifacts.threshold)
        scores["is_strong_seq_anom"] = scores["is_seq_anom"] & (
            (scores["reconstruction_score"] > float(artifacts.reconstruction_threshold))
            | (scores["feature_context_score"] > float(artifacts.feature_context_threshold))
        )

        scores = _apply_persistence(scores, artifacts.persistence_k, artifacts.persistence_m, flag_col="is_strong_seq_anom")

        scores = scores.merge(hybrid_ghost, on=["icao", "timestamp"], how="left")
        scores["is_hybrid_ghost"] = scores["is_hybrid_ghost"].fillna(False).astype(bool)
        scores["is_final_anom"] = scores["is_persistent_anom"] | scores["is_hybrid_ghost"]

    ml_anoms = pd.DataFrame()
    if not scores.empty:
        ml_anoms = scores[scores["is_final_anom"]].copy()
        ml_anoms["anomaly_type"] = np.where(
            ml_anoms["is_hybrid_ghost"],
            "Hybrid Ghost Aircraft Candidate",
            "ML Sequence Anomaly (Persistent)",
        )
        ml_anoms["persistence_k"] = int(artifacts.persistence_k)
        ml_anoms["persistence_m"] = int(artifacts.persistence_m)

    # direct ghost anomalies so ghost tracks can still be reported
    direct_ghost_anoms = hybrid_ghost[hybrid_ghost["is_hybrid_ghost"]].copy()
    if not direct_ghost_anoms.empty:
        direct_ghost_anoms["anomaly_score"] = np.nan
        direct_ghost_anoms["reconstruction_score"] = np.nan
        direct_ghost_anoms["feature_context_score"] = np.nan
        direct_ghost_anoms["threshold"] = float(artifacts.threshold)
        direct_ghost_anoms["reconstruction_threshold"] = float(artifacts.reconstruction_threshold)
        direct_ghost_anoms["feature_context_threshold"] = float(artifacts.feature_context_threshold)
        direct_ghost_anoms["is_seq_anom"] = False
        direct_ghost_anoms["is_strong_seq_anom"] = False
        direct_ghost_anoms["seq_anom_int"] = 0
        direct_ghost_anoms["anom_count_last_m"] = 0
        direct_ghost_anoms["is_persistent_anom"] = False
        direct_ghost_anoms["is_final_anom"] = True
        direct_ghost_anoms["anomaly_type"] = "Hybrid Ghost Aircraft Candidate"
        direct_ghost_anoms["persistence_k"] = int(artifacts.persistence_k)
        direct_ghost_anoms["persistence_m"] = int(artifacts.persistence_m)

    anomaly_records: List[dict] = []
    if not ml_anoms.empty:
        anomaly_records.extend(ml_anoms.to_dict(orient="records"))
    if not direct_ghost_anoms.empty:
        anomaly_records.extend(direct_ghost_anoms.to_dict(orient="records"))
    anoms = pd.DataFrame.from_records(anomaly_records) if anomaly_records else pd.DataFrame()

    if not anoms.empty:
        anoms = anoms.drop_duplicates(subset=["icao", "timestamp", "anomaly_type"]).sort_values(["timestamp", "icao"])

    if not scores.empty:
        scores = scores.sort_values(["timestamp", "icao"])

    return scores, anoms
