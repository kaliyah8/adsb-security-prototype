from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, List, cast, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

    # gating + persistence metadata (for reproducibility)
    min_quality: float
    persistence_k: int
    persistence_m: int
    max_dt_sec: int  # stored so scoring matches training


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

    - Drops duplicate (icao, timestamp)
    - Only builds sequences where all internal gaps satisfy 0 < dt <= max_dt_sec
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

        g["dt"] = g[ts_col].diff().astype("float")
        g["dt_ok"] = (g["dt"] > 0) & (g["dt"] <= float(max_dt_sec))

        feats = _select_cols_df(g, feature_cols).to_numpy(dtype=np.float32)

        cont = (
            g["dt_ok"]
            .rolling(window=seq_len, min_periods=seq_len)
            .min()
            .fillna(0)
            .to_numpy(dtype=int)
        )

        for i in range(seq_len - 1, len(g)):
            if cont[i] != 1:
                continue
            window = feats[i - seq_len + 1 : i + 1]
            sequences.append(window)
            keys.append((str(icao), int(g.iloc[i][ts_col])))

    if not sequences:
        return (
            np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.zeros((0, 2), dtype=object),
        )

    X = np.stack(sequences, axis=0)
    K = np.array(keys, dtype=object)
    return X, K


def _time_split(df: pd.DataFrame, ts_col: str = "timestamp", train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple global time split to avoid training on future data."""
    d = df.sort_values(ts_col).copy()
    if len(d) < 10:
        return d, d.iloc[0:0].copy()

    split_idx = int(len(d) * train_frac)
    train = d.iloc[:split_idx].copy()
    val = d.iloc[split_idx:].copy()
    return train, val


def train_lstm_autoencoder(
    df_proc_in: pd.DataFrame,
    *,
    seq_len: int = 30,
    threshold_percentile: float = 99.5,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    min_quality: float = 0.7,
    persistence_k: int = 3,
    persistence_m: int = 5,
    max_dt_sec: int = 5,
    print_dt_stats: bool = True,
) -> Artifacts:
    """
    Train LSTM autoencoder on CLEAN, GATED sequences.
    Saves artifacts to ./artifacts.
    """
    df_proc = _require_df(df_proc_in).copy()

    df_feat = build_feature_frame(
        df_proc,
        id_col="icao",
        ts_col="timestamp",
        smooth=True,
        smooth_window=3,
        impute="median",
        add_quality=True,
    )

    if print_dt_stats:
        df_debug = df_feat.sort_values(["icao", "timestamp"]).copy()
        df_debug["dt"] = df_debug.groupby("icao")["timestamp"].diff()
        print("\n==== DT STATISTICS (seconds) ====")
        print(df_debug["dt"].describe())
        print("================================\n")

    # Quality gating for training
    if "data_quality_score" in df_feat.columns:
        df_feat = df_feat[
            (df_feat["data_quality_score"] >= float(min_quality)) & (df_feat.get("bad_point", 0) == 0)
        ].copy()

    if df_feat.empty:
        raise ValueError("After quality gating, no data remains for training. Lower min_quality or check preprocessing.")

    feature_cols = list(FEATURE_COLS)

    train_df, val_df = _time_split(df_feat, ts_col="timestamp", train_frac=0.8)
    if val_df.empty:
        val_df = train_df.copy()

    _ensure_dir(ARTIFACT_DIR)

    # scaler fit on TRAIN ONLY
    scaler = _fit_scaler(train_df, feature_cols)
    train_scaled = _transform_with_scaler(train_df, scaler, feature_cols)
    val_scaled = _transform_with_scaler(val_df, scaler, feature_cols)

    X_train, _ = _make_sequences(train_scaled, feature_cols, seq_len, id_col="icao", ts_col="timestamp", max_dt_sec=max_dt_sec)
    if X_train.shape[0] == 0:
        raise ValueError(
            "Not enough training sequences.\n"
            "- Reduce seq_len, OR\n"
            "- Accumulate more history (Live mode), OR\n"
            "- Increase max_dt_sec if timestamps update slower."
        )

    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_size=64)
    model.fit(X_train, epochs=epochs, batch_size=batch_size, lr=lr)

    # threshold from VALIDATION if possible
    X_val, _ = _make_sequences(val_scaled, feature_cols, seq_len, id_col="icao", ts_col="timestamp", max_dt_sec=max_dt_sec)
    errs = model.reconstruction_error(X_val) if X_val.shape[0] else model.reconstruction_error(X_train)
    threshold = float(np.percentile(errs, threshold_percentile))

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
        "max_dt_sec": int(max_dt_sec),
        "version": "v3_quality_gated_persistent_dt_guard",
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

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
    )


def _apply_persistence(scores: pd.DataFrame, k: int, m: int) -> pd.DataFrame:
    """
    Alert if at least k of last m sequences are above threshold (per aircraft).
    Requires: icao, timestamp, is_seq_anom
    """
    s = scores.sort_values(["icao", "timestamp"]).copy()
    s["seq_anom_int"] = s["is_seq_anom"].astype(int)

    s["anom_count_last_m"] = (
        s.groupby("icao", dropna=False)["seq_anom_int"]
        .transform(lambda x: x.rolling(window=m, min_periods=1).sum())
    )
    s["is_persistent_anom"] = (s["anom_count_last_m"] >= int(k))
    return s


def score_sequences(df_proc_in: pd.DataFrame, artifacts: Artifacts) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score sequences with quality gating + persistence.
    Returns:
      scores: per-sequence table
      anoms: persistent anomalies table
    """
    df_proc = _require_df(df_proc_in).copy()

    df_feat = build_feature_frame(
        df_proc,
        id_col="icao",
        ts_col="timestamp",
        smooth=True,
        smooth_window=3,
        impute="median",
        add_quality=True,
    )

    # same quality gate as training
    if "data_quality_score" in df_feat.columns:
        df_feat = df_feat[
            (df_feat["data_quality_score"] >= float(artifacts.min_quality)) & (df_feat.get("bad_point", 0) == 0)
        ].copy()

    if df_feat.empty:
        return pd.DataFrame(), pd.DataFrame()

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
        return pd.DataFrame(), pd.DataFrame()

    errs = artifacts.model.reconstruction_error(X).astype(float)

    scores = pd.DataFrame(
        {
            "icao": K[:, 0].astype(str),
            "timestamp": K[:, 1].astype(int),
            "anomaly_score": errs,
        }
    )

    scores["threshold"] = float(artifacts.threshold)
    scores["is_seq_anom"] = scores["anomaly_score"] > float(artifacts.threshold)

    scores = _apply_persistence(scores, artifacts.persistence_k, artifacts.persistence_m)

    anoms = scores[scores["is_persistent_anom"]].copy()
    anoms["anomaly_type"] = "ML Sequence Anomaly (Persistent)"
    anoms["persistence_k"] = int(artifacts.persistence_k)
    anoms["persistence_m"] = int(artifacts.persistence_m)

    return scores.sort_values(["timestamp", "icao"]), anoms.sort_values(["timestamp", "icao"])