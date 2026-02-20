# ml_pipeline.py
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, List, cast

import numpy as np
import pandas as pd

# sklearn is used for scaling/features (if your project uses it)
from sklearn.preprocessing import StandardScaler

# torch / model bits (your project already has these modules)
from lstm_autoencoder import LSTMAutoencoder


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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _require_df(obj: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Force DataFrame type (fixes Pylance unions + prevents accidental Series)."""
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    return obj


def _select_cols_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Always return a DataFrame (never Series)."""
    return df.loc[:, cols].copy()


def _make_sequences(df_in: pd.DataFrame, feature_cols: List[str], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences per ICAO sorted by timestamp.
    Returns:
        X: (N, seq_len, F)
        keys: (N, 2) with [icao, last_timestamp] for each sequence
    """
    df = _require_df(df_in).copy()

    # Enforce required columns
    for c in ["icao", "timestamp"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Ensure all features exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    df["icao"] = df["icao"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

    df = df.sort_values(["icao", "timestamp"]).reset_index(drop=True)

    sequences: List[np.ndarray] = []
    keys: List[Tuple[str, int]] = []

    for icao, g in df.groupby("icao", sort=False):
        g = cast(pd.DataFrame, g)  # keep Pylance happy
        if len(g) < seq_len:
            continue

        feats = _select_cols_df(g, feature_cols).to_numpy(dtype=np.float32)

        # sliding windows
        for i in range(seq_len - 1, len(g)):
            window = feats[i - seq_len + 1 : i + 1]
            sequences.append(window)
            keys.append((str(icao), int(g.iloc[i]["timestamp"])))

    if not sequences:
        return np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32), np.zeros((0, 2), dtype=object)

    X = np.stack(sequences, axis=0)
    K = np.array(keys, dtype=object)
    return X, K


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


def train_lstm_autoencoder(
    df_in: pd.DataFrame,
    seq_len: int = 30,
    threshold_percentile: float = 99.5,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Artifacts:
    """
    Train LSTM autoencoder on sequences built from df_in.
    Saves artifacts to ./artifacts.
    """
    df = _require_df(df_in).copy()

    # Feature columns: keep aligned with your data_processing output
    feature_cols = [c for c in ["speed_mps", "vs_mps", "turn_rate_dps", "accel_mps2"] if c in df.columns]
    if not feature_cols:
        # fallback if your processing uses different names
        feature_cols = [c for c in ["velocity", "vertical_rate", "heading"] if c in df.columns]
    if not feature_cols:
        raise ValueError("No usable ML feature columns found in processed dataframe.")

    _ensure_dir(ARTIFACT_DIR)

    scaler = _fit_scaler(df, feature_cols)
    df_scaled = _transform_with_scaler(df, scaler, feature_cols)

    X, K = _make_sequences(df_scaled, feature_cols, seq_len)
    if X.shape[0] == 0:
        raise ValueError("Not enough data to build sequences for training.")

    # Train model
    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_size=64)
    model.fit(X, epochs=epochs, batch_size=batch_size, lr=lr)

    # Reconstruction error per sequence -> threshold
    errs = model.reconstruction_error(X)
    threshold = float(np.percentile(errs, threshold_percentile))

    # Save artifacts
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    model.save(MODEL_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({"seq_len": seq_len, "threshold": threshold, "feature_cols": feature_cols}, f)

    return Artifacts(seq_len=seq_len, threshold=threshold, feature_cols=feature_cols, scaler=scaler, model=model)


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

    model = LSTMAutoencoder(n_features=len(feature_cols), hidden_size=64)
    model.load(MODEL_PATH)

    return Artifacts(seq_len=seq_len, threshold=threshold, feature_cols=feature_cols, scaler=scaler, model=model)


def score_sequences(df_in: pd.DataFrame, artifacts: Artifacts) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _require_df(df_in).copy()

    df_scaled = _transform_with_scaler(df, artifacts.scaler, artifacts.feature_cols)
    X, K = _make_sequences(df_scaled, artifacts.feature_cols, artifacts.seq_len)

    if X.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    errs = artifacts.model.reconstruction_error(X)

    scores = pd.DataFrame(
        {
            "icao": K[:, 0].astype(str),
            "timestamp": K[:, 1].astype(int),
            "anomaly_score": errs.astype(float),
        }
    )

    anoms = scores[scores["anomaly_score"] > artifacts.threshold].copy()
    anoms["anomaly_type"] = "ML Sequence Anomaly"
    return scores.sort_values(["timestamp", "icao"]), anoms.sort_values(["timestamp", "icao"])
