from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd

from data_processing import process_adsb_data
from data_simulation import generate_aircraft_data
from ml_pipeline import train_lstm_autoencoder

ARTIFACT_DIR = "artifacts"
DATA_HISTORY_PATH = os.path.join(ARTIFACT_DIR, "training_history.csv")

# Keep the history from growing forever
MAX_HISTORY_ROWS = 30000


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _ensure_artifact_dir() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _load_history() -> pd.DataFrame:
    if os.path.exists(DATA_HISTORY_PATH):
        try:
            df = pd.read_csv(DATA_HISTORY_PATH)
            print(f"Loaded previous clean history: {len(df)} rows")
            return df
        except Exception as e:
            print(f"WARNING: Failed to load existing history file: {e}")
    return pd.DataFrame()


def _save_history(df: pd.DataFrame) -> None:
    df.to_csv(DATA_HISTORY_PATH, index=False)
    print(f"Saved clean history: {len(df)} rows -> {DATA_HISTORY_PATH}")


def _dedupe_history(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    dedupe_cols = [c for c in ["icao", "timestamp"] if c in d.columns]
    if dedupe_cols:
        d = d.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
    else:
        d = d.drop_duplicates().reset_index(drop=True)

    return d


def _cap_history(df: pd.DataFrame, max_rows: int = MAX_HISTORY_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()

    if "timestamp" in df.columns:
        return df.sort_values("timestamp").tail(max_rows).reset_index(drop=True)

    return df.tail(max_rows).reset_index(drop=True)


def main() -> None:
    _set_seed(42)
    _ensure_artifact_dir()

    print("=== TRAINING SCRIPT STARTED ===")
    print("Generating new clean baseline simulated ADS-B data...")

    # New clean normal baseline only
    df_raw = generate_aircraft_data()
    print("New raw rows generated:", len(df_raw))

    df_proc_new = process_adsb_data(df_raw)
    print("New processed rows:", len(df_proc_new))

    # Load previous clean history and append new clean baseline
    df_hist = _load_history()

    if df_hist.empty:
        df_train = df_proc_new.copy()
    else:
        df_train = pd.concat([df_hist, df_proc_new], ignore_index=True)

    df_train = _dedupe_history(df_train)
    df_train = _cap_history(df_train, MAX_HISTORY_ROWS)

    print("Total clean training dataset after merge/dedupe/cap:", len(df_train))

    # Save updated clean history BEFORE training
    _save_history(df_train)

    artifacts = train_lstm_autoencoder(
        df_train,
        seq_len=20,
        epochs=24,
        batch_size=64,
        lr=1e-4,
        threshold_percentile=99.3,
        threshold_mad_scale=6.0,
        max_dt_sec=10,
        min_quality=0.80,
        persistence_k=5,
        persistence_m=6,
        ghost_birth_grace_sec=20,
        ghost_age_window_sec=20,
        smooth_window=5,
        impute="knn",
        impute_neighbors=5,
        combined_score_weight=0.20,
        calibration_percentile=99.7,
        print_dt_stats=True,
    )

    print("=== TRAINING COMPLETE ===")
    print(f"Saved threshold: {artifacts.threshold:.6f}")
    print(f"Saved seq_len: {artifacts.seq_len}")
    print(f"History file: {DATA_HISTORY_PATH}")


if __name__ == "__main__":
    main()
