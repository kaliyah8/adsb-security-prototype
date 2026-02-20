# ml_features.py
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "implied_speed_mps",
    "velocity",
    "accel_mps2",
    "vert_rate_mps",
    "turn_rate_dps",
    "speed_mismatch_mps",
]

def build_feature_frame(df_proc: pd.DataFrame) -> pd.DataFrame:
    if df_proc is None or df_proc.empty:
        return pd.DataFrame()

    d = df_proc.copy()
    if "speed_mismatch_mps" not in d.columns:
        d["speed_mismatch_mps"] = (d["implied_speed_mps"] - d["velocity"]).abs()

    for c in FEATURE_COLS:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    return d
