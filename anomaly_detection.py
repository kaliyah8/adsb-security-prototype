# anomaly_detection.py
import numpy as np
import pandas as pd

# Conservative envelope bounds (SI)
MAX_SPEED_MPS = 400.0          # hard cap for implied speed (teleport-like)
MAX_ACCEL_MPS2 = 12.0          # strong accel spike threshold
MAX_VERT_MPS = 40.0            # vertical speed threshold
MAX_TURN_DPS = 10.0            # very aggressive turn rate

# Data quality
MIN_DT = 0.5
MAX_DT = 15.0                  # ignore longer gaps for rule checks

# Spoofing persistence
SPOOF_MISMATCH_MPS = 60.0       # mismatch threshold between implied and reported speed
SPOOF_K_CONSEC = 3              # require k consecutive hits per aircraft


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-based anomaly rules (defensible and unit-consistent).
    Returns rows flagged with anomaly_type and anomaly_score.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    required = [
        "icao", "timestamp", "has_prev", "delta_t",
        "implied_speed_mps", "accel_mps2", "vert_rate_mps", "turn_rate_dps",
        "velocity",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"detect_anomalies missing required columns: {missing}")

    d = df.copy()

    # must have prev point
    d = d[d["has_prev"]].copy()
    if d.empty:
        return pd.DataFrame()

    # quality dt
    d = d[(d["delta_t"] >= MIN_DT) & (d["delta_t"] <= MAX_DT)].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values(["icao", "timestamp"]).reset_index(drop=True)

    out = []

    # 1) Teleport / Position Jump
    m_jump = d["implied_speed_mps"] > MAX_SPEED_MPS
    if m_jump.any():
        a = d[m_jump].copy()
        a["anomaly_type"] = "Position Jump / Teleportation"
        a["anomaly_score"] = (a["implied_speed_mps"] - MAX_SPEED_MPS).clip(lower=0)
        out.append(a)

    # 2) Accel spike
    m_acc = d["accel_mps2"].abs() > MAX_ACCEL_MPS2
    if m_acc.any():
        a = d[m_acc].copy()
        a["anomaly_type"] = "Implausible Acceleration"
        a["anomaly_score"] = (a["accel_mps2"].abs() - MAX_ACCEL_MPS2).clip(lower=0)
        out.append(a)

    # 3) Vertical speed
    m_vz = d["vert_rate_mps"].abs() > MAX_VERT_MPS
    if m_vz.any():
        a = d[m_vz].copy()
        a["anomaly_type"] = "Implausible Vertical Speed"
        a["anomaly_score"] = (a["vert_rate_mps"].abs() - MAX_VERT_MPS).clip(lower=0)
        out.append(a)

    # 4) Turn rate
    m_turn = d["turn_rate_dps"].abs() > MAX_TURN_DPS
    if m_turn.any():
        a = d[m_turn].copy()
        a["anomaly_type"] = "Implausible Turn Rate"
        a["anomaly_score"] = (a["turn_rate_dps"].abs() - MAX_TURN_DPS).clip(lower=0)
        out.append(a)

    # 5) Spoofing suspected: implied vs reported mismatch, persistent (true consecutive)
    d["speed_mismatch_mps"] = (d["implied_speed_mps"] - d["velocity"]).abs()
    d["spoof_hit"] = (d["speed_mismatch_mps"] > SPOOF_MISMATCH_MPS).astype(int)

    d["hit_run"] = (
        d.groupby("icao")["spoof_hit"]
        .transform(lambda s: s.rolling(SPOOF_K_CONSEC, min_periods=SPOOF_K_CONSEC).sum())
    )

    m_persist = d["hit_run"] >= SPOOF_K_CONSEC
    if m_persist.any():
        a = d[m_persist].copy()
        a["anomaly_type"] = "GPS Spoofing Suspected (Speed Mismatch)"
        a["anomaly_score"] = (a["speed_mismatch_mps"] - SPOOF_MISMATCH_MPS).clip(lower=0)
        out.append(a)

    if not out:
        return pd.DataFrame()

    res = pd.concat(out, ignore_index=True)

    keep_cols = list(df.columns) + ["speed_mismatch_mps", "anomaly_type", "anomaly_score"]
    keep_cols = [c for c in keep_cols if c in res.columns]
    res = res[keep_cols].drop_duplicates(subset=["icao", "timestamp", "anomaly_type"])

    return res.sort_values(["timestamp", "icao"]).reset_index(drop=True)