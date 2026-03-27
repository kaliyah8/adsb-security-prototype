from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_processing import process_adsb_data
from data_simulation import (
    generate_aircraft_data,
    inject_ghost_aircraft_attack,
    inject_gps_spoofing_attack,
    inject_teleportation_attack,
)
from ml_pipeline import Artifacts, load_artifacts, score_sequences

ARTIFACT_DIR = "artifacts"
EVAL_METRICS_PATH = os.path.join(ARTIFACT_DIR, "evaluation_metrics.json")
EVAL_SEQ_PATH = os.path.join(ARTIFACT_DIR, "evaluation_seq_details.csv")
EVAL_ROW_PATH = os.path.join(ARTIFACT_DIR, "evaluation_row_details.csv")
EVAL_SCENARIO_PATH = os.path.join(ARTIFACT_DIR, "evaluation_scenario_summary.csv")

SCENARIOS = ["normal", "teleportation", "gps_spoofing", "ghost_aircraft"]


def _ensure_artifact_dir() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _pick_target_and_window(df_raw: pd.DataFrame) -> Tuple[str, int, int]:
    if df_raw.empty or "icao" not in df_raw.columns:
        raise RuntimeError("Cannot pick a target aircraft from empty simulation data.")

    icaos = df_raw["icao"].dropna().astype(str).unique().tolist()
    if not icaos:
        raise RuntimeError("Simulation data does not contain any ICAO values.")

    target_icao = icaos[0]
    g = (
        df_raw[df_raw["icao"].astype(str) == target_icao]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(g) < 20:
        raise RuntimeError(f"Target aircraft {target_icao} does not have enough points for evaluation.")

    start_step = max(5, len(g) // 3)
    end_step = min(len(g) - 2, start_step + 8)
    return target_icao, start_step, end_step


def _label_rows(
    df_raw: pd.DataFrame,
    scenario: str,
    target_icao: str,
    ghost_icao: str,
    start_step: int,
    end_step: int,
    trailing_buffer_steps: int = 0,
) -> pd.DataFrame:
    labels = df_raw[["icao", "timestamp"]].copy()
    labels["icao"] = labels["icao"].astype(str)
    labels["timestamp"] = pd.to_numeric(labels["timestamp"], errors="coerce").fillna(0).astype(int)
    labels["scenario"] = str(scenario)
    labels["label"] = 0
    labels["anomaly_type"] = "normal"

    if scenario == "normal":
        return labels

    if scenario == "ghost_aircraft":
        mask = labels["icao"] == str(ghost_icao)
        labels.loc[mask, "label"] = 1
        labels.loc[mask, "anomaly_type"] = "ghost_aircraft"
        return labels

    target_rows = (
        df_raw[df_raw["icao"].astype(str) == str(target_icao)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    buffered_end_step = min(len(target_rows) - 1, int(end_step) + max(0, int(trailing_buffer_steps)))
    attack_ts = target_rows.iloc[start_step : buffered_end_step + 1]["timestamp"].astype(int).tolist()
    if not attack_ts:
        return labels

    attack_start_ts = int(min(attack_ts))
    attack_end_ts = int(max(attack_ts))
    mask = (
        (labels["icao"] == str(target_icao))
        & (labels["timestamp"] >= attack_start_ts)
        & (labels["timestamp"] <= attack_end_ts)
    )
    labels.loc[mask, "label"] = 1
    labels.loc[mask, "anomaly_type"] = str(scenario)
    return labels


def _build_sequence_labels(row_labels: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    d = row_labels.copy()
    d = d.sort_values(["icao", "timestamp"]).reset_index(drop=True)
    d["label"] = pd.to_numeric(d["label"], errors="coerce").fillna(0).astype(int)
    d["seq_label"] = (
        d.groupby("icao", dropna=False)["label"]
        .transform(lambda s: s.rolling(window=int(seq_len), min_periods=int(seq_len)).max())
        .fillna(0)
        .astype(int)
    )
    return d[["icao", "timestamp", "scenario", "seq_label"]].drop_duplicates(subset=["icao", "timestamp"])


def _safe_metric(func: Any, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(func(y_true, y_pred, zero_division=0))
    except TypeError:
        return float(func(y_true, y_pred))
    except Exception:
        return float("nan")


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, fn: Any) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float("nan")


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "precision": _safe_metric(precision_score, y_true, y_pred),
        "recall": _safe_metric(recall_score, y_true, y_pred),
        "f1": _safe_metric(f1_score, y_true, y_pred),
    }

    if y_score is not None:
        metrics["roc_auc"] = _safe_auc(y_true, y_score, roc_auc_score)
        metrics["pr_auc"] = _safe_auc(y_true, y_score, average_precision_score)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
        metrics["tp"] = float(tp)
        metrics["false_positive_rate"] = float(fp / max(fp + tn, 1))
    except Exception:
        metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = float("nan")
        metrics["false_positive_rate"] = float("nan")

    return metrics


def _evaluate_single_run(
    artifacts: Artifacts,
    scenario: str,
    run_id: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df_raw = generate_aircraft_data(seed=int(seed))
    target_icao, start_step, end_step = _pick_target_and_window(df_raw)
    ghost_icao = f"GHOST{run_id:03d}"

    if scenario == "teleportation":
        df_raw = inject_teleportation_attack(df_raw, target_icao=target_icao, start_step=start_step, end_step=end_step)
    elif scenario == "gps_spoofing":
        df_raw = inject_gps_spoofing_attack(df_raw, target_icao=target_icao, start_step=start_step, end_step=end_step)
    elif scenario == "ghost_aircraft":
        df_raw = inject_ghost_aircraft_attack(df_raw, ghost_icao=ghost_icao, start_step=start_step, end_step=end_step)

    trailing_buffer_steps = 0
    if scenario in {"teleportation", "gps_spoofing"}:
        trailing_buffer_steps = max(0, int(artifacts.seq_len) + int(artifacts.persistence_m) - 2)

    row_labels = _label_rows(
        df_raw,
        scenario,
        target_icao,
        ghost_icao,
        start_step,
        end_step,
        trailing_buffer_steps=trailing_buffer_steps,
    )
    seq_labels = _build_sequence_labels(row_labels, artifacts.seq_len)
    df_proc = process_adsb_data(df_raw)
    scores_df, anoms_df = score_sequences(df_proc, artifacts)

    seq_eval = scores_df.copy()
    if seq_eval.empty:
        seq_eval = pd.DataFrame(columns=["icao", "timestamp", "anomaly_score", "is_final_anom"])

    seq_eval = seq_eval.merge(seq_labels, on=["icao", "timestamp"], how="left")
    seq_eval["scenario"] = seq_eval["scenario"].fillna(str(scenario))
    seq_eval["seq_label"] = pd.to_numeric(seq_eval["seq_label"], errors="coerce").fillna(0).astype(int)
    seq_eval["pred_label"] = seq_eval.get("is_final_anom", False).fillna(False).astype(int)
    seq_eval["run_id"] = int(run_id)

    row_eval = row_labels.copy()
    pred_rows = pd.DataFrame(columns=["icao", "timestamp", "pred_label"])
    if not anoms_df.empty:
        pred_rows = anoms_df[["icao", "timestamp"]].drop_duplicates().copy()
        pred_rows["pred_label"] = 1

    row_eval = row_eval.merge(pred_rows, on=["icao", "timestamp"], how="left")
    row_eval["pred_label"] = pd.to_numeric(row_eval["pred_label"], errors="coerce").fillna(0).astype(int)
    row_eval["run_id"] = int(run_id)

    summary = {
        "scenario": str(scenario),
        "run_id": int(run_id),
        "seed": int(seed),
        "target_icao": str(target_icao),
        "ghost_icao": str(ghost_icao),
        "rows": int(len(row_eval)),
        "sequence_rows": int(len(seq_eval)),
        "positive_rows": int(row_eval["label"].sum()),
        "positive_sequences": int(seq_eval["seq_label"].sum()),
        "detected_positive_rows": int(((row_eval["label"] == 1) & (row_eval["pred_label"] == 1)).sum()),
        "detected_positive_sequences": int(((seq_eval["seq_label"] == 1) & (seq_eval["pred_label"] == 1)).sum()),
        "any_detection": bool(
            (((row_eval["label"] == 1) & (row_eval["pred_label"] == 1)).any())
            or (((seq_eval["seq_label"] == 1) & (seq_eval["pred_label"] == 1)).any())
        ),
    }
    return seq_eval, row_eval, summary


def run_evaluation(
    *,
    artifacts: Optional[Artifacts] = None,
    runs_per_scenario: int = 5,
    base_seed: int = 100,
    save: bool = True,
) -> Dict[str, Any]:
    active_artifacts = artifacts or load_artifacts()
    if active_artifacts is None:
        raise RuntimeError("No trained artifacts found. Train the model before evaluation.")

    seq_frames: List[pd.DataFrame] = []
    row_frames: List[pd.DataFrame] = []
    scenario_rows: List[Dict[str, Any]] = []

    run_id = 0
    for scenario in SCENARIOS:
        for offset in range(int(runs_per_scenario)):
            seq_eval, row_eval, summary = _evaluate_single_run(
                active_artifacts,
                scenario=scenario,
                run_id=run_id,
                seed=int(base_seed + run_id),
            )
            seq_frames.append(seq_eval)
            row_frames.append(row_eval)
            scenario_rows.append(summary)
            run_id += 1

    seq_all = pd.concat(seq_frames, ignore_index=True, sort=False) if seq_frames else pd.DataFrame()
    row_all = pd.concat(row_frames, ignore_index=True, sort=False) if row_frames else pd.DataFrame()
    scenario_df = pd.DataFrame(scenario_rows)

    seq_metrics = _binary_metrics(
        seq_all["seq_label"].to_numpy(dtype=int) if not seq_all.empty else np.zeros((0,), dtype=int),
        seq_all["pred_label"].to_numpy(dtype=int) if not seq_all.empty else np.zeros((0,), dtype=int),
        seq_all["anomaly_score"].to_numpy(dtype=float) if ("anomaly_score" in seq_all.columns and not seq_all.empty) else None,
    )
    row_metrics = _binary_metrics(
        row_all["label"].to_numpy(dtype=int) if not row_all.empty else np.zeros((0,), dtype=int),
        row_all["pred_label"].to_numpy(dtype=int) if not row_all.empty else np.zeros((0,), dtype=int),
        None,
    )

    per_scenario: Dict[str, Dict[str, float]] = {}
    for scenario in SCENARIOS:
        seq_s = seq_all[seq_all["scenario"] == scenario].copy() if not seq_all.empty else pd.DataFrame()
        row_s = row_all[row_all["scenario"] == scenario].copy() if not row_all.empty else pd.DataFrame()
        per_scenario[scenario] = {
            "sequence_precision": _safe_metric(
                precision_score,
                seq_s["seq_label"].to_numpy(dtype=int) if not seq_s.empty else np.zeros((0,), dtype=int),
                seq_s["pred_label"].to_numpy(dtype=int) if not seq_s.empty else np.zeros((0,), dtype=int),
            ),
            "sequence_recall": _safe_metric(
                recall_score,
                seq_s["seq_label"].to_numpy(dtype=int) if not seq_s.empty else np.zeros((0,), dtype=int),
                seq_s["pred_label"].to_numpy(dtype=int) if not seq_s.empty else np.zeros((0,), dtype=int),
            ),
            "row_recall": _safe_metric(
                recall_score,
                row_s["label"].to_numpy(dtype=int) if not row_s.empty else np.zeros((0,), dtype=int),
                row_s["pred_label"].to_numpy(dtype=int) if not row_s.empty else np.zeros((0,), dtype=int),
            ),
            "detections_any_rate": float(
                scenario_df[scenario_df["scenario"] == scenario]["any_detection"].mean()
            ) if not scenario_df.empty else float("nan"),
        }

    metrics = {
        "runs_per_scenario": int(runs_per_scenario),
        "scenario_count": int(len(SCENARIOS)),
        "sequence_metrics": seq_metrics,
        "row_metrics": row_metrics,
        "per_scenario": per_scenario,
        "artifact_threshold": float(active_artifacts.threshold),
        "artifact_seq_len": int(active_artifacts.seq_len),
        "artifact_impute_strategy": str(active_artifacts.impute_strategy),
        "artifact_smooth_window": int(active_artifacts.smooth_window),
    }

    result = {
        "metrics": metrics,
        "sequence_details": seq_all,
        "row_details": row_all,
        "scenario_summary": scenario_df,
    }

    if save:
        _ensure_artifact_dir()
        with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        seq_all.to_csv(EVAL_SEQ_PATH, index=False)
        row_all.to_csv(EVAL_ROW_PATH, index=False)
        scenario_df.to_csv(EVAL_SCENARIO_PATH, index=False)

    return result
