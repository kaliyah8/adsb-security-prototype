from __future__ import annotations

import pandas as pd

from data_processing import process_adsb_data
from data_simulation import (
    generate_aircraft_data,
    inject_teleportation_attack,
    inject_gps_spoofing_attack,
    inject_ghost_aircraft_attack,
)
from ml_pipeline import load_artifacts, score_sequences


def _pick_target_and_window(df_raw: pd.DataFrame) -> tuple[str, int, int]:
    if df_raw.empty:
        raise RuntimeError("Generated dataframe is empty.")

    if "icao" not in df_raw.columns:
        raise RuntimeError("Generated dataframe does not contain an 'icao' column.")

    icaos = df_raw["icao"].dropna().astype(str).unique().tolist()
    if not icaos:
        raise RuntimeError("No ICAO values found in generated dataframe.")

    target_icao = icaos[0]

    g = (
        df_raw[df_raw["icao"].astype(str) == target_icao]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(g) < 20:
        raise RuntimeError(
            f"Target aircraft {target_icao} does not have enough points for attack injection."
        )

    start_step = max(5, len(g) // 3)
    end_step = min(len(g) - 2, start_step + 8)

    return target_icao, start_step, end_step


def run_test(scenario: str) -> None:
    artifacts = load_artifacts()
    if artifacts is None:
        raise RuntimeError("No trained artifacts found. Train the model first.")

    df_raw = generate_aircraft_data()
    target_icao, start_step, end_step = _pick_target_and_window(df_raw)

    if scenario == "teleportation":
        df_raw = inject_teleportation_attack(
            df_raw,
            target_icao=target_icao,
            start_step=start_step,
            end_step=end_step,
        )
    elif scenario == "gps_spoofing":
        df_raw = inject_gps_spoofing_attack(
            df_raw,
            target_icao=target_icao,
            start_step=start_step,
            end_step=end_step,
        )
    elif scenario == "ghost_aircraft":
        df_raw = inject_ghost_aircraft_attack(
            df_raw,
            ghost_icao="ghost001",
            start_step=start_step,
            end_step=end_step,
        )
    elif scenario == "normal":
        pass
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    df_proc = process_adsb_data(df_raw)
    scores, anoms = score_sequences(df_proc, artifacts)

    print(f"\n=== SCENARIO: {scenario} ===")
    print(f"Target ICAO: {target_icao}")
    print(f"Attack window: start_step={start_step}, end_step={end_step}")
    print(f"Processed rows: {len(df_proc)}")
    print(f"Sequence scores rows: {len(scores)}")
    print(f"Persistent anomaly rows: {len(anoms)}")

    if not scores.empty:
        print("\nTop 10 sequence scores:")
        print(
            scores.sort_values("anomaly_score", ascending=False).head(10)[
                [
                    "icao",
                    "timestamp",
                    "anomaly_score",
                    "reconstruction_score",
                    "feature_context_score",
                    "threshold",
                    "is_seq_anom",
                    "is_persistent_anom",
                ]
            ].to_string(index=False)
        )

    if not anoms.empty:
        print("\nPersistent anomalies:")
        print(
            anoms[
                [
                    "icao",
                    "timestamp",
                    "anomaly_score",
                    "threshold",
                    "is_seq_anom",
                    "is_persistent_anom",
                    "anomaly_type",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nNo persistent anomalies detected.")


if __name__ == "__main__":
    run_test("normal")
    run_test("teleportation")
    run_test("gps_spoofing")
    run_test("ghost_aircraft")
