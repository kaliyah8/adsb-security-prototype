from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from data_processing import process_adsb_data
from data_simulation import generate_aircraft_data, inject_ghost_aircraft_attack
from ml_evaluation import run_evaluation
from ml_features import FEATURE_COLS, build_feature_frame
from ml_pipeline import load_artifacts, score_sequences, train_lstm_autoencoder


class MLSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        self._tmpdir.cleanup()

    def test_feature_frame_supports_knn_imputation_and_expected_columns(self) -> None:
        df_raw = generate_aircraft_data(num_aircraft=5, time_steps=35, seed=11)
        df_proc = process_adsb_data(df_raw)
        df_proc.loc[df_proc.index[::7], "velocity"] = pd.NA
        df_proc.loc[df_proc.index[::9], "heading"] = pd.NA

        feat = build_feature_frame(df_proc, smooth_window=5, impute="knn", impute_neighbors=3, add_quality=True)

        self.assertFalse(feat.empty)
        for col in FEATURE_COLS:
            self.assertIn(col, feat.columns)
            self.assertFalse(feat[col].isna().any(), msg=f"{col} still contains NaNs")

    def test_training_loading_and_ghost_detection_end_to_end(self) -> None:
        df_train = process_adsb_data(generate_aircraft_data(num_aircraft=8, time_steps=60, seed=13))
        artifacts = train_lstm_autoencoder(
            df_train,
            seq_len=10,
            epochs=1,
            batch_size=32,
            lr=1e-3,
            max_dt_sec=10,
            threshold_percentile=99.0,
            persistence_k=2,
            persistence_m=3,
            ghost_birth_grace_sec=10,
            ghost_age_window_sec=15,
            smooth_window=3,
            impute="median",
            combined_score_weight=0.1,
            print_dt_stats=False,
        )

        loaded = load_artifacts()
        self.assertIsNotNone(loaded)
        self.assertEqual(artifacts.seq_len, loaded.seq_len)

        df_attack_raw = generate_aircraft_data(num_aircraft=8, time_steps=60, seed=17)
        df_attack_raw = inject_ghost_aircraft_attack(df_attack_raw, ghost_icao="ghost001", start_step=20, end_step=28)
        df_attack = process_adsb_data(df_attack_raw)

        _, anoms = score_sequences(df_attack, loaded)
        self.assertFalse(anoms.empty)
        self.assertTrue((anoms["anomaly_type"] == "Hybrid Ghost Aircraft Candidate").any())

    def test_evaluation_pipeline_writes_outputs(self) -> None:
        df_train = process_adsb_data(generate_aircraft_data(num_aircraft=6, time_steps=50, seed=19))
        train_lstm_autoencoder(
            df_train,
            seq_len=8,
            epochs=1,
            batch_size=32,
            lr=1e-3,
            max_dt_sec=10,
            threshold_percentile=99.0,
            persistence_k=2,
            persistence_m=3,
            ghost_birth_grace_sec=10,
            ghost_age_window_sec=15,
            smooth_window=3,
            impute="median",
            combined_score_weight=0.1,
            print_dt_stats=False,
        )

        result = run_evaluation(runs_per_scenario=2, base_seed=200, save=True)

        self.assertIn("metrics", result)
        self.assertTrue(os.path.exists("artifacts/evaluation_metrics.json"))
        self.assertTrue(os.path.exists("artifacts/evaluation_seq_details.csv"))
        self.assertTrue(os.path.exists("artifacts/evaluation_row_details.csv"))
        self.assertTrue(os.path.exists("artifacts/evaluation_scenario_summary.csv"))


if __name__ == "__main__":
    unittest.main()
