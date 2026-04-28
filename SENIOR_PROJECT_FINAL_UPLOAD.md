# ADS-B Security Monitoring Platform

## Senior Project Final Upload

### Student

Kelsey G

### Project Title

ADS-B Security Monitoring Platform

### Project Type

Aviation cybersecurity, anomaly detection, machine learning, and interactive dashboard visualization

---

## 1. Project Overview

This project is an aviation cybersecurity platform designed to detect suspicious aircraft behavior from ADS-B-style surveillance data. The system focuses on identifying anomalies such as teleportation, GPS spoofing, and ghost aircraft injection. It combines rule-based physics validation with machine learning and interactive visualization to create a monitoring platform that is both technically structured and understandable for broader audiences.

The final system includes:

- simulated ADS-B data generation
- feature engineering and preprocessing
- rule-based anomaly detection
- an LSTM autoencoder anomaly model
- hybrid ghost aircraft detection
- evaluation scripts and benchmark artifacts
- a Streamlit dashboard for interactive review and presentation

The project is implemented in Python and combines cybersecurity reasoning, motion physics, machine learning, testing, and dashboard-based visualization into one complete workflow.

---

## 2. Problem Statement

ADS-B is widely used in aviation for broadcasting aircraft position and movement information. However, it was not originally built with strong security protections. Because of this, surveillance data may be vulnerable to attacks such as:

- impossible aircraft jumps
- false position injection
- spoofed movement
- ghost or fabricated aircraft tracks

These attacks can create confusion in the airspace picture and reduce trust in surveillance systems. The purpose of this project is to build a structured system that can detect and explain these threats using both deterministic rules and machine learning.

---

## 3. Project Goals

The primary goals of the project were:

- detect multiple types of suspicious aircraft behavior
- reduce false positives on normal flights
- use explainable physics-based rules
- apply machine learning to learn normal behavior patterns
- improve ghost aircraft detection
- provide visual tools to understand and demonstrate anomalies
- create a polished, presentation-ready interface

---

## 4. Detection Approach

The platform uses three main detection systems.

### 4.1 Rule-Based Flight Validation

This system detects physically implausible aircraft behavior based on flight realism. It checks for:

- unrealistic speed
- unrealistic acceleration
- unrealistic vertical movement
- unrealistic turn rate
- mismatches between reported and implied movement

Current rule thresholds:

- maximum implied speed: `400.0 m/s`
- maximum acceleration: `12.0 m/s^2`
- maximum vertical speed: `40.0 m/s`
- maximum turn rate: `10.0 deg/s`
- spoofing mismatch threshold: `60.0 m/s`
- consecutive spoofing requirement: `3`

File:

- `anomaly_detection.py`

### 4.2 Sequence Anomaly Model

This system uses an LSTM autoencoder to learn normal aircraft behavior from clean training data. New flight sequences are scored for anomaly likelihood based on reconstruction error and feature context.

Files:

- `lstm_autoencoder.py`
- `ml_pipeline.py`
- `train_ml.py`

### 4.3 Hybrid Ghost Aircraft Detection

Ghost aircraft are not always captured well by sequence-only ML, especially when the tracks are short-lived or appear suddenly. A hybrid detector was added to specifically identify suspicious ghost track behavior.

File:

- `ml_pipeline.py`

---

## 5. Data and Simulation

The project currently uses simulated ADS-B data for model development and evaluation.

The simulated data includes:

- `icao`
- `timestamp`
- `latitude`
- `longitude`
- `velocity`
- `heading`
- `baro_altitude`
- `vertical_rate`

Default simulator settings:

- number of aircraft: `25`
- time steps per run: `120`
- time step size: `1.0 second`
- airspace center: `(33.9416, -118.4085)`
- default seed: `7`

Default generated aircraft ranges:

- initial speed: `90.0` to `250.0 m/s`
- initial altitude: `500.0` to `11000.0 m`
- initial heading: `0.0` to `360.0 deg`

The simulator can inject multiple attacks:

- teleportation
- GPS spoofing
- ghost aircraft

File:

- `data_simulation.py`

This approach makes it possible to train, test, and compare the detection systems in a controlled environment.

Attack implementation notes:

- teleportation injects an abrupt large coordinate jump
- GPS spoofing applies a progressive coordinate drift across the attack window
- ghost aircraft creates a new short-lived false aircraft track with its own ICAO identifier

---

## 6. Data Processing and Feature Engineering

Before detection, the project processes raw surveillance data into kinematic and temporal features.

### 6.1 Data Processing

The processing stage computes:

- distance traveled
- implied speed
- acceleration
- vertical rate
- heading difference
- turn rate

File:

- `data_processing.py`

### 6.2 Feature Engineering

The ML model uses the following engineered features:

- `implied_speed_mps`
- `velocity`
- `accel_mps2`
- `vert_rate_mps`
- `turn_rate_dps`
- `speed_mismatch_mps`
- `altitude_m`
- `heading_delta_deg`
- `yaw_rate_dps`
- `speed_delta_mps`
- `relative_speed_to_median_mps`
- `velocity_window_mean`
- `velocity_window_std`
- `accel_window_mean`
- `accel_window_std`

The preprocessing pipeline also uses KNN imputation to handle missing data more robustly.

File:

- `ml_features.py`

---

## 7. Machine Learning Model

The machine learning model used in this project is an LSTM autoencoder.

### 7.1 Why This Model Was Chosen

Aircraft behavior is sequential and time-dependent. An LSTM autoencoder is appropriate because it can learn patterns across a window of movement data rather than relying on a single point at a time.

### 7.2 Current Model Settings

The current saved model state is:

- sequence length: `20`
- feature count: `15`
- imputation strategy: `knn`
- smoothing window: `5`
- final threshold: `2.210689`
- reconstruction threshold: `1.646354`
- feature-context threshold: `3.490756`

Latest verified production training configuration:

- epochs: `24`
- batch size: `64`
- learning rate: `1e-4`
- threshold percentile: `99.3`
- threshold MAD scale: `6.0`
- max allowed internal sequence gap: `10 seconds`
- minimum quality score: `0.80`
- persistence rule: `5 of 6`
- ghost birth grace period: `20 seconds`
- ghost age window: `20 seconds`
- KNN neighbors: `5`
- combined score weight: `0.20`
- calibration percentile: `99.7`

Artifacts:

- `artifacts/lstm_autoencoder.pt`
- `artifacts/scaler.pkl`
- `artifacts/meta.pkl`

---

## 8. System Architecture

The overall system flow is:

```text
Simulated or input ADS-B data
    ->
Preprocessing and kinematic calculation
    ->
Feature engineering and imputation
    ->
Rule-based detection + ML anomaly scoring + Hybrid ghost logic
    ->
Evaluation artifacts + interactive dashboard output
```

Main files involved:

- `data_simulation.py`
- `data_processing.py`
- `ml_features.py`
- `anomaly_detection.py`
- `ml_pipeline.py`
- `ml_evaluation.py`
- `app.py`

---

## 9. Dashboard and Visualization

The project includes a Streamlit dashboard to display system status, airspace behavior, attacks, and model output.

Main dashboard pages:

- `Overview`
- `Operations`
- `ML Intelligence`
- `Threat Library`
- `Settings`

The dashboard supports:

- aircraft monitoring
- incident visualization
- anomaly summaries
- benchmark result review
- animated attack explanation scenes

Operational modes supported by the dashboard:

- simulation mode
- live OpenSky snapshot mode

Additional dashboard capabilities:

- aircraft inspector
- incident map review
- ML benchmark summaries
- active-scenario animation views
- run-state and alert summaries

Files:

- `app.py`
- `ui_components.py`
- `anomaly_scene.py`
- `anomaly_playbook.py`

---

## 10. Training and Evaluation Workflow

### 10.1 Training

To train the model:

```bash
cd /Users/kelseyg/Desktop/adsb_security_prototype_updated
source venv/bin/activate
cd adsb_security_prototype
python train_ml.py
```

Training actions:

- generates clean baseline simulation data
- builds features
- trains the LSTM autoencoder
- calibrates thresholds
- saves artifacts

Latest verified training run details:

- new raw rows generated: `3000`
- new processed rows: `3000`
- retained clean history: `3000`
- rows after feature build: `3000`
- rows after quality gating: `2975`
- training rows: `2380`
- validation rows: `595`
- `X_train` shape: `(1905, 20, 15)`
- `X_val` shape: `(120, 20, 15)`

### 10.2 Evaluation

To evaluate the saved model:

```bash
python evaluate_ml.py
```

Evaluation outputs:

- `artifacts/evaluation_metrics.json`
- `artifacts/evaluation_seq_details.csv`
- `artifacts/evaluation_row_details.csv`
- `artifacts/evaluation_scenario_summary.csv`

### 10.3 Scenario Verification

To run practical scenario checks:

```bash
python test_ml.py
```

This tests:

- normal traffic
- teleportation
- GPS spoofing
- ghost aircraft

---

## 11. Testing and Validation

The project was validated in several ways.

### 11.1 Code Health

- key modules compile successfully with `py_compile`
- `app.py` imports successfully

### 11.2 Automated Tests

Regression tests:

```bash
python -m unittest discover -s tests -v
```

Current result:

- `3/3` tests passed

Test file:

- `tests/test_ml_system.py`

The current tests verify:

- KNN imputation and feature-column completeness
- end-to-end training, artifact loading, and ghost-aircraft detection
- evaluation pipeline output generation

### 11.3 Math and Calculation Checks

The project’s distance and movement calculations were also spot-checked:

- 1000 meter move test returned `1000.0 m`
- known straight-line processing test returned:
  - distance `1000.1002 m`
  - implied speed `100.0100 m/s`
  - acceleration `0.0`
  - turn rate `0.0`

This confirms the core kinematic calculations are behaving correctly.

Additional validation completed:

- fresh model training completed successfully
- fresh held-out evaluation completed successfully
- fresh scenario verification completed successfully
- the application imports successfully without a Python runtime exception

---

## 12. Final Results

The latest verified benchmark after retraining is:

### 12.1 Sequence-Level Metrics

- Precision: `0.1880`
- Recall: `0.3785`
- F1: `0.2512`
- ROC-AUC: `0.7757`
- PR-AUC: `0.4328`
- False-positive rate: `0.0154`

### 12.2 Row-Level Metrics

- Precision: `0.2141`
- Recall: `0.5573`
- F1: `0.3094`
- False-positive rate: `0.0129`

Confusion counts:

#### Sequence level

- true negatives: `48720`
- false positives: `760`
- false negatives: `289`
- true positives: `176`

#### Row level

- true negatives: `58903`
- false positives: `767`
- false negatives: `166`
- true positives: `209`

Source:

- `artifacts/evaluation_metrics.json`

---

## 13. Per-Scenario Performance

### Normal

- no persistent anomalies
- no held-out detections

Interpretation:

- the system stayed quiet on normal traffic

### Teleportation

- sequence precision: `0.4863`
- sequence recall: `0.2840`
- row recall: `0.4303`
- scenario detection rate: `0.6`

### GPS Spoofing

- sequence precision: `0.3191`
- sequence recall: `0.4884`
- row recall: `0.5939`
- scenario detection rate: `1.0`

### Ghost Aircraft

- row recall: `0.8889`
- scenario detection rate: `1.0`

Interpretation:

- ghost aircraft is being detected primarily by the hybrid detector rather than the sequence model alone

---

## 14. Scenario Verification Results

The latest end-to-end verification after retraining produced:

- `normal`: `0` persistent anomalies
- `teleportation`: `23` persistent anomalies
- `gps_spoofing`: `18` persistent anomalies
- `ghost_aircraft`: `8` hybrid ghost detections

This shows the system is detecting the expected attack scenarios while staying quiet on the normal scenario.

---

## 15. Files and Project Structure

### Core Dashboard

- `app.py`
- `ui_components.py`

### Data and Simulation

- `data_simulation.py`
- `data_processing.py`
- `data_fetcher.py`

### Detection and ML

- `anomaly_detection.py`
- `ml_features.py`
- `ml_pipeline.py`
- `lstm_autoencoder.py`

### Evaluation and Scripts

- `train_ml.py`
- `evaluate_ml.py`
- `test_ml.py`
- `ml_evaluation.py`

### Supporting Visual Files

- `anomaly_scene.py`
- `anomaly_playbook.py`

### Utility File

- `utils.py`

### Tests

- `tests/test_ml_system.py`

### Saved Artifacts

- `artifacts/lstm_autoencoder.pt`
- `artifacts/scaler.pkl`
- `artifacts/meta.pkl`
- `artifacts/training_history.csv`
- `artifacts/evaluation_metrics.json`
- `artifacts/evaluation_seq_details.csv`
- `artifacts/evaluation_row_details.csv`
- `artifacts/evaluation_scenario_summary.csv`

### Dependencies

The current dependency set from `requirements.txt` is:

- `streamlit`
- `pandas`
- `numpy`
- `requests`
- `pydeck`
- `torch`
- `scikit-learn`
- `joblib`

---

## 16. Strengths of the Project

The strongest parts of this project are:

- multi-layer detection design
- explainable rule-based logic
- time-series ML anomaly scoring
- hybrid handling for ghost aircraft
- reproducible training and evaluation scripts
- saved benchmark artifacts
- interactive visual dashboard
- technical depth across ML, data engineering, and cybersecurity
- clear separation between simulation, preprocessing, detection, evaluation, and presentation layers

---

## 17. Limitations

The project also has important limitations.

- the system is still mostly validated on simulated data
- precision and recall can still be improved
- ghost sequence-level metrics are weak because ghost tracks are often short
- real-world validation is still limited
- UI polish and user experience can still be improved

This means the project is strong and functional, but it should not be described as a perfectly guaranteed real-world defense system.

---

## 18. Future Work

Recommended next steps:

- evaluate on real-world ADS-B captures
- improve precision and recall further
- add more automated regression tests
- add one-command verification for the whole project
- improve dashboard visual consistency
- add exportable incident reports
- improve live-feed robustness and monitoring
- expand deployment and reproducibility documentation

---

## 19. Conclusion

This project demonstrates a complete aviation cybersecurity monitoring system that combines simulation, data processing, rule-based validation, machine learning, evaluation, and interactive visualization. It successfully detects teleportation, GPS spoofing, and ghost aircraft behavior in controlled testing. It also includes a structured evaluation process and a presentation-ready dashboard, making it a strong senior project that shows applied skill in cybersecurity, machine learning, and software development.

Based on the current verified repository state, the project is functioning correctly at the code, training, evaluation, and scenario-verification levels. The largest remaining opportunities are real-world validation and final user-interface refinement, not basic system correctness.

---

## 20. Appendix

### Run the Dashboard

```bash
cd /Users/kelseyg/Desktop/adsb_security_prototype_updated
source venv/bin/activate
cd adsb_security_prototype
streamlit run app.py
```

### Retrain the Model

```bash
python train_ml.py
```

### Evaluate the Model

```bash
python evaluate_ml.py
```

### Run Scenario Verification

```bash
python test_ml.py
```

### Run Regression Tests

```bash
python -m unittest discover -s tests -v
```

### Current Verified Artifact State

- threshold: `2.210689`
- sequence length: `20`
- reconstruction threshold: `1.646354`
- feature-context threshold: `3.490756`
- feature count: `15`
