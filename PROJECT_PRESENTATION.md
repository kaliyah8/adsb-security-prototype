# ADS-B Security Monitoring Platform

## Presentation Deck

### Slide 1. Title

**ADS-B Security Monitoring Platform**  
An interactive aviation cybersecurity system for detecting suspicious aircraft behavior using rule-based physics checks, machine learning, and hybrid ghost-aircraft detection.

Presenter: Kelsey G  
Project type: Aviation cybersecurity / anomaly detection / interactive dashboard

---

### Slide 2. Project Summary

This project monitors aircraft behavior from ADS-B-style surveillance data and highlights suspicious activity that may indicate:

- Teleportation or impossible jumps
- GPS spoofing or position displacement
- Ghost aircraft or false track injection

The platform combines:

- Physics-based detection rules
- An LSTM autoencoder anomaly model
- A hybrid ghost-aircraft detector
- An interactive Streamlit dashboard for analysis and explanation

---

### Slide 3. Why This Matters

ADS-B is widely used for aircraft position broadcasting, but it was not originally designed with strong security protections.

This creates opportunities for:

- False aircraft injection
- Position manipulation
- Impossible movement patterns
- Confusing or misleading airspace pictures

The goal of this project is to create a system that can:

- Detect suspicious aircraft behavior
- Explain why the behavior looks abnormal
- Visualize the event clearly for technical and non-technical audiences

---

### Slide 4. Main Goals

The project was designed to:

- Detect multiple attack types with high sensitivity
- Reduce false positives on normal flights
- Combine explainable rule-based logic with ML
- Support presentation-quality visualization
- Provide an interface that feels like a real monitoring platform

---

### Slide 5. Detection Systems

The platform uses **3 detection systems**:

#### 1. Rule-Based Flight Validation

This checks whether the aircraft motion is physically plausible.

Examples:

- Impossible speed
- Implausible acceleration
- Implausible vertical movement
- Implausible turn rate
- Position mismatch behavior

File: `anomaly_detection.py`

#### 2. Sequence Anomaly Model

This is the ML system. It learns normal aircraft behavior using an LSTM autoencoder and flags unusual sequences.

File: `ml_pipeline.py`

#### 3. Hybrid Ghost Aircraft Detection

This is a dedicated logic layer for detecting short-lived or suspicious ghost tracks that may not form strong LSTM sequences.

File: `ml_pipeline.py`

---

### Slide 6. System Architecture

```text
Raw ADS-B Data
    ->
Data Processing
    ->
Feature Engineering
    ->
Rule-Based Detection + ML Detection + Hybrid Ghost Detection
    ->
Evaluation + Visualization + Dashboard
```

Core pipeline files:

- `data_simulation.py`
- `data_processing.py`
- `ml_features.py`
- `anomaly_detection.py`
- `ml_pipeline.py`
- `ml_evaluation.py`
- `app.py`

---

### Slide 7. Data Sources

The project currently uses **simulated ADS-B data** as the primary dataset.

The simulator generates:

- aircraft ICAO identifiers
- timestamps
- latitude and longitude
- velocity
- heading
- barometric altitude
- vertical rate

The simulator can also inject attacks for controlled testing:

- teleportation
- GPS spoofing
- ghost aircraft

File: `data_simulation.py`

---

### Slide 8. Data Processing

The raw data is standardized and converted into motion-based kinematic features.

The processing stage computes:

- distance traveled
- implied speed
- acceleration
- vertical rate
- heading change
- turn rate

Important file:

- `data_processing.py`

This stage enforces required columns and prevents silent failures if the input schema is wrong.

---

### Slide 9. Feature Engineering

The ML model uses a richer feature set than the original version.

Current feature set:

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

The preprocessing also supports stronger imputation:

- KNN imputation

File: `ml_features.py`

---

### Slide 10. Machine Learning Model

The ML model is an **LSTM autoencoder**.

How it works:

1. Train the model on clean normal flight behavior
2. Feed aircraft motion sequences into the model
3. Measure reconstruction error
4. Combine reconstruction error with terminal feature context
5. Flag high-scoring sequences as anomalous

Current saved model settings:

- sequence length: `20`
- imputation strategy: `knn`
- smoothing window: `5`
- final threshold: `2.210689`
- reconstruction threshold: `1.646354`
- feature-context threshold: `3.490756`

Files:

- `lstm_autoencoder.py`
- `ml_pipeline.py`
- `train_ml.py`

---

### Slide 11. Physics-Based Rules

The rule-based system helps detect obviously impossible or suspicious motion.

Examples of rule logic:

- speed above realistic flight limit
- acceleration outside plausible bounds
- excessive vertical movement
- unrealistic turn rates
- persistent speed mismatch patterns

This is useful because:

- it is explainable
- it catches obvious anomalies quickly
- it provides a second layer besides ML

File: `anomaly_detection.py`

---

### Slide 12. Ghost Aircraft Detection

Ghost aircraft are difficult because they can be:

- short-lived
- late appearing
- weakly represented in sequence windows
- not strong enough for sequence-only ML scoring

To address that, the project uses **hybrid ghost detection**.

It looks for:

- young tracks
- late-born tracks
- suspiciously slow or motionless tracks
- feature patterns inconsistent with normal traffic

This allows ghost detections even when sequence-level ML metrics are weak.

---

### Slide 13. Dashboard

The project includes an interactive Streamlit dashboard that supports:

- overview of system state
- operations review
- aircraft inspection
- incident visualization
- ML intelligence metrics
- threat animation / explanation views
- settings and diagnostics

Main application file:

- `app.py`

Supporting UI file:

- `ui_components.py`

---

### Slide 14. Dashboard Pages

Current top-level dashboard pages:

- `Overview`
- `Operations`
- `ML Intelligence`
- `Threat Library`
- `Settings`

What they do:

- `Overview`: high-level system snapshot and monitoring story
- `Operations`: aircraft inspection, live map, alert review
- `ML Intelligence`: benchmark metrics and model output
- `Threat Library`: visual explanation of attack types
- `Settings`: controls, diagnostics, and technical details

---

### Slide 15. Training Workflow

To train the model:

```bash
cd /Users/kelseyg/Desktop/adsb_security_prototype_updated
source venv/bin/activate
cd adsb_security_prototype
python train_ml.py
```

What this does:

- generates or refreshes clean baseline simulated data
- builds features
- trains the LSTM autoencoder
- calibrates thresholds
- saves model artifacts

Saved artifacts:

- `artifacts/lstm_autoencoder.pt`
- `artifacts/scaler.pkl`
- `artifacts/meta.pkl`
- `artifacts/training_history.csv`

---

### Slide 16. Evaluation Workflow

To evaluate the retrained model:

```bash
python evaluate_ml.py
```

This creates:

- `artifacts/evaluation_metrics.json`
- `artifacts/evaluation_seq_details.csv`
- `artifacts/evaluation_row_details.csv`
- `artifacts/evaluation_scenario_summary.csv`

This measures:

- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- false-positive rate

Files:

- `evaluate_ml.py`
- `ml_evaluation.py`

---

### Slide 17. Scenario Verification Workflow

To test expected attack behavior directly:

```bash
python test_ml.py
```

This runs:

- normal traffic
- teleportation
- GPS spoofing
- ghost aircraft

This is the quickest operational check that the model and pipeline are behaving correctly.

File:

- `test_ml.py`

---

### Slide 18. Verified Results

The latest verified held-out benchmark after retraining:

#### Sequence-Level Metrics

- Precision: `0.1880`
- Recall: `0.3785`
- F1: `0.2512`
- ROC-AUC: `0.7757`
- PR-AUC: `0.4328`
- False-positive rate: `0.0154`

#### Row-Level Metrics

- Precision: `0.2141`
- Recall: `0.5573`
- F1: `0.3094`
- False-positive rate: `0.0129`

Source:

- `artifacts/evaluation_metrics.json`

---

### Slide 19. Per-Scenario Results

#### Normal

- sequence recall: `0.0`
- row recall: `0.0`
- detections any rate: `0.0`

Interpretation:

- the system stayed quiet on normal held-out data

#### Teleportation

- sequence precision: `0.4863`
- sequence recall: `0.2840`
- row recall: `0.4303`
- detections any rate: `0.6`

#### GPS Spoofing

- sequence precision: `0.3191`
- sequence recall: `0.4884`
- row recall: `0.5939`
- detections any rate: `1.0`

#### Ghost Aircraft

- sequence precision: `0.0`
- sequence recall: `0.0`
- row recall: `0.8889`
- detections any rate: `1.0`

Interpretation:

- ghost aircraft is being detected mainly by the hybrid detector, not the sequence model

---

### Slide 20. Fresh Scenario Test Results

Latest end-to-end scenario verification after retraining:

- `normal`: `0` persistent anomalies
- `teleportation`: `23` persistent anomalies
- `gps_spoofing`: `18` persistent anomalies
- `ghost_aircraft`: `8` hybrid ghost detections

Interpretation:

- normal traffic does not produce persistent ML alerts
- teleportation is strongly detected
- GPS spoofing is strongly detected
- ghost aircraft is detected through the hybrid logic

---

### Slide 21. What Works Well

Strengths of the project:

- multi-layer detection, not ML-only
- explainable physics-based rules
- sequence-based ML for temporal anomaly detection
- hybrid ghost aircraft logic
- evaluation pipeline with saved benchmark artifacts
- presentation-focused dashboard
- attack explanation visuals
- reproducible training and testing commands

---

### Slide 22. Limitations

Current limitations:

- training and testing still rely heavily on simulated data
- precision is still lower than ideal for a production system
- ghost aircraft sequence metrics are weak because many ghost tracks are too short for sequence formation
- browser-side UI polish still needs refinement in some views
- real-world validation on live ADS-B data is still limited

Important honesty statement:

This project is strong and working, but it does **not** guarantee perfect detection in every real-world environment.

---

### Slide 23. Improvements Already Made

Major improvements completed during development:

- richer feature engineering
- KNN imputation
- dynamic threshold calibration
- hybrid ghost detection
- better evaluation labeling alignment
- improved GPS spoofing simulation behavior
- safer app state handling
- dashboard restructuring and modernized views

---

### Slide 24. Recommended Next Steps

If the project continues, the highest-value next steps are:

- validate on real-world ADS-B captures
- add more regression tests
- improve precision and recall further
- simplify and polish dashboard layout
- build a single-command project verification script
- create exportable incident reports

---

### Slide 25. File Map

#### Core Application

- `app.py` -> main Streamlit dashboard
- `ui_components.py` -> reusable UI styling/helpers

#### Data and Detection

- `data_simulation.py` -> flight simulation and attack injection
- `data_processing.py` -> preprocessing and kinematic calculations
- `anomaly_detection.py` -> rule-based anomaly detection
- `ml_features.py` -> feature engineering and imputation
- `ml_pipeline.py` -> model training, scoring, thresholds, hybrid ghost logic
- `lstm_autoencoder.py` -> neural network model definition

#### Evaluation and Execution

- `train_ml.py` -> training entry point
- `evaluate_ml.py` -> held-out benchmark evaluation
- `test_ml.py` -> scenario verification runner
- `ml_evaluation.py` -> metric computation and evaluation output generation

#### Supporting Files

- `data_fetcher.py` -> data loading / fetching support
- `anomaly_scene.py` -> scenario visuals
- `anomaly_playbook.py` -> attack guidance and descriptions
- `tests/test_ml_system.py` -> automated regression tests

#### Artifacts

- `artifacts/lstm_autoencoder.pt`
- `artifacts/scaler.pkl`
- `artifacts/meta.pkl`
- `artifacts/training_history.csv`
- `artifacts/evaluation_metrics.json`
- `artifacts/evaluation_seq_details.csv`
- `artifacts/evaluation_row_details.csv`
- `artifacts/evaluation_scenario_summary.csv`

---

### Slide 26. Demo Script

Suggested live demo order:

1. Start on `Overview`
2. Explain the project goal in one sentence
3. Show the three detection systems
4. Switch to `Operations`
5. Run `Teleportation`
6. Show the incident map and aircraft path
7. Run `GPS Spoofing`
8. Show alert behavior and comparison
9. Open `ML Intelligence`
10. Show benchmark metrics and score separation
11. Open `Threat Library`
12. Show attack animation examples

---

### Slide 27. One-Minute Explanation

This project is an aviation cybersecurity dashboard that detects suspicious aircraft behavior using both physics-based checks and machine learning. It was built to identify problems like teleportation, GPS spoofing, and ghost aircraft, then explain those detections visually through an interactive monitoring platform. The system was retrained and verified with automated tests, scenario checks, and held-out evaluation metrics. It performs well on simulated attacks, stays quiet on normal traffic, and provides a strong foundation for future real-world validation.

---

### Slide 28. Closing Statement

This project shows:

- applied machine learning
- cybersecurity reasoning
- aviation-domain understanding
- data engineering
- evaluation and testing discipline
- interactive product design

It is more than a prototype dashboard. It is a structured aviation security monitoring system with measurable detection behavior and a clear path for future improvement.

---

## Appendix

### Commands

#### Run the app

```bash
cd /Users/kelseyg/Desktop/adsb_security_prototype_updated
source venv/bin/activate
cd adsb_security_prototype
streamlit run app.py
```

#### Retrain

```bash
python train_ml.py
```

#### Evaluate

```bash
python evaluate_ml.py
```

#### Scenario test

```bash
python test_ml.py
```

#### Regression tests

```bash
python -m unittest discover -s tests -v
```

### Current Verified State

- App imports successfully
- Core modules compile successfully
- Regression tests pass
- Training completes
- Evaluation completes
- Scenario checks complete successfully

