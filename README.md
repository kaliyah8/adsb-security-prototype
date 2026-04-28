# Enhanced ADS-B Security Monitoring Prototype

This project implements an enhanced physics-based anomaly detection prototype for securing the Automatic Dependent Surveillance-Broadcast (ADS-B) system. It now features integration with live ADS-B data sources, advanced anomaly detection capabilities, and comprehensive cyber attack simulation with visualization of impacts on Air Traffic Control (ATC), pilots, and aircraft systems.

## Features

*   **Data Acquisition:** Fetch live ADS-B data from the OpenSky Network API or use simulated data.
*   **Data Processing:** Prepares raw ADS-B data for anomaly detection, including unit conversions and feature engineering.
*   **Enhanced Anomaly Detection:** Applies physics-based rules (Position Consistency, Speed/Altitude Consistency, Ghost Aircraft Detection, GPS Spoofing Detection) to identify anomalous aircraft behavior.
*   **Cyber Attack Simulation:** Injects various cyber attack scenarios (Teleportation, Ghost Aircraft, GPS Spoofing) into the data stream to test detection and visualize impact.
*   **Interactive Visualization:** A Streamlit dashboard displays air traffic on a map, highlighting detected anomalies and actively attacked aircraft in real-time, along with simulated alerts for ATC, pilots, and system impact.
*   **ML Option (LSTM Sequence Model):** Optional LSTM autoencoder that learns normal motion over time and flags sequences with high reconstruction error.

## Project Structure

```
adsb_security_prototype/
├── app.py
├── data_simulation.py
├── data_processing.py
├── anomaly_detection.py
├── data_fetcher.py
├── utils.py
├── ml_features.py
├── lstm_autoencoder.py
├── ml_pipeline.py
├── models/               # generated at runtime when you train
├── requirements.txt
├── README.md
└── resources.md
```

## Setup and Running the Project

To run this project locally, follow these steps:

1.  **Navigate to the project directory:**

    ```bash
    cd adsb_security_prototype/adsb_security_prototype 2
    ```

2.  **Create a Python virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**

    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This command will open the Streamlit application in your default web browser. If it doesn't open automatically, navigate to the URL displayed in your terminal (usually `http://localhost:8501`).

## Using the Dashboard

*   **Select Data Source:** Choose between "Simulated Data" and "Live ADS-B Data" in the sidebar.
    *   For **Simulated Data**: Click "Run Simulation" to generate data with injected attacks. You can select the type of attack to simulate.
    *   For **Live ADS-B Data**: Click "Fetch Live Data" to retrieve current aircraft states from the OpenSky Network API.
*   Use the "Select Time Step" slider to navigate through the data and observe aircraft movement and anomaly detections on the map.
*   Normal aircraft are displayed in green, anomalous aircraft in red, and actively attacked aircraft in orange.
*   Simulated alerts for ATC, pilots, and system impact will be displayed when an active attack is being simulated.

### ML Mode (LSTM)

*   Select **Detection Method → ML (LSTM Sequence Model)**.
*   Click **Train / Retrain Model** (first time only).
    * The app trains on the current dataset after filtering out obvious rule-based anomalies.
*   The model is saved under `models/lstm_artifacts.joblib` and will auto-load next run.

## Cyber Attack Simulation Details

This prototype includes the following simulated cyber attacks:

*   **Teleportation Attack:** An aircraft's reported position suddenly jumps to a distant location.
*   **Ghost Aircraft Attack:** A non-existent aircraft appears in the airspace, potentially causing false collision alerts.
*   **GPS Spoofing Attack:** An aircraft's reported GPS coordinates are subtly shifted, making it appear in a slightly incorrect location without drastic changes in speed or heading. This mimics real-world GPS spoofing incidents.

## Resources

For a detailed list of data sources, research papers, and libraries used in this project, please refer to `resources.md`.
