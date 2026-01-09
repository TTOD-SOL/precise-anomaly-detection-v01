# precise-anomaly-detection





\# Precise Anomaly Detection (Precision-First Alerts)



An anomaly detection project designed for low false positives. It simulates multivariate sensor data (vibration, temperature, current), injects realistic fault patterns, then detects anomalies using robust statistics + rule stacking with a precision gate.



\## Why "precision-first"?

In operations, false alerts waste time. This project prioritizes precision by requiring multiple conditions before raising an alert.



\## Quickstart

pip install -r requirements.txt
python src/generate_sensor_data.py
python src/detect_anomalies.py
python src/evaluate.py



