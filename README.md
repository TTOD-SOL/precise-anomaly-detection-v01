# precise-anomaly-detection





\# Precise Anomaly Detection (Precision-First Alerts)



An anomaly detection project designed for low false positives. It simulates multivariate sensor data (vibration, temperature, current), injects realistic fault patterns, then detects anomalies using robust statistics + rule stacking with a precision gate.



\## Why "precision-first"?

In operations, false alerts waste time. This project prioritizes precision by requiring multiple conditions before raising an alert.



\## Quickstart



pip install -r requirements.txt



python src/generate\_sensor\_data.py



python src/detect\_anomalies.py



python src/evaluate.py





\## Results (local run)

\- Generated 12,000 synthetic multivariate sensor rows



\- Ground-truth anomalies injected: 2,600



\- Precision-first detector produced: 27 alerts



\- Confusion matrix (example run):

&nbsp; - TP=90, FP=1, FN=2510, TN=9160



\- Metrics:

&nbsp; - Precision: 0.989

&nbsp; - Recall: 0.035

&nbsp; - F1: 0.067



\### Interpretation

This configuration prioritizes precision (minimizing false positives) to reduce alert fatigue. Thresholds can be tuned in `src/detect\_anomalies.py` to increase recall when needed.



\### Alert characteristics

\- Total alerts: 27 (out of 12,000 samples)



\- All alerts triggered by multi-signal agreement (VIB + AMPS)



\- Each alert includes an interpretable reason code and severity score



Example alert fields:

\- timestamp



\- vibration\_g, temp\_c, current\_a



\- anomaly score



\- reason (e.g., VIB+AMPS)



