from pathlib import Path
import numpy as np
import pandas as pd

INP = Path("data/raw/sensor_stream.csv")
OUT = Path("outputs/alerts.csv")

def mad(arr: np.ndarray) -> float:
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    m = mad(x)
    if m == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / m  # consistent with std under normality

def main():
    df = pd.read_csv(INP, parse_dates=["timestamp"])

    # Rolling baseline (more realistic than global stats)
    window = 240  # 4 hours
    for col in ["vibration_g", "temp_c", "current_a"]:
        df[f"{col}_rz"] = (
            df[col]
            .rolling(window=window, min_periods=window)
            .apply(lambda s: robust_z(s.values)[-1], raw=False)
        )

    # Precision-first logic:
    # Alert only when at least TWO signals are "highly abnormal"
    # This reduces false positives.
    vib_hit = df["vibration_g_rz"].abs() > 4.5
    temp_hit = df["temp_c_rz"].abs() > 4.0
    amp_hit = df["current_a_rz"].abs() > 4.0

    # Gate: require 2-of-3
    hits = vib_hit.astype(int) + temp_hit.astype(int) + amp_hit.astype(int)
    df["alert"] = (hits >= 2)

    # Add reason codes for interpretability
    reasons = []
    for v, t, a in zip(vib_hit, temp_hit, amp_hit):
        r = []
        if v: r.append("VIB")
        if t: r.append("TEMP")
        if a: r.append("AMPS")
        reasons.append("+".join(r) if r else "")
    df["reason"] = reasons

    # Simple anomaly score: sum of absolute robust z-scores (only where defined)
    df["score"] = df[["vibration_g_rz", "temp_c_rz", "current_a_rz"]].abs().sum(axis=1)

    alerts = df[df["alert"]].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    alerts[["timestamp", "vibration_g", "temp_c", "current_a", "score", "reason"]].to_csv(OUT, index=False)

    print(f"✅ Alerts saved to {OUT} ({len(alerts)} alerts)")
    print("Tip: adjust thresholds in detect_anomalies.py to trade precision vs recall.")

if __name__ == "__main__":
    main()
