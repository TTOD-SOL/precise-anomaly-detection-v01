from pathlib import Path
import pandas as pd

INP = Path("data/raw/sensor_stream.csv")

def main():
    df = pd.read_csv(INP, parse_dates=["timestamp"])

    # Re-run alert logic (kept minimal for evaluation)
    # If you want, import from detect_anomalies.py, but keeping standalone is simpler for GitHub.
    window = 240

    def robust_z_last(s):
        x = s.values
        med = x.mean()  # fallback if MAD window is too small
        # using rolling mean baseline for evaluation simplicity
        return x[-1] - med

    # We'll approximate alert using rolling z-like deviation to avoid duplicated robust helpers.
    # For production, compute from detect_anomalies.py and save full stream with alert labels.
    df["vib_dev"] = df["vibration_g"] - df["vibration_g"].rolling(window, min_periods=window).median()
    df["temp_dev"] = df["temp_c"] - df["temp_c"].rolling(window, min_periods=window).median()
    df["amp_dev"] = df["current_a"] - df["current_a"].rolling(window, min_periods=window).median()

    vib_hit = df["vib_dev"].abs() > 0.35
    temp_hit = df["temp_dev"].abs() > 3.5
    amp_hit = df["amp_dev"].abs() > 3.2
    hits = vib_hit.astype(int) + temp_hit.astype(int) + amp_hit.astype(int)
    df["alert"] = (hits >= 2)

    # Drop early NaNs
    df = df.dropna(subset=["vib_dev","temp_dev","amp_dev"])

    y_true = df["is_anomaly"].astype(int)
    y_pred = df["alert"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("=== Evaluation ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

if __name__ == "__main__":
    main()
