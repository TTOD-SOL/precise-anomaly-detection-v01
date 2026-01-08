"""
Generate synthetic multivariate sensor data for anomaly detection.

Simulates realistic industrial signals:
- Vibration (g)
- Temperature (°C)
- Electrical current (A)

Injects labeled anomaly patterns:
1. Bearing wear (gradual vibration drift + spikes)
2. Cooling failure (temperature shift)
3. Mechanical jam (short high current + vibration)

Output:
- data/raw/sensor_stream.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)

    # Output directory
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Time series length (minutes)
    n = 12_000
    start_time = pd.Timestamp("2025-01-01")
    timestamps = start_time + pd.to_timedelta(np.arange(n), unit="m")

    # Baseline sensor signals
    vibration = rng.normal(loc=1.0, scale=0.08, size=n)     # g-force
    temperature = rng.normal(loc=55.0, scale=1.2, size=n)   # °C
    current = rng.normal(loc=12.0, scale=0.6, size=n)       # Amps

    # Ground-truth anomaly labels
    is_anomaly = np.zeros(n, dtype=int)

    # ---- Inject anomaly patterns ----

    # 1) Bearing wear: gradual vibration increase + spikes
    wear_start, wear_end = 3500, 5200
    vibration[wear_start:wear_end] += np.linspace(0, 0.6, wear_end - wear_start)
    spike_idx = np.arange(wear_start, wear_end, 90)
    vibration[spike_idx] += rng.normal(0.9, 0.15, size=len(spike_idx))
    is_anomaly[wear_start:wear_end] = 1

    # 2) Cooling system issue: sustained temperature increase
    cool_start, cool_end = 7000, 7800
    temperature[cool_start:cool_end] += 6.0
    is_anomaly[cool_start:cool_end] = 1

    # 3) Mechanical jam: short high current + vibration spike
    jam_start, jam_end = 9800, 9900
    current[jam_start:jam_end] += 6.0
    vibration[jam_start:jam_end] += 0.5
    is_anomaly[jam_start:jam_end] = 1

    # Build DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "vibration_g": vibration,
        "temp_c": temperature,
        "current_a": current,
        "is_anomaly": is_anomaly
    })

    # Write output
    output_path = out_dir / "sensor_stream.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Sensor data generated: {output_path}")
    print(f"Rows: {len(df)} | Anomalies: {df['is_anomaly'].sum()}")


if __name__ == "__main__":
    main()
