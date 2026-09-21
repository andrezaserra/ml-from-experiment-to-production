"""Generate the synthetic satellite telemetry used in the minicourse.

This dataset is entirely synthetic and exists only for teaching software/MLOps
concepts. The numerical ranges and anomaly mechanisms must not be interpreted
as operational limits of any real satellite.
"""

from pathlib import Path
import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_SAMPLES = 6000


def generate_telemetry(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)

    # Simplified 90-minute orbital pattern used only to create correlated signals.
    orbit_minute = t % 90
    eclipse = (orbit_minute >= 55).astype(int)

    battery_voltage = (
        28.2
        - 0.8 * eclipse
        + 0.18 * np.sin(2 * np.pi * t / 180)
        + rng.normal(0, 0.18, n_samples)
    )

    battery_current = (
        1.8
        - 3.6 * eclipse
        + 0.35 * np.sin(2 * np.pi * t / 90)
        + rng.normal(0, 0.35, n_samples)
    )

    battery_temperature = (
        24.5
        + 2.0 * eclipse
        + 1.1 * np.sin(2 * np.pi * t / 500)
        + rng.normal(0, 0.7, n_samples)
    )

    solar_panel_current = 4.8 * (1 - eclipse) + rng.normal(0, 0.35, n_samples)
    solar_panel_current = np.clip(solar_panel_current, 0, None)

    bus_voltage = 28.0 + rng.normal(0, 0.12, n_samples)
    attitude_error = np.abs(rng.normal(0.025, 0.018, n_samples))

    # Inject several types of anomalies.
    anomaly = np.zeros(n_samples, dtype=int)
    fault_indices = rng.choice(n_samples, size=520, replace=False)
    anomaly[fault_indices] = 1
    fault_types = rng.integers(0, 5, size=len(fault_indices))

    for idx, fault_type in zip(fault_indices, fault_types):
        if fault_type == 0:
            battery_voltage[idx] -= rng.uniform(1.5, 3.5)
        elif fault_type == 1:
            battery_temperature[idx] += rng.uniform(7.0, 14.0)
        elif fault_type == 2:
            bus_voltage[idx] -= rng.uniform(0.8, 2.0)
        elif fault_type == 3:
            attitude_error[idx] += rng.uniform(0.12, 0.40)
        else:
            if eclipse[idx] == 0:
                solar_panel_current[idx] *= rng.uniform(0.05, 0.40)
            else:
                battery_current[idx] -= rng.uniform(1.0, 2.5)

    # Small label noise prevents a perfectly deterministic toy problem.
    flip_indices = rng.choice(n_samples, size=60, replace=False)
    anomaly[flip_indices] = 1 - anomaly[flip_indices]

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_samples, freq="min"),
            "battery_voltage": battery_voltage,
            "battery_current": battery_current,
            "battery_temperature": battery_temperature,
            "solar_panel_current": solar_panel_current,
            "bus_voltage": bus_voltage,
            "attitude_error": attitude_error,
            "eclipse": eclipse,
            "anomaly": anomaly,
        }
    )


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "data" / "telemetry.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_telemetry().to_csv(output, index=False)
    print(f"Dataset written to: {output}")
