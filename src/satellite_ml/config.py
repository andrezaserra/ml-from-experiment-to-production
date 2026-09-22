from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "telemetry.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

FEATURES = [
    "battery_voltage",
    "battery_current",
    "battery_temperature",
    "solar_panel_current",
    "bus_voltage",
    "attitude_error",
    "eclipse",
]

TARGET = "anomaly"

RANDOM_STATE = 42
TEST_SIZE = 0.25
