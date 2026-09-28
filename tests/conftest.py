import numpy as np
import pandas as pd
import pytest

STATIONS = [
    {"station_id": "A", "name": "A", "lat": 28.60, "lon": 77.20, "elevation_m": 216},
    {"station_id": "B", "name": "B", "lat": 28.65, "lon": 77.25, "elevation_m": 210},
    {"station_id": "C", "name": "C", "lat": 28.55, "lon": 77.10, "elevation_m": 230},
]


@pytest.fixture
def stations():
    return [dict(s) for s in STATIONS]


@pytest.fixture
def hourly():
    """Three stations, 20 days of smooth synthetic hourly weather."""
    ts = pd.date_range("2024-08-01", periods=20 * 24, freq="h", tz="UTC")
    hour = ts.hour.to_numpy()
    frames = []
    for i, s in enumerate(STATIONS):
        frames.append(pd.DataFrame({
            "station_id": s["station_id"],
            "ts": ts,
            "temp_c": 30 + 5 * np.sin(2 * np.pi * hour / 24) + i * 0.3,
            "pressure_hpa": 1000 + np.cos(4 * np.pi * hour / 24) - i * 0.2,
            "rh_pct": 60 - 15 * np.sin(2 * np.pi * hour / 24) + i,
        }))
    return pd.concat(frames, ignore_index=True)
