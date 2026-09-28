"""Synthetic faults and storms with labels, for testing and for the live demo.

Owner: M1

All functions return a modified copy and set the label columns
(label_temp_c, label_pressure_hpa, label_rh_pct, event_id).

Fault models (magnitudes are typical ranges, pick randomly with the seed):
  SPIKE         x[t] += A for 1-2 steps     T ±5-25 °C, P ±5-40 hPa, RH ±20-50 % (clip RH 0-100)
  FROZEN        x[t0:t0+L] = x[t0]          L = 6-48 h
  DRIFT         x[t] += r * (t - t0)        total T 2-5 °C, P 2-6 hPa, RH 8-20 % over 3-10 days
  DROPOUT       NaN, or -9999 / 0           L = 1-12 h
  NOISE         x += N(0, sigma)            sigma = 1.5-4 x normal hourly std, L = 6-24 h
  OUT_OF_RANGE  stuck at a rail             T 85 or -40, RH 100 or 0, P 1100

Storm (a genuine event, labels stay NONE, event_id = "STORM_<n>"):
  every station within radius_km of the centre, delayed by distance / 30 km/h:
  T drops 4-10 °C over 1-2 h, RH rises 15-35 %, P rises 1-4 hPa, all relax over ~6 h.
"""

import pandas as pd


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add label_* = "NONE" and event_id = "" if missing."""
    raise NotImplementedError


def inject_fault(df: pd.DataFrame, station_id: str, var: str, fault_type: str,
                 t0: pd.Timestamp, duration_h: int, magnitude: float | None = None,
                 seed: int = 42) -> pd.DataFrame:
    """Inject one fault into one variable of one station, starting at t0."""
    raise NotImplementedError


def inject_storm(df: pd.DataFrame, stations: list[dict], center_id: str, t0: pd.Timestamp,
                 radius_km: float = 80, strength: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Inject a coherent thunderstorm across neighbouring stations (use skyguard.geo)."""
    raise NotImplementedError


def build_test_set(df: pd.DataFrame, stations: list[dict], seed: int = 42) -> pd.DataFrame:
    """Many random faults (about 40 of each type) plus ~10 storms on the given split.

    Faults should not overlap each other or the storms.
    Save the result to data/injected/<split>.parquet from the caller.
    """
    raise NotImplementedError
