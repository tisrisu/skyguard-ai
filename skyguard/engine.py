"""Runs the full pipeline over the data, one hour at a time.

Owner: M3

For each station and variable at time ts:
  physics     = check_physics(...)
  spatial     = spatial_check(...)
  features    = make_features(...)
  ml          = model.score(features, var)
  status, confidence, severity = decide(...)
  if not NORMAL:  fault_type = fingerprint(...); reasons, shap_top = explain(...)
  if SENSOR_FAULT or SUSPECT:  corrected_value = impute(...)
  trust       = trust.update(...)

If a module raises (not ready yet, or a bad row), fall back gracefully:
missing ML -> score 0.0, missing spatial -> consistent None, and keep going.
The dashboard's demo buttons modify `engine.df` (future rows) through the
injector, and the replay continues from there.
"""

import pandas as pd


class Engine:
    def __init__(self, df: pd.DataFrame, stations: list[dict], clim, model=None,
                 cfg: dict | None = None):
        self.df = df                 # long-format data, may contain injected faults
        self.stations = stations
        self.clim = clim
        self.model = model
        # TODO: TrustTracker, neighbour lists per station (skyguard.geo.neighbours)

    def step(self, ts: pd.Timestamp) -> list[dict]:
        """Results (schemas.Result.to_dict()) for every station and variable at ts,
        each including "latency_ms" in scores."""
        raise NotImplementedError

    def run(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Call step() for every hour in [start, end] and return all results as a DataFrame."""
        raise NotImplementedError
