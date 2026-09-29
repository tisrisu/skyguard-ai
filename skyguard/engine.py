

from __future__ import annotations

import time
import pandas as pd
import numpy as np

from skyguard.config import load_config
from skyguard.data.climatology import Climatology
from skyguard.fusion.decide import decide
from skyguard.fusion.explain import explain
from skyguard.fusion.fingerprint import fingerprint
from skyguard.geo import neighbours
from skyguard.heal.impute import impute
from skyguard.health.trust import TrustTracker
from skyguard.models.features import make_features
from skyguard.physics.rules import check_physics
from skyguard.schemas import Result, Status, VARIABLES
from skyguard.spatial.consensus import spatial_check


class Engine:
    def __init__(self, df: pd.DataFrame, stations: list[dict], clim, model=None,
                 cfg: dict | None = None):
        self.df = df.copy()
        self.df["ts"] = pd.to_datetime(self.df["ts"], utc=True)
        self.df = self.df.sort_values(["station_id", "ts"]).reset_index(drop=True)
        self.stations = stations
        self.clim = clim
        self.model = model
        self.cfg = cfg or load_config()
        self.trust = TrustTracker(self.cfg)
        self._neighbours = {
            s["station_id"]: neighbours(
                stations, s["station_id"], self.cfg["spatial"]["radius_km"]
            )
            for s in stations
        }

    def _history(self, station_id: str, ts: pd.Timestamp) -> pd.DataFrame:
        return self.df[
            self.df["station_id"].eq(station_id) & (self.df["ts"] <= ts)
        ].sort_values("ts").reset_index(drop=True)

    def _window(self, ts: pd.Timestamp) -> pd.DataFrame:
        w = pd.Timedelta(hours=self.cfg["spatial"]["time_window_h"])
        return self.df[(self.df["ts"] >= ts - w) & (self.df["ts"] <= ts + w)]

    def _neighbour_rows(self, station_id: str, ts: pd.Timestamp) -> pd.DataFrame:
        ids = {n["station_id"] for n in self._neighbours.get(station_id, [])}
        if not ids:
            return self.df.iloc[0:0].copy()
        w = pd.Timedelta(hours=self.cfg["spatial"]["time_window_h"])
        return self.df[
            self.df["station_id"].isin(ids)
            & (self.df["ts"] >= ts - w)
            & (self.df["ts"] <= ts + w)
        ]

    def step(self, ts: pd.Timestamp) -> list[dict]:
        """Return one Result dict per station × variable at *ts*."""
        started = time.perf_counter()
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        current = self.df[self.df["ts"].eq(ts)].sort_values("station_id")
        around = self._window(ts)
        out: list[dict] = []

        for station_id in [s["station_id"] for s in self.stations]:
            station_rows = current[current["station_id"].eq(station_id)]
            if station_rows.empty:
                continue
            hist = self._history(station_id, ts)
            neighbours_now = self._neighbour_rows(station_id, ts)

            for var in VARIABLES:
                row = station_rows.iloc[-1]
                raw = row[var]
                value = None if pd.isna(raw) else float(raw)

                physics = check_physics(hist, var, self.cfg)
                spatial = spatial_check(around, self.stations, self.clim,
                                        station_id, ts, var, self.cfg)

                features = make_features(hist, self.clim, var)
                ml_score = 0.0
                if self.model is not None:
                    try:
                        ml_score = float(self.model.score(features, var))
                    except Exception:
                        ml_score = 0.0

                try:
                    mean, std = self.clim.expected(station_id, ts, var)
                    dev_sigma = 0.0 if not np.isfinite(std) or std == 0 or value is None \
                        else (value - mean) / std
                except Exception:
                    dev_sigma = 0.0

                status, confidence, severity = decide(
                    physics, spatial, ml_score, dev_sigma, self.cfg
                )

                fault_type = None
                reasons: list[str] = []
                shap_top = []
                corrected = None

                if status != Status.NORMAL:
                    fault_type = fingerprint(hist, var, physics, spatial, self.cfg)
                    try:
                        reasons, shap_top = explain(
                            physics, spatial, ml_score, features, var, self.model
                        )
                    except Exception:
                        reasons = list(physics.get("reasons", []))
                        shap_top = []

                if status in (Status.SENSOR_FAULT, Status.SUSPECT):
                    try:
                        corrected = impute(
                            hist.iloc[:-1].copy(), neighbours_now, self.clim,
                            station_id, ts, var
                        )
                    except Exception:
                        corrected = None

                trust = self.trust.update(station_id, var, status, severity)

                scores = {
                    "physics": 1.0 if physics.get("hard") else (0.5 if physics.get("soft") else 0.0),
                    "physics_rule": physics.get("rule_ids", [None])[0] if physics.get("rule_ids") else None,
                    "iforest": float(ml_score),
                    "spatial_z": spatial.get("z"),
                    "dev_sigma": float(dev_sigma) if np.isfinite(dev_sigma) else 0.0,
                }

                res = Result(
                    station_id=station_id,
                    ts=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    variable=var,
                    value=value,
                    status=status,
                    fault_type=fault_type,
                    confidence=float(confidence),
                    severity=severity,
                    reasons=reasons,
                    shap_top=shap_top,
                    corrected_value=None if corrected is None or pd.isna(corrected) else float(corrected),
                    trust=float(trust),
                    scores=scores,
                )
                out.append(res.to_dict())

        latency_ms = (time.perf_counter() - started) * 1000.0
        for r in out:
            r["scores"]["latency_ms"] = latency_ms
            r["latency_ms"] = latency_ms
        return out

    def run(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Process every available hour in [start, end] and return results."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")

        times = pd.DatetimeIndex(sorted(self.df.loc[
            (self.df["ts"] >= start) & (self.df["ts"] <= end), "ts"
        ].unique()))
        rows = []
        for ts in times:
            rows.extend(self.step(ts))
        return pd.DataFrame(rows)
