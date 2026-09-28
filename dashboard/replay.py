"""Hour-by-hour replay of the station network for the dashboard.

The dashboard asks a source for the results at each hour. EngineSource wraps
skyguard.engine.Engine. Until the engine is implemented, PreviewSource stands in:
it reports what the fault injector put into the data, so the interface can be
built and tested. The header always shows which source is running.
"""

import time

import numpy as np
import pandas as pd

from skyguard.data.climatology import Climatology
from skyguard.data.io import load_data, select_split
from skyguard.geo import neighbours
from skyguard.inject.injector import add_label_columns, inject_fault, inject_storm
from skyguard.schemas import (LABEL_COLUMNS, NAMES, ORIG_COLUMNS, UNITS, VARIABLES,
                              FaultType, Result, Severity, Status)

KEEP_HOURS = 7 * 24     # results older than this are dropped
WARMUP_HOURS = 24       # computed on every jump so the charts are not empty

PREVIEW_SEVERITY = {
    FaultType.SPIKE: Severity.CRITICAL,
    FaultType.OUT_OF_RANGE: Severity.CRITICAL,
    FaultType.DROPOUT: Severity.HIGH,
    FaultType.FROZEN: Severity.HIGH,
    FaultType.NOISE: Severity.MEDIUM,
    FaultType.DRIFT: Severity.MEDIUM,
}


def is_valid(value, var, cfg) -> bool:
    lo, hi = cfg["physics"]["range"][var]
    return value is not None and not pd.isna(value) and lo <= value <= hi


class PreviewSource:
    """Stand-in for the detector: reports the injector's labels, with simple facts as reasons."""

    name = "Preview"

    def __init__(self, stations, cfg):
        self.cfg = cfg
        radius = cfg["spatial"]["radius_km"]
        self.nearby = {s["station_id"]: [n["station_id"] for n in neighbours(stations, s["station_id"], radius)]
                       for s in stations}
        self.trust = {}

    def reset(self):
        self.trust.clear()

    def step(self, df, ts):
        now = df[df["ts"] == ts].set_index("station_id")
        before = df[df["ts"] == ts - pd.Timedelta(hours=1)].set_index("station_id")
        out = []
        for sid, row in now.iterrows():
            for var in VARIABLES:
                out.append(self._result(sid, ts, var, row, now, before).to_dict())
        return out

    def _result(self, sid, ts, var, row, now, before):
        value = float(row[var])
        res = Result(station_id=sid, ts=ts.strftime("%Y-%m-%dT%H:%M:%SZ"), variable=var,
                     value=None if pd.isna(value) else value)
        label = row[LABEL_COLUMNS[var]]
        if label != FaultType.NONE:
            res.status, res.fault_type = Status.SENSOR_FAULT, label
            res.severity, res.confidence = PREVIEW_SEVERITY[label], 0.9
            res.corrected_value = float(row[ORIG_COLUMNS[var]])
        elif str(row["event_id"]).startswith("STORM"):
            res.status, res.confidence = Status.GENUINE_EVENT, 0.85

        if res.status != Status.NORMAL:
            res.reasons = self._facts(sid, var, value, now, before)
        res.trust = self._update_trust(sid, var, res.status, res.severity)
        return res

    def _facts(self, sid, var, value, now, before):
        unit, name = UNITS[var], NAMES[var]
        facts = []
        if not is_valid(value, var, self.cfg):
            facts.append(f"Reported {value:g} {unit}, outside the valid range" if not pd.isna(value)
                         else "No reading received")
        elif sid in before.index and is_valid(before.at[sid, var], var, self.cfg):
            change = value - before.at[sid, var]
            facts.append(f"{name} {'rose' if change >= 0 else 'fell'} {abs(change):.1f} {unit} in the last hour")

        others = [now.at[n, var] for n in self.nearby[sid] if n in now.index]
        others = [v for v in others if is_valid(v, var, self.cfg)]
        if others:
            facts.append(f"Neighbour median {np.median(others):.1f} {unit} ({len(others)} stations)")
        return facts

    def _update_trust(self, sid, var, status, severity):
        cfg = self.cfg["trust"]
        score = self.trust.get((sid, var), cfg["start"])
        if status == Status.SENSOR_FAULT:
            score -= cfg["penalty"][severity]
        elif status == Status.NORMAL:
            score += cfg["recovery_per_clean"]
        score = min(max(score, 0.0), cfg["start"])
        self.trust[(sid, var)] = score
        return score


class EngineSource:
    """The real detector from skyguard.engine."""

    name = "Engine"

    def __init__(self, df, stations, cfg):
        from skyguard.engine import Engine
        from skyguard.models.iforest import IFModel

        self._make = Engine
        self.stations, self.cfg = stations, cfg
        self.clim = Climatology().fit(select_split(load_data(), "train", cfg))
        try:
            self.model = IFModel.load()
        except (NotImplementedError, FileNotFoundError):
            self.model = None
        self.engine = Engine(df, stations, self.clim, self.model, cfg)

    def reset(self):
        # a fresh engine also means fresh trust scores
        self.engine = self._make(self.engine.df, self.stations, self.clim, self.model, self.cfg)

    def step(self, df, ts):
        self.engine.df = df
        return self.engine.step(ts)


def pick_source(df, stations, cfg, probe_ts):
    """The engine if it works, otherwise the preview and a note saying why."""
    try:
        source = EngineSource(df, stations, cfg)
        source.step(df, probe_ts)
        source.reset()
        return source, None
    except NotImplementedError:
        return PreviewSource(stations, cfg), "Detection engine not connected yet"
    except Exception as exc:
        return PreviewSource(stations, cfg), f"Engine error: {type(exc).__name__}: {exc}"


class Replay:
    def __init__(self, df, stations, cfg, start):
        self.cfg = cfg
        self.stations = stations
        self.clean = add_label_columns(df)
        self.df = self.clean
        self.hours = pd.DatetimeIndex(sorted(df["ts"].unique()))
        self.source, self.notice = pick_source(self.df, stations, cfg, self.hours[WARMUP_HOURS])
        self.results = pd.DataFrame()
        self.step_ms = []
        self.pos = 0
        self.jump(start)

    @property
    def now(self) -> pd.Timestamp:
        return self.hours[self.pos]

    @property
    def at_end(self) -> bool:
        return self.pos >= len(self.hours) - 1

    def jump(self, ts):
        pos = int(self.hours.searchsorted(pd.Timestamp(ts)))
        self.pos = min(max(pos, WARMUP_HOURS), len(self.hours) - 1)
        self.source.reset()
        self.results = pd.DataFrame()
        self.step_ms = []
        self._run(range(self.pos - WARMUP_HOURS, self.pos + 1))

    def advance(self, hours=1):
        end = min(self.pos + hours, len(self.hours) - 1)
        self._run(range(self.pos + 1, end + 1))
        self.pos = end

    def _run(self, positions):
        rows = []
        for p in positions:
            start = time.perf_counter()
            rows.extend(self.source.step(self.df, self.hours[p]))
            self.step_ms.append((time.perf_counter() - start) * 1000)
        if not rows:
            return
        new = pd.DataFrame(rows)
        new["ts"] = pd.to_datetime(new["ts"], utc=True)
        # fixed types, so an hour where every corrected_value is None still concatenates cleanly
        new = new.astype({"value": float, "corrected_value": float, "confidence": float, "trust": float,
                          "fault_type": object, "severity": object})
        latest = self.hours[positions[-1]]
        combined = new if self.results.empty else pd.concat([self.results, new], ignore_index=True)
        self.results = combined[combined["ts"] > latest - pd.Timedelta(hours=KEEP_HOURS)]
        self.step_ms = self.step_ms[-KEEP_HOURS:]

    # Demo actions. Each one changes the data from the next hour on.

    def next_hour(self) -> pd.Timestamp:
        return self.hours[min(self.pos + 1, len(self.hours) - 1)]

    def inject_spike(self, station_id, target_c=55.0):
        t = self.next_hour()
        current = self.df.loc[(self.df["station_id"] == station_id) & (self.df["ts"] == t), "temp_c"]
        if current.empty or pd.isna(current.iloc[0]):
            return None
        self.df = inject_fault(self.df, station_id, "temp_c", FaultType.SPIKE, t, 1,
                               magnitude=target_c - float(current.iloc[0]))
        return t

    def inject_freeze(self, station_id, var="rh_pct", hours=12):
        t = self.next_hour()
        self.df = inject_fault(self.df, station_id, var, FaultType.FROZEN, t, hours)
        return t

    def inject_storm(self, station_id, strength=1.2):
        """Returns None if a storm is already passing this station (stacking two looks absurd)."""
        t = self.next_hour()
        coming = self.df[(self.df["station_id"] == station_id) & (self.df["ts"] >= t - pd.Timedelta(hours=8))
                         & (self.df["ts"] <= t + pd.Timedelta(hours=8))]
        if coming["event_id"].str.startswith("STORM").any():
            return None
        self.df = inject_storm(self.df, self.stations, station_id, t, strength=strength,
                               seed=int(t.timestamp()))
        return t

    def clear_injections(self):
        self.df = self.clean
