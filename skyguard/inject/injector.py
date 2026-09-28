"""Synthetic faults and storms with labels, for testing and for the live demo.

Owner: M1

Run:  python -m skyguard.inject.injector
Writes: data/injected/val.parquet and data/injected/test.parquet

All functions return a modified copy. The clean value is kept in orig_<var>, the
fault type goes in label_<var> and the event in event_id ("SPIKE_003", ...).
Storms are genuine weather: labels stay NONE and event_id is "STORM_<n>".
Magnitudes and durations are in config.yaml -> inject.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from skyguard.config import INJECTED_DIR, load_config
from skyguard.data.io import load_data, load_stations, select_split
from skyguard.geo import haversine_km
from skyguard.schemas import LABEL_COLUMNS, ORIG_COLUMNS, VARIABLES, FaultType


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Float value columns, plus orig_*, label_* ("NONE") and event_id ("") where missing."""
    df = df.copy()
    for var in VARIABLES:
        df[var] = df[var].astype(float)
        if ORIG_COLUMNS[var] not in df.columns:
            df[ORIG_COLUMNS[var]] = df[var]
        if LABEL_COLUMNS[var] not in df.columns:
            df[LABEL_COLUMNS[var]] = FaultType.NONE
    if "event_id" not in df.columns:
        df["event_id"] = ""
    return df


def _next_event_id(df: pd.DataFrame, prefix: str) -> str:
    n = df.loc[df["event_id"].str.startswith(prefix + "_"), "event_id"].nunique()
    return f"{prefix}_{n + 1:03d}"


# Each fault function changes df[var] in place for the rows in idx.

def _spike(df, var, idx, rng, magnitude, cfg):
    if magnitude is None:
        lo, hi = cfg["inject"]["SPIKE"][var]
        if var == "rh_pct":
            # go towards the side with room, otherwise clipping at 0/100 hides the spike
            sign = -1 if df.loc[idx, var].mean() > 50 else 1
        else:
            sign = rng.choice([-1, 1])
        magnitude = sign * rng.uniform(lo, hi)
    df.loc[idx, var] += magnitude


def _frozen(df, var, idx, rng, magnitude, cfg):
    valid = df.loc[idx, var].dropna()
    if len(valid):
        df.loc[idx, var] = valid.iloc[0]


def _drift(df, var, idx, rng, magnitude, cfg):
    if magnitude is None:
        lo, hi = cfg["inject"]["DRIFT"][var]
        magnitude = rng.choice([-1, 1]) * rng.uniform(lo, hi)
    df.loc[idx, var] += np.linspace(0, magnitude, len(idx))


def _dropout(df, var, idx, rng, magnitude, cfg):
    # 0 is an obvious failure only for pressure; 0 °C or 0 % RH can look plausible
    modes = {"nan": np.nan, "sentinel": -9999.0}
    if var == "pressure_hpa":
        modes["zero"] = 0.0
    df.loc[idx, var] = modes[rng.choice(list(modes))]


def _noise(df, var, idx, rng, magnitude, cfg):
    # scale by this station's normal hour-to-hour change, ignoring rows that are already faulty
    station = df.at[idx[0], "station_id"]
    lo, hi = cfg["physics"]["range"][var]
    series = df.loc[(df["station_id"] == station) & (df[LABEL_COLUMNS[var]] == FaultType.NONE), var]
    step_std = series.where(series.between(lo, hi)).diff().std()
    mult = magnitude if magnitude is not None else rng.uniform(*cfg["inject"]["NOISE_SIGMA_MULT"])
    df.loc[idx, var] += rng.normal(0, mult * step_std, size=len(idx))


def _out_of_range(df, var, idx, rng, magnitude, cfg):
    lo, hi = cfg["physics"]["range"][var]
    margin = cfg["inject"]["OUT_OF_RANGE_MARGIN"][var]
    df.loc[idx, var] = float(rng.choice([lo - margin, hi + margin]))


_INJECTORS = {
    FaultType.SPIKE: _spike,
    FaultType.FROZEN: _frozen,
    FaultType.DRIFT: _drift,
    FaultType.DROPOUT: _dropout,
    FaultType.NOISE: _noise,
    FaultType.OUT_OF_RANGE: _out_of_range,
}


def inject_fault(df: pd.DataFrame, station_id: str, var: str, fault_type: str,
                 t0: pd.Timestamp, duration_h: int, magnitude: float | None = None,
                 event_id: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Inject one fault into one variable of one station for duration_h hours from t0.

    magnitude overrides the random size, sign included (e.g. +23 for the demo spike).
    For NOISE it is the multiple of the station's normal hourly change.
    """
    if fault_type not in _INJECTORS:
        raise ValueError(f"unknown fault type {fault_type!r}, expected one of {FaultType.FAULTS}")

    df = add_label_columns(df)
    t0 = pd.Timestamp(t0)
    rows = (df["station_id"] == station_id) & (df["ts"] >= t0) & (df["ts"] < t0 + pd.Timedelta(hours=duration_h))
    idx = df.index[rows]
    if len(idx) == 0:
        return df

    _INJECTORS[fault_type](df, var, idx, np.random.default_rng(seed), magnitude, load_config())
    if var == "rh_pct" and fault_type in (FaultType.SPIKE, FaultType.DRIFT, FaultType.NOISE):
        df.loc[idx, var] = df.loc[idx, var].clip(0, 100)

    df.loc[idx, LABEL_COLUMNS[var]] = fault_type
    df.loc[idx, "event_id"] = event_id or _next_event_id(df, fault_type)
    return df


def inject_storm(df: pd.DataFrame, stations: list[dict], center_id: str,
                 t0: pd.Timestamp, radius_km: float | None = None, strength: float = 1.0,
                 event_id: str | None = None, seed: int = 42) -> pd.DataFrame:
    """A thunderstorm at every station within radius_km of center_id.

    The start at each station is delayed by distance / front speed. Temperature drops
    while humidity and pressure rise over onset_h hours, then all relax exponentially.
    """
    cfg = load_config()["inject"]["STORM"]
    df = add_label_columns(df)
    rng = np.random.default_rng(seed)
    radius_km = radius_km or cfg["radius_km"]

    onset_h = int(rng.integers(cfg["onset_h"][0], cfg["onset_h"][1] + 1))
    shape = np.concatenate([
        np.arange(1, onset_h + 1) / onset_h,
        np.exp(-np.arange(cfg["relax_h"]) / cfg["relax_tau_h"]),
    ])
    amplitude = {
        "temp_c": -rng.uniform(*cfg["temp_drop"]) * strength,
        "rh_pct": rng.uniform(*cfg["rh_rise"]) * strength,
        "pressure_hpa": rng.uniform(*cfg["pressure_rise"]) * strength,
    }
    event_id = event_id or _next_event_id(df, "STORM")
    center = next(s for s in stations if s["station_id"] == center_id)
    t0 = pd.Timestamp(t0)

    for stn in stations:
        dist = haversine_km(center["lat"], center["lon"], stn["lat"], stn["lon"])
        if dist > radius_km:
            continue
        start = t0 + pd.Timedelta(hours=round(dist / cfg["front_speed_kmh"]))
        end = start + pd.Timedelta(hours=len(shape))
        idx = df.index[(df["station_id"] == stn["station_id"]) & (df["ts"] >= start) & (df["ts"] < end)]
        step = ((df.loc[idx, "ts"] - start) // pd.Timedelta(hours=1)).to_numpy()
        for var, amp in amplitude.items():
            df.loc[idx, var] += amp * shape[step]
        df.loc[idx, "rh_pct"] = df.loc[idx, "rh_pct"].clip(0, 100)
        df.loc[idx, "event_id"] = event_id
    return df


def build_test_set(df: pd.DataFrame, stations: list[dict] | None = None,
                   seed: int = 42) -> pd.DataFrame:
    """Storms first, then faults of every type, on the given data slice.

    Nothing overlaps at a station: every event keeps inject.test_set.gap_h clear hours
    around it. Counts come from config.yaml -> inject.test_set.
    """
    cfg = load_config()
    plan, storm = cfg["inject"]["test_set"], cfg["inject"]["STORM"]
    stations = stations or load_stations()
    df = add_label_columns(df)
    rng = np.random.default_rng(seed)

    station_ids = sorted(df["station_id"].unique())
    hours = pd.DatetimeIndex(sorted(df["ts"].unique()))
    busy = {sid: np.zeros(len(hours), dtype=bool) for sid in station_ids}
    gap = plan["gap_h"]

    def free_slot(sids, length):
        for _ in range(200):
            start = int(rng.integers(0, max(1, len(hours) - length)))
            window = slice(max(0, start - gap), start + length + gap)
            if not any(busy[s][window].any() for s in sids):
                return start
        return None

    def reserve(sids, start, length):
        for s in sids:
            busy[s][max(0, start - gap): start + length + gap] = True

    # longest a storm can last at the farthest station
    storm_len = storm["onset_h"][1] + storm["relax_h"] + math.ceil(storm["radius_km"] / storm["front_speed_kmh"])
    n_storms = 0
    for _ in range(plan["storms"]):
        center_id = str(rng.choice(station_ids))
        center = next(s for s in stations if s["station_id"] == center_id)
        affected = [
            s["station_id"] for s in stations
            if s["station_id"] in busy
            and haversine_km(center["lat"], center["lon"], s["lat"], s["lon"]) <= storm["radius_km"]
        ]
        start = free_slot(affected, storm_len)
        if start is None:
            continue
        n_storms += 1
        df = inject_storm(df, stations, center["station_id"], hours[start],
                          strength=rng.uniform(*storm["strength"]),
                          event_id=f"STORM_{n_storms:03d}", seed=int(rng.integers(2**31)))
        reserve(affected, start, storm_len)

    counts = dict.fromkeys(FaultType.FAULTS, 0)
    for fault_type in FaultType.FAULTS:
        lo, hi = cfg["inject"]["DURATION_RANGE"][fault_type]
        for _ in range(plan["faults_per_type"]):
            sid = str(rng.choice(station_ids))
            var = str(rng.choice(VARIABLES))
            duration = int(rng.integers(lo, hi + 1))
            start = free_slot([sid], duration)
            if start is None:
                continue
            counts[fault_type] += 1
            df = inject_fault(df, sid, var, fault_type, hours[start], duration,
                              event_id=f"{fault_type}_{counts[fault_type]:03d}",
                              seed=int(rng.integers(2**31)))
            reserve([sid], start, duration)

    print(f"  storms: {n_storms}, faults: {counts}")
    return df


def main() -> None:
    cfg = load_config()
    df, stations = load_data(), load_stations()
    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    for offset, split in enumerate(("val", "test")):
        print(f"{split}:")
        injected = build_test_set(select_split(df, split, cfg), stations, seed=cfg["seed"] + offset)
        path = INJECTED_DIR / f"{split}.parquet"
        injected.to_parquet(path, index=False)
        print(f"  saved {len(injected):,} rows to {path}")


if __name__ == "__main__":
    main()
