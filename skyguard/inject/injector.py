"""Synthetic faults and storms with labels, for testing and for the live demo.

Owner: M1

All functions return a modified copy and set the label columns
(label_temp_c, label_pressure_hpa, label_rh_pct, event_id).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skyguard.geo import haversine_km
from skyguard.schemas import VARIABLES, LABEL_COLUMNS, FaultType
from skyguard.config import load_config


# ===================================================================
# Helpers
# ===================================================================

def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add label_* = "NONE" and event_id = "" if missing."""
    df = df.copy()
    for col in LABEL_COLUMNS.values():
        if col not in df.columns:
            df[col] = FaultType.NONE
    if "event_id" not in df.columns:
        df["event_id"] = ""
    return df


def _mask(df: pd.DataFrame, station_id: str, t0: pd.Timestamp,
          duration_h: int) -> pd.Series:
    """Boolean mask for rows of *station_id* from t0 to t0 + duration_h."""
    t_end = t0 + pd.Timedelta(hours=duration_h)
    return (df["station_id"] == station_id) & (df["ts"] >= t0) & (df["ts"] < t_end)


# ===================================================================
# Individual fault injectors
# ===================================================================

def _inject_spike(df: pd.DataFrame, var: str, idx: pd.Index,
                  rng: np.random.Generator, magnitude: float | None, cfg: dict) -> pd.DataFrame:
    if magnitude is not None:
        mag = magnitude
    else:
        lo, hi = cfg["inject"]["SPIKE"][var]
        mag = rng.uniform(lo, hi) * rng.choice([-1, 1])
    df.loc[idx, var] = df.loc[idx, var].astype(float) + mag
    return df


def _inject_frozen(df: pd.DataFrame, var: str, idx: pd.Index,
                   _rng, _mag, _cfg) -> pd.DataFrame:
    if len(idx) == 0:
        return df
    valid_vals = df.loc[idx, var].dropna()
    if len(valid_vals) == 0:
        return df
    frozen_val = valid_vals.iloc[0]
    df.loc[idx, var] = frozen_val
    return df


def _inject_drift(df: pd.DataFrame, var: str, idx: pd.Index,
                  rng: np.random.Generator, magnitude: float | None, cfg: dict) -> pd.DataFrame:
    n = len(idx)
    if n == 0:
        return df
    if magnitude is not None:
        total = magnitude
    else:
        lo, hi = cfg["inject"]["DRIFT"][var]
        total = rng.uniform(lo, hi) * rng.choice([-1, 1])
    ramp = np.linspace(0, total, n)
    df.loc[idx, var] += ramp
    return df


def _inject_dropout(df: pd.DataFrame, var: str, idx: pd.Index,
                    rng: np.random.Generator, _mag, _cfg) -> pd.DataFrame:
    if var == "pressure_hpa":
        mode = rng.choice(["nan", "sentinel", "zero"])
    else:
        mode = rng.choice(["nan", "sentinel"])
        
    if mode == "nan":
        df.loc[idx, var] = np.nan
    elif mode == "sentinel":
        df.loc[idx, var] = -9999.0
    else:
        df.loc[idx, var] = 0.0
    return df


def _inject_noise(df: pd.DataFrame, var: str, idx: pd.Index,
                  rng: np.random.Generator, magnitude: float | None, cfg: dict) -> pd.DataFrame:
    base_std = df[var].diff().std()
    mult = magnitude if magnitude is not None else rng.uniform(*cfg["inject"]["NOISE_SIGMA_MULT"])
    sigma = mult * base_std
    noise = rng.normal(0, sigma, size=len(idx))
    df.loc[idx, var] += noise
    return df


def _inject_oor(df: pd.DataFrame, var: str, idx: pd.Index,
                rng: np.random.Generator, _mag, cfg: dict) -> pd.DataFrame:
    ranges = cfg["physics"]["range"]
    if var == "temp_c":
        rail = rng.choice([ranges[var][0] - 5, ranges[var][1] + 5])
    elif var == "pressure_hpa":
        rail = rng.choice([ranges[var][0] - 10, ranges[var][1] + 10])
    elif var == "rh_pct":
        rail = rng.choice([-5, 105])
    df.loc[idx, var] = float(rail)
    return df


_INJECTORS = {
    FaultType.SPIKE: _inject_spike,
    FaultType.FROZEN: _inject_frozen,
    FaultType.DRIFT: _inject_drift,
    FaultType.DROPOUT: _inject_dropout,
    FaultType.NOISE: _inject_noise,
    FaultType.OUT_OF_RANGE: _inject_oor,
}


# ===================================================================
# Public API
# ===================================================================

def inject_fault(df: pd.DataFrame, station_id: str, var: str, fault_type: str,
                 t0: pd.Timestamp, duration_h: int, magnitude: float | None = None,
                 event_id: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Inject one fault into one variable of one station, starting at t0."""
    if fault_type not in _INJECTORS:
        raise ValueError(f"Unknown fault_type {fault_type!r}. Choose from {list(_INJECTORS.keys())}")

    df = add_label_columns(df)
    rng = np.random.default_rng(seed)
    cfg = load_config()
    label_col = LABEL_COLUMNS[var]

    sel = _mask(df, station_id, t0, duration_h)
    idx = df.index[sel]

    if len(idx) == 0:
        return df

    df = _INJECTORS[fault_type](df, var, idx, rng, magnitude, cfg)

    # clip RH to [0, 100]
    if var == "rh_pct" and fault_type not in (FaultType.DROPOUT, FaultType.OUT_OF_RANGE):
        df.loc[idx, var] = df.loc[idx, var].clip(0, 100)

    # set labels
    df.loc[idx, label_col] = fault_type
    
    if event_id is None:
        existing = df["event_id"].str.startswith(f"{fault_type}_").sum()
        event_id = f"{fault_type}_{existing + 1:03d}"
    
    df.loc[idx, "event_id"] = event_id

    return df


def inject_storm(df: pd.DataFrame, stations: list[dict], center_id: str,
                 t0: pd.Timestamp, radius_km: float = 80, strength: float = 1.0,
                 event_id: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Inject a coherent thunderstorm across neighbouring stations."""
    df = add_label_columns(df)
    rng = np.random.default_rng(seed)

    t_drop = rng.uniform(4, 10) * strength
    rh_rise = rng.uniform(15, 35) * strength
    p_rise = rng.uniform(1, 4) * strength
    onset_h = rng.integers(1, 3)
    relax_h = 6

    if event_id is None:
        existing = df.loc[df["event_id"].str.startswith("STORM_", na=False), "event_id"].nunique()
        event_id = f"STORM_{existing + 1:03d}"

    center = next(s for s in stations if s["station_id"] == center_id)

    for stn in stations:
        sid = stn["station_id"]
        dist = haversine_km(center["lat"], center["lon"], stn["lat"], stn["lon"])
        if dist > radius_km:
            continue

        delay_h = int(round(dist / 30.0))
        local_t0 = t0 + pd.Timedelta(hours=delay_h)

        total_h = onset_h + relax_h
        t_profile = np.zeros(total_h)
        for i in range(onset_h):
            t_profile[i] = -t_drop * (i + 1) / onset_h
        for i in range(relax_h):
            t_profile[onset_h + i] = -t_drop * np.exp(-i / 3.0)

        rh_profile = np.zeros(total_h)
        for i in range(onset_h):
            rh_profile[i] = rh_rise * (i + 1) / onset_h
        for i in range(relax_h):
            rh_profile[onset_h + i] = rh_rise * np.exp(-i / 3.0)

        p_profile = np.zeros(total_h)
        for i in range(onset_h):
            p_profile[i] = p_rise * (i + 1) / onset_h
        for i in range(relax_h):
            p_profile[onset_h + i] = p_rise * np.exp(-i / 3.0)

        for step in range(total_h):
            ts = local_t0 + pd.Timedelta(hours=step)
            row_mask = (df["station_id"] == sid) & (df["ts"] == ts)
            if row_mask.any():
                df.loc[row_mask, "temp_c"] = df.loc[row_mask, "temp_c"].astype(float) + t_profile[step]
                df.loc[row_mask, "rh_pct"] = (df.loc[row_mask, "rh_pct"].astype(float) + rh_profile[step]).clip(0, 100)
                df.loc[row_mask, "pressure_hpa"] = df.loc[row_mask, "pressure_hpa"].astype(float) + p_profile[step]
                df.loc[row_mask, "event_id"] = event_id

    return df


def build_test_set(df: pd.DataFrame, stations: list[dict] | None = None,
                   seed: int = 42) -> pd.DataFrame:
    """~40 faults of each type + ~10 storms, non-overlapping, on the given data slice."""
    if stations is None:
        from skyguard.data.io import load_stations
        stations = load_stations()
        
    df = add_label_columns(df)
    rng = np.random.default_rng(seed)
    cfg = load_config()

    station_ids = sorted(df["station_id"].unique())
    all_ts = sorted(df["ts"].unique())
    n_ts = len(all_ts)

    # Track occupied time slots per station (across ALL variables) to avoid overlaps
    occupied: dict[str, set[int]] = {
        sid: set() for sid in station_ids
    }

    from typing import Optional
    def _find_free_slot(sid: str, dur_h: int, attempts: int = 200) -> Optional[int]:
        occ = occupied[sid]
        for _ in range(attempts):
            start = rng.integers(0, max(1, n_ts - dur_h))
            span = set(range(start, start + dur_h))
            if not span & occ:
                return int(start)
        return None

    def _find_free_slot_multi(sids: list[str], dur_h: int, attempts: int = 200) -> Optional[int]:
        """Find a slot free across ALL given stations."""
        for _ in range(attempts):
            start = rng.integers(0, max(1, n_ts - dur_h))
            span = set(range(start, start + dur_h))
            if all(not (span & occupied[s]) for s in sids):
                return int(start)
        return None

    def _mark_occupied(sid: str, start_idx: int, dur_h: int):
        buf = 6
        occupied[sid].update(range(max(0, start_idx - buf),
                                   min(n_ts, start_idx + dur_h + buf)))

    # --- Inject ~10 storms FIRST ---
    storm_count = 0
    storm_seed = int(rng.integers(0, 2**31))
    for _ in range(10):
        center_id = rng.choice(station_ids)
        
        # Find stations within storm radius
        center_stn = next(s for s in stations if s["station_id"] == center_id)
        affected_sids = []
        for stn in stations:
            dist = haversine_km(center_stn["lat"], center_stn["lon"], stn["lat"], stn["lon"])
            if dist <= 80 and stn["station_id"] in station_ids:
                affected_sids.append(stn["station_id"])
        
        # Check ALL affected stations are free
        slot = _find_free_slot_multi(affected_sids, 8)
        if slot is None:
            continue

        t0 = pd.Timestamp(all_ts[slot])
        storm_strength = rng.uniform(0.7, 1.3)
        storm_seed += 1
        
        event_id = f"STORM_{storm_count + 1:03d}"
        df = inject_storm(df, stations, center_id, t0,
                          radius_km=80, strength=storm_strength, 
                          event_id=event_id, seed=storm_seed)

        for sid in affected_sids:
            _mark_occupied(sid, slot, 10)

        storm_count += 1

    # --- Inject ~40 of each fault type ---
    injected_count: dict[str, int] = {ft: 0 for ft in FaultType.FAULTS}
    fault_seed = int(rng.integers(0, 2**31))

    for fault_type in FaultType.FAULTS:
        target = 40
        lo_dur, hi_dur = cfg["inject"]["DURATION_RANGE"][fault_type]

        for _ in range(target):
            sid = rng.choice(station_ids)
            var = rng.choice(VARIABLES)
            dur = int(rng.integers(lo_dur, hi_dur + 1))

            slot = _find_free_slot(sid, dur)
            if slot is None:
                continue

            t0 = pd.Timestamp(all_ts[slot])
            fault_seed += 1
            
            event_id = f"{fault_type}_{injected_count[fault_type] + 1:03d}"
            df = inject_fault(df, sid, var, fault_type, t0, dur, event_id=event_id, seed=fault_seed)
            _mark_occupied(sid, slot, dur)
            injected_count[fault_type] += 1

    print(f"Injected faults: {injected_count}")
    print(f"Injected storms: {storm_count}")

    return df
