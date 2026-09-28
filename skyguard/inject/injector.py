"""Synthetic faults and storms with labels, for testing and for the live demo.

Owner: M1

All functions return a modified copy and set the label columns
(label_temp_c, label_pressure_hpa, label_rh_pct, event_id).

Fault models (magnitudes are typical ranges, pick randomly with the seed):
  SPIKE         x[t] += A for 1-2 steps     T +/-5-25 C, P +/-5-40 hPa, RH +/-20-50 % (clip RH 0-100)
  FROZEN        x[t0:t0+L] = x[t0]          L = 6-48 h
  DRIFT         x[t] += r * (t - t0)        total T 2-5 C, P 2-6 hPa, RH 8-20 % over 3-10 days
  DROPOUT       NaN, or -9999 / 0           L = 1-12 h
  NOISE         x += N(0, sigma)            sigma = 1.5-4 x normal hourly std, L = 6-24 h
  OUT_OF_RANGE  stuck at a rail             T 85 or -40, RH 100 or 0, P 1100

Storm (a genuine event, labels stay NONE, event_id = "STORM_<n>"):
  every station within radius_km of the centre, delayed by distance / 30 km/h:
  T drops 4-10 C over 1-2 h, RH rises 15-35 %, P rises 1-4 hPa, all relax over ~6 h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skyguard.geo import haversine_km

# ---------------------------------------------------------------------------
# Magnitude defaults per variable  (low, high)
# ---------------------------------------------------------------------------
_SPIKE_MAG = {"temp_c": (5, 25), "pressure_hpa": (5, 40), "rh_pct": (20, 50)}
_DRIFT_TOTAL = {"temp_c": (2, 5), "pressure_hpa": (2, 6), "rh_pct": (8, 20)}
_NOISE_SIGMA_MULT = (1.5, 4.0)  # multiplier of the variable's hourly std
_RAILS = {
    "temp_c": [-40, 85],
    "pressure_hpa": [1100],
    "rh_pct": [0, 100],
}

FAULT_TYPES = ["SPIKE", "FROZEN", "DRIFT", "DROPOUT", "NOISE", "OUT_OF_RANGE"]
VARIABLES = ["temp_c", "pressure_hpa", "rh_pct"]
LABEL_COLS = [f"label_{v}" for v in VARIABLES]


# ===================================================================
# Helpers
# ===================================================================

def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add label_* = "NONE" and event_id = "" if missing."""
    df = df.copy()
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = "NONE"
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
                  rng: np.random.Generator, magnitude: float | None) -> pd.DataFrame:
    lo, hi = _SPIKE_MAG[var]
    mag = magnitude if magnitude is not None else rng.uniform(lo, hi)
    sign = rng.choice([-1, 1])
    df.loc[idx, var] = df.loc[idx, var].astype(float) + sign * mag
    return df


def _inject_frozen(df: pd.DataFrame, var: str, idx: pd.Index,
                   _rng, _mag) -> pd.DataFrame:
    if len(idx) == 0:
        return df
    frozen_val = df.loc[idx[0], var]
    df.loc[idx, var] = frozen_val
    return df


def _inject_drift(df: pd.DataFrame, var: str, idx: pd.Index,
                  rng: np.random.Generator, magnitude: float | None) -> pd.DataFrame:
    lo, hi = _DRIFT_TOTAL[var]
    total = magnitude if magnitude is not None else rng.uniform(lo, hi)
    sign = rng.choice([-1, 1])
    n = len(idx)
    if n == 0:
        return df
    ramp = np.linspace(0, sign * total, n)
    df.loc[idx, var] += ramp
    return df


def _inject_dropout(df: pd.DataFrame, var: str, idx: pd.Index,
                    rng: np.random.Generator, _mag) -> pd.DataFrame:
    # randomly pick NaN, -9999, or 0 for the dropout
    mode = rng.choice(["nan", "sentinel", "zero"])
    if mode == "nan":
        df.loc[idx, var] = np.nan
    elif mode == "sentinel":
        df.loc[idx, var] = -9999.0
    else:
        df.loc[idx, var] = 0.0
    return df


def _inject_noise(df: pd.DataFrame, var: str, idx: pd.Index,
                  rng: np.random.Generator, magnitude: float | None) -> pd.DataFrame:
    base_std = df[var].std()
    mult = magnitude if magnitude is not None else rng.uniform(*_NOISE_SIGMA_MULT)
    sigma = mult * base_std
    noise = rng.normal(0, sigma, size=len(idx))
    df.loc[idx, var] += noise
    return df


def _inject_oor(df: pd.DataFrame, var: str, idx: pd.Index,
                rng: np.random.Generator, _mag) -> pd.DataFrame:
    rail = rng.choice(_RAILS[var])
    df.loc[idx, var] = float(rail)
    return df


_INJECTORS = {
    "SPIKE": _inject_spike,
    "FROZEN": _inject_frozen,
    "DRIFT": _inject_drift,
    "DROPOUT": _inject_dropout,
    "NOISE": _inject_noise,
    "OUT_OF_RANGE": _inject_oor,
}

# Default duration ranges per fault type (hours)
_DURATION_RANGE = {
    "SPIKE": (1, 2),
    "FROZEN": (6, 48),
    "DRIFT": (72, 240),      # 3-10 days
    "DROPOUT": (1, 12),
    "NOISE": (6, 24),
    "OUT_OF_RANGE": (6, 48),
}


# ===================================================================
# Public API
# ===================================================================

def inject_fault(df: pd.DataFrame, station_id: str, var: str, fault_type: str,
                 t0: pd.Timestamp, duration_h: int, magnitude: float | None = None,
                 seed: int = 42) -> pd.DataFrame:
    """Inject one fault into one variable of one station, starting at t0."""
    if fault_type not in _INJECTORS:
        raise ValueError(f"Unknown fault_type {fault_type!r}. Choose from {FAULT_TYPES}")

    df = add_label_columns(df)
    rng = np.random.default_rng(seed)
    label_col = f"label_{var}"

    sel = _mask(df, station_id, t0, duration_h)
    idx = df.index[sel]

    if len(idx) == 0:
        return df

    df = _INJECTORS[fault_type](df, var, idx, rng, magnitude)

    # clip RH to [0, 100]
    if var == "rh_pct" and fault_type not in ("DROPOUT", "OUT_OF_RANGE"):
        df.loc[idx, var] = df.loc[idx, var].clip(0, 100)

    # set labels
    df.loc[idx, label_col] = fault_type

    return df


def inject_storm(df: pd.DataFrame, stations: list[dict], center_id: str,
                 t0: pd.Timestamp, radius_km: float = 80, strength: float = 1.0,
                 seed: int = 42) -> pd.DataFrame:
    """Inject a coherent thunderstorm across neighbouring stations.

    Storm physics (scaled by *strength*):
      - Temperature drops 4-10 C over 1-2 hours, then relaxes over ~6 h
      - RH rises 15-35 %
      - Pressure rises 1-4 hPa
      - Onset delayed by distance / 30 km/h for each station
    Labels stay NONE (genuine event).  event_id = "STORM_<n>".
    """
    df = add_label_columns(df)
    rng = np.random.default_rng(seed)

    # pick random storm magnitudes
    t_drop = rng.uniform(4, 10) * strength
    rh_rise = rng.uniform(15, 35) * strength
    p_rise = rng.uniform(1, 4) * strength
    onset_h = rng.integers(1, 3)  # 1 or 2 hours for the drop
    relax_h = 6

    # unique storm id
    existing = df["event_id"].str.startswith("STORM_").sum()
    storm_id = f"STORM_{existing + 1:03d}"

    center = next(s for s in stations if s["station_id"] == center_id)

    for stn in stations:
        sid = stn["station_id"]
        dist = haversine_km(center["lat"], center["lon"], stn["lat"], stn["lon"])
        if dist > radius_km:
            continue

        # delay by distance / 30 km/h, rounded to nearest hour
        delay_h = int(round(dist / 30.0))
        local_t0 = t0 + pd.Timedelta(hours=delay_h)

        # build the storm profile: drop over onset_h, then relax over relax_h
        total_h = onset_h + relax_h
        profile_t = np.arange(total_h)

        # temperature: drop then exponential relaxation
        t_profile = np.zeros(total_h)
        for i in range(onset_h):
            t_profile[i] = -t_drop * (i + 1) / onset_h
        for i in range(relax_h):
            t_profile[onset_h + i] = -t_drop * np.exp(-i / 3.0)

        # RH: rise then relax (mirror of temperature)
        rh_profile = np.zeros(total_h)
        for i in range(onset_h):
            rh_profile[i] = rh_rise * (i + 1) / onset_h
        for i in range(relax_h):
            rh_profile[onset_h + i] = rh_rise * np.exp(-i / 3.0)

        # pressure: rise then relax
        p_profile = np.zeros(total_h)
        for i in range(onset_h):
            p_profile[i] = p_rise * (i + 1) / onset_h
        for i in range(relax_h):
            p_profile[onset_h + i] = p_rise * np.exp(-i / 3.0)

        # apply to the dataframe
        for step in range(total_h):
            ts = local_t0 + pd.Timedelta(hours=step)
            row_mask = (df["station_id"] == sid) & (df["ts"] == ts)
            if row_mask.any():
                df.loc[row_mask, "temp_c"] = df.loc[row_mask, "temp_c"].astype(float) + t_profile[step]
                df.loc[row_mask, "rh_pct"] = (df.loc[row_mask, "rh_pct"].astype(float) + rh_profile[step]).clip(0, 100)
                df.loc[row_mask, "pressure_hpa"] = df.loc[row_mask, "pressure_hpa"].astype(float) + p_profile[step]
                df.loc[row_mask, "event_id"] = storm_id

    return df


def build_test_set(df: pd.DataFrame, stations: list[dict] | None = None,
                   seed: int = 42) -> pd.DataFrame:
    """~40 faults of each type + ~10 storms, non-overlapping, on the given data slice.

    Returns the modified df with label columns set.
    """
    if stations is None:
        from skyguard.data.io import load_stations
        stations = load_stations()
    df = add_label_columns(df)
    rng = np.random.default_rng(seed)

    station_ids = sorted(df["station_id"].unique())
    all_ts = sorted(df["ts"].unique())
    n_ts = len(all_ts)

    # --- Track occupied time slots per (station_id, var) to avoid overlaps ---
    # Each entry: set of hour indices that are already used
    occupied: dict[tuple[str, str], set[int]] = {
        (sid, var): set() for sid in station_ids for var in VARIABLES
    }


    from typing import Optional
    def _find_free_slot(sid: str, var: str, dur_h: int, attempts: int = 200) -> Optional[int]:
        """Find a random starting index where dur_h consecutive hours are free."""
        occ = occupied[(sid, var)]
        for _ in range(attempts):
            start = rng.integers(0, max(1, n_ts - dur_h))
            span = set(range(start, start + dur_h))
            if not span & occ:
                return int(start)
        return None

    def _mark_occupied(sid: str, var: str, start_idx: int, dur_h: int):
        # add a small buffer around faults to avoid adjacent injection
        buf = 6
        occupied[(sid, var)].update(range(max(0, start_idx - buf),
                                         min(n_ts, start_idx + dur_h + buf)))

    injected_count: dict[str, int] = {ft: 0 for ft in FAULT_TYPES}
    fault_seed = int(rng.integers(0, 2**31))

    # --- Inject ~40 of each fault type ---
    for fault_type in FAULT_TYPES:
        target = 40
        lo_dur, hi_dur = _DURATION_RANGE[fault_type]

        for _ in range(target):
            sid = rng.choice(station_ids)
            var = rng.choice(VARIABLES)
            dur = int(rng.integers(lo_dur, hi_dur + 1))

            slot = _find_free_slot(sid, var, dur)
            if slot is None:
                continue

            t0 = pd.Timestamp(all_ts[slot])
            fault_seed += 1
            df = inject_fault(df, sid, var, fault_type, t0, dur, seed=fault_seed)
            _mark_occupied(sid, var, slot, dur)
            injected_count[fault_type] += 1

    # --- Inject ~10 storms ---
    # Storms affect all variables across multiple stations, so mark broadly
    storm_count = 0
    storm_seed = int(rng.integers(0, 2**31))
    for _ in range(10):
        center_id = rng.choice(station_ids)
        # find a free-ish slot (check temp_c on the center station, 8 hours wide)
        slot = _find_free_slot(center_id, "temp_c", 8)
        if slot is None:
            continue

        t0 = pd.Timestamp(all_ts[slot])
        storm_strength = rng.uniform(0.7, 1.3)
        storm_seed += 1
        df = inject_storm(df, stations, center_id, t0,
                          radius_km=80, strength=storm_strength, seed=storm_seed)

        # mark the storm hours as occupied for ALL stations and variables
        for sid in station_ids:
            for var in VARIABLES:
                _mark_occupied(sid, var, slot, 10)

        storm_count += 1

    print(f"Injected faults: {injected_count}")
    print(f"Injected storms: {storm_count}")

    return df
