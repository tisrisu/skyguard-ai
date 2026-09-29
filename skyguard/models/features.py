"""Features for one variable at the latest timestamp of a station's history.

Owner: M2

The same function is used for training (applied row by row, or a vectorised
version over a whole DataFrame) and for live replay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skyguard.physics.dewpoint import dewpoint_c

# Suggested starting set; adjust, but keep names and plain-English labels in sync.
FEATURE_NAMES = [
    "value",
    "delta_1h",
    "delta_3h",
    "roll_std_6h",
    "roll_std_24h",
    "dev_sigma",          # (value - climatology mean) / climatology std
    "same_value_run",     # consecutive identical readings
    "dewpoint",
    "dewpoint_delta_1h",
    "hour_sin",
    "hour_cos",
]

# Used by explain() to turn SHAP feature names into readable reasons.
FEATURE_LABELS = {
    "value": "reported value",
    "delta_1h": "sudden 1-hour change",
    "delta_3h": "3-hour change",
    "roll_std_6h": "short-term variability",
    "roll_std_24h": "daily variability",
    "dev_sigma": "distance from normal for this hour and season",
    "same_value_run": "identical readings in a row",
    "dewpoint": "implied dewpoint",
    "dewpoint_delta_1h": "change in moisture content",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _same_value_run_length(series: pd.Series) -> int:
    """Count how many consecutive identical values end at the last row.

    E.g. [30, 30, 31, 31, 31] → 3 (three 31s in a row at the end).
    Returns 1 when the series has only one element (one reading = run of 1).
    """
    if len(series) == 0:
        return 0
    vals = series.values
    last = vals[-1]
    if np.isnan(last):
        return 1
    run = 1
    for i in range(len(vals) - 2, -1, -1):
        if vals[i] == last:
            run += 1
        else:
            break
    return run


def _same_value_run_vectorised(series: pd.Series) -> pd.Series:
    """Vectorised same-value run length for a station-sorted series.

    For each row, counts how many immediately preceding rows (inclusive) share
    the same value.  E.g. [30, 30, 31, 31, 31] → [1, 2, 1, 2, 3].
    """
    vals = series.values
    n = len(vals)
    runs = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        if vals[i] == vals[i - 1] and not np.isnan(vals[i]):
            runs[i] = runs[i - 1] + 1
    return pd.Series(runs, index=series.index)


# ---------------------------------------------------------------------------
# Single-row extraction  (used during live replay by Engine.step)
# ---------------------------------------------------------------------------

def make_features(station_hist: pd.DataFrame, clim, var: str) -> np.ndarray:
    """Feature vector ``(len(FEATURE_NAMES),)`` for the **last** row of *station_hist*.

    Parameters
    ----------
    station_hist : pd.DataFrame
        One station's rows up to and including the current *ts*, **sorted by ts**.
        Must contain columns: ``ts``, ``temp_c``, ``pressure_hpa``, ``rh_pct``.
    clim : Climatology
        Fitted ``Climatology`` object (from ``skyguard.data.climatology``).
    var : str
        The target variable name (one of ``schemas.VARIABLES``).

    Returns
    -------
    np.ndarray of shape ``(11,)`` — one entry per ``FEATURE_NAMES``.
    All NaNs are replaced with 0.0 so the downstream IsolationForest never
    sees missing data.
    """
    col = station_hist[var]
    last_idx = station_hist.index[-1]
    current_val = float(col.iloc[-1]) if not pd.isna(col.iloc[-1]) else 0.0

    # --- value ---
    value = current_val

    # --- delta_1h ---
    delta_1h = float(col.diff(1).iloc[-1]) if len(col) >= 2 else 0.0

    # --- delta_3h ---
    delta_3h = float(col.diff(3).iloc[-1]) if len(col) >= 4 else 0.0

    # --- rolling std (6h, 24h) ---
    roll_std_6h = float(col.rolling(6, min_periods=2).std().iloc[-1]) if len(col) >= 2 else 0.0
    roll_std_24h = float(col.rolling(24, min_periods=2).std().iloc[-1]) if len(col) >= 2 else 0.0

    # --- deviation from climatology in sigma units ---
    last_row = station_hist.iloc[-1]
    ts = pd.Timestamp(last_row["ts"])
    station_id = str(last_row["station_id"])
    mean, std = clim.expected(station_id, ts, var)
    if np.isnan(mean) or np.isnan(std) or std == 0:
        dev_sigma = 0.0
    else:
        dev_sigma = (current_val - mean) / std

    # --- same_value_run ---
    same_value_run = float(_same_value_run_length(col))

    # --- dewpoint and dewpoint_delta_1h ---
    temp = station_hist["temp_c"]
    rh = station_hist["rh_pct"]
    dp_series = dewpoint_c(temp.values, rh.values)
    dp_current = float(dp_series[-1]) if not np.isnan(dp_series[-1]) else 0.0
    if len(dp_series) >= 2 and not np.isnan(dp_series[-2]):
        dp_delta = dp_current - float(dp_series[-2])
    else:
        dp_delta = 0.0

    # --- hour sin/cos (encodes time-of-day cyclically) ---
    hour = ts.hour + ts.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    # --- assemble and NaN-fill ---
    feat = np.array([
        value, delta_1h, delta_3h,
        roll_std_6h, roll_std_24h,
        dev_sigma, same_value_run,
        dp_current, dp_delta,
        hour_sin, hour_cos,
    ], dtype=np.float64)

    np.nan_to_num(feat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


# ---------------------------------------------------------------------------
# Vectorised extraction  (used for training and batch evaluation)
# ---------------------------------------------------------------------------

def make_feature_frame(df: pd.DataFrame, clim, var: str) -> pd.DataFrame:
    """Vectorised feature extraction over a whole long-format DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data (multiple stations) with columns ``station_id``,
        ``ts``, ``temp_c``, ``pressure_hpa``, ``rh_pct``.  Must be sorted by
        ``[station_id, ts]``.
    clim : Climatology
        Fitted ``Climatology`` object.
    var : str
        Target variable.

    Returns
    -------
    pd.DataFrame with columns matching ``FEATURE_NAMES`` and the same index as *df*.
    """
    out = pd.DataFrame(index=df.index)

    grouped = df.groupby("station_id", sort=False)
    col = df[var]

    # --- value ---
    out["value"] = col

    # --- delta_1h, delta_3h ---
    out["delta_1h"] = grouped[var].diff(1)
    out["delta_3h"] = grouped[var].diff(3)

    # --- rolling std (per station) ---
    out["roll_std_6h"] = grouped[var].transform(
        lambda s: s.rolling(6, min_periods=2).std()
    )
    out["roll_std_24h"] = grouped[var].transform(
        lambda s: s.rolling(24, min_periods=2).std()
    )

    # --- deviation from climatology ---
    out["dev_sigma"] = clim.deviation_sigma(df, var)

    # --- same_value_run (per station) ---
    out["same_value_run"] = grouped[var].transform(_same_value_run_vectorised)

    # --- dewpoint and dewpoint_delta_1h ---
    dp = dewpoint_c(df["temp_c"].values, df["rh_pct"].values)
    out["dewpoint"] = dp
    out["dewpoint_delta_1h"] = pd.Series(dp, index=df.index).groupby(
        df["station_id"], sort=False
    ).diff(1)

    # --- hour sin/cos ---
    ts = pd.to_datetime(df["ts"], utc=True)
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)

    # --- NaN → 0.0 (safety net) ---
    out = out[FEATURE_NAMES].fillna(0.0)

    return out
