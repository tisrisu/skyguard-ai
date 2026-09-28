"""Corrected value for a faulty reading.

Owner: M1

Two estimates, blended with the weights in config.yaml -> impute:
  spatial   climatology mean for this station + median anomaly of the neighbours
  temporal  straight-line extrapolation from the last two valid readings
Readings that are NaN, sentinels or outside physics.range are ignored.
"""

import numpy as np
import pandas as pd

from skyguard.config import load_config


def impute(history: pd.DataFrame, neighbours_now: pd.DataFrame, clim, station_id: str,
           ts: pd.Timestamp, var: str) -> float:
    """Best estimate of what the sensor should have read at ts.

    history         this station's rows before ts (faulty rows already excluded if possible)
    neighbours_now  neighbour rows around ts (± spatial.time_window_h)
    """
    cfg = load_config()
    lo, hi = cfg["physics"]["range"][var]
    expected_mean, _ = clim.expected(station_id, ts, var)

    spatial = np.nan
    nearby = neighbours_now[neighbours_now[var].between(lo, hi)]
    if pd.notna(expected_mean) and not nearby.empty:
        anomalies = [
            value - clim.expected(sid, t, var)[0]
            for sid, t, value in nearby[["station_id", "ts", var]].itertuples(index=False)
        ]
        anomalies = [a for a in anomalies if pd.notna(a)]
        if anomalies:
            spatial = expected_mean + float(np.median(anomalies))

    temporal = np.nan
    good = history.loc[history[var].between(lo, hi), var] if not history.empty else pd.Series(dtype=float)
    if len(good) >= 2:
        temporal = 2 * good.iloc[-1] - good.iloc[-2]
    elif len(good) == 1:
        temporal = good.iloc[-1]

    weights = cfg["impute"]
    if pd.notna(spatial) and pd.notna(temporal):
        w_s, w_t = weights["spatial_weight"], weights["temporal_weight"]
        return float((w_s * spatial + w_t * temporal) / (w_s + w_t))
    for estimate in (spatial, temporal, expected_mean):
        if pd.notna(estimate):
            return float(estimate)
    return float("nan")
