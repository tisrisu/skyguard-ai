"""Neighbour/spatial consistency check.

Owner: M3
"""

from __future__ import annotations

import math
import pandas as pd
import numpy as np

from skyguard.config import load_config
from skyguard.geo import neighbours


def spatial_check(df_window: pd.DataFrame, stations: list[dict], clim, station_id: str,
                  ts: pd.Timestamp, var: str, cfg: dict | None = None) -> dict:
    """Compare climatology-normalised anomaly against nearby stations.

    The configured ±time_window_h is searched and the offset with the smallest
    absolute robust z-score is retained.
    """
    cfg = cfg or load_config()
    scfg = cfg["spatial"]
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    near = neighbours(stations, station_id, scfg["radius_km"])
    ids = {s["station_id"] for s in near}
    if not ids:
        return {"z": None, "n": 0, "consistent": None, "reason": "No neighbouring stations are available"}

    window = int(scfg["time_window_h"])
    home = df_window[df_window["station_id"].eq(station_id)]
    if home.empty:
        return {"z": None, "n": 0, "consistent": None, "reason": "Current station reading is unavailable"}

    home_row = home.iloc[(pd.to_datetime(home["ts"], utc=True) - ts).abs().argmin()]
    try:
        home_mean, _ = clim.expected(station_id, ts, var)
        home_anomaly = float(home_row[var]) - home_mean
    except Exception:
        home_anomaly = float(home_row[var])

    best = None
    for offset in range(-window, window + 1):
        target = ts + pd.Timedelta(hours=offset)
        anomalies = []
        used = []
        for nid in ids:
            rows = df_window[df_window["station_id"].eq(nid)]
            if rows.empty:
                continue
            rows = rows.copy()
            d = (pd.to_datetime(rows["ts"], utc=True) - target).abs()
            i = d.argmin()
            if d.iloc[i] > pd.Timedelta(minutes=30):
                continue
            r = rows.iloc[i]
            if pd.isna(r[var]):
                continue
            try:
                mean, _ = clim.expected(nid, target, var)
                anomaly = float(r[var]) - mean
            except Exception:
                anomaly = float(r[var])
            if math.isfinite(anomaly):
                anomalies.append(anomaly)
                used.append(nid)

        if len(anomalies) < int(scfg["min_neighbours"]):
            continue

        med = float(np.median(anomalies))
        mad = float(np.median(np.abs(np.asarray(anomalies) - med)))
        floor = float(scfg["mad_floor"][var])
        denom = 1.4826 * mad + floor
        z = (home_anomaly - med) / denom
        candidate = (abs(z), z, len(anomalies), offset, used)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        return {
            "z": None, "n": 0, "consistent": None,
            "reason": f"Fewer than {scfg['min_neighbours']} neighbouring stations have usable {var} data",
        }

    _, z, n, offset, used = best
    threshold = float(scfg["z_threshold"])
    consistent = abs(z) <= threshold
    direction = "agree with" if consistent else "disagree with"
    timing = "at the same hour" if offset == 0 else f"within ±{window} h"
    reason = (
        f"{n} neighbouring stations {direction} the station's {var.replace('_', ' ')} "
        f"{timing} (z = {z:.1f})"
    )
    return {"z": float(z), "n": int(n), "consistent": bool(consistent), "reason": reason}
